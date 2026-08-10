use crate::Error;
use crate::metal::step_kernel::{N_LAYERS, StepRuntime, build_step_runtime};
use crate::metal::step_kv::MonolithicEncoderCache;
use crate::sample::Rng;
use crate::tokenizer::Tokenizer;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use super::config::{StepGenerateConfig, smoke_config};
use crate::flags::progress_enabled;

/// Longest common prefix length (in tokens) of two id slices.
pub(super) fn longest_common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Whether shortening KV from `old_len` to `new_len` must re-prefill the kept
/// prefix. After a sliding ring wraps, slots for early absolute positions hold
/// post-wrap K/V; clamping `kv_len` alone would leave them stale (serve
/// finalize after a long aborted write → next-turn alpha soup).
///
/// A sliding layer stores position `p` at slot `p & (ring-1)`, so the ring only
/// ever holds the last `ring` **written** positions. Two subtleties make this
/// exact, and getting either wrong is a corruption or a perf bug:
///
/// 1. **Written ≠ committed.** Denoise writes its canvas at `[kv_len,
///    kv_len+CANVAS)` unconditionally (`kv_write_end = u32::MAX`), even for a
///    block that is never committed — so an aborted long generation leaves
///    post-wrap K/V in the early slots while `old_len` still sits below the ring
///    size. Testing `old_len > ring` (the highest *committed* position) misses
///    every `old_len ∈ (ring-CANVAS, ring]`, which is a live corruption.
/// 2. **Only the last window matters.** After truncating to `new_len`, the
///    deepest position any later query can read is `new_len - (window-1)` (the
///    first re-issued query sits at `new_len`). Truncations shallower than that
///    need no rebuild at *any* context length — rebuilding them anyway costs a
///    full re-prefill of the whole conversation for a 1-token rewind.
///
/// So: rebuild iff the deepest position we will still read has already had its
/// slot reused by a higher one. The `saturating_sub`s are load-bearing — they
/// are what keeps a not-yet-wrapped ring (incl. `DGQ_KV_RING_UNCAPPED` and any
/// `max_seq <= ring`, where the ring provably never wraps) off the rebuild path.
pub(crate) fn kv_truncate_needs_ring_rebuild(
    old_len: usize,
    new_len: usize,
    ring: Option<usize>,
    window: usize,
) -> bool {
    let Some(ring) = ring else {
        return false; // all-linear layers: slots are never aliased.
    };
    if new_len >= old_len {
        return false;
    }
    let written_end = old_len + crate::metal::CANVAS;
    let oldest_live = written_end.saturating_sub(ring);
    let deepest_needed = new_len.saturating_sub(window.saturating_sub(1));
    deepest_needed < oldest_live
}

/// A saved KV position captured by [`StepGenerateSession::checkpoint`] and
/// restored by [`StepGenerateSession::rollback_to`]. Plain values (no borrow of
/// the session), so a conversation manager can hold one across turns.
///
/// Parked KV-rewind scaffolding: the checkpoint/rollback pair is the primitive
/// the conversation manager would use to drop `thought`-channel reasoning from
/// persisted context, but no production caller is wired to it yet. Kept (rather
/// than deleted) because it is the tested, ring-wrap-correct primitive for that
/// path; see the `checkpoint`/`rollback_to` docs below.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCheckpoint {
    kv_len: u32,
    tokens: usize,
}

/// A saved conversation's KV: the raw KV-cache bytes plus the causal token log
/// they correspond to. Produced by [`StepGenerateSession::snapshot_kv`] and
/// loaded by [`StepGenerateSession::restore_kv`]. The conversation manager keeps
/// these in an LRU pool (RAM in v1; SSD in v2) to swap conversations through the
/// single hot GPU buffer. `kv_bytes.len()` is the pool cost.
pub struct KvSnapshot {
    pub(super) kv_bytes: Vec<u8>,
    pub(super) tokens: Vec<u32>,
}

impl KvSnapshot {
    /// Bytes this snapshot occupies in the pool (the KV buffer copy).
    pub fn byte_len(&self) -> usize {
        self.kv_bytes.len()
    }

    /// The causal token log this snapshot represents.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// The raw KV bytes (for spilling to the SSD tier). The token log is stored
    /// separately by the manager, so only these bytes need to hit disk.
    pub fn kv_bytes(&self) -> &[u8] {
        &self.kv_bytes
    }

    /// Reconstruct a snapshot from SSD bytes + the conversation's token log.
    pub fn from_parts(kv_bytes: Vec<u8>, tokens: Vec<u32>) -> Self {
        Self { kv_bytes, tokens }
    }

    /// Test-only: a snapshot with a recorded byte cost but no real KV, so the
    /// conversation manager's routing/LRU/accounting can be exercised without a
    /// GPU session.
    #[cfg(test)]
    pub fn for_test(byte_len: usize) -> Self {
        Self {
            kv_bytes: vec![0u8; byte_len],
            tokens: Vec::new(),
        }
    }
}

/// Reusable monolithic runtime across prompts (M4.3).
pub struct StepGenerateSession {
    pub(super) rt: StepRuntime,
    pub(super) model_dir: PathBuf,
    pub(super) layers: usize,
    pub(super) encoder: Option<MonolithicEncoderCache>,
    pub(super) step_text_tokenizer: Option<Tokenizer>,
    /// Token sequence whose *causal* KV is currently valid in `rt` (== the KV
    /// length). Cross-turn reuse prefills only the delta past the common prefix
    /// of this and the next prompt. Cleared by `reset_kv`.
    pub(super) kv_valid_tokens: Vec<u32>,
}

impl StepGenerateSession {
    pub fn open(
        model_dir: &Path,
        cfg: &StepGenerateConfig,
        prefill_token_ids: Option<Vec<u32>>,
    ) -> Result<(Self, Duration), Error> {
        let layers = cfg.layers.clamp(1, N_LAYERS);
        let (rt, build) =
            build_step_runtime(model_dir, &smoke_config(cfg, prefill_token_ids.clone()))?;
        if progress_enabled() {
            eprintln!(
                "step-generate: runtime ready (total={:.2?}, compile={:.2?})",
                build.total, build.compile
            );
        }
        Ok((
            Self {
                rt,
                model_dir: model_dir.to_path_buf(),
                layers,
                encoder: None,
                step_text_tokenizer: None,
                // The open-time prefill is causal KV for exactly these tokens
                // (build errors on a kv_len mismatch), so the log records them
                // for begin_turn's reuse guard to verify against.
                kv_valid_tokens: prefill_token_ids.unwrap_or_default(),
            },
            build.compile,
        ))
    }

    /// Drop the prefilled KV so the next `generate_with_session` re-prefills from
    /// scratch. Use for *independent* prompts (smoketest); chat relies on the
    /// cross-turn KV-reuse continuation path instead.
    pub fn reset_kv(&mut self) {
        self.rt.set_kv_len(0);
        self.kv_valid_tokens.clear();
    }

    /// A saved KV position (`checkpoint`) that `rollback_to` can return to. It
    /// pins both the runtime `kv_len` and the length of the causal token log so
    /// the two stay consistent. Values only — safe to hold across turns.
    ///
    /// The conversation manager uses this to keep ephemeral content — e.g. a
    /// turn's `thought`-channel reasoning — OUT of the persisted context:
    /// checkpoint after the user message, generate, roll back, then extend with
    /// only the sanitized answer.
    ///
    /// When the sequence never wrapped a sliding ring, rollback is O(1)
    /// (`kv_len` clamp; bytes past the new length are unread). After a ring
    /// wrap, rollback must re-prefill the kept prefix — early ring slots were
    /// overwritten by absolute positions `≥ ring` and truncating `kv_len` alone
    /// would leave those positions with the wrong K/V (serve finalize repro:
    /// aborted long write → truncate → next turn alpha-soup).
    #[allow(dead_code)] // parked: see KvCheckpoint
    pub fn checkpoint(&self) -> KvCheckpoint {
        KvCheckpoint {
            kv_len: self.rt.read_params().kv_len,
            tokens: self.kv_valid_tokens.len(),
        }
    }

    /// Sliding-ring slot count (power-of-two window), or `None` if every layer
    /// is linear (no wrap possible).
    fn sliding_ring_len(&self) -> Option<usize> {
        self.rt
            .layout()
            .layers
            .iter()
            .find_map(|l| (l.kv_ring_mask != 0).then_some(l.kv_ring_mask as usize + 1))
    }

    /// Rebuild causal positions `[0, tokens)` by reset + prefill. Used when a
    /// ring wrap has invalidated the kept prefix's sliding-layer slots.
    fn rebuild_kv_prefix(&mut self, tokens: &[u32]) -> Result<(), Error> {
        self.reset_kv();
        if tokens.is_empty() {
            return Ok(());
        }
        self.extend_kv(tokens)
    }

    /// Return the session's KV to a previously captured `checkpoint`, discarding
    /// everything causally appended since (see [`checkpoint`](Self::checkpoint)).
    /// The checkpoint must not be ahead of the current KV — a stale checkpoint
    /// from a longer past state is ignored (clamped) rather than trusted.
    #[allow(dead_code)] // parked: see KvCheckpoint
    pub fn rollback_to(&mut self, cp: &KvCheckpoint) -> Result<(), Error> {
        let old = self.kv_valid_tokens.len();
        let tokens = cp.tokens.min(old);
        if kv_truncate_needs_ring_rebuild(
            old,
            tokens,
            self.sliding_ring_len(),
            self.rt.sliding_window(),
        ) {
            let kept = self.kv_valid_tokens[..tokens].to_vec();
            return self.rebuild_kv_prefix(&kept);
        }
        self.rt
            .set_kv_len(cp.kv_len.min(self.kv_valid_tokens.len() as u32));
        self.kv_valid_tokens.truncate(tokens);
        Ok(())
    }

    /// Truncate the resident KV to its first `n_tokens` causal positions.
    /// Twin of `rollback_to` for the conversation manager's turn finalize:
    /// roll back to the reuse point, then `extend_kv` the canonical
    /// (thought-free) tail.
    ///
    /// If `n_tokens < len` after the sequence wrapped a sliding ring, rebuilds
    /// the kept prefix via re-prefill (see [`checkpoint`](Self::checkpoint)).
    /// Otherwise O(1): bytes past `n_tokens` are left stale but unread.
    /// Would [`truncate_kv_to`](Self::truncate_kv_to)`(n_tokens)` pay a ring
    /// rebuild (deep rewind past the O(1) slack)? Rebuild cost ≈ a fresh
    /// prefill of the kept prefix, so callers implementing the
    /// no-rebuild-to-salvage policy check this and drop/re-prefill instead.
    pub fn truncate_needs_rebuild(&self, n_tokens: usize) -> bool {
        let old = self.kv_valid_tokens.len();
        let n = n_tokens.min(old);
        n < old
            && kv_truncate_needs_ring_rebuild(
                old,
                n,
                self.sliding_ring_len(),
                self.rt.sliding_window(),
            )
    }

    pub fn truncate_kv_to(&mut self, n_tokens: usize) -> Result<(), Error> {
        let old = self.kv_valid_tokens.len();
        let n = n_tokens.min(old);
        if n == old {
            return Ok(());
        }
        if kv_truncate_needs_ring_rebuild(old, n, self.sliding_ring_len(), self.rt.sliding_window())
        {
            let kept = self.kv_valid_tokens[..n].to_vec();
            let rebuild_started = Instant::now();
            let result = self.rebuild_kv_prefix(&kept);
            // Unconditional: rebuilds are rare, cost seconds, and the line is
            // the only visibility into them once serve runs quiet.
            eprintln!(
                "truncate_kv_to: ring rebuild {old} -> {n} tokens (ring={:?}) ({:.2?})",
                self.sliding_ring_len(),
                rebuild_started.elapsed()
            );
            return result;
        }
        self.rt.set_kv_len(n as u32);
        self.kv_valid_tokens.truncate(n);
        Ok(())
    }

    /// The most causal tokens the KV can hold while leaving room for one canvas
    /// block (every denoise writes at `[kv_len, kv_len+CANVAS)`, and `set_kv_len`
    /// asserts that headroom). Callers that extend the KV with variable-length
    /// content (turn finalize, tool-response extensions) must stay within this.
    pub fn extend_capacity(&self) -> usize {
        self.rt.max_seq().saturating_sub(crate::metal::CANVAS)
    }

    /// Causally prefill `tokens` onto the end of the current KV (no denoise),
    /// extending `kv_valid_tokens`. Used by the conversation manager to finalize
    /// a turn: after `rollback_to` a checkpoint, extend with only the sanitized
    /// answer so the persisted KV is the thought-free canonical continuation.
    /// (Also puts the final answer block — normally bidirectional and excluded —
    /// into the reusable causal KV immediately.) Total: an extension past
    /// [`extend_capacity`](Self::extend_capacity) returns a typed error (no
    /// partial write) instead of tripping the `set_kv_len` overflow assert —
    /// reachable from user input via a near-context-full turn finalize.
    pub fn extend_kv(&mut self, tokens: &[u32]) -> Result<(), Error> {
        if tokens.is_empty() {
            return Ok(());
        }
        let offset = self.kv_valid_tokens.len();
        if offset + tokens.len() > self.extend_capacity() {
            return Err(Error::Format(
                "KV extend exceeds capacity (tokens + CANVAS headroom > max_seq)",
            ));
        }
        self.rt.prefill_chunks_from(offset, tokens)?;
        self.kv_valid_tokens.extend_from_slice(tokens);
        Ok(())
    }

    /// Replace the session's state with a deterministic pseudorandom KV
    /// declared to be `n_tokens` long (see `StepRuntime::synthetic_fill_kv`).
    /// The synthetic causal token log is seed-derived (ids well inside vocab,
    /// so an accidental ring rebuild embeds valid — if meaningless — tokens
    /// rather than tripping asserts; note a rebuild REPLACES the synthetic
    /// bytes with real KV, so byte-consistency tests must stay inside the
    /// O(1)-truncate slack). Test infrastructure only.
    pub fn synthetic_fill_kv(&mut self, n_tokens: usize, seed: u64) -> Result<(), Error> {
        if n_tokens + crate::metal::CANVAS > self.rt.max_seq() {
            return Err(Error::Format(
                "synthetic fill exceeds max_seq - CANVAS headroom",
            ));
        }
        self.rt.synthetic_fill_kv(n_tokens, seed);
        let mut s = seed ^ 0x9E37_79B9_7F4A_7C15 | 1;
        self.kv_valid_tokens = (0..n_tokens)
            .map(|_| {
                s ^= s >> 12;
                s ^= s << 25;
                s ^= s >> 27;
                ((s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) % 200_000) as u32
            })
            .collect();
        Ok(())
    }

    /// FNV-1a over the session's READABLE state: the causal token log plus the
    /// live KV (`StepRuntime::live_kv_fnv` — linear layers whole prefix, ring
    /// layers window-only). The token pipeline's rewind/extension gates probe
    /// this; it is invariant across ring-slot residue that no future read can
    /// observe.
    pub fn live_kv_fingerprint(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &t in &self.kv_valid_tokens {
            for b in t.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        self.rt.live_kv_fnv(self.kv_valid_tokens.len(), h)
    }

    /// Full snapshot of the session's KV state (buffer bytes + causal token log),
    /// for saving a conversation out of the single hot buffer. Restore with
    /// [`restore_kv`](Self::restore_kv). See [`KvSnapshot`].
    pub fn snapshot_kv(&self) -> KvSnapshot {
        KvSnapshot {
            kv_bytes: self.rt.snapshot_kv(self.kv_valid_tokens.len()),
            tokens: self.kv_valid_tokens.clone(),
        }
    }

    /// Load a conversation's KV back into the hot buffer, replacing whatever was
    /// resident. After this the session continues that conversation as if it had
    /// never been swapped out.
    pub fn restore_kv(&mut self, snap: &KvSnapshot) {
        self.rt.restore_kv(snap.tokens.len(), &snap.kv_bytes);
        self.rt.set_kv_len(snap.tokens.len() as u32);
        self.kv_valid_tokens = snap.tokens.clone();
    }

    /// The causal token sequence currently resident in KV (== `kv_len`). Lets the
    /// conversation manager route by longest-common-prefix without re-tokenizing.
    pub fn kv_valid_tokens(&self) -> &[u32] {
        &self.kv_valid_tokens
    }

    /// TEST-ONLY: zero the monolithic KV + f32 side ring and reset the
    /// side-ring cursor (see `StepRuntime::debug_scrub_kv`). Lineage probes
    /// use this to rule cross-build residue in or out.
    #[cfg(test)]
    pub(crate) fn scrub_kv_for_test(&mut self) {
        self.reset_kv();
        self.rt.debug_scrub_kv();
    }

    /// TEST-ONLY: selective scrub (see `StepRuntime::debug_scrub_kv_parts`).
    #[cfg(test)]
    #[allow(dead_code)] // diagnostic probe kept alongside scrub_kv_for_test
    pub(crate) fn scrub_kv_parts_for_test(&mut self, monolithic: bool, side: bool) {
        self.reset_kv();
        self.rt.debug_scrub_kv_parts(monolithic, side);
    }

    /// Direct KV buffer + layout access for oracle-style tests that MUTATE
    /// stored KV between prefill and denoise (fusion replay: rewrite aged
    /// full-layer rows, then re-enter generation on the doctored cache).
    /// Diagnostic only — production code goes through the runtime.
    #[allow(dead_code)]
    pub(crate) fn kv_buffer_for_test(
        &self,
    ) -> &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer> {
        self.rt.kvcache()
    }

    #[allow(dead_code)]
    pub(crate) fn layout_for_test(&self) -> &crate::metal::step_kernel::ModelLayout {
        self.rt.layout()
    }

    /// Make the session's KV safe to reuse for `prompt`. Cross-turn reuse assumes
    /// the cached causal KV is a *prefix* of the next prompt (append-only chat).
    /// A stateless server sees independent prompts that may diverge from — or be
    /// shorter than — the cached sequence; reusing that KV would answer from the
    /// wrong context. So: keep the KV only when the cached tokens are a genuine
    /// prefix of `prompt` (an extension → reuse prefills just the delta);
    /// otherwise drop it and re-prefill from scratch.
    #[allow(dead_code)]
    pub fn reset_kv_unless_extends(&mut self, prompt: &[u32]) {
        let kept = longest_common_prefix(&self.kv_valid_tokens, prompt);
        if kept < self.kv_valid_tokens.len() {
            self.reset_kv();
        }
    }
}

/// DIAGNOSTIC: multiply every live f16 KV value by (1 + eps*u),
/// u uniform in [-1, 1] — models the fast path's per-value computation noise
/// on top of a KNOWN-GOOD engine prefill. f16 sessions only (q8 skipped).
pub(super) fn perturb_live_kv_f16(
    rt: &mut crate::metal::step_kernel::StepRuntime,
    kv_len: usize,
    eps: f32,
    seed: u64,
) {
    use crate::shaders::f16::{f16_bits_to_f32, f32_to_f16_bits};
    use objc2_metal::MTLBuffer as _;
    if crate::flags::kv_format(rt.max_seq()) != crate::shaders::kv_quant::KvFormat::F16 {
        eprintln!("DGQ_KV_NOISE: q8 session — skipped");
        return;
    }
    let layout = *rt.layout();
    let buf = rt.kvcache();
    let mut rng = Rng::new(seed ^ 0x5eed);
    let mut n = 0u64;
    for layer in 0..crate::metal::step_kernel::N_LAYERS {
        let l = &layout.layers[layer];
        let nkv = l.n_kv_heads as usize;
        let hd = l.head_dim as usize;
        let token_stride = nkv * hd * 2;
        let slots = if l.kv_ring_mask != 0 {
            kv_len.min(l.kv_ring_mask as usize + 1)
        } else {
            kv_len
        };
        let base = l.kv_region as usize / 2;
        let ptr = unsafe { (buf.contents().as_ptr() as *mut u16).add(base) };
        for i in 0..slots * token_stride {
            let p = unsafe { ptr.add(i) };
            let v = f16_bits_to_f32(unsafe { *p });
            let u = rng.next_f32() * 2.0 - 1.0;
            unsafe { *p = f32_to_f16_bits(v * (1.0 + eps * u)) };
            n += 1;
        }
    }
    eprintln!("DGQ_KV_NOISE: perturbed {n} f16 KV values by rel eps {eps}");
}
