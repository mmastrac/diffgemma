//! M2/M4: end-to-end monolithic generate loop (prefill → denoise blocks → KV extend).

use crate::Error;
use crate::denoise_trace::{DenoiseTrace, SCHEMA_VERSION, step_trace_from_stats};
use crate::generate::GenerateOutput;
use crate::metal::step_kernel::{
    CANVAS, N_LAYERS, StepFinishMode, StepRuntime, StepSmokeConfig, VOCAB, build_step_runtime,
    denoise_parity_log_enabled, final_entropy_log_enabled, log_denoise_parity_step,
    log_final_per_token_entropy, step_params_from_sampler, step_text_log_enabled,
    trace_entropy_enabled,
};
use crate::metal::step_kv::{
    MonolithicEncoderCache, MonolithicPrefillTiming, extend_monolithic_kv_chunked,
    extend_monolithic_kv_with_cache, prefill_monolithic_kv_with_cache,
};
use crate::metal::{ForwardTelemetry, SessionTelemetry, StepPhaseTelemetry};
use crate::sample::{Rng, SamplerConfig, step_entropy_stats};
use crate::tokenizer::Tokenizer;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Per-denoise-step progress snapshot for live UIs (chat streaming/spinner).
/// `argmax` is the **active** canvas slice only (not stale rows past
/// `active_canvas` after E6 shrink-on-retry). Under the MLX-exact sampler the
/// block's final commit IS that slice, so a stable prefix is a faithful preview.
pub struct StepProgressEvent<'a> {
    /// 1-based committed-block index this step belongs to.
    pub block_idx: usize,
    /// Total blocks budgeted for this generate (`max_new_tokens / CANVAS`).
    pub max_blocks: usize,
    /// 1-based denoise step within the block.
    pub step_in_block: u32,
    pub max_steps: usize,
    /// Active-canvas argmax of this step (`active_canvas` entries).
    pub argmax: &'a [u32],
    /// Positions accepted by the entropy-bound rule this step.
    pub accept_count: u32,
    /// Mean per-position entropy (nats) this step.
    pub mean_entropy: f32,
    /// True when this block's argmax is finalized and will not be E6-retried.
    /// Mid-attempt "would-stop" steps stay false so stream mappers never commit
    /// a discarded empty/degenerate canvas (which used to splice a second
    /// `<|channel>thought` into `reasoning_content`).
    pub block_done: bool,
}

/// Called after every denoise step with the current canvas snapshot. Must be
/// cheap; runs on the generation thread.
pub type StepObserver = std::sync::Arc<dyn Fn(&StepProgressEvent<'_>) + Send + Sync>;

/// Called once per committed denoise block after its stats (incl. token ids and
/// stop metadata) are finalized. Serve uses this to flush `model-*.json` to
/// disk before the next block so a killed process still leaves partial logs.
pub type BlockCommitObserver =
    std::sync::Arc<dyn Fn(&crate::generate::BlockDenoiseStats) + Send + Sync>;

#[derive(Clone)]
pub struct StepGenerateConfig {
    pub seed: u64,
    pub max_new_tokens: usize,
    pub max_seq: usize,
    pub layers: usize,
    pub sampler: SamplerConfig,
    pub no_early_stop: bool,
    /// Override random canvas (256 ids) for parity with MLX/HF traces.
    pub initial_canvas_ids: Option<Vec<u32>>,
    /// End-of-turn / EOS ids that terminate a "full message". When non-empty,
    /// generation stops (and the sequence is truncated) as soon as a committed
    /// block emits any of these. Empty preserves the fixed `max_new_tokens`
    /// budget behavior used by parity/golden paths.
    pub stop_token_ids: Vec<u32>,
    /// When set (serve tool-mode), a stop token does not end the turn if the
    /// reply still looks unfinished: open `call:NAME{…}`, or non-empty prose
    /// after a closed tool call (`…}<tool_call|>Wait, I`). Stop is trimmed and
    /// generation continues into the next block.
    pub continue_incomplete_tool_calls: bool,
    /// E6 empty/degenerate-reply canvas re-roll (with `DGQ_EMPTY_REPLY_RETRY>0`).
    /// Given the first block's committed argmax, returns true when it renders as
    /// an empty user-facing reply (eos-first canvas OR `<|channel>thought`
    /// ceremony) — checked against the real decoded+sanitized output, not a
    /// token blocklist. `None` disables the re-roll. Built by
    /// `chat_template::empty_reply_check`.
    pub degenerate_reply_check: Option<std::sync::Arc<dyn Fn(&[u32]) -> bool + Send + Sync>>,
    /// Optional per-step progress callback (chat streaming / spinner).
    pub step_observer: Option<StepObserver>,
    /// Optional per-block commit callback (serve model-log flush).
    pub block_commit_observer: Option<BlockCommitObserver>,
}

impl std::fmt::Debug for StepGenerateConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StepGenerateConfig")
            .field("seed", &self.seed)
            .field("max_new_tokens", &self.max_new_tokens)
            .field("max_seq", &self.max_seq)
            .field("layers", &self.layers)
            .field("sampler", &self.sampler)
            .field("no_early_stop", &self.no_early_stop)
            .field("stop_token_ids", &self.stop_token_ids)
            .field(
                "continue_incomplete_tool_calls",
                &self.continue_incomplete_tool_calls,
            )
            .field(
                "degenerate_reply_check",
                &self.degenerate_reply_check.is_some(),
            )
            .field("step_observer", &self.step_observer.is_some())
            .field(
                "block_commit_observer",
                &self.block_commit_observer.is_some(),
            )
            .finish()
    }
}

impl StepGenerateConfig {
    pub fn from_generate(
        seed: u64,
        max_new_tokens: usize,
        max_seq: usize,
        layers: usize,
        sampler: SamplerConfig,
        no_early_stop: bool,
    ) -> Self {
        Self {
            seed,
            max_new_tokens,
            max_seq,
            layers,
            sampler,
            no_early_stop,
            initial_canvas_ids: None,
            stop_token_ids: Vec::new(),
            continue_incomplete_tool_calls: false,
            degenerate_reply_check: None,
            step_observer: None,
            block_commit_observer: None,
        }
    }
}

fn smoke_config(cfg: &StepGenerateConfig, prefill_token_ids: Option<Vec<u32>>) -> StepSmokeConfig {
    StepSmokeConfig {
        layers: cfg.layers.min(N_LAYERS).max(1),
        steps: cfg.sampler.max_denoising_steps.max(1),
        kv_len: 0,
        seed: cfg.seed,
        max_seq: cfg.max_seq,
        finish: StepFinishMode::Full,
        prefill_token_ids,
        no_early_stop: false,
    }
}

use crate::flags::progress_enabled;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DenoiseStopReason {
    None,
    Confident,
    Plateau,
    MaxSteps,
}

impl DenoiseStopReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Confident => "confident",
            Self::Plateau => "plateau",
            Self::MaxSteps => "max_steps",
        }
    }
}

fn log_denoise_step_progress(
    block_idx: usize,
    max_blocks: usize,
    step_idx: u32,
    max_steps: usize,
    stats: &crate::sample::StepEntropyStats,
    mean_entropy_gpu: f32,
    prefix_mean: Option<f32>,
    region_end: Option<usize>,
    answer_text: Option<&str>,
    canvas_stable: u32,
    prefix_stable: u32,
    full_argmax_diff: Option<usize>,
    prefix_argmax_diff: Option<usize>,
    argmax_hist_len: u32,
    accept_plateau: u32,
    step_elapsed: Duration,
    block_elapsed: Duration,
    denoise_elapsed: Duration,
    stop: DenoiseStopReason,
) {
    if !progress_enabled() {
        return;
    }
    let stop_note = match stop {
        DenoiseStopReason::None => "",
        DenoiseStopReason::Confident => " confident_stop",
        DenoiseStopReason::Plateau => " plateau_stop",
        DenoiseStopReason::MaxSteps => " max_steps",
    };
    let mut extra = String::new();
    if let Some(pm) = prefix_mean {
        extra.push_str(&format!(" prefix_mean={pm:.4}"));
    }
    if let Some(re) = region_end {
        extra.push_str(&format!(" ans_len={re}"));
    }
    if let Some(text) = answer_text {
        // Show the TAIL of the converging text: the head is stable after the
        // first steps, the interesting churn is at the end of the canvas.
        const SHOW: usize = 120;
        let one_line = text.replace(['\n', '\r'], "\\n");
        let shown = match one_line.char_indices().nth_back(SHOW) {
            Some((i, _)) => format!("…{}", &one_line[i..]),
            None => one_line,
        };
        extra.push_str(&format!(" text={shown:?}"));
    }
    if let Some(d) = full_argmax_diff {
        extra.push_str(&format!(" full_diff={d}"));
    }
    if let Some(d) = prefix_argmax_diff {
        extra.push_str(&format!(" prefix_diff={d}"));
    }
    eprintln!(
        "step-generate: block {block_idx}/{max_blocks} step {step_idx}/{max_steps} accept={} low_ent={} min_ent={:.4} mean_ent={mean_entropy_gpu:.4} canvas_stable={canvas_stable} prefix_stable={prefix_stable} hist_len={argmax_hist_len} plateau={accept_plateau}{extra} step={step_elapsed:.2?} block={block_elapsed:.2?} denoise={denoise_elapsed:.2?}{stop_note}",
        stats.accept_count, stats.low_entropy_positions, stats.min_entropy,
    );
}

fn step_answer_text(
    tokenizer: Option<&Tokenizer>,
    prev_argmax: &[u32],
    ids: &[u32],
    eos_token_id: u32,
) -> (usize, Option<String>) {
    let region_end = crate::sample::answer_region_end(ids, eos_token_id);
    let prefix = crate::sample::answer_prefix_ids(prev_argmax, ids, eos_token_id);
    let text = tokenizer.map(|tok| tok.decode(prefix));
    (region_end, text)
}

/// Longest common prefix length (in tokens) of two id slices.
fn longest_common_prefix(a: &[u32], b: &[u32]) -> usize {
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
    kv_bytes: Vec<u8>,
    tokens: Vec<u32>,
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
    rt: StepRuntime,
    model_dir: PathBuf,
    layers: usize,
    encoder: Option<MonolithicEncoderCache>,
    step_text_tokenizer: Option<Tokenizer>,
    /// Token sequence whose *causal* KV is currently valid in `rt` (== the KV
    /// length). Cross-turn reuse prefills only the delta past the common prefix
    /// of this and the next prompt. Cleared by `reset_kv`.
    kv_valid_tokens: Vec<u32>,
}

impl StepGenerateSession {
    pub fn open(
        model_dir: &Path,
        cfg: &StepGenerateConfig,
        prefill_token_ids: Option<Vec<u32>>,
    ) -> Result<(Self, Duration), Error> {
        let layers = cfg.layers.min(N_LAYERS).max(1);
        let (rt, build) = build_step_runtime(model_dir, &smoke_config(cfg, prefill_token_ids))?;
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
                kv_valid_tokens: Vec::new(),
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
    pub fn truncate_kv_to(&mut self, n_tokens: usize) -> Result<(), Error> {
        let old = self.kv_valid_tokens.len();
        let n = n_tokens.min(old);
        if n == old {
            return Ok(());
        }
        if kv_truncate_needs_ring_rebuild(old, n, self.sliding_ring_len(), self.rt.sliding_window())
        {
            let kept = self.kv_valid_tokens[..n].to_vec();
            if crate::flags::progress_enabled() {
                eprintln!(
                    "truncate_kv_to: ring rebuild {old} -> {n} tokens (ring={:?})",
                    self.sliding_ring_len()
                );
            }
            return self.rebuild_kv_prefix(&kept);
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

    /// Direct KV buffer + layout access for oracle-style tests that MUTATE
    /// stored KV between prefill and denoise (E16 fusion replay: rewrite aged
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

/// DIAGNOSTIC (task #67): multiply every live f16 KV value by (1 + eps*u),
/// u uniform in [-1, 1] — models the fast path's per-value computation noise
/// on top of a KNOWN-GOOD engine prefill. f16 sessions only (q8 skipped).
fn perturb_live_kv_f16(
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

/// Monolithic generate: prefill prompt → denoise blocks → extend KV (matches `generate_inner` structure).
pub fn generate_monolithic(
    model_dir: &Path,
    prompt_token_ids: &[u32],
    cfg: &StepGenerateConfig,
    prompt_label: &str,
) -> Result<GenerateOutput, Error> {
    let (mut session, _) =
        StepGenerateSession::open(model_dir, cfg, Some(prompt_token_ids.to_vec()))?;
    generate_with_session(&mut session, prompt_token_ids, cfg, prompt_label)
}

pub fn generate_with_session(
    session: &mut StepGenerateSession,
    prompt_token_ids: &[u32],
    cfg: &StepGenerateConfig,
    prompt_label: &str,
) -> Result<GenerateOutput, Error> {
    if prompt_token_ids.is_empty() {
        return Err(Error::Runtime("generate requires a non-empty prompt"));
    }
    let canvas_len = CANVAS;
    let layers = session.layers;
    let max_blocks = cfg.max_new_tokens.div_ceil(canvas_len).max(1);
    let model_dir = session.model_dir.as_path();
    let shared_blob = session.rt.shared_dgq_blob();
    let rt = &mut session.rt;

    // The step-text log AND the whitespace-collapse guard both need to decode
    // canvas tokens; load the tokenizer lazily once per session for either.
    if (step_text_log_enabled() || crate::flags::ws_block_stop_enabled())
        && session.step_text_tokenizer.is_none()
    {
        let tok_path = session.model_dir.join("tokenizer.json");
        match Tokenizer::load(&tok_path) {
            Ok(tok) => session.step_text_tokenizer = Some(tok),
            Err(err) => {
                eprintln!(
                    "step-generate: step-text/ws-guard tokenizer load failed for {}: {err}",
                    tok_path.display()
                );
            }
        }
    }

    let prefill_started = Instant::now();
    let existing_kv = rt.read_params().kv_len as usize;
    let n_prompt = prompt_token_ids.len();
    // Cross-turn KV reuse: the session's `kv_valid_tokens` are already causal in
    // the KV. Reuse the longest common prefix (capped at what the runtime
    // actually holds) and prefill only the delta at that offset.
    let reuse = if crate::flags::kv_reuse_enabled() {
        longest_common_prefix(&session.kv_valid_tokens, prompt_token_ids).min(existing_kv)
    } else {
        0
    };
    let (kv_len, prefill_timing, prefill_elapsed) = if existing_kv >= n_prompt && existing_kv > 0 {
        // Whole prompt already prefilled (session-open prefill, or full reuse).
        if progress_enabled() {
            eprintln!("step-generate: using step-kernel prefill kv_len={existing_kv}");
        }
        (
            existing_kv.min(n_prompt),
            MonolithicPrefillTiming::default(),
            Duration::ZERO,
        )
    } else if reuse > 0 {
        // Keep KV[0..reuse] (causally valid), prefill the delta [reuse..] at
        // that offset. Short deltas take the fast quantized resume; a delta
        // past the fast-prefill trust cap (task #64: the bf16 stream
        // accumulates error with length) extends via the f32 engine in
        // canvas-sized blocks instead — slower, correct.
        let delta = &prompt_token_ids[reuse..];
        let max = crate::flags::fast_prefill_max_tokens();
        let kv_len = if max == 0 || delta.len() <= max {
            rt.prefill_chunks_from(reuse, delta)?
        } else {
            if session.encoder.is_none() {
                session.encoder = Some(MonolithicEncoderCache::open_opt(
                    model_dir,
                    canvas_len,
                    cfg.max_seq,
                    Some(std::sync::Arc::clone(&shared_blob)),
                )?);
            }
            let encoder = session.encoder.as_mut().expect("encoder cache");
            let off = extend_monolithic_kv_chunked(
                encoder,
                rt.kvcache(),
                rt.layout(),
                reuse,
                delta,
                cfg.max_seq,
                layers,
            )?;
            rt.set_kv_len(off as u32);
            off
        };
        let prefill_elapsed = prefill_started.elapsed();
        if progress_enabled() {
            eprintln!(
                "step-generate: cross-turn KV reuse: kept {reuse}/{n_prompt}, prefilled {} delta tokens ({prefill_elapsed:.2?})",
                n_prompt - reuse
            );
        }
        (kv_len, MonolithicPrefillTiming::default(), prefill_elapsed)
    } else if crate::metal::step_kernel::should_fast_prefill(prompt_token_ids.len()) {
        // Fast monolithic prefill: quantized + causal forward over prompt chunks,
        // writing the b4 KV directly (no f32 engine, no pack conversion).
        let kv_len = rt.prefill_chunks(prompt_token_ids)?;
        let prefill_elapsed = prefill_started.elapsed();
        if progress_enabled() {
            eprintln!("step-generate: fast-prefill kv_len={kv_len} ({prefill_elapsed:.2?})");
        }
        (kv_len, MonolithicPrefillTiming::default(), prefill_elapsed)
    } else {
        if session.encoder.is_none() {
            let encoder_started = Instant::now();
            session.encoder = Some(MonolithicEncoderCache::open_opt(
                model_dir,
                canvas_len,
                cfg.max_seq,
                Some(std::sync::Arc::clone(&shared_blob)),
            )?);
            if progress_enabled() {
                eprintln!(
                    "step-generate: encoder cache ready ({:.2?})",
                    encoder_started.elapsed()
                );
            }
        }
        let encoder = session.encoder.as_mut().expect("encoder cache");
        let (kv_len, prefill_timing) = prefill_monolithic_kv_with_cache(
            encoder,
            prompt_token_ids,
            rt.kvcache(),
            rt.layout(),
            cfg.max_seq,
            layers,
        )?;
        rt.set_kv_len(kv_len as u32);
        let prefill_elapsed = prefill_started.elapsed();
        (kv_len, prefill_timing, prefill_elapsed)
    };
    // DIAGNOSTIC (task #67 sensitivity probe): DGQ_KV_NOISE=<rel eps>
    // perturbs every live f16 KV value after prefill (whichever path ran) by
    // a random relative factor — separates "the model is knife-edge
    // sensitive to KV noise at this length" from "the fast path has a real
    // defect".
    if let Some(eps) = crate::flags::kv_noise() {
        perturb_live_kv_f16(rt, kv_len, eps, cfg.seed);
    }
    // Canvas denoise writes at [kv_len..kv_len+CANVAS]; ensure the runtime's
    // kv_len is exactly the prompt length (the reuse/extend paths update KV but
    // not params; a full-reuse KV may also hold stale tokens past the prompt).
    rt.set_kv_len(kv_len as u32);
    if progress_enabled() && prefill_elapsed > Duration::ZERO {
        eprintln!(
            "step-generate: prefilled kv_len={kv_len} ({prefill_elapsed:.2?}, gpu_forward={:.1}ms kv_pack={:.1}ms)",
            prefill_timing.gpu_forward_ms, prefill_timing.kv_pack_ms
        );
    }

    let mut sequences = prompt_token_ids.to_vec();
    let mut rng = Rng::new(cfg.seed);
    let mut denoise_steps_run = 0usize;
    let mut blocks_committed = 0usize;
    let mut block_steps_eff = Vec::new();
    let mut last_block_accept_hist = Vec::new();
    let mut last_block_min_entropy_hist = Vec::new();
    let mut denoise_elapsed = Duration::ZERO;
    let mut extend_elapsed = Duration::ZERO;
    let mut session_telemetry = SessionTelemetry::default();
    let mut step_traces = Vec::new();
    let max_steps = cfg.sampler.max_denoising_steps.max(1);
    let mut initial_canvas_ids: Option<Vec<u32>> = None;
    let mut stopped_on_eot = false;
    let mut stop_token_id: Option<u32> = None;
    let mut stop_block_idx: Option<usize> = None;
    let mut stop_offset: Option<usize> = None;
    let mut block_stats: Vec<crate::generate::BlockDenoiseStats> = Vec::new();

    for _block in 0..max_blocks {
        if sequences.len() >= prompt_token_ids.len() + cfg.max_new_tokens {
            break;
        }

        let remaining = prompt_token_ids.len() + cfg.max_new_tokens - sequences.len();
        let is_last_block = remaining <= canvas_len;
        let block_idx = blocks_committed + 1;

        let params = step_params_from_sampler(
            &cfg.sampler,
            rt.read_params().kv_len,
            cfg.no_early_stop,
            rt.read_params().eos_token_id,
        );
        rt.reset_block(VOCAB, &mut rng, params);
        if let Some(ref ids) = cfg.initial_canvas_ids {
            rt.set_canvas_ids(ids)?;
        }
        if initial_canvas_ids.is_none() {
            initial_canvas_ids = Some(rt.read_canvas_state().ids.to_vec());
            if denoise_parity_log_enabled() {
                let c = initial_canvas_ids.as_ref().expect("initial canvas");
                eprintln!(
                    "denoise-parity: initial_canvas[:8]={:?}",
                    &c[..8.min(c.len())]
                );
            }
        }

        if progress_enabled() {
            eprintln!(
                "step-generate: block {block_idx}/{max_blocks} starting denoise (kv_len={}, max_steps={max_steps}, new_tokens_remaining={remaining})",
                rt.read_params().kv_len
            );
        }

        let block_started = Instant::now();
        // E6: empty/degenerate-reply retry — first block only. If the committed
        // canvas would trim to empty (position 0 is a stop/eos/control token),
        // re-roll the canvas and re-denoise up to N times (DGQ_EMPTY_REPLY_RETRY).
        let empty_retry_max = if block_idx == 1 {
            crate::flags::empty_reply_retry()
        } else {
            0
        };
        let mut empty_retry_attempt = 0u32;
        // Non-convergence commit guard (`DGQ_BLOCK_COMMIT_MAX_ENT`): re-roll
        // budget, and the abandon verdict when the budget is spent.
        let mut nc_retry_attempt = 0u32;
        let mut abandon_turn = false;
        let (
            st,
            block_step_count,
            accept_hist,
            min_entropy_hist,
            mean_entropy_hist,
            low_ent_hist,
            denoise_stop,
        ) = 'attempt: loop {
            // Shrink-on-retry (E3/E6): attempt 0 uses the full canvas (handles
            // long answers); each degenerate retry narrows it (256→128→64),
            // where the empty/ceremony attractor is far weaker (72%→3% by width).
            let attempt_canvas = match empty_retry_attempt {
                0 => CANVAS,
                1 => 128,
                _ => 64,
            };
            rt.set_active_canvas(attempt_canvas);
            let active = rt.active_canvas();
            let mut block_step_count = 0u32;
            let mut accept_hist = Vec::new();
            let mut min_entropy_hist = Vec::new();
            let mut mean_entropy_hist = Vec::new();
            let mut low_ent_hist = Vec::new();
            let mut last_st;
            let mut prev_step_argmax: Option<[u32; CANVAS]> = None;
            let prefix_stable_streak = 0u32;
            let last_denoise_stop = loop {
                let step_started = Instant::now();
                rt.run_denoise_step()?;
                let check_logits = crate::metal::step_kernel::logits_finite_check_enabled();
                rt.check_logits_finite()?;
                let step_elapsed = step_started.elapsed();
                let step_ms = step_elapsed.as_secs_f64() * 1000.0;
                let readback_bytes = StepRuntime::denoise_step_host_readback_bytes(check_logits);
                let mut forward = ForwardTelemetry::monolithic_gpu_step(readback_bytes);
                rt.fill_expert_forward_telemetry(&mut forward);
                session_telemetry.steps.push(StepPhaseTelemetry {
                    decoder_ms: step_ms,
                    sampler_ms: 0.0,
                    forward,
                });
                denoise_steps_run += 1;
                block_step_count += 1;
                let st = rt.read_canvas_state();
                last_st = st;
                if denoise_parity_log_enabled() {
                    log_denoise_parity_step(
                        &format!("block={block_idx} step_index={block_step_count}"),
                        &st,
                        &rt.read_params(),
                        rt.logits(),
                    );
                }
                if crate::flags::sc_log_enabled() {
                    eprintln!(
                        "monolithic denoise: step_index={block_step_count} st.step={} sc_active_next={}",
                        st.step,
                        st.step >= 1
                    );
                }
                // Slice canvas stats/argmax to the ACTIVE rows — stale rows
                // [active..CANVAS) hold reset sentinels (u32::MAX / 0) that would
                // skew stats and (on decode) corrupt the reply.
                let stats = step_entropy_stats(&st.entropy[..active], &st.accept[..active]);
                accept_hist.push(stats.accept_count);
                min_entropy_hist.push(stats.min_entropy);
                mean_entropy_hist.push(st.mean_entropy);
                low_ent_hist.push(stats.low_entropy_positions);
                let max_steps_reached = st.step >= max_steps as u32;
                let params = rt.read_params();
                let region_end =
                    crate::sample::answer_region_end(&st.ids[..active], params.eos_token_id);
                let (full_diff, prefix_diff, prefix_stable_streak) = match prev_step_argmax {
                    Some(prev) => {
                        let full = crate::sample::count_argmax_diff(&st.prev_argmax, &prev, active);
                        let prefix =
                            crate::sample::count_argmax_diff(&st.prev_argmax, &prev, region_end);
                        let streak = if prefix == 0 {
                            prefix_stable_streak.saturating_add(1)
                        } else {
                            0
                        };
                        (Some(full), Some(prefix), streak)
                    }
                    None => (None, None, 0),
                };
                prev_step_argmax = Some(st.prev_argmax);
                let early_stop = crate::sample::decode_early_stop_flag(st.stop_flag);
                let snap = crate::sample::EarlyStopSnapshot {
                    canvas_stable: st.canvas_stable,
                    mean_entropy: st.mean_entropy,
                    accept_plateau: st.accept_plateau,
                    conf_threshold: params.conf_threshold,
                    accept_plateau_threshold: params.accept_plateau_threshold,
                    plateau_prefix_mean_max: params.plateau_prefix_mean_max,
                };
                let cpu_early =
                    !cfg.no_early_stop && crate::sample::early_stop_from_snapshot(&snap);
                let gpu_early =
                    !cfg.no_early_stop && crate::sample::is_early_stop_flag(st.stop_flag);
                let stop_reason = match early_stop {
                    Some(crate::sample::EarlyStopKind::Confident) => DenoiseStopReason::Confident,
                    Some(crate::sample::EarlyStopKind::Plateau) => DenoiseStopReason::Plateau,
                    Some(crate::sample::EarlyStopKind::MaxSteps) => DenoiseStopReason::MaxSteps,
                    None if max_steps_reached => DenoiseStopReason::MaxSteps,
                    None => DenoiseStopReason::None,
                };
                if crate::flags::log_early_stop_enabled() {
                    if gpu_early != cpu_early {
                        eprintln!(
                            "step-generate: early-stop mismatch step={} gpu_flag={} gpu_early={gpu_early} cpu_early={cpu_early} accept_plateau={} mean_ent={:.4} stable={} threshold={:.4}",
                            st.step,
                            st.stop_flag,
                            st.accept_plateau,
                            st.mean_entropy,
                            st.canvas_stable,
                            params.conf_threshold,
                        );
                    } else if gpu_early {
                        let kind = match early_stop {
                            Some(crate::sample::EarlyStopKind::Confident) => "confident_stable",
                            Some(crate::sample::EarlyStopKind::Plateau) => "plateau_stop",
                            _ => "early_stop",
                        };
                        eprintln!(
                            "step-generate: early-stop step={} reason={kind} stop_flag={} accept_plateau={} mean_ent={:.4} stable={} accept={}",
                            st.step,
                            st.stop_flag,
                            st.accept_plateau,
                            st.mean_entropy,
                            st.canvas_stable,
                            stats.accept_count,
                        );
                    }
                }
                let (prefix_mean_log, region_end_log, answer_text_log) = if step_text_log_enabled()
                {
                    // Slice to the ACTIVE canvas: the readback buffers are
                    // PREFILL_M-sized, and rows [active..] hold stale data
                    // (was: ans_len=1024 + stale positions polluting
                    // prefix_mean/text on every step line).
                    let ids = &st.ids[..active.min(st.ids.len())];
                    let entropy = &st.entropy[..active.min(st.entropy.len())];
                    let pm = crate::sample::mean_entropy_answer_prefix(
                        entropy,
                        ids,
                        params.eos_token_id,
                    );
                    let (re, text) = step_answer_text(
                        session.step_text_tokenizer.as_ref(),
                        &st.prev_argmax,
                        ids,
                        params.eos_token_id,
                    );
                    (Some(pm), Some(re), text)
                } else {
                    (None, None, None)
                };
                let answer_text_ref = answer_text_log.as_deref();
                step_traces.push(step_trace_from_stats(
                    block_idx as u32,
                    block_step_count,
                    max_steps,
                    &stats,
                    &st.prev_argmax,
                    if trace_entropy_enabled() {
                        Some(&st.entropy)
                    } else {
                        None
                    },
                    stop_reason != DenoiseStopReason::None,
                ));
                log_denoise_step_progress(
                    block_idx,
                    max_blocks,
                    block_step_count,
                    max_steps,
                    &stats,
                    st.mean_entropy,
                    prefix_mean_log,
                    region_end_log,
                    answer_text_ref,
                    st.canvas_stable,
                    prefix_stable_streak,
                    full_diff,
                    prefix_diff,
                    st.argmax_hist_len,
                    st.accept_plateau,
                    step_elapsed,
                    block_started.elapsed(),
                    denoise_elapsed + block_started.elapsed(),
                    stop_reason,
                );
                if let Some(ref observer) = cfg.step_observer {
                    // Never mark block_done here: E6 may still discard this canvas.
                    // Active slice only — rows past `active` are stale after shrink.
                    observer(&StepProgressEvent {
                        block_idx,
                        max_blocks,
                        step_in_block: block_step_count,
                        max_steps,
                        argmax: &st.prev_argmax[..active],
                        accept_count: stats.accept_count,
                        mean_entropy: st.mean_entropy,
                        block_done: false,
                    });
                }
                if stop_reason != DenoiseStopReason::None {
                    break stop_reason;
                }
            };
            let st = last_st;
            // Empty/degenerate-reply detection: does the committed argmax render as
            // an empty user-facing reply (eos-first canvas OR `<|channel>thought`
            // ceremony)? Checked against the real decoded+sanitized output (E6
            // attractor). Re-roll the canvas from the advancing rng and retry.
            let degenerate = cfg
                .degenerate_reply_check
                .as_ref()
                .is_some_and(|check| check(&st.prev_argmax[..active]));
            if empty_retry_attempt < empty_retry_max && degenerate {
                empty_retry_attempt += 1;
                if progress_enabled() {
                    eprintln!(
                        "step-generate: block {block_idx} empty/degenerate reply; re-rolling canvas (attempt {empty_retry_attempt}/{empty_retry_max})"
                    );
                }
                let params = step_params_from_sampler(
                    &cfg.sampler,
                    rt.read_params().kv_len,
                    cfg.no_early_stop,
                    rt.read_params().eos_token_id,
                );
                rt.reset_block(VOCAB, &mut rng, params);
                continue 'attempt;
            }
            // Non-convergence commit guard: a block that burned the whole step
            // schedule and still shows late-window mean entropy above the floor
            // committed ~45%-accepted garble in the OpenCode collapse — and the
            // committed flood is self-consistent, so later blocks converge onto
            // it. Re-roll with fresh noise; if it still cannot converge, end
            // the turn instead of committing the attractor.
            let commit_max_ent = crate::flags::block_commit_max_ent();
            let non_converged =
                commit_max_ent > 0.0 && last_denoise_stop == DenoiseStopReason::MaxSteps && {
                    let late0 = mean_entropy_hist.len().saturating_sub(8);
                    let late_min = mean_entropy_hist[late0..]
                        .iter()
                        .copied()
                        .fold(f32::INFINITY, f32::min);
                    late_min.is_finite() && late_min > commit_max_ent
                };
            if non_converged {
                if nc_retry_attempt < crate::flags::block_commit_retry() {
                    nc_retry_attempt += 1;
                    if progress_enabled() {
                        eprintln!(
                            "step-generate: block {block_idx} ended max_steps NON-CONVERGED (late mean_ent > {commit_max_ent}); re-rolling canvas (attempt {nc_retry_attempt}/{})",
                            crate::flags::block_commit_retry()
                        );
                    }
                    let params = step_params_from_sampler(
                        &cfg.sampler,
                        rt.read_params().kv_len,
                        cfg.no_early_stop,
                        rt.read_params().eos_token_id,
                    );
                    rt.reset_block(VOCAB, &mut rng, params);
                    continue 'attempt;
                }
                // Budget spent: abandon without the block_done notification —
                // this canvas must never reach a streamer as final.
                abandon_turn = true;
                break 'attempt (
                    st,
                    block_step_count,
                    accept_hist,
                    min_entropy_hist,
                    mean_entropy_hist,
                    low_ent_hist,
                    last_denoise_stop,
                );
            }
            // Accepted: notify streamers that this block's active argmax is final.
            if let Some(ref observer) = cfg.step_observer {
                observer(&StepProgressEvent {
                    block_idx,
                    max_blocks,
                    step_in_block: block_step_count,
                    max_steps,
                    argmax: &st.prev_argmax[..active],
                    accept_count: accept_hist.last().copied().unwrap_or(0),
                    mean_entropy: st.mean_entropy,
                    block_done: true,
                });
            }
            break 'attempt (
                st,
                block_step_count,
                accept_hist,
                min_entropy_hist,
                mean_entropy_hist,
                low_ent_hist,
                last_denoise_stop,
            );
        };
        let block_elapsed = block_started.elapsed();
        denoise_elapsed += block_elapsed;
        block_steps_eff.push(block_step_count);
        last_block_accept_hist = accept_hist.clone();
        last_block_min_entropy_hist = min_entropy_hist.clone();
        let late = accept_hist.len().saturating_sub(8);
        let late_accept: u32 = accept_hist.get(late..).unwrap_or(&[]).iter().sum();
        let late_min_ent = min_entropy_hist
            .get(late..)
            .and_then(|s| s.iter().copied().reduce(f32::min))
            .unwrap_or(f32::NAN);
        let late_low_ent = low_ent_hist
            .get(late..)
            .and_then(|s| s.iter().copied().reduce(u32::max))
            .unwrap_or(0);
        let late_mean_ent = mean_entropy_hist
            .get(late..)
            .and_then(|s| s.iter().copied().reduce(f32::min))
            .unwrap_or(f32::NAN);
        if progress_enabled() {
            eprintln!(
                "step-generate: block {} denoise={block_elapsed:.2?} steps_eff={block_step_count} accept/step={accept_hist:?}",
                block_idx
            );
            eprintln!(
                "step-generate: block {} min_ent/step={min_entropy_hist:?}",
                block_idx
            );
            eprintln!(
                "step-generate: block {} mean_ent/step={mean_entropy_hist:?}",
                block_idx
            );
            eprintln!(
                "step-generate: block {} low_ent(<0.1)/step={low_ent_hist:?}",
                block_idx
            );
            eprintln!(
                "step-generate: block {} late-window (last 8 steps): accept_sum={late_accept} min_ent={late_min_ent:.4} mean_ent={late_mean_ent:.4} max_low_ent={late_low_ent} (early stop: accept plateau>={} or prefix mean_ent<{:.4} + stable argmax)",
                block_idx,
                crate::sample::ACCEPT_PLATEAU_THRESHOLD,
                cfg.sampler.confidence_threshold,
            );
        }
        if final_entropy_log_enabled() {
            log_final_per_token_entropy(
                &format!("block {block_idx} final"),
                &st,
                st.stop_flag,
                rt.read_params().eos_token_id,
            );
        }

        // Commit-guard abandon: the block ran out of schedule AND retries while
        // non-converged. Commit nothing (the canvas is the garble attractor),
        // record the failed block's stats for diagnosis, and end the turn with
        // whatever earlier blocks produced.
        if abandon_turn {
            eprintln!(
                "step-generate: block {block_idx} NON-CONVERGED after {} re-rolls (late mean_ent={late_mean_ent:.3} > {}); ending turn without committing ({} new tokens kept)",
                crate::flags::block_commit_retry(),
                crate::flags::block_commit_max_ent(),
                sequences.len() - prompt_token_ids.len()
            );
            let stats = crate::generate::BlockDenoiseStats {
                block_idx,
                max_blocks,
                steps_eff: block_step_count,
                denoise_secs: block_elapsed.as_secs_f64(),
                accept_per_step: accept_hist,
                min_ent_per_step: min_entropy_hist,
                mean_ent_per_step: mean_entropy_hist,
                low_ent_per_step: low_ent_hist,
                late_accept_sum: late_accept,
                late_min_ent,
                late_mean_ent,
                late_max_low_ent: late_low_ent,
                denoise_stop: "non_converged_abandon".to_string(),
                kept_tokens: 0,
                token_ids: Vec::new(),
                stop_token_id: None,
                stop_offset: None,
                continued_past_stop: false,
            };
            if let Some(ref obs) = cfg.block_commit_observer {
                obs(&stats);
            }
            block_stats.push(stats);
            break;
        }

        // The committed block is only the ACTIVE canvas rows (shrink-on-retry
        // narrows a degenerate first block); rows [active..CANVAS) are stale
        // sentinels. `rt.active_canvas()` after the loop is the successful
        // attempt's width (no set_active_canvas ran after its break).
        let committed_canvas = rt.active_canvas();
        let argmax_tokens: Vec<u32> = st.prev_argmax[..committed_canvas].to_vec();
        let block_base = sequences.len();
        sequences.extend_from_slice(&argmax_tokens);
        blocks_committed += 1;

        let mut stats = crate::generate::BlockDenoiseStats {
            block_idx,
            max_blocks,
            steps_eff: block_step_count,
            denoise_secs: block_elapsed.as_secs_f64(),
            accept_per_step: accept_hist,
            min_ent_per_step: min_entropy_hist,
            mean_ent_per_step: mean_entropy_hist,
            low_ent_per_step: low_ent_hist,
            late_accept_sum: late_accept,
            late_min_ent,
            late_mean_ent,
            late_max_low_ent: late_low_ent,
            denoise_stop: denoise_stop.as_str().to_string(),
            kept_tokens: committed_canvas,
            token_ids: Vec::new(),
            stop_token_id: None,
            stop_offset: None,
            continued_past_stop: false,
        };

        // Full-message stop: end the turn as soon as the committed block emits a
        // stop token (e.g. <turn|> or <eos>). Trim it and everything after so the
        // reply is exactly the model's turn, and skip the KV extend — unless
        // `continue_incomplete_tool_calls` and the reply still looks unfinished
        // (open `call:NAME{…}`, or trailing prose after a closed tool call).
        let mut end_turn = false;
        if !cfg.stop_token_ids.is_empty() {
            if let Some(rel) = argmax_tokens
                .iter()
                .position(|id| cfg.stop_token_ids.contains(id))
            {
                sequences.truncate(block_base + rel);
                stats.kept_tokens = rel;
                stats.stop_token_id = Some(argmax_tokens[rel]);
                stats.stop_offset = Some(rel);

                let defer_stop = cfg.continue_incomplete_tool_calls && {
                    if session.step_text_tokenizer.is_none() {
                        let tok_path = session.model_dir.join("tokenizer.json");
                        if let Ok(tok) = Tokenizer::load(&tok_path) {
                            session.step_text_tokenizer = Some(tok);
                        }
                    }
                    let reply = &sequences[prompt_token_ids.len()..];
                    let cleaned = crate::sample::strip_degenerate_token_ids(reply);
                    session.step_text_tokenizer.as_ref().is_some_and(|tok| {
                        crate::tools::should_continue_past_stop(&tok.decode(&cleaned))
                    })
                };

                if defer_stop {
                    stats.continued_past_stop = true;
                    eprintln!(
                        "serve: unfinished tool turn after stop token {}; continuing to next block",
                        argmax_tokens[rel],
                    );
                } else {
                    stopped_on_eot = true;
                    stop_token_id = Some(argmax_tokens[rel]);
                    stop_block_idx = Some(block_idx);
                    stop_offset = Some(rel);
                    end_turn = true;
                    if progress_enabled() {
                        eprintln!(
                            "step-generate: block {block_idx} hit stop token {} at offset {rel}; ending turn ({} new tokens)",
                            argmax_tokens[rel],
                            sequences.len() - prompt_token_ids.len()
                        );
                    }
                }
            }
        }
        let kept = stats.kept_tokens;
        stats.token_ids = argmax_tokens[..kept].to_vec();
        if let Some(ref obs) = cfg.block_commit_observer {
            obs(&stats);
        }
        block_stats.push(stats);
        if end_turn {
            break;
        }
        let extend_tokens = &argmax_tokens[..kept];

        // Whitespace-collapse STOPGAP (opt-in, see flags::ws_block_stop_enabled:
        // the attractor is being treated as an unfixed bug and a default-on
        // stopper would hide the evidence): a committed block whose text is
        // pure whitespace / all pad-filler ends the turn instead of crawling
        // toward the context wall at max_steps per block.
        if crate::flags::ws_block_stop_enabled() {
            let cleaned = crate::sample::strip_degenerate_token_ids(extend_tokens);
            let all_ws = cleaned.is_empty()
                || session
                    .step_text_tokenizer
                    .as_ref()
                    .is_some_and(|tok| tok.decode(&cleaned).trim().is_empty());
            if all_ws {
                sequences.truncate(block_base);
                if progress_enabled() {
                    eprintln!(
                        "step-generate: block {block_idx} committed pure whitespace; ending turn (DGQ_WS_BLOCK_STOP, {} new tokens kept)",
                        sequences.len() - prompt_token_ids.len()
                    );
                }
                break;
            }
        }

        if !is_last_block && kept > 0 {
            let extend_started = Instant::now();
            let kv_before = rt.read_params().kv_len as usize;
            let new_kv_len = if crate::flags::fast_block_extend_enabled() {
                // Fast quantized causal extend of the committed canvas — the
                // same offset-resume as the cross-turn delta prefill. The
                // engine extend costs ~10s per 256-token block (the dominant
                // cost of multi-block replies); this is one prefill-chunk
                // forward (~0.85s).
                rt.prefill_chunks_from(kv_before, extend_tokens)?
            } else {
                if session.encoder.is_none() {
                    session.encoder = Some(MonolithicEncoderCache::open_opt(
                        model_dir,
                        canvas_len,
                        cfg.max_seq,
                        Some(std::sync::Arc::clone(&shared_blob)),
                    )?);
                }
                let encoder = session.encoder.as_mut().expect("encoder cache");
                extend_monolithic_kv_with_cache(
                    encoder,
                    rt.kvcache(),
                    rt.layout(),
                    kv_before,
                    extend_tokens,
                    cfg.max_seq,
                    layers,
                )?
            };
            rt.set_kv_len(new_kv_len as u32);
            let block_extend = extend_started.elapsed();
            extend_elapsed += block_extend;
            if progress_enabled() {
                eprintln!(
                    "step-generate: extended kv {kv_before} -> {new_kv_len} (+{} tokens) ({block_extend:.2?})",
                    extend_tokens.len()
                );
            }
        }
    }

    if progress_enabled() && !session_telemetry.steps.is_empty() {
        let n = session_telemetry.steps.len().max(1) as f64;
        let agg = session_telemetry.aggregate_forward();
        eprintln!(
            "step-generate: P2.1 hot path mean {:.2} syncs/step, {:.1} KiB readback/step (DGQ_CHECK_LOGITS for opt-in logits scan)",
            agg.gpu_syncs as f64 / n,
            agg.gpu_readback_bytes as f64 / 1024.0 / n
        );
    }

    // Record the tokens whose causal KV is now valid (== the runtime kv_len):
    // the prompt plus any committed blocks that were causally extended. The
    // final (last/stopped) block's canvas KV is bidirectional, so it is NOT
    // counted — the next turn re-prefills it as prompt content. This is what
    // the next turn's cross-turn reuse diffs against.
    let valid_len = (rt.read_params().kv_len as usize).min(sequences.len());
    session.kv_valid_tokens = sequences[..valid_len].to_vec();

    Ok(GenerateOutput {
        token_ids: sequences.clone(),
        denoise_steps_run,
        blocks_committed,
        stopped_on_eot,
        stop_token_id,
        stop_block_idx,
        stop_offset,
        block_stats,
        block_steps_eff,
        last_block_accept_hist,
        last_block_min_entropy_hist,
        prefill_elapsed,
        denoise_elapsed,
        extend_elapsed,
        session_telemetry,
        denoise_trace: Some(DenoiseTrace {
            schema_version: SCHEMA_VERSION,
            source: "rust-monolithic".into(),
            prompt: prompt_label.to_string(),
            prompt_token_ids: prompt_token_ids.to_vec(),
            seed: cfg.seed,
            max_denoise_steps: max_steps,
            layers,
            max_new_tokens: cfg.max_new_tokens,
            weights_profile: Some(crate::generate_golden::monolithic_weights_profile().into()),
            entropy_bound: cfg.sampler.entropy_bound,
            step_traces,
            denoise_steps_run,
            blocks_committed,
            output_token_ids: sequences.clone(),
            initial_canvas_ids,
            canvas_rng: Some("rust-lcg".into()),
        }),
    })
}

#[cfg(test)]
#[path = "step_generate_tests.rs"]
mod tests;
