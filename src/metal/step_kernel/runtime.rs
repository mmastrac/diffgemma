//! `StepRuntime`: buffers, pipelines, and session state; drives `StepEnc`
//! per step. Split from exec.rs; construction lives in build.rs.

use super::*;

pub struct StepRuntime {
    /// Byte grant covering this runtime's footprint (weights blob + KV +
    /// scratch slack). Held for the runtime's lifetime; releasing IS dropping
    /// the runtime, so the budget cannot be held wrong. See `membudget`.
    pub(super) _mem_permit: crate::membudget::MemPermit,
    pub(super) ctx: MetalContext,
    pub(super) pipelines: &'static StepPipelines,
    /// fp16-arena pipeline set (DGQ_PREFILL_F16): identical kernels
    /// compiled with K_ARENA_F16 — the fast prefill dispatches through these
    /// while denoise keeps the gate-validated bf16 set. None when off.
    pub(super) pipelines_prefill_f16: Option<&'static StepPipelines>,
    /// While true, dispatch_and_wait encodes against the fp16 set (bracketed
    /// around prefill_chunks_from's chunk loop).
    pub(super) arena_f16_mode: bool,
    pub(super) bufs: StepBuffers,
    pub(super) gpu_blob: std::sync::Arc<DgqGpuBlob>,
    pub(super) weight_cache: GpuDecoderWeightCache,
    pub(super) text_config: TextConfig,
    pub(super) dims: crate::metal::step_config::ModelDims,
    pub(super) block_profile: StepBlockProfile,
    pub(super) attn_format: crate::metal::step_quant::DenseWeightFormat,
    pub(super) dense_format: crate::metal::step_quant::DenseWeightFormat,
    pub(super) sc_format: crate::metal::step_quant::DenseWeightFormat,
    pub(super) embed_bf16: bool,
    pub(super) layout: ModelLayout,
    pub(super) tensor_offsets: HashMap<String, u64>,
    pub layers: usize,
    /// KV-cache capacity (positions per layer) the buffers were sized for. Every
    /// denoise block writes a CANVAS-wide canvas at [kv_len..kv_len+CANVAS], so
    /// kv_len + CANVAS must never exceed this — checked in `set_kv_len`.
    pub(super) max_seq: usize,
    /// KV position the f32 side ring is valid up to. A fast prefill resuming
    /// at a different offset re-hydrates the window from the monolithic cache
    /// first (rollback / restore / conversation swap all invalidate silently —
    /// the mismatch check heals every case).
    pub(super) kv_f32_side_valid: usize,
    /// Active canvas width for the NEXT denoise step (shrink-on-retry).
    /// `CANVAS` (256) normally; the block loop narrows it (128/64) when re-
    /// rolling a degenerate reply. Clamped to [1, CANVAS].
    pub(super) active_canvas: usize,
}

impl StepRuntime {
    /// KV-cache capacity (positions per layer) the buffers were sized for.
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    pub fn layout(&self) -> &ModelLayout {
        &self.layout
    }

    /// Config-derived model geometry (see `ModelDims`).
    pub fn dims(&self) -> &crate::metal::step_config::ModelDims {
        &self.dims
    }

    /// Model sliding-window size (Gemma-4: 1024). A sliding layer's query at
    /// position `q` reads keys `[q - (window-1), q]`.
    pub fn sliding_window(&self) -> usize {
        self.text_config.sliding_window
    }

    pub fn kvcache(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.bufs.kvcache
    }

    pub fn logits(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.bufs.logits
    }

    pub fn shared_dgq_blob(&self) -> std::sync::Arc<DgqGpuBlob> {
        std::sync::Arc::clone(&self.gpu_blob)
    }

    pub fn read_params(&self) -> StepParams {
        read_struct(&self.bufs.params)
    }

    pub fn write_params(&mut self, params: StepParams) {
        write_struct(&self.bufs.params, &params);
    }

    pub fn set_kv_len(&mut self, kv_len: u32) {
        // A denoise block (and each prefill chunk) writes a CANVAS-wide canvas at
        // [kv_len..kv_len+CANVAS] into the KV cache. If that exceeds the cache
        // capacity, the write silently spills into the next layer's region (or off
        // the buffer) and corrupts attention into word-salad. Fail loudly instead —
        // callers must size max_seq >= prompt + generated + CANVAS.
        assert!(
            kv_len as usize + self.dims.canvas <= self.max_seq,
            "KV cache overflow: kv_len={kv_len} + canvas={} > max_seq={}; \
             size max_seq >= prompt_len + max_new_tokens + canvas",
            self.dims.canvas,
            self.max_seq,
        );
        let mut params = self.read_params();
        params.kv_len = kv_len;
        self.write_params(params);
    }

    /// Set the first position `qk_rope_kv` must NOT write to the KV cache
    /// (see `StepParams::kv_write_end`). Prefill brackets its chunk loop with
    /// this (prompt end during, `u32::MAX` after) so padded tail rows never
    /// clobber live ring slots while denoise canvas writes stay unaffected.
    pub fn set_kv_write_end(&mut self, end: u32) {
        let mut params = self.read_params();
        params.kv_write_end = end;
        self.write_params(params);
    }

    /// FNV-1a (continuing `seed_hash`) over the READABLE KV state at `kv_len`:
    /// full (linear) layers hash their whole `[0, kv_len)` prefix; sliding
    /// (ring) layers hash only the last `min(kv_len, window)` positions in
    /// position order. Ring slots below the window are unreachable by any
    /// future read and legitimately hold residue after extends near ring
    /// aliasing (an extend writes slot `pos & mask`; a later truncate cannot
    /// un-write the dead position that slot used to alias) — so the token
    /// pipeline's rewind gates probe THIS, not the raw ring bytes.
    pub fn live_kv_fnv(&self, kv_len: usize, seed_hash: u64) -> u64 {
        let fmt = crate::flags::kv_format(self.max_seq);
        let src = unsafe {
            std::slice::from_raw_parts(
                self.bufs.kvcache.contents().as_ptr() as *const u8,
                self.bufs.kvcache.length(),
            )
        };
        let mut h = seed_hash;
        let mut eat = |bytes: &[u8]| {
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        let window = self.sliding_window();
        for l in &self.layout.layers {
            let group =
                crate::metal::step_kv::kv_region_bytes(l.n_kv_heads, l.head_dim, 1, fmt) as usize;
            let base = l.kv_region as usize;
            if l.kv_ring_mask == 0 {
                eat(&src[base..base + kv_len * group]);
            } else {
                for p in kv_len.saturating_sub(window)..kv_len {
                    let off = base + (p & l.kv_ring_mask as usize) * group;
                    eat(&src[off..off + group]);
                }
            }
        }
        h
    }

    /// TEST-ONLY diagnostic: zero the monolithic KV cache and the f32 side
    /// ring, and invalidate the side-ring cursor — removes all cross-build
    /// residue so lineage probes can distinguish true path-dependence from
    /// stale-state leakage.
    #[cfg(test)]
    pub fn debug_scrub_kv(&mut self) {
        self.debug_scrub_kv_parts(true, true);
    }

    /// TEST-ONLY: selectively zero the monolithic cache and/or the f32 side
    /// ring — lets lineage probes identify WHICH buffer's residue a stale
    /// read is picking up.
    #[cfg(test)]
    pub fn debug_scrub_kv_parts(&mut self, monolithic: bool, side: bool) {
        if monolithic {
            unsafe {
                std::ptr::write_bytes(
                    self.bufs.kvcache.contents().as_ptr() as *mut u8,
                    0,
                    self.bufs.kvcache.length(),
                );
            }
        }
        if side && let Some(s) = &self.bufs.kv_f32_side {
            unsafe {
                std::ptr::write_bytes(s.contents().as_ptr() as *mut u8, 0, s.length());
            }
        }
        if side {
            self.kv_f32_side_valid = 0;
        }
    }

    /// Fill the ENTIRE monolithic KV cache with a deterministic pseudorandom
    /// f16-safe pattern (fixed small exponent, random sign + mantissa →
    /// ±[0.125, 0.25)) and declare `n_tokens` of it causally valid. Test
    /// infrastructure for the token pipeline's long-context byte-consistency
    /// gates: a "100k-token" context in ~a second instead of a ~7-minute
    /// prefill. The values are semantically meaningless but numerically tame,
    /// so forwards over them stay finite and bit-deterministic — which is all
    /// the order-of-operations gates assert.
    pub fn synthetic_fill_kv(&mut self, n_tokens: usize, seed: u64) {
        let len_bytes = self.bufs.kvcache.length();
        let ptr = self.bufs.kvcache.contents().as_ptr() as *mut u16;
        let words = len_bytes / 2;
        let mut s = seed | 1;
        for i in 0..words {
            // xorshift64* — cheap, deterministic, no external state.
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let r = (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u16;
            let v = (r & 0x03FF) | 0x3000 | ((r & 0x0400) << 5);
            unsafe { *ptr.add(i) = v };
        }
        self.set_kv_len(n_tokens as u32);
    }

    /// Compact byte snapshot of the live `[0, kv_len)` KV, for saving a
    /// conversation out of the single hot buffer (multi-conversation swap).
    /// Gathers only each layer's valid prefix (see `gather_kv_prefix`), so the
    /// blob is proportional to the conversation length, not the `max_seq`
    /// capacity. Pair with the session's `kv_valid_tokens` and [`restore_kv`].
    pub fn snapshot_kv(&self, kv_len: usize) -> Vec<u8> {
        crate::metal::step_kv::gather_kv_prefix(
            &self.bufs.kvcache,
            &self.layout,
            self.max_seq,
            kv_len,
        )
    }

    /// Restore a compact snapshot from [`snapshot_kv`] into the hot buffer. The
    /// caller must also set `kv_len` (via `set_kv_len`) to match, and pass the
    /// same `kv_len` here so each layer's slice length is reconstructed.
    pub fn restore_kv(&mut self, kv_len: usize, bytes: &[u8]) {
        crate::metal::step_kv::scatter_kv_prefix(
            &self.bufs.kvcache,
            &self.layout,
            self.max_seq,
            kv_len,
            bytes,
        );
    }

    /// Fast prefill on the monolithic kernels: process the prompt in CANVAS-sized
    /// chunks, each a KV-only CAUSAL forward writing K/V into the b4 cache at
    /// [chunk_start, chunk_start+chunk_len). The last chunk is padded to CANVAS;
    /// padding K/V lands beyond prompt_len (overwritten by the first denoise block)
    /// and causal masking keeps real tokens from attending to it. Replaces the
    /// ~70s f32-engine prefill. Returns kv_len (= prompt length). Causal w/o window
    /// is correct for prompts <= sliding_window (1024); longer prompts would need
    /// windowing on sliding layers (not yet implemented).
    pub fn prefill_chunks(&mut self, prompt_token_ids: &[u32]) -> Result<usize, Error> {
        self.prefill_chunks_from(0, prompt_token_ids)
    }

    /// Refill each sliding layer's f32 side ring from the monolithic
    /// cache for the last `min(upto, ring)` positions below `upto` (f16→f32
    /// widening — bakes in the one rounding those values already carry, no
    /// further compounding). One dispatch for all layers, ~ms.
    fn hydrate_kv_f32_side(&mut self, upto: usize) -> Result<(), Error> {
        let layout = self.layout;
        let layers = self.layers;
        self.dispatch_and_wait(|enc| {
            let pipe = enc
                .ps
                .kv_f32_side_hydrate
                .clone()
                .ok_or(Error::Gpu("kv_f32_side_hydrate pipeline missing"))?;
            let sbuf = enc
                .bufs
                .kv_f32_side
                .clone()
                .ok_or(Error::Runtime("kv_f32_side buffer missing"))?;
            for layer in 0..layers {
                let l = &layout.layers[layer];
                let slots = if l.kv_ring_mask != 0 {
                    (l.kv_ring_mask as usize) + 1
                } else {
                    upto // full layers are linear: hydrate everything
                };
                let count = upto.min(slots);
                let pos0 = upto - count;
                enc.sink_set_pipeline(&pipe);
                enc.bind_kvcache(0);
                enc.sink_set_buffer(&sbuf, enc.bufs.kv_f32_side_offs[layer] as usize, 1);
                let shape = [count as u32, pos0 as u32, l.n_kv_heads, l.head_dim];
                enc.sink_set_bytes(&shape, 2);
                enc.sink_set_bytes(&{ l.kv_region }, 3);
                enc.sink_set_bytes(&l.kv_ring_mask, 4);
                let grid = MTLSize {
                    width: count,
                    height: l.n_kv_heads as usize,
                    depth: l.head_dim as usize,
                };
                let tg = MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                };
                enc.sink_dispatch(grid, tg);
            }
            Ok(())
        })
    }

    /// Fast (quantized, causal) prefill of `delta_token_ids` starting at KV
    /// position `offset`. The delta chunks attend causally to the KV already
    /// present at [0..offset] (e.g. the reused cross-turn prefix), so this
    /// resumes a prefill without recomputing the prefix. Because each position's
    /// KV is fixed by its causal context (independent of chunk grouping), the
    /// result at [offset..] is identical to a full `prefill_chunks` of the whole
    /// sequence when [0..offset] was itself fast-prefilled. Returns the new
    /// kv_len (`offset + delta.len()`).
    pub fn prefill_chunks_from(
        &mut self,
        offset: usize,
        delta_token_ids: &[u32],
    ) -> Result<usize, Error> {
        let layout = self.layout;
        let layers = self.layers;
        let n = offset + delta_token_ids.len();
        let mut pos = offset;
        let batch = crate::flags::prefill_batch_enabled();
        // M0 range trace (DGQ_TRACE_RANGES=1): after each chunk forward the
        // arena planes hold the LAST layer's stage outputs — sample max|x| and
        // non-finites per plane across all chunks, so position-dependence is
        // covered. Answers "do prefill activations fit fp16 (max 65504)?".
        type RangePeaks = std::collections::BTreeMap<&'static str, (f32, bool, Option<usize>)>;
        let mut range_peaks: Option<RangePeaks> =
            crate::flags::trace_ranges_enabled().then(RangePeaks::new);
        let probe_planes = |this: &Self, m_rows: usize, peaks: &mut RangePeaks| {
            let a = &this.bufs.arena_map;
            for (label, off, per_row) in [
                ("hidden", a.hidden_off(), this.dims.hid),
                ("attnq(Q)", a.attnq_off(), 4096),
                ("attn_out", a.attno_off(), 4096),
                ("tmp(o_proj)", a.tmp_off(), this.dims.hid),
                ("ffg(gate_up)", a.ffg_off(), this.dims.dense_ff),
                ("dense", a.dense_off(), this.dims.hid),
                ("moein", a.moein_off(), this.dims.hid),
            ] {
                // half_buffer_stats returns (finite, max_abs) — bool true = healthy.
                let (finite, mx) = if this.arena_f16_mode {
                    f16_buffer_stats(&this.bufs.arena, off as usize, m_rows * per_row, 8192)
                } else {
                    half_buffer_stats(&this.bufs.arena, off as usize, m_rows * per_row, 8192)
                };
                let e = peaks.entry(label).or_insert((0.0, false, None));
                e.0 = e.0.max(mx);
                e.1 |= !finite;
                if !finite && e.2.is_none() {
                    // Locate the first offending ROW (full scan, diagnostic-only):
                    // rows >= the chunk's real token count are inert zero-pad rows
                    // whose all-masked softmax yields NaN by construction.
                    let ptr = unsafe {
                        this.bufs.arena.contents().as_ptr().add(off as usize) as *const u16
                    };
                    for i in 0..m_rows * per_row {
                        let v = crate::shaders::bf16::bf16_bits_to_f32(unsafe { *ptr.add(i) });
                        if !v.is_finite() {
                            e.2 = Some(i / per_row);
                            break;
                        }
                    }
                }
            }
        };
        // Suppress KV writes for the zero-PADDED tail rows (positions >= n):
        // on sliding layers a pad position wraps onto (pos & ring_mask) and
        // would clobber the oldest live window slots.
        self.set_kv_write_end(n as u32);
        // The prefill attention reads sliding K/V from the f32 side ring;
        // make its window current for a resume at `offset` (fresh prefill at
        // 0 needs nothing — every read position gets written first).
        if self.bufs.kv_f32_side.is_some() && offset > 0 && self.kv_f32_side_valid != offset {
            self.hydrate_kv_f32_side(offset)?;
        }
        self.arena_f16_mode = self.pipelines_prefill_f16.is_some();
        while pos < n {
            let remaining = n - pos;
            // Batched super-chunk: n_subs full-CANVAS causal sub-chunks as one
            // forward (needs kv headroom for the whole super-chunk). The tail
            // (< 2 full chunks) falls back to plain 256-chunks.
            let n_subs = if batch {
                (remaining / self.dims.canvas).min(PREFILL_SUBS)
            } else {
                1
            };
            if n_subs >= 2 && pos + n_subs * self.dims.canvas + self.dims.canvas <= self.max_seq {
                let m = n_subs * self.dims.canvas;
                self.set_canvas_ids(&delta_token_ids[pos - offset..pos - offset + m])?;
                self.set_kv_len(pos as u32);
                self.write_params_sub(pos as u32, n_subs);
                self.dispatch_and_wait(|enc| enc.encode_prefill_super(&layout, layers, n_subs))?;
                if let Some(peaks) = range_peaks.as_mut() {
                    probe_planes(self, m, peaks);
                }
                pos += m;
                continue;
            }
            let chunk_len = remaining.min(self.dims.canvas);
            let mut ids = [0u32; CANVAS];
            ids[..chunk_len]
                .copy_from_slice(&delta_token_ids[pos - offset..pos - offset + chunk_len]);
            self.set_canvas_ids(&ids)?;
            self.set_kv_len(pos as u32);
            self.dispatch_and_wait(|enc| enc.encode_prefill_chunk(&layout, layers))?;
            if let Some(peaks) = range_peaks.as_mut() {
                probe_planes(self, self.dims.canvas, peaks);
            }
            pos += chunk_len;
        }
        if let Some(peaks) = range_peaks {
            for (label, (mx, nf, nf_row)) in &peaks {
                eprintln!(
                    "prefill-trace: plane={label} max_abs={mx:.1} non_finite={nf} first_nf_row={nf_row:?} (last-layer stage outputs, all chunks, offset={offset} n={n})"
                );
            }
        }
        self.arena_f16_mode = false;
        if self.bufs.kv_f32_side.is_some() {
            self.kv_f32_side_valid = n;
        }
        self.set_kv_write_end(u32::MAX);
        self.set_kv_len(n as u32);
        // The prefill dirtied scratch (arena hidden/dense, MoE routing buffers,
        // logits); re-zero to the same clean state the (self-contained) engine
        // prefill leaves — mirrors the post-open zeros minus kvcache (holds the
        // prompt KV). Leaving residual here made some short prompts degenerate.
        zero_buffer(&self.bufs.arena);
        zero_buffer(&self.bufs.logits);
        zero_buffer(&self.bufs.expert_layer_unique);
        zero_buffer(&self.bufs.moe_grouped_indirect);
        Ok(n)
    }

    pub fn read_canvas_state(&self) -> CanvasState {
        read_struct(&self.bufs.state)
    }

    /// Read the sampler rowstat plane `{mx, sum}` (f32 pairs, tempered
    /// distribution) for the first `rows` canvas rows. `p_max = 1/sum` since
    /// the softmax is centered on the max logit. Trace-only readback
    /// (`DGQ_TRACE_PMAX_JSONL`) — never on a hot path.
    pub fn read_sample_rowstats(&self, rows: usize) -> Vec<[f32; 2]> {
        let byte_off = self.bufs.arena_map.rs_samp_off() as usize;
        let ptr = unsafe { self.bufs.arena.contents().as_ptr().add(byte_off) as *const f32 };
        (0..rows.min(self.dims.canvas))
            .map(|r| unsafe { [*ptr.add(r * 2), *ptr.add(r * 2 + 1)] })
            .collect()
    }

    /// Last-layer hidden rows for the canvas, `rows x HID` row-major.
    ///
    /// Valid only AFTER a full forward: no finish stage writes `hidden_off`, so
    /// the final decoder layer's output survives lm_head and the sampler. Used
    /// by the output-token classifier at COMMIT — one readback per block, and
    /// only when `DGQ_TOKEN_CLASS` is set, so the hot path is untouched when
    /// the feature is off.
    pub fn read_hidden_rows(&self, rows: usize) -> Vec<f32> {
        let byte_off = self.bufs.arena_map.hidden_off() as usize;
        let n = rows.min(self.dims.canvas);
        crate::metal::step_kernel::diag_bench::read_arena_buffer_f32(
            &self.bufs.arena,
            byte_off,
            n * self.dims.hid,
        )
    }

    pub fn set_canvas_ids(&mut self, ids: &[u32]) -> Result<(), Error> {
        // CANVAS for denoise / plain prefill chunks; up to PREFILL_M for a
        // batched prefill super-chunk.
        if ids.len() != self.dims.canvas
            && (ids.len() > self.dims.prefill_m() || !ids.len().is_multiple_of(self.dims.canvas))
        {
            return Err(Error::Format(
                "canvas ids length must be self.dims.canvas..=self.dims.prefill_m()",
            ));
        }
        let mut state = self.read_canvas_state();
        for (i, &id) in ids.iter().enumerate() {
            state.ids[i] = id;
        }
        self.write_canvas_state(&state);
        Ok(())
    }

    /// Write the per-sub-chunk StepParams slots for a batched prefill
    /// super-chunk: slot c = current params with kv_len = base + c*CANVAS.
    fn write_params_sub(&mut self, base_kv_len: u32, n_subs: usize) {
        let mut p = read_struct::<StepParams>(&self.bufs.params);
        let ptr = self.bufs.params_sub.contents().as_ptr() as *mut StepParams;
        for c in 0..n_subs {
            p.kv_len = base_kv_len + (c * self.dims.canvas) as u32;
            unsafe { std::ptr::write(ptr.add(c), p) };
        }
    }

    pub fn write_canvas_state(&mut self, state: &CanvasState) {
        write_struct(&self.bufs.state, state);
    }

    /// New denoise block: fresh random canvas, reset step/stop, patch sampler params.
    /// Active denoise canvas width for subsequent steps (shrink-on-retry).
    /// Clamped to [1, CANVAS]. A `DGQ_FORCE_CANVAS` override still wins at the
    /// dispatch site. The block loop reads it back via `active_canvas()`.
    pub fn set_active_canvas(&mut self, w: usize) {
        self.active_canvas = w.clamp(1, self.dims.canvas);
    }

    /// Effective active canvas width (the `DGQ_FORCE_CANVAS` test override, else
    /// the value set by `set_active_canvas`). The CPU block loop slices canvas
    /// stats/trim to this many rows.
    pub fn active_canvas(&self) -> usize {
        crate::flags::force_canvas()
            .map(|w| (w as usize).clamp(1, self.dims.canvas))
            .unwrap_or(self.active_canvas)
    }

    pub fn reset_block(&mut self, vocab: usize, rng: &mut Rng, params: StepParams) {
        let mut state = init_canvas_state_from_rng(vocab, rng);
        state.step = 0;
        state.stop_flag = 0;
        state.argmax_hist_len = 0;
        state.argmax_hist_base = 0;
        state.argmax_hist = [0; CANVAS * ARGMAX_HIST_MAX];
        state.canvas_stable = 0;
        state.mean_entropy = 0.0;
        state.accept_plateau = 0;
        state.prev_accept_sig = 0;
        state.frozen = [0; FROZEN_WORDS];
        self.write_canvas_state(&state);
        self.write_params(params);
    }

    pub fn run_denoise_step(&mut self) -> Result<(), Error> {
        zero_buffer(&self.bufs.expert_layer_unique);
        self.run_forward_once(StepFinishMode::Full)
    }

    /// Populate forward telemetry from per-layer expert counts (grouped MoE path).
    pub fn fill_expert_forward_telemetry(&self, forward: &mut crate::metal::ForwardTelemetry) {
        let ptr = self.bufs.expert_layer_unique.contents().as_ptr() as *const u32;
        let counts = unsafe { std::slice::from_raw_parts(ptr, self.layers) };
        let weight_bytes =
            grouped_expert_blob_bytes_per_expert(self.block_profile.format, &self.dims);
        forward.record_expert_layers_grouped(counts, weight_bytes);
    }

    /// Host readback size for one `CanvasState` poll (shared buffer, no extra sync).
    pub const CANVAS_STATE_BYTES: usize = std::mem::size_of::<CanvasState>();

    /// P2.1 budget: host bytes touched per denoise step on the generate hot path.
    pub fn denoise_step_host_readback_bytes(
        check_logits: bool,
        dims: &crate::metal::step_config::ModelDims,
    ) -> u64 {
        // Forward reads state once on CPU to seed preamble; generate polls once after sync.
        let mut bytes = (Self::CANVAS_STATE_BYTES as u64) * 2;
        if check_logits && logits_finite_check_enabled() {
            bytes += logits_finite_sample_bytes(dims);
        }
        bytes
    }

    /// Opt-in full-tensor logits scan (`DGQ_CHECK_LOGITS=1`). Off by default (P2.1).
    pub fn check_logits_finite(&self) -> Result<(), Error> {
        if !logits_finite_check_enabled() {
            return Ok(());
        }
        let sample = logits_finite_sample_count().min(self.dims.canvas * self.dims.vocab);
        let (finite, max_abs) = half_buffer_stats(
            &self.bufs.logits,
            0,
            self.dims.canvas * self.dims.vocab,
            sample,
        );
        if !finite {
            eprintln!("non-finite logits (max_abs={max_abs:.4}, sample={sample})");
            return Err(Error::Runtime("non-finite logits"));
        }
        Ok(())
    }

    fn check_debug_status(&self) -> Result<(), Error> {
        if let Some(ref dbg) = self.bufs.debug_status {
            let st = crate::metal::debug_status::read_buffer(dbg);
            crate::metal::debug_status::check_status(st)?;
        }
        Ok(())
    }

    pub(super) fn dispatch_and_wait<F>(&mut self, f: F) -> Result<(), Error>
    where
        F: FnOnce(&mut StepEnc<'_>) -> Result<(), Error>,
    {
        let cmd = self
            .ctx
            .queue
            .commandBuffer()
            .ok_or(Error::Gpu("command buffer alloc failed"))?;
        let ps: &StepPipelines = if self.arena_f16_mode {
            self.pipelines_prefill_f16.unwrap_or(self.pipelines)
        } else {
            self.pipelines
        };
        // Some pipelines (tunable STACKED — segment layout only known at
        // dispatch) compile LAZILY inside the encode closure via
        // runtime_step_variant(); scope the arena-f16 compile mode to this
        // dispatch so those lazy compiles (and their cache keys) inherit the
        // active set's arena dtype.
        let saved_af16 = crate::shaders::variant::arena_f16_compile_enabled();
        crate::shaders::variant::set_arena_f16_compile(saved_af16 || self.arena_f16_mode);
        let mut enc = StepEnc {
            enc: cmd
                .computeCommandEncoder()
                .ok_or(Error::Gpu("compute encoder alloc failed"))?,
            ctx: &self.ctx,
            ps,
            bufs: &self.bufs,
            dims: &self.dims,
            block_profile: self.block_profile,
            tensor_offsets: &self.tensor_offsets,
            partial_lm_m: self.dims.canvas as u32,
            attn_format: self.attn_format,
            dense_format: self.dense_format,
            sc_format: self.sc_format,
            embed_bf16: self.embed_bf16,
            prefill_causal: false,
            forward_m: self.dims.canvas,
            active_canvas: self.dims.canvas,
            sub_c: 0,
            use_params_sub: false,
            sliding_window: self.text_config.sliding_window as u32,
        };
        let encode_result = f(&mut enc);
        crate::shaders::variant::set_arena_f16_compile(saved_af16);
        encode_result?;
        enc.enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if let Some(err) = cmd.error() {
            return Err(Error::Format(
                format!(
                    "step dispatch command buffer failed: {}",
                    err.localizedDescription()
                )
                .leak(),
            ));
        }
        Ok(())
    }

    /// Attention + dense FFN + GPU router; MoE expert matmuls on CPU (matches `.dgq` Q4 oracle).
    pub fn fill_moe_out_dgq_cpu(&mut self, layer: usize) -> Result<(), Error> {
        let route: RouteScratch = read_struct(&self.bufs.route);
        let routes = routes_from_route_scratch(&route);
        let moe_in = read_arena_buffer_f32(
            &self.bufs.arena,
            self.bufs.arena_map.moein_off() as usize,
            self.dims.canvas * self.dims.hid,
        );
        let mut moe_out = vec![0.0f32; self.dims.canvas * self.dims.hid];
        let mut scratch = MoeScratch::new(self.dims.canvas, &self.text_config);
        experts_forward_dgq_cpu(
            &mut moe_out,
            &moe_in,
            &self.weight_cache,
            layer,
            &self.text_config,
            self.dims.canvas,
            &routes,
            &mut scratch,
        )?;
        write_f32_arena(&self.bufs.arena, self.bufs.arena_map.moeout_off(), &moe_out);
        Ok(())
    }

    /// One decoder layer: GPU router + grouped MoE + GPU post-combine (single submit).
    pub fn encode_full_layer(&mut self, layer: usize) -> Result<(), Error> {
        let layout = self.layout;
        self.dispatch_and_wait(|enc| enc.encode_full_layer(layer, &layout))?;
        Ok(())
    }

    /// One forward step with per-phase GPU sync (for profiling; ~4 extra submits vs monolithic).
    pub(super) fn profile_forward_once(
        &mut self,
        finish: StepFinishMode,
    ) -> Result<StepProfileResult, Error> {
        use std::time::Instant;
        let layout = self.layout;
        let layers = self.layers;
        let st_before: CanvasState = read_struct(&self.bufs.state);
        let first_step = if st_before.step == 0 { 1u32 } else { 0u32 };

        let t0 = Instant::now();
        self.dispatch_and_wait(|enc| enc.encode_step_preamble(&layout, first_step))?;
        let preamble = t0.elapsed();

        let t1 = Instant::now();
        self.dispatch_and_wait(|enc| {
            for layer in 0..layers {
                enc.encode_layer(layer, &layout)?;
            }
            Ok(())
        })?;
        let layer_pre_moe = t1.elapsed();

        let t2 = Instant::now();
        self.dispatch_and_wait(|enc| {
            for layer in 0..layers {
                enc.encode_layer_moe_grouped(layer, &layout)?;
            }
            Ok(())
        })?;
        let layer_moe = t2.elapsed();

        let t3 = Instant::now();
        self.dispatch_and_wait(|enc| {
            for layer in 0..layers {
                enc.encode_layer_moe_post(layer, &layout)?;
            }
            Ok(())
        })?;
        let layer_post = t3.elapsed();

        let t4 = Instant::now();
        self.dispatch_and_wait(|enc| enc.encode_step_finish(&layout, finish))?;
        let finish_t = t4.elapsed();

        let total = preamble + layer_pre_moe + layer_moe + layer_post + finish_t;
        Ok(StepProfileResult {
            compile: std::time::Duration::ZERO,
            preamble,
            layer_pre_moe,
            layer_moe,
            layer_post,
            finish: finish_t,
            total,
            layers,
            block_format: self.block_profile.format,
        })
    }

    fn time_enc_stage<F>(&mut self, f: F) -> Result<std::time::Duration, Error>
    where
        F: FnOnce(&mut StepEnc<'_>) -> Result<(), Error>,
    {
        use std::time::Instant;
        let t0 = Instant::now();
        self.dispatch_and_wait(f)?;
        Ok(t0.elapsed())
    }

    /// Per-stage bf16 activation-range trace: runs the forward stage-by-stage
    /// (one submit per stage so buffers are valid) and records max|x| of each
    /// stage's bf16 arena output. Answers "does any activation exceed f16's 65504
    /// range?" before trying f16/scaled-f16 arenas. `DGQ_TRACE_RANGES=1`.
    pub(super) fn trace_step_ranges(&mut self) -> Result<(), Error> {
        use std::collections::BTreeMap;
        let layout = self.layout;
        let layers = self.layers;
        let st_before: CanvasState = read_struct(&self.bufs.state);
        let first_step = if st_before.step == 0 { 1u32 } else { 0u32 };
        const SAMPLE: usize = 4096;
        // (label) -> (max_abs across layers, any_non_finite)
        let mut peak: BTreeMap<&'static str, (f32, bool)> = BTreeMap::new();
        let mut probe = |this: &Self, label: &'static str, off: u64, elems: usize| {
            // half_buffer_stats returns (finite, max_abs) — bool true = healthy.
            let f16 = this.arena_f16_mode || crate::flags::arena_f16_all_enabled();
            let (finite, mx) = if f16 {
                f16_buffer_stats(&this.bufs.arena, off as usize, elems, SAMPLE)
            } else {
                half_buffer_stats(&this.bufs.arena, off as usize, elems, SAMPLE)
            };
            let e = peak.entry(label).or_insert((0.0, false));
            e.0 = e.0.max(mx);
            e.1 |= !finite;
            if mx == 0.0 || !finite {
                let ptr =
                    unsafe { this.bufs.arena.contents().as_ptr().add(off as usize) as *const u16 };
                let bits: Vec<String> = (0..8)
                    .map(|i| format!("{:04x}", unsafe { *ptr.add(i) }))
                    .collect();
                eprintln!("    {label}: first bits [{}]", bits.join(" "));
            }
        };

        self.dispatch_and_wait(|enc| enc.encode_step_preamble(&layout, first_step))?;
        probe(
            self,
            "preamble:soft",
            self.bufs.arena_map.soft_off(),
            self.dims.canvas * self.dims.hid,
        );
        probe(
            self,
            "preamble:dense(sc_mlp)",
            self.bufs.arena_map.dense_off(),
            self.dims.canvas * self.dims.hid,
        );
        probe(
            self,
            "preamble:hidden(embed+sc)",
            self.bufs.arena_map.hidden_off(),
            self.dims.canvas * self.dims.hid,
        );

        for layer in 0..layers {
            let a = &self.bufs.arena_map;
            let (hidden, attnq, attno, tmp, ffg, dense, moein) = (
                a.hidden_off(),
                a.attnq_off(),
                a.attno_off(),
                a.tmp_off(),
                a.ffg_off(),
                a.dense_off(),
                a.moein_off(),
            );
            self.time_enc_stage(|e| e.encode_layer_qkv_gemm(layer, &layout))?;
            probe(self, "layer:qkv(Q)", attnq, self.dims.canvas * 4096);
            self.time_enc_stage(|e| e.encode_layer_qk_rope_kv_dispatch(layer, &layout))?;
            self.time_enc_stage(|e| e.encode_layer_attention_dispatch(layer, &layout))?;
            probe(self, "layer:attn_out", attno, self.dims.canvas * 4096);
            self.time_enc_stage(|e| e.encode_layer_o_proj_gemm(layer, &layout))?;
            probe(self, "layer:o_proj", tmp, self.dims.canvas * self.dims.hid);
            self.time_enc_stage(|e| e.encode_layer_o_proj_tail(layer, &layout))?;
            probe(
                self,
                "layer:resid_pre_moe(hidden)",
                hidden,
                self.dims.canvas * self.dims.hid,
            );
            let l = &layout.layers[layer];
            self.time_enc_stage(|e| {
                e.rmsnorm(
                    e.arena().stream_off(),
                    e.arena().tmp_off(),
                    l.pre_ff_ln,
                    e.dims.hid as u32,
                    e.dims.canvas,
                );
                Ok(())
            })?;
            self.time_enc_stage(|e| e.encode_layer_dense_gate_up(layer, &layout))?;
            probe(
                self,
                "layer:gate_up",
                ffg,
                self.dims.canvas * self.dims.dense_ff,
            );
            self.time_enc_stage(|e| {
                e.glu(
                    e.arena().ffg_off(),
                    e.arena().ffu_off(),
                    e.arena().ffg_off(),
                    e.dims.canvas * e.dims.dense_ff,
                );
                Ok(())
            })?;
            probe(
                self,
                "layer:swiglu",
                ffg,
                self.dims.canvas * self.dims.dense_ff,
            );
            self.time_enc_stage(|e| e.encode_layer_dense_down(layer, &layout))?;
            probe(
                self,
                "layer:dense_down",
                dense,
                self.dims.canvas * self.dims.hid,
            );
            self.time_enc_stage(|e| {
                e.rmsnorm(
                    e.arena().dense_off(),
                    e.arena().dense_off(),
                    l.post_ff_ln_1,
                    e.dims.hid as u32,
                    e.dims.canvas,
                );
                Ok(())
            })?;
            self.time_enc_stage(|e| e.encode_layer_router_buckets(layer, &layout))?;
            // MoE
            self.time_enc_stage(|e| e.encode_moe_batched_gate_up(layer, &layout))?;
            self.time_enc_stage(|e| e.encode_moe_batched_swiglu())?;
            self.time_enc_stage(|e| e.encode_moe_batched_down(layer, &layout))?;
            self.time_enc_stage(|e| e.encode_moe_batched_scatter())?;
            self.time_enc_stage(|e| e.encode_layer_moe_post_norm(layer, &layout))?;
            probe(
                self,
                "layer:moe_norm(moein)",
                moein,
                self.dims.canvas * self.dims.hid,
            );
            self.time_enc_stage(|e| e.encode_layer_moe_post_combine(layer, &layout))?;
            probe(
                self,
                "layer:resid_post_moe(hidden)",
                hidden,
                self.dims.canvas * self.dims.hid,
            );
            // Per-layer residual peak (the f16-overflow suspect).
            let (_, hmx) = half_buffer_stats(
                &self.bufs.arena,
                hidden as usize,
                self.dims.canvas * self.dims.hid,
                SAMPLE,
            );
            eprintln!("  layer {layer:>2}: residual(hidden) max|x| = {hmx:.1}");
        }

        eprintln!(
            "=== bf16 activation ranges (max|x| across {layers} layers) — f16 max = 65504 ==="
        );
        for (label, (mx, nf)) in &peak {
            let flag = if *mx > 65504.0 {
                "  <-- OVERFLOWS f16"
            } else if *mx > 16384.0 {
                "  (tight)"
            } else {
                ""
            };
            eprintln!(
                "  {label:<28} {mx:>12.1}{}{}",
                if *nf { " [non-finite!]" } else { "" },
                flag
            );
        }
        Ok(())
    }

    /// Per-stage GPU timing inside `encode_layer` + MoE grouped/post (one submit per stage×layer).
    pub(super) fn profile_encode_subprofile(&mut self) -> Result<EncodeSubProfileResult, Error> {
        let layout = self.layout;
        let layers = self.layers;
        let st_before: CanvasState = read_struct(&self.bufs.state);
        let first_step = if st_before.step == 0 { 1u32 } else { 0u32 };

        self.dispatch_and_wait(|enc| enc.encode_step_preamble(&layout, first_step))?;

        let mut layer_prof = LayerEncodeSubProfile::default();
        let mut moe_prof = MoeEncodeSubProfile::default();

        for layer in 0..layers {
            layer_prof.qkv_gemm +=
                self.time_enc_stage(|e| e.encode_layer_qkv_gemm(layer, &layout))?;
            layer_prof.qk_rope_kv +=
                self.time_enc_stage(|e| e.encode_layer_qk_rope_kv_dispatch(layer, &layout))?;
            layer_prof.attention +=
                self.time_enc_stage(|e| e.encode_layer_attention_dispatch(layer, &layout))?;
            layer_prof.o_proj_gemm +=
                self.time_enc_stage(|e| e.encode_layer_o_proj_gemm(layer, &layout))?;
            layer_prof.o_proj_tail +=
                self.time_enc_stage(|e| e.encode_layer_o_proj_tail(layer, &layout))?;

            let l = &layout.layers[layer];
            layer_prof.dense_pre_norm += self.time_enc_stage(|e| {
                e.rmsnorm(
                    e.arena().stream_off(),
                    e.arena().tmp_off(),
                    l.pre_ff_ln,
                    e.dims.hid as u32,
                    e.dims.canvas,
                );
                Ok(())
            })?;
            layer_prof.dense_gate_up +=
                self.time_enc_stage(|e| e.encode_layer_dense_gate_up(layer, &layout))?;
            layer_prof.dense_glu += self.time_enc_stage(|e| {
                e.glu(
                    e.arena().ffg_off(),
                    e.arena().ffu_off(),
                    e.arena().ffg_off(),
                    e.dims.canvas * e.dims.dense_ff,
                );
                Ok(())
            })?;
            layer_prof.dense_down +=
                self.time_enc_stage(|e| e.encode_layer_dense_down(layer, &layout))?;
            layer_prof.dense_post_norm += self.time_enc_stage(|e| {
                e.rmsnorm(
                    e.arena().dense_off(),
                    e.arena().dense_off(),
                    l.post_ff_ln_1,
                    e.dims.hid as u32,
                    e.dims.canvas,
                );
                Ok(())
            })?;
            layer_prof.router +=
                self.time_enc_stage(|e| e.encode_layer_router_buckets(layer, &layout))?;
        }

        for layer in 0..layers {
            moe_prof.gather +=
                self.time_enc_stage(|e| e.encode_moe_batched_gather_bf16_to_f32())?;
            moe_prof.gate_up +=
                self.time_enc_stage(|e| e.encode_moe_batched_gate_up(layer, &layout))?;
            moe_prof.swiglu += self.time_enc_stage(|e| e.encode_moe_batched_swiglu())?;
            moe_prof.down += self.time_enc_stage(|e| e.encode_moe_batched_down(layer, &layout))?;
            moe_prof.scatter += self.time_enc_stage(|e| e.encode_moe_batched_scatter())?;
        }

        for layer in 0..layers {
            moe_prof.post +=
                self.time_enc_stage(|e| e.encode_layer_moe_post_norm(layer, &layout))?;
            moe_prof.post +=
                self.time_enc_stage(|e| e.encode_layer_moe_post_combine(layer, &layout))?;
        }

        Ok(EncodeSubProfileResult {
            compile: std::time::Duration::ZERO,
            layers,
            layer: layer_prof,
            moe: moe_prof,
        })
    }

    /// P2.2 Phase A: one command buffer + one GPU sync per denoise step.
    /// Holistic prefill proxy: time ONE M=1024 super-chunk forward at
    /// kv=`kv_len` — all stages (QKV/o-proj/dense GEMMs, rope, attention per sub,
    /// router, MoE-expert GEMM, norms) interleaved in one command buffer exactly
    /// as production, with real weights. This is the true per-super-chunk cost;
    /// summed over the kv brackets a prompt sweeps it estimates real prefill,
    /// but at ~1-2s instead of the full 100-560s. Timing is KV-value-independent
    /// (attention streams kv_len rows regardless), so no actual prefill is
    /// needed — just set kv_len. Uses whatever tile config the flags compiled the
    /// pipelines with. Returns mean ms/super-chunk (min-of-rounds).
    pub(crate) fn bench_prefill_super(
        &mut self,
        kv_len: u32,
        iters: usize,
        n_subs: usize,
    ) -> Result<std::time::Duration, Error> {
        use std::time::Instant;
        let layout = self.layout;
        let layers = self.layers;
        let n_subs = n_subs.clamp(1, PREFILL_SUBS);
        let m = n_subs * self.dims.canvas;
        let ids = vec![0u32; m];
        self.set_canvas_ids(&ids)?;
        self.set_kv_len(kv_len);
        self.write_params_sub(kv_len, n_subs);
        self.arena_f16_mode = self.pipelines_prefill_f16.is_some();
        // 1 warm-up round + min over timed rounds (factors out clock ramp).
        let mut best = std::time::Duration::MAX;
        for round in 0..(iters.max(1) + 1) {
            let t = Instant::now();
            self.dispatch_and_wait(|enc| enc.encode_prefill_super(&layout, layers, n_subs))?;
            if round > 0 {
                best = best.min(t.elapsed());
            }
        }
        self.arena_f16_mode = false;
        Ok(best)
    }

    /// Floor decomposition (diagnostic): time the full super-chunk, then re-time
    /// with each stage-GROUP ablated — the delta is that group's cost. Reuses the
    /// real weights/shapes at M=1024; timing is data-independent so ablated
    /// stages just feed stale arena data downstream (no correctness needed).
    /// Returns (full, [(label, group_ms)]).
    pub(crate) fn bench_prefill_super_stages(
        &mut self,
        kv_len: u32,
        iters: usize,
        n_subs: usize,
    ) -> Result<(f64, Vec<(&'static str, f64)>), Error> {
        use step_schedule::StepStage;
        use step_schedule::StepStage::*;
        let groups: &[(&str, &[StepStage])] = &[
            ("qk_rope_kv", &[LayerQkRopeKv]),
            ("attn_only", &[LayerAttention]),
            ("qkv_proj+inorm", &[LayerInputNormQkv]),
            ("o_proj", &[LayerOProjPostAttn]),
            ("router", &[LayerRouter]),
            (
                "moe_experts",
                &[
                    MoeBatchedGather,
                    MoeBatchedGateUp,
                    MoeBatchedSwiglu,
                    MoeBatchedDown,
                    MoeBatchedScatter,
                    MoeGroupedScalar,
                ],
            ),
            ("dense_ffn", &[LayerDenseFfn]),
            (
                "moe_postnorm+combine",
                &[LayerMoePostNorm, LayerMoePostCombine],
            ),
            ("embed", &[EmbedGather]),
        ];
        let full = self
            .bench_prefill_super(kv_len, iters, n_subs)?
            .as_secs_f64()
            * 1e3;
        let mut out = Vec::new();
        for (label, stages) in groups {
            set_prefill_bench_skip(stages);
            let ablated = self
                .bench_prefill_super(kv_len, iters, n_subs)?
                .as_secs_f64()
                * 1e3;
            set_prefill_bench_skip(&[]);
            out.push((*label, (full - ablated).max(0.0)));
        }
        Ok((full, out))
    }

    pub(super) fn run_forward_once(&mut self, finish: StepFinishMode) -> Result<(), Error> {
        if let Some(ref dbg) = self.bufs.debug_status {
            crate::metal::debug_status::zero_buffer(dbg);
        }
        let layout = self.layout;
        let layers = self.layers;
        let st_before: CanvasState = read_struct(&self.bufs.state);
        if crate::metal::embed::sc_log_enabled() && st_before.step >= 1 {
            let elems = self.dims.canvas * self.dims.vocab;
            let sample = elems.min(8192);
            let (nf, mx) = half_buffer_stats(&self.bufs.logits, 0, elems, sample);
            eprintln!(
                "monolithic pre-sc: st.step={} logits_max_abs={:.4} non_finite_sample={}",
                st_before.step, mx, nf
            );
        }
        let first_step = if st_before.step == 0 { 1u32 } else { 0u32 };
        let partial_lm_m = partial_lm_active_rows(&st_before);
        // Denoise (Full) may run a narrowed canvas (shrink-on-retry);
        // ForwardOnly (prefill/dump) always uses the full CANVAS sub width.
        let active_canvas = if finish == StepFinishMode::Full {
            crate::flags::force_canvas()
                .map(|w| (w as usize).clamp(1, self.dims.canvas))
                .unwrap_or(self.active_canvas)
                .clamp(1, self.dims.canvas)
        } else {
            self.dims.canvas
        };
        self.dispatch_and_wait(|enc| {
            enc.partial_lm_m = partial_lm_m;
            enc.active_canvas = active_canvas;
            enc.forward_m = active_canvas;
            enc.interpret_step(&layout, layers, first_step, finish)
        })?;
        self.check_debug_status()
    }
}
