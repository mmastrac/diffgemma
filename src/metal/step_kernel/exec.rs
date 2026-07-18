//! The execution child of `step_kernel`: `StepEnc` (per-step encode of
//! every kernel dispatch), `StepRuntime` (buffers/pipelines/session state),
//! and `build_step_runtime` + the shared-pipeline cache. Carved from mod.rs
//! as ONE unit (PLAN: the struct-literal construction and the 50-field
//! pipeline struct move together); a child module, so parent privates
//! (StepPipelines, StepBuffers, params, arena helpers) are reachable by
//! ancestry.

use super::*;

pub(super) struct StepEnc<'a> {
    enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    pub(super) ctx: &'a MetalContext,
    ps: &'a StepPipelines,
    pub(super) bufs: &'a StepBuffers,
    pub(super) block_profile: StepBlockProfile,
    tensor_offsets: &'a HashMap<String, u64>,
    /// Active canvas rows for lm_head (P2.5); `CANVAS` when full lm_head.
    partial_lm_m: u32,
    /// Attention + dense-FFN weights are stored q8 (mixed-precision .dgq): route
    /// their GEMMs through the q8 kernel and skip the q4-only fused stacked path.
    attn_ffn_q8: bool,
    /// Attention + dense-FFN weights are stored bf16 (Raw): route their GEMMs
    /// through the bf16 kernel and skip the q4-only fused stacked path.
    attn_ffn_bf16: bool,
    /// Embed (tied lm_head + SC soft-embed) is stored bf16 (Raw) rather than
    /// q8-per-row: dispatch the bf16 gather / lm_head / softembed paths.
    embed_bf16: bool,
    /// Prefill mode: attention is CAUSAL (scalar kernel only; mma variants have no
    /// causal mask) and the SC/sampler/lm_head stages are skipped (KV-only forward).
    prefill_causal: bool,
    /// Rows in this forward: CANVAS for denoise / plain prefill chunks;
    /// n_subs*CANVAS for a batched prefill super-chunk. Every row-independent
    /// stage (embed, norms, QKV/o_proj/FFN GEMMs, router, MoE) dispatches at
    /// this M; attention + rope/KV-write stay per-CANVAS-sub-chunk.
    forward_m: usize,
    /// Active canvas width for a DENOISE step (E3/E6 shrink-on-retry): the
    /// number of canvas rows actually denoised, `CANVAS` (256) normally, less
    /// on a narrowed retry. Drives the denoise-only width sites (attention
    /// seq-len + grid, SC softembed, lm_head, sampler) and `forward_m`. Stays
    /// `CANVAS` for prefill chunks/super-chunks (their sub width is 256).
    active_canvas: usize,
    /// Current sub-chunk (0..PREFILL_SUBS) for the per-sub attention/rope
    /// dispatches of a super-chunk: arena rows offset by sub_c*CANVAS and
    /// StepParams come from the params_sub slot (kv_len differs per sub).
    sub_c: usize,
    /// Bind params from bufs.params_sub[sub_c] instead of bufs.params.
    use_params_sub: bool,
    /// Model sliding-window size (Gemma-4: 1024) for sliding-attention layers.
    sliding_window: u32,
}

impl<'a> StepEnc<'a> {
    #[inline]
    pub(super) fn arena(&self) -> &ArenaLayout {
        &self.bufs.arena_map
    }
}

impl StepEnc<'_> {
    fn sink_set_pipeline(&mut self, ps: &ComputePipeline) {
        self.enc.setComputePipelineState(&ps.pipeline);
    }

    fn sink_set_buffer(
        &mut self,
        buf: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
        index: usize,
    ) {
        unsafe {
            self.enc.setBuffer_offset_atIndex(Some(buf), offset, index);
        }
    }

    fn sink_set_bytes<T: Copy>(&mut self, val: &T, index: usize) {
        crate::metal::batch::set_bytes(&self.enc, val, index);
    }

    #[allow(dead_code)]
    fn bind_arena_layout_buf(&mut self, index: usize) {
        self.sink_set_buffer(&self.bufs.arena_layout_buf, 0, index);
    }

    fn bind_debug_status(&mut self, index: usize) {
        if let Some(ref dbg) = self.bufs.debug_status {
            self.sink_set_buffer(dbg, 0, index);
        }
    }

    fn sink_dispatch(&mut self, grid: MTLSize, tg: MTLSize) {
        self.enc
            .dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }

    /// Buffer-scope memory barrier on the live encoder.
    fn sink_memory_barrier(&mut self) {
        self.enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
    }

    fn sink_dispatch_indirect(&mut self, indirect_offset: usize, _n: u32, tg: MTLSize) {
        unsafe {
            self.enc
                .dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                    &self.bufs.moe_grouped_indirect,
                    indirect_offset,
                    tg,
                );
        }
    }

    fn bind_blob(&mut self, idx: usize) {
        self.sink_set_buffer(&self.bufs.blob, 0, idx);
    }

    /// Expert weights live in the second blob region on split blobs; job
    /// offsets are rebased to match (layer_moe_block_jobs_impl expert_base).
    fn bind_blob_experts(&mut self, idx: usize) {
        let buf = self.bufs.blob_experts.clone();
        self.sink_set_buffer(&buf, 0, idx);
    }

    fn bind_params(&mut self, idx: usize) {
        if self.use_params_sub {
            self.sink_set_buffer(
                &self.bufs.params_sub,
                self.sub_c * std::mem::size_of::<StepParams>(),
                idx,
            );
        } else {
            self.sink_set_buffer(&self.bufs.params, 0, idx);
        }
    }

    /// kv_len the NEXT rope/attention dispatch will see (per-sub slot during a
    /// batched prefill super-chunk, else the shared params).
    fn dispatch_kv_len(&self) -> u32 {
        if self.use_params_sub {
            let ptr = self.bufs.params_sub.contents().as_ptr() as *const StepParams;
            unsafe { (*ptr.add(self.sub_c)).kv_len }
        } else {
            read_struct::<StepParams>(&self.bufs.params).kv_len
        }
    }

    fn bind_kvcache(&mut self, idx: usize) {
        self.sink_set_buffer(&self.bufs.kvcache, 0, idx);
    }

    fn bind_state(&mut self, idx: usize) {
        self.sink_set_buffer(&self.bufs.state, 0, idx);
    }

    fn bind_logits(&mut self, idx: usize) {
        self.sink_set_buffer(&self.bufs.logits, 0, idx);
    }

    fn bind_sc_probs(&mut self, idx: usize) {
        self.sink_set_buffer(&self.bufs.sc_probs, 0, idx);
    }

    fn bind_route(&mut self, idx: usize) {
        self.sink_set_buffer(&self.bufs.route, 0, idx);
    }

    fn dispatch_1d(&mut self, ps: &ComputePipeline, count: usize, tpg: usize) {
        self.sink_set_pipeline(ps);
        let tg_w = tpg.min(count.max(1));
        let grid = MTLSize {
            width: div_up(count, tg_w),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
    }

    /// Split 1D dispatches that would exceed Metal's 65535 threadgroup grid width.
    fn dispatch_1d_ranged(
        &mut self,
        ps: &ComputePipeline,
        count: usize,
        tpg: usize,
        mut encode: impl FnMut(&mut Self, u32, u32),
    ) {
        const MAX_GROUPS: usize = 65535;
        let chunk_max = MAX_GROUPS * tpg;
        let mut base = 0usize;
        while base < count {
            let chunk = (count - base).min(chunk_max);
            self.sink_set_pipeline(ps);
            encode(self, base as u32, chunk as u32);
            let tg_w = tpg.min(chunk.max(1));
            let grid = MTLSize {
                width: div_up(chunk, tg_w),
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            };
            self.sink_dispatch(grid, tg);
            base += chunk;
        }
    }

    /// Softcap logits (matches ranged logit_softcapping dispatch pattern).
    pub(super) fn dispatch_softcap(&mut self) {
        let len = CANVAS * VOCAB;
        self.dispatch_1d_ranged(&self.ps.softcap, len, 256, |this, base, chunk| {
            this.sink_set_buffer(&this.bufs.logits, 0, 0);
            this.sink_set_bytes(&base, 1);
            this.sink_set_bytes(&chunk, 2);
            this.sink_set_buffer(&this.bufs.dummy_dump, 0, 3);
            this.bind_debug_status(4);
        });
    }

    fn gemm_q4(
        &mut self,
        x_off: u64,
        y_off: u64,
        w_off: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), Error> {
        // Attention/dense-FFN weights are the only tensors routed through gemm_q4;
        // in mixed-precision .dgq they are stored bf16 (or q8 on older checkpoints),
        // so dispatch the matching kernel.
        if self.attn_ffn_bf16 {
            return self.gemm_bf16(x_off, y_off, w_off, m, n, k);
        }
        if self.attn_ffn_q8 {
            return self.gemm_q8(x_off, y_off, w_off, m, n, k);
        }
        let ps = self.ps.block_gemm(self.block_profile.format, n, k)?;
        self.sink_set_pipeline(ps);
        self.sink_set_buffer(&self.bufs.arena, x_off as usize, 0);
        self.sink_set_buffer(&self.bufs.arena, y_off as usize, 1);
        self.bind_blob(2);
        self.sink_set_bytes(&w_off, 3);
        self.sink_set_bytes(&m, 4);
        let grid = MTLSize {
            width: div_up(n as usize, crate::flags::gemm_tune_tile().1),
            height: div_up(m as usize, crate::flags::gemm_tune_tile().0),
            depth: 1,
        };
        let tg = MTLSize {
            width: GEMM_THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    fn gemm_q4_stacked(
        &mut self,
        x_off: u64,
        segs: &[crate::shaders::gemm_block_stacked::GemmStackedSeg],
        m: u32,
        k: u32,
        n_total: u32,
    ) -> Result<(), Error> {
        debug_assert!(segs.len() <= STACKED_SEG_MAX, "too many stacked segments");
        let ps = crate::shaders::gemm_tunable::stacked_pipeline_for(
            self.ctx,
            n_total,
            k,
            self.block_profile.format,
            segs,
        )?;
        self.sink_set_pipeline(ps.as_ref());
        self.sink_set_buffer(&self.bufs.arena, x_off as usize, 0);
        self.sink_set_buffer(&self.bufs.arena, 0, 1);
        self.bind_blob(2);
        self.sink_set_bytes(&m, 3);
        let grid = MTLSize {
            width: div_up(n_total as usize, crate::flags::gemm_tune_tile().1),
            height: div_up(m as usize, crate::flags::gemm_tune_tile().0),
            depth: 1,
        };
        let tg = MTLSize {
            width: GEMM_THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    /// bf16 fused stacked GEMM (QKV / gate+up on the bf16 path) — same N-segment
    /// layout as `gemm_q4_stacked` but reads bf16 weights (no dequant).
    fn gemm_bf16_stacked(
        &mut self,
        x_off: u64,
        segs: &[crate::shaders::gemm_block_stacked::GemmStackedSeg],
        m: u32,
        k: u32,
        n_total: u32,
    ) -> Result<(), Error> {
        debug_assert!(segs.len() <= STACKED_SEG_MAX, "too many stacked segments");
        let ps = crate::shaders::gemm_tunable::stacked_pipeline_for(
            self.ctx,
            n_total,
            k,
            crate::shaders::QuantFormat::Raw,
            segs,
        )?;
        let grid = MTLSize {
            width: div_up(n_total as usize, crate::flags::gemm_tune_tile().1),
            height: div_up(m as usize, crate::flags::gemm_tune_tile().0),
            depth: 1,
        };
        self.sink_set_pipeline(ps.as_ref());
        self.sink_set_buffer(&self.bufs.arena, x_off as usize, 0);
        self.sink_set_buffer(&self.bufs.arena, 0, 1);
        self.bind_blob(2);
        self.sink_set_bytes(&m, 3);
        let tg = MTLSize {
            width: GEMM_THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    fn memzero_bytes(&mut self, byte_off: u64, nbytes: u64) {
        self.sink_set_pipeline(&self.ps.memzero);
        self.sink_set_buffer(&self.bufs.arena, byte_off as usize, 0);
        // memzero_bytes.metal zeros one uchar4 (4 bytes) per thread, so count is
        // div_up(nbytes, 4). (Was /16 — only cleared a quarter of the range,
        // which left the chunked SC f32 accumulator stale past row 64 → NONDET-SC-1.)
        let count = div_up(nbytes as usize, 4);
        self.dispatch_1d(&self.ps.memzero, count, 256);
    }

    /// Zero an arbitrary buffer (e.g. `gemm_b` scratch) — used by chunked SC softembed f32 accumulator.
    fn memzero_buffer(
        &mut self,
        buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        nbytes: u64,
    ) {
        self.sink_set_pipeline(&self.ps.memzero);
        self.sink_set_buffer(buf, 0, 0);
        // 4 bytes (one uchar4) per thread — see memzero_bytes.
        let count = div_up(nbytes as usize, 4);
        self.dispatch_1d(&self.ps.memzero, count, 256);
    }

    pub(super) fn rmsnorm(&mut self, x_off: u64, y_off: u64, w_off: u64, dim: u32, rows: usize) {
        self.sink_set_pipeline(&self.ps.rmsnorm);
        self.sink_set_buffer(&self.bufs.arena, x_off as usize, 0);
        self.sink_set_buffer(&self.bufs.arena, y_off as usize, 1);
        self.bind_blob(2);
        self.sink_set_bytes(&w_off, 3);
        self.sink_set_bytes(&dim, 4);
        let grid = MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
    }

    fn rmsnorm_f32(&mut self, x_off: u64, y_off: u64, w_off: u64, dim: u32, rows: usize) {
        self.sink_set_pipeline(&self.ps.rmsnorm_f32);
        self.sink_set_buffer(&self.bufs.arena, x_off as usize, 0);
        self.sink_set_buffer(&self.bufs.arena, y_off as usize, 1);
        self.bind_blob(2);
        self.sink_set_bytes(&w_off, 3);
        self.sink_set_bytes(&dim, 4);
        let grid = MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
    }

    fn gemm_q8(
        &mut self,
        x_off: u64,
        y_off: u64,
        w_off: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), Error> {
        let ps = self.ps.dense_q8(n, k)?;
        let grid = MTLSize {
            width: div_up(n as usize, crate::flags::gemm_tune_tile().1),
            height: div_up(m as usize, crate::flags::gemm_tune_tile().0),
            depth: 1,
        };
        self.sink_set_pipeline(ps);
        self.sink_set_buffer(&self.bufs.arena, x_off as usize, 0);
        self.sink_set_buffer(&self.bufs.arena, y_off as usize, 1);
        self.bind_blob(2);
        self.sink_set_bytes(&w_off, 3);
        self.sink_set_bytes(&m, 4);
        let tg = MTLSize {
            width: GEMM_THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    /// bf16-weight GEMM (mixed-precision attention/dense-FFN). Same shape/dispatch
    /// as gemm_q8; weights at `w_off` are bf16 [N,K] (no dequant).
    fn gemm_bf16(
        &mut self,
        x_off: u64,
        y_off: u64,
        w_off: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), Error> {
        let ps = self.ps.dense_raw(n, k)?;
        let grid = MTLSize {
            width: div_up(n as usize, crate::flags::gemm_tune_tile().1),
            height: div_up(m as usize, crate::flags::gemm_tune_tile().0),
            depth: 1,
        };
        self.sink_set_pipeline(ps);
        self.sink_set_buffer(&self.bufs.arena, x_off as usize, 0);
        self.sink_set_buffer(&self.bufs.arena, y_off as usize, 1);
        self.bind_blob(2);
        self.sink_set_bytes(&w_off, 3);
        self.sink_set_bytes(&m, 4);
        let tg = MTLSize {
            width: GEMM_THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    /// Tied lm_head with bf16 embed: logits = hidden @ embed^T. Reuses the bf16
    /// GEMM kernel (writes the logits buffer instead of the arena).
    fn gemm_bf16_logits(
        &mut self,
        x_off: u64,
        w_off: u64,
        m: u32,
        n: u32,
        k: u32,
        logits_byte_off: usize,
    ) -> Result<(), Error> {
        let ps = self.ps.dense_raw(n, k)?;
        let grid = MTLSize {
            width: div_up(n as usize, crate::flags::gemm_tune_tile().1),
            height: div_up(m as usize, crate::flags::gemm_tune_tile().0),
            depth: 1,
        };
        self.sink_set_pipeline(ps);
        self.sink_set_buffer(&self.bufs.arena, x_off as usize, 0);
        self.sink_set_buffer(&self.bufs.logits, logits_byte_off, 1);
        self.bind_blob(2);
        self.sink_set_bytes(&w_off, 3);
        self.sink_set_bytes(&m, 4);
        let tg = MTLSize {
            width: GEMM_THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    pub(super) fn gemm_q8_logits(
        &mut self,
        x_off: u64,
        w_off: u64,
        m: u32,
        n: u32,
        k: u32,
        logits_byte_off: usize,
    ) -> Result<(), Error> {
        let ps = self.ps.dense_q8(n, k)?;
        let grid = MTLSize {
            width: div_up(n as usize, crate::flags::gemm_tune_tile().1),
            height: div_up(m as usize, crate::flags::gemm_tune_tile().0),
            depth: 1,
        };
        self.sink_set_pipeline(ps);
        self.sink_set_buffer(&self.bufs.arena, x_off as usize, 0);
        self.sink_set_buffer(&self.bufs.logits, logits_byte_off, 1);
        self.bind_blob(2);
        self.sink_set_bytes(&w_off, 3);
        self.sink_set_bytes(&m, 4);
        let tg = MTLSize {
            width: GEMM_THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    /// probs [M,K] half buffer → arena y_off [M,N] via q8 weights indexed by K.
    #[allow(dead_code)]
    fn gemm_q8_rowk_half(
        &mut self,
        x_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        y_off: u64,
        w_off: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), Error> {
        // x (sc_probs) is fp16 (10-mantissa probs, sc_probs.metal): use the
        // fp16-input pipeline so the prob precision survives into the GEMM tile.
        let ps = self.ps.q8_rowk_xfp16(n, k)?;
        self.sink_set_pipeline(ps);
        self.sink_set_buffer(x_buf, 0, 0);
        self.sink_set_buffer(&self.bufs.arena, y_off as usize, 1);
        self.bind_blob(2);
        self.sink_set_bytes(&w_off, 3);
        self.sink_set_bytes(&m, 4);
        let grid = MTLSize {
            width: div_up(n as usize, 32),
            height: div_up(m as usize, crate::shaders::gemm_common::M_TILE),
            depth: 1,
        };
        let tg = MTLSize {
            width: GEMM_THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    /// probs [M,K] half buffer @ sc_probs → arena y_off [M,N] via q8 weights.
    fn gemm_q8_rowk_acc_f32(
        &mut self,
        y_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        w_off: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), Error> {
        let ps = self
            .ps
            .gemm_q8_rowk_acc_f32
            .get(&(n, k))
            .ok_or(Error::Gpu("missing gemm_q8_rowk_acc_f32 pipeline"))?;
        self.sink_set_pipeline(ps);
        self.sink_set_buffer(&self.bufs.sc_probs, 0, 0);
        self.sink_set_buffer(y_buf, 0, 1);
        self.bind_blob(2);
        self.sink_set_bytes(&w_off, 3);
        self.sink_set_bytes(&m, 4);
        let grid = MTLSize {
            width: div_up(n as usize, 32),
            height: div_up(m as usize, crate::shaders::gemm_common::M_TILE),
            depth: 1,
        };
        let tg = MTLSize {
            width: GEMM_THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    /// bf16-embed variant of `gemm_q8_rowk_acc_f32` (chunked SC softembed accumulate).
    fn gemm_bf16_rowk_acc_f32(
        &mut self,
        y_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        w_off: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), Error> {
        let ps = self
            .ps
            .gemm_bf16_rowk_acc_f32
            .get(&(n, k))
            .ok_or(Error::Gpu("missing gemm_bf16_rowk_acc_f32 pipeline"))?;
        self.sink_set_pipeline(ps);
        self.sink_set_buffer(&self.bufs.sc_probs, 0, 0);
        self.sink_set_buffer(y_buf, 0, 1);
        self.bind_blob(2);
        self.sink_set_bytes(&w_off, 3);
        self.sink_set_bytes(&m, 4);
        let grid = MTLSize {
            width: div_up(n as usize, 32),
            height: div_up(m as usize, crate::shaders::gemm_common::M_TILE),
            depth: 1,
        };
        let tg = MTLSize {
            width: GEMM_THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    /// Convert f32 buffer → bf16 arena slot with scale: `arena[base+i] = f32_buf[i] * scale`.
    fn f32_to_half_scale(
        &mut self,
        src_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
        y_off: u64,
        len: usize,
        scale: f32,
    ) {
        // convert_scale (src_f32=true, dst_f32=false): src @0, arena dst @1.
        // Arena is bound at `y_off`, so both base offsets are 0 (the binding
        // offset already places the slot; passing `y_off` again would double it).
        self.sink_set_pipeline(&self.ps.f32_to_half_scale);
        self.sink_set_buffer(src_buf, 0, 0);
        self.sink_set_buffer(&self.bufs.arena, y_off as usize, 1);
        self.sink_set_bytes(&0u32, 2); // src_base
        self.sink_set_bytes(&0u32, 3); // dst_base
        self.sink_set_bytes(&(len as u32), 4);
        self.sink_set_bytes(&scale, 5);
        self.sink_set_buffer(&self.bufs.dummy_dump, 0, 6);
        self.dispatch_1d(&self.ps.f32_to_half_scale, len, 256);
    }

    #[allow(dead_code)]
    fn scale_half_arena(&mut self, y_off: u64, elems: usize, scale: f32) {
        // convert_scale (arena->arena, in-place): same buffer @0 and @1.
        self.sink_set_pipeline(&self.ps.half_scale);
        self.sink_set_buffer(&self.bufs.arena, y_off as usize, 0);
        self.sink_set_buffer(&self.bufs.arena, y_off as usize, 1);
        self.sink_set_bytes(&0u32, 2); // src_base
        self.sink_set_bytes(&0u32, 3); // dst_base
        self.sink_set_bytes(&(elems as u32), 4);
        self.sink_set_bytes(&scale, 5);
        self.sink_set_buffer(&self.bufs.dummy_dump, 0, 6);
        self.dispatch_1d(&self.ps.half_scale, elems, 256);
    }

    fn scale_half_logits(&mut self, elems: usize, scale: f32) {
        // Same convert_scale kernel as scale_half_arena; only the bound buffer
        // differs (logits vs arena). In-place bf16 scale, same buffer @0 and @1.
        self.sink_set_pipeline(&self.ps.half_scale);
        self.bind_logits(0);
        self.bind_logits(1);
        self.sink_set_bytes(&0u32, 2); // src_base
        self.sink_set_bytes(&0u32, 3); // dst_base
        self.sink_set_bytes(&(elems as u32), 4);
        self.sink_set_bytes(&scale, 5);
        self.sink_set_buffer(&self.bufs.dummy_dump, 0, 6);
        self.dispatch_1d(&self.ps.half_scale, elems, 256);
    }

    pub(super) fn encode_sc_logit_rowstats(&mut self) {
        self.sink_set_pipeline(&self.ps.logit_rowstats);
        self.bind_logits(0);
        self.sink_set_buffer(&self.bufs.arena, self.arena().rs_sc_off() as usize, 1);
        let dims = [CANVAS as u32, VOCAB as u32];
        self.sink_set_bytes(&dims, 2);
        self.bind_debug_status(3);
        let (grid, tg) = crate::shaders::logit_rowstats::dispatch_shape(CANVAS);
        self.sink_dispatch(grid, tg);
    }

    fn encode_sc_softembed(&mut self, layout: &ModelLayout) -> Result<(), Error> {
        if self.embed_bf16 && sc_sparse_enabled() {
            return self.encode_sc_softembed_sparse(layout);
        }
        self.encode_sc_softembed_exact(layout)
    }

    fn dispatch_sc_prob_cols(&mut self, v0: u32, chunk: u32) {
        self.sink_set_pipeline(&self.ps.sc_prob_cols);
        self.bind_logits(0);
        self.sink_set_buffer(&self.bufs.arena, self.arena().rs_sc_off() as usize, 1);
        self.bind_sc_probs(2);
        let params = [CANVAS as u32, VOCAB as u32, v0, chunk];
        self.sink_set_bytes(&params, 3);
        self.bind_debug_status(4);
        let (grid, tg) = crate::shaders::sc_prob_cols::dispatch_shape(CANVAS, chunk as usize);
        self.sink_dispatch(grid, tg);
    }

    /// Vocab-chunked softembed: rowstats once, then chunk GEMMs (no full prob matrix).
    /// Accumulates in f32 (in `gemm_b`) to match full-path precision; converts to half once at the end.
    fn encode_sc_softembed_chunked(&mut self, layout: &ModelLayout) -> Result<(), Error> {
        use crate::dgq::layout::q8_row_bytes;
        use crate::model::embed::LM_HEAD_CHUNK;

        // f32 accumulator in gemm_b (free during preamble, before layer GEMMs).
        let acc_bytes = (CANVAS * HID * std::mem::size_of::<f32>()) as u64;
        self.memzero_buffer(&self.bufs.gemm_b, acc_bytes);
        self.sink_memory_barrier(); // memzero gemm_b before the first `+=`

        let row_bytes = q8_row_bytes(HID) as u64;
        let chunk_max = LM_HEAD_CHUNK as u32;
        let mut v0 = 0u32;
        while v0 < VOCAB as u32 {
            let chunk = (VOCAB as u32 - v0).min(chunk_max);
            self.dispatch_sc_prob_cols(v0, chunk);
            if self.embed_bf16 {
                let w_off = layout.embed + (v0 as u64) * (HID as u64) * 2;
                self.gemm_bf16_rowk_acc_f32(
                    &self.bufs.gemm_b,
                    w_off,
                    CANVAS as u32,
                    HID as u32,
                    chunk,
                )?;
            } else {
                let w_off = layout.embed + (v0 as u64) * row_bytes;
                self.gemm_q8_rowk_acc_f32(
                    &self.bufs.gemm_b,
                    w_off,
                    CANVAS as u32,
                    HID as u32,
                    chunk,
                )?;
            }
            self.sink_memory_barrier(); // serialize the cross-chunk `+=` into gemm_b
            v0 += chunk;
        }
        // Convert f32 accumulator → bf16 arena soft slot, applying embed_scale
        // (== sqrt(HID)) and dividing out the SC_PROB_GEMM_SCALE that sc_prob_cols
        // multiplied into the probs to keep them in fp16's normal range.
        let scale = (HID as f32).sqrt() / SC_PROB_GEMM_SCALE;
        self.f32_to_half_scale(
            &self.bufs.gemm_b,
            self.arena().soft_off(),
            CANVAS * HID,
            scale,
        );
        Ok(())
    }

    /// Sparse SC softembed: select per-row survivors (prob within e^-10 of row
    /// max), then gather-weighted-sum their embed rows — instead of the full vocab
    /// GEMM. APPROXIMATE (drops the prob tail). Scratch: prob+cnt in gemm_a, idx +
    /// f32 accumulator in gemm_b (both free during preamble). rowstat from the
    /// prior ScLogitRowstats stage.
    fn encode_sc_softembed_sparse(&mut self, layout: &ModelLayout) -> Result<(), Error> {
        let maxk = SC_SPARSE_MAXK;
        // gemm_b: [0..acc_bytes) = f32 accumulator; idx at IDX_OFF.
        // gemm_a: [0..) = fp16 prob; cnt at CNT_OFF.
        const IDX_OFF: usize = 4 * 1024 * 1024;
        const PROB_OFF: usize = 0;
        const CNT_OFF: usize = 4 * 1024 * 1024;

        // Pass 1: per-row threshold select + compact.
        self.sink_set_pipeline(&self.ps.sc_sparse_select);
        self.bind_logits(0);
        self.sink_set_buffer(&self.bufs.arena, self.arena().rs_sc_off() as usize, 1);
        self.sink_set_buffer(&self.bufs.gemm_b, IDX_OFF, 2);
        self.sink_set_buffer(&self.bufs.gemm_a, PROB_OFF, 3);
        self.sink_set_buffer(&self.bufs.gemm_a, CNT_OFF, 4);
        let p1 = [CANVAS as u32, VOCAB as u32, maxk, 0u32];
        self.sink_set_bytes(&p1, 5);
        self.sink_dispatch(
            MTLSize {
                width: CANVAS,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        self.sink_memory_barrier();

        // Pass 2: gather-weighted-sum embed rows → f32 accumulator (gemm_b[0..]).
        self.sink_set_pipeline(&self.ps.sc_sparse_gather);
        self.sink_set_buffer(&self.bufs.gemm_b, IDX_OFF, 0);
        self.sink_set_buffer(&self.bufs.gemm_a, PROB_OFF, 1);
        self.sink_set_buffer(&self.bufs.gemm_a, CNT_OFF, 2);
        self.bind_blob(3);
        self.sink_set_bytes(&layout.embed, 4);
        self.sink_set_buffer(&self.bufs.gemm_b, 0, 5);
        let p2 = [CANVAS as u32, HID as u32, maxk, 0u32];
        self.sink_set_bytes(&p2, 6);
        self.sink_dispatch(
            MTLSize {
                width: CANVAS,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        self.sink_memory_barrier();

        // Finalize: f32 accumulator → bf16 soft slot (÷ SC_PROB_GEMM_SCALE, × √HID).
        let scale = (HID as f32).sqrt() / SC_PROB_GEMM_SCALE;
        self.f32_to_half_scale(
            &self.bufs.gemm_b,
            self.arena().soft_off(),
            CANVAS * HID,
            scale,
        );
        Ok(())
    }

    /// Exact (non-sparse) softembed = the chunked path; sparse is the
    /// default approximation on bf16-embed models (DGQ_SC_SPARSE=0 opts out).
    pub(super) fn encode_sc_softembed_exact(&mut self, layout: &ModelLayout) -> Result<(), Error> {
        self.encode_sc_softembed_chunked(layout)
    }

    fn residual(&mut self, a_off: u64, b_off: u64, y_off: u64, scal_off: u64, elems: usize) {
        self.sink_set_buffer(&self.bufs.arena, a_off as usize, 0);
        self.sink_set_buffer(&self.bufs.arena, b_off as usize, 1);
        self.sink_set_buffer(&self.bufs.arena, y_off as usize, 2);
        self.bind_blob(3);
        self.sink_set_bytes(&scal_off, 4);
        self.dispatch_1d(&self.ps.residual, elems, 256);
    }

    fn glu(&mut self, gate_off: u64, up_off: u64, y_off: u64, elems: usize) {
        self.sink_set_pipeline(&self.ps.glu);
        let dims = [elems as u32, 0u32];
        self.sink_set_buffer(&self.bufs.arena, gate_off as usize, 0);
        self.sink_set_buffer(&self.bufs.arena, up_off as usize, 1);
        self.sink_set_buffer(&self.bufs.arena, y_off as usize, 2);
        self.sink_set_bytes(&dims, 3);
        self.dispatch_1d(&self.ps.glu, elems, 256);
    }

    pub(super) fn encode_layer(&mut self, layer: usize, layout: &ModelLayout) -> Result<(), Error> {
        self.encode_layer_qkv_gemm(layer, layout)?;
        self.encode_layer_qk_rope_kv_dispatch(layer, layout)?;
        self.encode_layer_attention_dispatch(layer, layout)?;
        self.encode_layer_o_proj_post_attn(layer, layout)?;
        self.encode_layer_dense_ffn(layer, layout)?;
        self.encode_layer_router_buckets(layer, layout)?;
        Ok(())
    }

    pub(super) fn encode_layer_o_proj_post_attn(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        self.encode_layer_o_proj_gemm(layer, layout)?;
        self.encode_layer_o_proj_tail(layer, layout)
    }

    fn encode_layer_o_proj_gemm(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        let fm = self.forward_m;
        let l = &layout.layers[layer];
        let o_k = if l.is_full != 0 { 8192 } else { 4096 };
        self.gemm_q4(
            self.arena().attno_off(),
            self.arena().tmp_off(),
            l.o_proj,
            fm as u32,
            HID as u32,
            o_k,
        )
    }

    fn encode_layer_o_proj_tail(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        let fm = self.forward_m;
        let l = &layout.layers[layer];
        self.rmsnorm(
            self.arena().tmp_off(),
            self.arena().tmp_off(),
            l.post_attn_ln,
            HID as u32,
            fm,
        );
        self.residual(
            self.arena().hidden_off(),
            self.arena().tmp_off(),
            self.arena().stream_off(),
            0,
            fm * HID,
        );
        Ok(())
    }

    pub(super) fn encode_layer_dense_ffn(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        let fm = self.forward_m;
        let l = &layout.layers[layer];
        self.rmsnorm(
            self.arena().stream_off(),
            self.arena().tmp_off(),
            l.pre_ff_ln,
            HID as u32,
            fm,
        );
        self.encode_layer_dense_gate_up(layer, layout)?;
        self.glu(
            self.arena().ffg_off(),
            self.arena().ffu_off(),
            self.arena().ffg_off(),
            fm * DENSE_FF as usize,
        );
        self.encode_layer_dense_down(layer, layout)?;
        self.rmsnorm(
            self.arena().dense_off(),
            self.arena().dense_off(),
            l.post_ff_ln_1,
            HID as u32,
            fm,
        );
        Ok(())
    }

    fn encode_layer_dense_down(&mut self, layer: usize, layout: &ModelLayout) -> Result<(), Error> {
        let fm = self.forward_m;
        let l = &layout.layers[layer];
        self.gemm_q4(
            self.arena().ffg_off(),
            self.arena().dense_off(),
            l.mlp_down,
            fm as u32,
            HID as u32,
            DENSE_FF,
        )
    }

    fn encode_layer_dense_gate_up(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        let fm = self.forward_m;
        let l = &layout.layers[layer];
        if fused_gate_up_enabled() && self.attn_ffn_bf16 {
            let (segs, n_total) = gate_up_stacked_segments(l, self.arena());
            self.gemm_bf16_stacked(
                self.arena().tmp_off(),
                &segs,
                fm as u32,
                HID as u32,
                n_total,
            )?;
        } else if fused_gate_up_enabled() && !self.attn_ffn_q8 && !self.attn_ffn_bf16 {
            let (segs, n_total) = gate_up_stacked_segments(l, self.arena());
            self.gemm_q4_stacked(
                self.arena().tmp_off(),
                &segs,
                fm as u32,
                HID as u32,
                n_total,
            )?;
        } else {
            self.gemm_q4(
                self.arena().tmp_off(),
                self.arena().ffg_off(),
                l.mlp_gate,
                fm as u32,
                DENSE_FF,
                HID as u32,
            )?;
            self.gemm_q4(
                self.arena().tmp_off(),
                self.arena().ffu_off(),
                l.mlp_up,
                fm as u32,
                DENSE_FF,
                HID as u32,
            )?;
        }
        Ok(())
    }

    pub(super) fn encode_layer_router_buckets(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        let fm = self.forward_m;
        let layer_off = layer_byte_offset(layer);
        let router_dims = crate::shaders::moe_router::RouterDims {
            canvas: fm as u32,
            hidden: HID as u32,
            n_experts: N_EXPERTS as u32,
            top_k: TOP_K as u32,
            router_hscale: (HID as f32).powf(-0.5),
            block_m: self.moe_block_m(),
        };
        if router_gemm_enabled() {
            // Router-as-GEMM: xn = rmsnorm_noscale(stream) * router_scale[d]
            // (exactly what fn rmsnorm computes with w=router_scale; same 1e-6
            // eps as MOE_ROUTER_RMS_EPS) -> bf16 GEMM against router_proj
            // (n=128 experts) into the free ffg plane -> top-k tail applies
            // the uniform router_hscale (linear in the input, so folding it
            // out of the GEMM is exact up to bf16 logit rounding).
            let l = &layout.layers[layer];
            self.rmsnorm(
                self.arena().stream_off(),
                self.arena().tmp_off(),
                l.router_scale,
                HID as u32,
                fm,
            );
            self.gemm_bf16(
                self.arena().tmp_off(),
                self.arena().ffg_off(),
                l.router_proj,
                fm as u32,
                N_EXPERTS as u32,
                HID as u32,
            )?;
            self.sink_set_pipeline(&self.ps.router_topk);
            self.sink_set_buffer(&self.bufs.arena, self.arena().ffg_off() as usize, 0);
            self.bind_blob(1);
            self.sink_set_buffer(&self.bufs.layout, layer_off as usize, 2);
            self.bind_route(3);
            self.sink_set_bytes(&router_dims, 4);
            self.bind_debug_status(5);
            let grid = MTLSize {
                width: fm.div_ceil(64),
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            };
            self.sink_dispatch(grid, tg);
        } else {
            self.sink_set_pipeline(&self.ps.router);
            self.sink_set_buffer(&self.bufs.arena, self.arena().stream_off() as usize, 0);
            self.bind_blob(1);
            self.sink_set_buffer(&self.bufs.layout, layer_off as usize, 2);
            self.bind_route(3);
            self.sink_set_bytes(&router_dims, 4);
            self.bind_debug_status(5);
            let grid = MTLSize {
                width: fm,
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            };
            self.sink_dispatch(grid, tg);
        }

        self.sink_set_pipeline(&self.ps.bucket_count);
        self.bind_route(0);
        let n_experts = N_EXPERTS as u32;
        self.sink_set_bytes(&n_experts, 1);
        self.dispatch_1d(&self.ps.bucket_count, 128, 128);

        for phase in 0u32..3 {
            self.sink_set_pipeline(&self.ps.bucket_fill);
            self.bind_route(0);
            self.sink_set_bytes(&phase, 1);
            self.sink_set_bytes(&router_dims, 2);
            self.sink_set_buffer(&self.bufs.expert_layer_unique, 0, 3);
            let layer_idx = layer as u32;
            self.sink_set_bytes(&layer_idx, 4);
            self.bind_debug_status(5);
            self.sink_set_buffer(&self.bufs.moe_grouped_indirect, 0, 6);
            let grid_info = moe_grouped_grid_info();
            self.sink_set_bytes(&grid_info, 7);
            let count = if phase == 1 { 1 } else { fm * TOP_K };
            self.dispatch_1d(&self.ps.bucket_fill, count, 256);
        }

        let l = &layout.layers[layer];
        self.rmsnorm(
            self.arena().stream_off(),
            self.arena().moein_off(),
            l.pre_ff_ln_2,
            HID as u32,
            fm,
        );
        self.memzero_bytes(self.arena().moeout_off(), (fm * HID * 4) as u64);
        Ok(())
    }

    /// Block height of the block-sparse expert GEMM for THIS forward. 32 in
    /// steady state; the wide weight-stationary height during batched-prefill
    /// super-chunks (forward_m > CANVAS) when the wide tunable pipelines were
    /// built. moe_bucket_fill phase 1 builds the block list at this height,
    /// so dispatch_block_linear_grouped MUST select the matching pipeline —
    /// both read this one function.
    fn moe_block_m(&self) -> u32 {
        if self.forward_m <= CANVAS {
            return 32;
        }
        let fmt = self.block_profile.format;
        // Wide pipelines are compiled for exactly the shapes/gather variants
        // the narrow set has; one representative presence check suffices (and
        // is only built for the block-expert formats q4/q6/nvfp4).
        let wide_ok = self
            .ps
            .sparse_tunable_wide_fmt(fmt, MOE_FF * 2, HID as u32, false)
            .is_some()
            && self
                .ps
                .sparse_tunable_wide_fmt(fmt, HID as u32, MOE_FF, false)
                .is_some();
        if wide_ok {
            if crate::flags::progress_enabled() {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    eprintln!(
                        "step-kernel: weight-stationary expert GEMM (block_m={})",
                        self.ps.sparse_wide_bm
                    );
                });
            }
            self.ps.sparse_wide_bm
        } else {
            32
        }
    }

    fn dispatch_block_linear_grouped(
        &mut self,
        a_on_gemm_a: bool,
        buf_a_off: usize,
        buf_c_off: usize,
        jobs: &[BlockGroupedJob; N_EXPERTS],
        _total_m: u32,
        k: u32,
        n: u32,
        indirect_slot: usize,
        gather: bool,
    ) -> Result<(), Error> {
        // Tunable block-sparse is the sole expert-GEMM path (q4/q6/nvfp4);
        // indirect slots 4/5 (BN-wide N-tiles). Wide = the weight-stationary
        // prefill block height (moe_block_m()); the block list for this forward
        // was built at that height (bucket_fill phase 1), so the consuming
        // pipeline's TUNE_BM must match.
        let fmt = self.block_profile.format;
        let wide = self.moe_block_m() != 32;
        // Fused-gather gate_up: A-load pulls bf16 `moein` rows via token_list
        // (buffer 7), so no separate gather pass / f32 staging buffer. The caller
        // skips the gather pass iff `gather`; if the pipeline for this shape is
        // missing we'd read a stale A buffer, so fail loud rather than silently.
        let gather_ps = if gather {
            if wide {
                self.ps.sparse_tunable_wide_fmt(fmt, n, k, true)
            } else {
                self.ps.sparse_tunable_fmt(fmt, n, k, true)
            }
        } else {
            None
        };
        if gather && gather_ps.is_none() {
            return Err(Error::Format(
                "fused MoE gather requested but no gather pipeline for this shape",
            ));
        }
        let use_gather = gather_ps.is_some();
        let grouped_ps = if let Some(p) = gather_ps {
            p
        } else if wide {
            self.ps
                .sparse_tunable_wide_fmt(fmt, n, k, false)
                .ok_or(Error::Format("missing wide tunable sparse pipeline"))?
        } else {
            self.ps
                .sparse_tunable_fmt(fmt, n, k, false)
                .ok_or(Error::Format("missing tunable sparse pipeline"))?
        };
        let row_start_off = std::mem::offset_of!(RouteScratch, row_start);
        self.sink_set_pipeline(grouped_ps);
        let a_buf = if a_on_gemm_a {
            &self.bufs.gemm_a
        } else {
            &self.bufs.gemm_b
        };
        self.sink_set_buffer(a_buf, buf_a_off, 0);
        // Expert weights: region-2 buffer on split blobs (job offsets are
        // rebased to match in layer_moe_block_jobs_impl).
        self.bind_blob_experts(1);
        self.sink_set_buffer(&self.bufs.gemm_b, buf_c_off, 2);
        self.sink_set_bytes(jobs, 3);
        self.sink_set_buffer(&self.bufs.route, row_start_off, 4);
        let num_jobs = N_EXPERTS as u32;
        self.sink_set_bytes(&num_jobs, 5);
        self.bind_route(6);
        if use_gather {
            self.sink_set_buffer(&self.bufs.arena, self.arena().moein_off() as usize, 7);
        }
        let tg = MTLSize {
            width: crate::shaders::gemm_common::THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        // Indirect slots: tunable sparse uses 4/5.
        let slot = indirect_slot + 4;
        let indirect_offset = slot * 3 * std::mem::size_of::<u32>();
        self.sink_dispatch_indirect(indirect_offset, n, tg);
        Ok(())
    }

    /// Batched MoE: gather → grouped block GEMM (gate/up, down) → swiglu → weighted scatter.
    fn encode_layer_moe_batched(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        if !moe_fuse_gather_enabled() {
            self.encode_moe_batched_gather()?;
        }
        self.encode_moe_batched_gate_up(layer, layout)?;
        self.encode_moe_batched_swiglu()?;
        self.encode_moe_batched_down(layer, layout)?;
        self.encode_moe_batched_scatter()?;
        Ok(())
    }

    pub(super) fn encode_moe_batched_gather(&mut self) -> Result<(), Error> {
        self.encode_moe_batched_gather_bf16_to_f32()
    }

    fn encode_moe_batched_gather_bf16_to_f32(&mut self) -> Result<(), Error> {
        let token_list_off = std::mem::offset_of!(RouteScratch, token_list);
        let gather_dims = [0u32, HID as u32];
        let slots = (self.forward_m * TOP_K) as u32;
        let gather_count = slots as usize * HID;
        self.dispatch_1d_ranged(
            &self.ps.gather_rows_bf16_to_f32,
            gather_count,
            256,
            |this, base, _chunk| {
                this.sink_set_buffer(&this.bufs.arena, this.arena().moein_off() as usize, 0);
                this.sink_set_buffer(&this.bufs.route, token_list_off, 1);
                this.sink_set_buffer(&this.bufs.gemm_b, moe_w_byte_off_a(), 2);
                this.sink_set_buffer(&this.bufs.dummy_dump, 0, 5);
                this.sink_set_bytes(&gather_dims, 3);
                this.sink_set_bytes(&slots, 4);
                this.sink_set_bytes(&base, 6);
            },
        );
        Ok(())
    }

    pub(super) fn encode_moe_batched_gate_up(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        let l = &layout.layers[layer];
        let (gate_jobs, _) = layer_moe_block_jobs_impl(
            l,
            self.block_profile.format,
            Some((layer, self.tensor_offsets)),
            self.bufs.blob_expert_base,
        );
        self.dispatch_block_linear_grouped(
            false,
            moe_w_byte_off_a(),
            moe_w_byte_off_gu(),
            &gate_jobs,
            MOE_SLOTS,
            HID as u32,
            MOE_FF * 2,
            0,
            moe_fuse_gather_enabled(),
        )
    }

    pub(super) fn encode_moe_batched_swiglu(&mut self) -> Result<(), Error> {
        let gu_off = moe_w_byte_off_gu();
        let slots = (self.forward_m * TOP_K) as u32;
        let act_elems = slots as usize * MOE_FF as usize;
        self.sink_set_pipeline(&self.ps.gelu_swiglu_gate_up);
        self.sink_set_buffer(&self.bufs.gemm_b, gu_off, 0);
        self.sink_set_buffer(&self.bufs.gemm_a, 0, 1);
        self.sink_set_buffer(&self.bufs.dummy_dump, 0, 3);
        let swiglu_dims = [slots, MOE_FF];
        self.sink_set_bytes(&swiglu_dims, 2);
        self.dispatch_1d(&self.ps.gelu_swiglu_gate_up, act_elems, 256);
        Ok(())
    }

    pub(super) fn encode_moe_batched_down(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        let l = &layout.layers[layer];
        let (_, down_jobs) = layer_moe_block_jobs_impl(
            l,
            self.block_profile.format,
            None,
            self.bufs.blob_expert_base,
        );
        self.dispatch_block_linear_grouped(
            true,
            0,
            moe_w_byte_off_a(),
            &down_jobs,
            MOE_SLOTS,
            MOE_FF,
            HID as u32,
            1,
            false,
        )
    }

    pub(super) fn encode_moe_batched_scatter(&mut self) -> Result<(), Error> {
        let fm = self.forward_m;
        self.sink_set_pipeline(&self.ps.moe_scatter_weighted);
        self.sink_set_buffer(&self.bufs.gemm_b, moe_w_byte_off_a(), 0);
        self.sink_set_buffer(&self.bufs.arena, self.arena().moeout_off() as usize, 1);
        self.bind_route(2);
        let hidden = HID as u32;
        let canvas = fm as u32;
        self.sink_set_bytes(&hidden, 3);
        self.sink_set_bytes(&canvas, 4);
        // One threadgroup per (token, 256-wide d-tile); 256 threads, one per d.
        let grid = MTLSize {
            width: div_up(hidden as usize, 256),
            height: canvas as usize,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    pub(super) fn encode_layer_moe_scalar(
        &mut self,
        layer: usize,
        _layout: &ModelLayout,
    ) -> Result<(), Error> {
        let fm = self.forward_m;
        let layer_off = layer_byte_offset(layer);
        self.sink_set_pipeline(self.ps.moe_scalar(self.block_profile.format));
        self.sink_set_buffer(&self.bufs.arena, self.arena().moein_off() as usize, 0);
        self.sink_set_buffer(&self.bufs.gemm_b, moe_w_byte_off_a(), 1);
        self.bind_blob(2);
        self.sink_set_buffer(&self.bufs.layout, layer_off as usize, 3);
        self.bind_route(4);
        let grouped_dims = crate::shaders::moe_grouped::GroupedDims {
            canvas: fm as u32,
            hidden: HID as u32,
            moe_ff: MOE_FF,
            n_experts: N_EXPERTS as u32,
        };
        self.sink_set_bytes(&grouped_dims, 5);
        if !self.block_profile.is_nvfp4() {
            self.sink_set_buffer(&self.bufs.dummy_dump, 0, 6);
        }
        let grid = MTLSize {
            width: fm,
            height: N_EXPERTS,
            depth: 1,
        };
        let tg = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        self.encode_moe_batched_scatter()
    }

    pub(super) fn encode_layer_moe_grouped(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        match self.block_profile.moe_style() {
            MoeExecutionStyle::BatchedGrouped => self.encode_layer_moe_batched(layer, layout),
            MoeExecutionStyle::ScalarPerExpert => self.encode_layer_moe_scalar(layer, layout),
        }
    }

    pub(super) fn encode_layer_moe_post_norm(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        let fm = self.forward_m;
        let l = &layout.layers[layer];
        self.rmsnorm_f32(
            self.arena().moeout_off(),
            self.arena().moein_off(),
            l.post_ff_ln_2,
            HID as u32,
            fm,
        );
        Ok(())
    }

    pub(super) fn encode_layer_moe_post_combine(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        let fm = self.forward_m;
        let l = &layout.layers[layer];
        self.residual(
            self.arena().dense_off(),
            self.arena().moein_off(),
            self.arena().tmp_off(),
            0,
            fm * HID,
        );
        self.rmsnorm(
            self.arena().tmp_off(),
            self.arena().tmp_off(),
            l.post_ff_ln,
            HID as u32,
            fm,
        );
        self.residual(
            self.arena().stream_off(),
            self.arena().tmp_off(),
            self.arena().hidden_off(),
            l.layer_scalar,
            fm * HID,
        );
        Ok(())
    }

    fn encode_layer_moe_post(&mut self, layer: usize, layout: &ModelLayout) -> Result<(), Error> {
        self.encode_layer_moe_post_norm(layer, layout)?;
        self.encode_layer_moe_post_combine(layer, layout)?;
        Ok(())
    }

    /// Attention + dense FFN + router + grouped MoE + post-combine (one encoder session).
    pub(super) fn encode_full_layer(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        self.encode_layer(layer, layout)?;
        self.encode_layer_moe_grouped(layer, layout)?;
        self.encode_layer_moe_post(layer, layout)?;
        Ok(())
    }

    /// MoE grouped kernel for one expert at one canvas row (router bypassed).
    pub(super) fn encode_layer_moe_single_expert_setup(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
        position: usize,
        expert_id: u32,
    ) {
        let l = &layout.layers[layer];
        self.rmsnorm(
            self.arena().stream_off(),
            self.arena().moein_off(),
            l.pre_ff_ln_2,
            HID as u32,
            CANVAS,
        );
        self.memzero_bytes(self.arena().moeout_off(), (CANVAS * HID * 4) as u64);
        write_single_expert_route(&self.bufs.route, position, expert_id);
    }

    /// Grouped MoE with K_DUMP_STAGE dump of threadgroup act (debug capture only).
    pub(super) fn encode_layer_moe_grouped_act_probe(
        &mut self,
        layer: usize,
        _layout: &ModelLayout,
    ) -> Result<(), Error> {
        if self.block_profile.is_nvfp4() {
            return Err(Error::Format(
                "moe_grouped dump mode is q4-only (use q8 .dgq weights)",
            ));
        }
        let layer_off = layer_byte_offset(layer);
        self.memzero_bytes(self.arena().moeout_off(), (CANVAS * HID * 4) as u64);
        self.memzero_bytes(
            self.arena().soft_off(),
            (MOE_ACT_PROBE_FLOATS * std::mem::size_of::<f32>()) as u64,
        );
        self.sink_set_pipeline(&self.ps.moe_grouped_dump);
        self.sink_set_buffer(&self.bufs.arena, self.arena().moein_off() as usize, 0);
        self.sink_set_buffer(&self.bufs.arena, self.arena().moeout_off() as usize, 1);
        self.bind_blob(2);
        self.sink_set_buffer(&self.bufs.layout, layer_off as usize, 3);
        self.bind_route(4);
        self.sink_set_buffer(&self.bufs.arena, self.arena().soft_off() as usize, 6);
        let grouped_dims = crate::shaders::moe_grouped::GroupedDims {
            canvas: CANVAS as u32,
            hidden: HID as u32,
            moe_ff: MOE_FF,
            n_experts: N_EXPERTS as u32,
        };
        self.sink_set_bytes(&grouped_dims, 5);
        let grid = MTLSize {
            width: CANVAS,
            height: N_EXPERTS,
            depth: 1,
        };
        let tg = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    /// Input layernorm + fused Q‖K(‖V) projections (stops before qk_rope_kv).
    pub(super) fn encode_layer_qkv_gemm(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        let fm = self.forward_m;
        let l = &layout.layers[layer];
        self.rmsnorm(
            self.arena().hidden_off(),
            self.arena().tmp_off(),
            l.input_ln,
            HID as u32,
            fm,
        );
        if fused_qkv_enabled() && self.attn_ffn_bf16 {
            let (segs, n_total) = qkv_stacked_segments(l, self.arena());
            self.gemm_bf16_stacked(
                self.arena().tmp_off(),
                &segs,
                fm as u32,
                HID as u32,
                n_total,
            )?;
        } else if fused_qkv_enabled() && !self.attn_ffn_q8 && !self.attn_ffn_bf16 {
            let (segs, n_total) = qkv_stacked_segments(l, self.arena());
            self.gemm_q4_stacked(
                self.arena().tmp_off(),
                &segs,
                fm as u32,
                HID as u32,
                n_total,
            )?;
        } else {
            let q_n = if l.is_full != 0 { 8192 } else { 4096 };
            let k_n = if l.is_full != 0 { 1024 } else { 2048 };
            self.gemm_q4(
                self.arena().tmp_off(),
                self.arena().attnq_off(),
                l.q_proj,
                fm as u32,
                q_n,
                HID as u32,
            )?;
            self.gemm_q4(
                self.arena().tmp_off(),
                self.arena().attnk_off(),
                l.k_proj,
                fm as u32,
                k_n,
                HID as u32,
            )?;
            if l.v_proj != 0 {
                self.gemm_q4(
                    self.arena().tmp_off(),
                    self.arena().attnv_off(),
                    l.v_proj,
                    fm as u32,
                    k_n,
                    HID as u32,
                )?;
            }
        }
        Ok(())
    }

    /// QKV GEMM dispatches only (caller must have normalized input in `tmp`).
    pub(super) fn dispatch_qkv_gemms(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
        stacked: bool,
    ) -> Result<(), Error> {
        let l = &layout.layers[layer];
        if stacked {
            let (segs, n_total) = qkv_stacked_segments(l, self.arena());
            self.gemm_q4_stacked(
                self.arena().tmp_off(),
                &segs,
                CANVAS as u32,
                HID as u32,
                n_total,
            )?;
        } else {
            let q_n = if l.is_full != 0 { 8192 } else { 4096 };
            let k_n = if l.is_full != 0 { 1024 } else { 2048 };
            self.gemm_q4(
                self.arena().tmp_off(),
                self.arena().attnq_off(),
                l.q_proj,
                CANVAS as u32,
                q_n,
                HID as u32,
            )?;
            self.gemm_q4(
                self.arena().tmp_off(),
                self.arena().attnk_off(),
                l.k_proj,
                CANVAS as u32,
                k_n,
                HID as u32,
            )?;
            if l.v_proj != 0 {
                self.gemm_q4(
                    self.arena().tmp_off(),
                    self.arena().attnv_off(),
                    l.v_proj,
                    CANVAS as u32,
                    k_n,
                    HID as u32,
                )?;
            }
        }
        Ok(())
    }

    /// Dense gate/up GEMM dispatches only (caller must have normalized input in `tmp`).
    pub(super) fn dispatch_gate_up_gemms(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
        stacked: bool,
    ) -> Result<(), Error> {
        let l = &layout.layers[layer];
        if stacked {
            let (segs, n_total) = gate_up_stacked_segments(l, self.arena());
            self.gemm_q4_stacked(
                self.arena().tmp_off(),
                &segs,
                CANVAS as u32,
                HID as u32,
                n_total,
            )?;
        } else {
            self.gemm_q4(
                self.arena().tmp_off(),
                self.arena().ffg_off(),
                l.mlp_gate,
                CANVAS as u32,
                DENSE_FF,
                HID as u32,
            )?;
            self.gemm_q4(
                self.arena().tmp_off(),
                self.arena().ffu_off(),
                l.mlp_up,
                CANVAS as u32,
                DENSE_FF,
                HID as u32,
            )?;
        }
        Ok(())
    }

    /// QK-RoPE-KV write (expects Q/K/V already in arena).
    fn encode_layer_qk_rope_kv_dispatch(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        let l = &layout.layers[layer];
        let qk_y = (16 + 2 * l.n_kv_heads) as usize;
        let layer_off = layer_byte_offset(layer);

        // E14: prefill writes sliding K/V to the f32 side ring too.
        let side = if self.prefill_causal {
            self.bufs
                .kv_f32_side
                .as_ref()
                .zip(self.ps.qk_rope_kv_side.as_ref())
        } else {
            None
        };
        match &side {
            Some((_, pipe)) => self.sink_set_pipeline(pipe),
            None => self.sink_set_pipeline(&self.ps.qk_rope_kv),
        }
        // Per-sub-chunk row offsets into the batched Q/K/V planes (sub_c = 0
        // outside a super-chunk). K/V planes are written at the layer's native
        // widths (n_kv*hd); Q at n_q*hd.
        let q_row = CANVAS * self.sub_c * STEP_NQ_HEADS * l.head_dim as usize * 2;
        let kv_row = CANVAS * self.sub_c * (l.n_kv_heads * l.head_dim) as usize * 2;
        self.sink_set_buffer(
            &self.bufs.arena,
            self.arena().attnq_off() as usize + q_row,
            0,
        );
        self.sink_set_buffer(
            &self.bufs.arena,
            self.arena().attnk_off() as usize + kv_row,
            1,
        );
        self.sink_set_buffer(
            &self.bufs.arena,
            self.arena().attnv_off() as usize + kv_row,
            2,
        );
        self.bind_kvcache(3);
        self.bind_blob(4);
        self.sink_set_buffer(&self.bufs.layout, layer_off as usize, 5);
        self.bind_params(6);
        let attn_dims = crate::shaders::qk_rope_kv::AttnDims {
            canvas: self.active_canvas as u32,
            n_q_heads: STEP_NQ_HEADS as u32,
            causal: 0,
            window: 0, // KV write only; the window applies at attention read time
        };
        self.sink_set_bytes(&attn_dims, 7);
        self.bind_debug_status(8);
        if let Some((sbuf, _)) = side {
            // Full layers never dereference the side pointer (kernel guards on
            // !full) — bind offset 0 for them.
            let off = self.bufs.kv_f32_side_offs[layer];
            let off = if off == u64::MAX { 0 } else { off as usize };
            self.sink_set_buffer(sbuf, off, 9);
        }
        let grid = MTLSize {
            width: self.active_canvas,
            height: qk_y,
            depth: 1,
        };
        let tg = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    /// E17 full-layer prefill attention: the [qk, softmax, pv] GEMM
    /// decomposition (see `shaders/attn/attention_gemm`).
    fn encode_attn_gemm(&mut self, layer: usize, layout: &ModelLayout) -> Result<(), Error> {
        self.encode_attn_decomp(layer, layout, AttnDecompKind::Gemm)
    }

    /// E20: top-k sparse attention dispatch for full layers — causal PREFILL
    /// (`DGQ_ATTN_TOPK`) and bidirectional DENOISE (`DGQ_ATTN_TOPK_DECODE`);
    /// `self.prefill_causal` selects the mask. Shares the E17 QK stage (same
    /// kernel, same S plane), then dispatches `attn_topk_softmax` (top-k
    /// selection + renormalization) and `attn_topk_pv` (gathered-V PV)
    /// instead of E17's softmax + PV.
    fn encode_attn_topk(&mut self, layer: usize, layout: &ModelLayout) -> Result<(), Error> {
        self.encode_attn_decomp(layer, layout, AttnDecompKind::TopK)
    }

    /// The shared E17/E20 driver: one QK scaffold (identical dims, S plane,
    /// head-chunk loop), per-variant stage 2/3. Merged from the two ~80%%-
    /// identical encoders so the QK stages can never drift apart.
    ///
    /// `causal`: E17 is only dispatched on the prefill path
    /// (`self.prefill_causal == true`), so deriving both variants' mask from
    /// `prefill_causal` is bit-identical to E17's old hard-coded `causal: 1`.
    fn encode_attn_decomp(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
        kind: AttnDecompKind,
    ) -> Result<(), Error> {
        use crate::shaders::attention_gemm::n_pad;
        use crate::shaders::attention_topk::{BM, BN, PV_BN, SOFTMAX_TPG};
        let l = &layout.layers[layer];
        let hd = l.head_dim as usize;
        let nkv = l.n_kv_heads as usize;
        let group = STEP_NQ_HEADS / nkv;
        let m = self.active_canvas;
        let kv_len = self.dispatch_kv_len() as usize;
        let t_total = kv_len + m;
        let np = n_pad(t_total);
        // Q/O sub-chunk row offset into the batched prefill planes.
        let qo_row = CANVAS * self.sub_c * STEP_NQ_HEADS * hd * 2;
        // Tunable tile geometry (task #87) — must match the compiled E17
        // pipelines. E20's tiles are compile-time consts.
        let (qk_bm_f, qk_bn_f, pv_bm_f, pv_bn_f, sm_tpg) = crate::flags::gemm_attn_tile();
        let (qk_bm, qk_bn) = match kind {
            AttnDecompKind::Gemm => (qk_bm_f, qk_bn_f),
            AttnDecompKind::TopK => (BM, BN),
        };

        let mut dims = crate::shaders::attention_gemm::AttnGemmDims {
            m: m as u32,
            n: t_total as u32,
            k: hd as u32,
            a_row_stride: (STEP_NQ_HEADS * hd) as u32,
            b_row_stride: (nkv * hd * 2) as u32,
            s_row_stride: np as u32,
            out_row_stride: (STEP_NQ_HEADS * hd) as u32,
            causal: u32::from(self.prefill_causal),
            kv_len: kv_len as u32,
            hd: hd as u32,
            group: group as u32,
            nkv: nkv as u32,
            s_head_stride: (m * np) as u32,
            head_base: 0,
        };
        // kv-adaptive k (DGQ_ATTN_TOPK_DYN): k grows with context, capped by
        // the compiled K_PAD. Fixed DGQ_ATTN_TOPK_K otherwise. (E20 only.)
        let k_param =
            crate::flags::attn_topk_k_for(t_total).min(crate::flags::attn_topk_k_pad()) as u32;
        let tg128 = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        let tg_sm = MTLSize {
            width: match kind {
                AttnDecompKind::Gemm => sm_tpg,
                AttnDecompKind::TopK => SOFTMAX_TPG,
            },
            height: 1,
            depth: 1,
        };
        let tg_pv_topk = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        }; // one simdgroup
        // E17a: process Q heads in batches of HC so the S/P scratch holds only
        // HC heads. Data offsets use the global head (head_base + tgid.z); the
        // scratch is indexed by the batch-local tgid.z. Both variants read the
        // SAME flag the scratch allocation reads (task #97).
        let hc = crate::flags::gemm_attn_head_chunk().min(STEP_NQ_HEADS);
        // E17b/E20-side: prefer the f32-side-KV pipelines when compiled. The
        // f32 side ring is a PREFILL mechanism — denoise-step canvas KV writes
        // only reach the main f16 cache, so decode dispatches must read it
        // (never the side ring) even when E14 is on.
        let side = match kind {
            AttnDecompKind::Gemm => self.ps.attn_gemm_side.is_some(),
            AttnDecompKind::TopK => self.ps.attn_topk_side.is_some() && self.prefill_causal,
        };
        let side_off = self.bufs.kv_f32_side_offs[layer] as usize;
        let pipes = match (kind, side) {
            (AttnDecompKind::Gemm, true) => self.ps.attn_gemm_side.as_ref().unwrap().clone(),
            (AttnDecompKind::Gemm, false) => self.ps.attn_gemm.as_ref().unwrap().clone(),
            (AttnDecompKind::TopK, true) => self.ps.attn_topk_side.as_ref().unwrap().clone(),
            (AttnDecompKind::TopK, false) => self.ps.attn_topk.as_ref().unwrap().clone(),
        };
        let mut h0 = 0usize;
        while h0 < STEP_NQ_HEADS {
            let hb = (STEP_NQ_HEADS - h0).min(hc);
            dims.head_base = h0 as u32;
            let grid_qk = MTLSize {
                width: t_total.div_ceil(qk_bn),
                height: m.div_ceil(qk_bm),
                depth: hb,
            };
            let grid_sm = MTLSize {
                width: m,
                height: hb,
                depth: 1,
            };

            // QK: S = Q.K^T — the shared stage. Q at the sub-chunk offset; K
            // from the f16 main cache (buffer 1 @ kv_region) or the f32 side
            // ring (buffer 9 @ side_off). E20 additionally writes the FC32
            // u16 key plane at 8 for the selection passes.
            self.sink_set_pipeline(&pipes[0]);
            self.sink_set_buffer(
                &self.bufs.arena,
                self.arena().attnq_off() as usize + qo_row,
                0,
            );
            self.sink_set_buffer(&self.bufs.kvcache, l.kv_region as usize, 1);
            self.sink_set_buffer(self.bufs.attn_gemm_s.as_ref().unwrap(), 0, 2);
            if let AttnDecompKind::TopK = kind {
                self.sink_set_buffer(self.bufs.attn_topk_pat.as_ref().unwrap(), 0, 8);
            }
            if side {
                self.sink_set_buffer(self.bufs.kv_f32_side.as_ref().unwrap(), side_off, 9);
            }
            self.sink_set_bytes(&dims, 3);
            self.sink_dispatch(grid_qk, tg128);
            self.sink_memory_barrier();

            match kind {
                AttnDecompKind::Gemm => {
                    let dims_pv = crate::shaders::attention_gemm::AttnGemmDims {
                        n: hd as u32,
                        k: t_total as u32,
                        a_row_stride: np as u32,
                        ..dims
                    };
                    let grid_pv = MTLSize {
                        width: hd.div_ceil(pv_bn_f),
                        height: m.div_ceil(pv_bm_f),
                        depth: hb,
                    };
                    // Softmax: P = exp(S - rowmax), masked; denom -> lrow.
                    self.sink_set_pipeline(&pipes[1]);
                    self.sink_set_buffer(self.bufs.attn_gemm_s.as_ref().unwrap(), 0, 0);
                    self.sink_set_buffer(self.bufs.attn_gemm_p.as_ref().unwrap(), 0, 1);
                    self.sink_set_buffer(self.bufs.attn_gemm_lrow.as_ref().unwrap(), 0, 2);
                    self.sink_set_bytes(&dims, 3);
                    self.sink_dispatch(grid_sm, tg_sm);
                    self.sink_memory_barrier();

                    // PV: O = (P.V) / L. V from the f16 cache (buffer 1) or f32
                    // side ring (buffer 9); O at the sub-chunk offset.
                    self.sink_set_pipeline(&pipes[2]);
                    self.sink_set_buffer(self.bufs.attn_gemm_p.as_ref().unwrap(), 0, 0);
                    self.sink_set_buffer(&self.bufs.kvcache, l.kv_region as usize, 1);
                    self.sink_set_buffer(
                        &self.bufs.arena,
                        self.arena().attno_off() as usize + qo_row,
                        2,
                    );
                    self.sink_set_buffer(self.bufs.attn_gemm_lrow.as_ref().unwrap(), 0, 3);
                    if side {
                        self.sink_set_buffer(self.bufs.kv_f32_side.as_ref().unwrap(), side_off, 9);
                    }
                    self.sink_set_bytes(&dims_pv, 4);
                    self.sink_dispatch(grid_pv, tg128);
                    self.sink_memory_barrier();
                }
                AttnDecompKind::TopK => {
                    let grid_pv = MTLSize {
                        width: hd.div_ceil(PV_BN),
                        height: m,
                        depth: hb,
                    };
                    // topk_softmax: S + u16 keys -> P (compressed), Idx, lrow.
                    self.sink_set_pipeline(&pipes[1]);
                    self.sink_set_buffer(self.bufs.attn_gemm_s.as_ref().unwrap(), 0, 0);
                    self.sink_set_buffer(self.bufs.attn_topk_p.as_ref().unwrap(), 0, 1);
                    self.sink_set_buffer(self.bufs.attn_topk_idx.as_ref().unwrap(), 0, 2);
                    self.sink_set_buffer(self.bufs.attn_topk_lrow.as_ref().unwrap(), 0, 3);
                    self.sink_set_buffer(self.bufs.attn_topk_pat.as_ref().unwrap(), 0, 6);
                    self.sink_set_bytes(&dims, 4);
                    self.sink_set_bytes(&k_param, 5);
                    self.sink_dispatch(grid_sm, tg_sm);
                    self.sink_memory_barrier();

                    // topk_pv: O = (P · V_gathered) / L.
                    self.sink_set_pipeline(&pipes[2]);
                    self.sink_set_buffer(self.bufs.attn_topk_p.as_ref().unwrap(), 0, 0);
                    self.sink_set_buffer(self.bufs.attn_topk_idx.as_ref().unwrap(), 0, 1);
                    self.sink_set_buffer(&self.bufs.kvcache, l.kv_region as usize, 2);
                    self.sink_set_buffer(
                        &self.bufs.arena,
                        self.arena().attno_off() as usize + qo_row,
                        3,
                    );
                    self.sink_set_buffer(self.bufs.attn_topk_lrow.as_ref().unwrap(), 0, 4);
                    if side {
                        self.sink_set_buffer(self.bufs.kv_f32_side.as_ref().unwrap(), side_off, 9);
                    }
                    self.sink_set_bytes(&dims, 5);
                    self.sink_dispatch(grid_pv, tg_pv_topk);
                    self.sink_memory_barrier();
                }
            }
            h0 += hb;
        }
        Ok(())
    }

    /// E18: sliding-layer PREFILL via fused flash (`DGQ_FLASH_PREFILL`).
    /// Window-aware + ring-aware; hd must be 256 (compile-time FL_HD). f16 KV.
    fn encode_attn_flash_sliding(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        let (_, bq, _bk) = crate::flags::flash_prefill();
        let l = &layout.layers[layer];
        let hd = l.head_dim as usize;
        debug_assert_eq!(hd, 256, "flash sliding is compiled for hd=256");
        let nkv = l.n_kv_heads as usize;
        let group = STEP_NQ_HEADS / nkv;
        let m = self.active_canvas;
        let kv_len = self.dispatch_kv_len() as usize;
        let t_total = kv_len + m;
        let qo_row = CANVAS * self.sub_c * STEP_NQ_HEADS * hd * 2;
        let window = if attn_window_enabled() {
            self.sliding_window
        } else {
            0
        };
        let dims = crate::shaders::attention_flash::FlashDims {
            m: m as u32,
            t_total: t_total as u32,
            hd: hd as u32,
            a_row_stride: (STEP_NQ_HEADS * hd) as u32,
            b_row_stride: (nkv * hd * 2) as u32,
            out_row_stride: (STEP_NQ_HEADS * hd) as u32,
            kv_len: kv_len as u32,
            group: group as u32,
            nkv: nkv as u32,
            head_base: 0,
            causal: 1,
            window,
            kv_ring_mask: l.kv_ring_mask,
        };
        let pipe = self.ps.attn_flash_sliding.as_ref().unwrap();
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let grid = MTLSize {
            width: m.div_ceil(bq),
            height: 1,
            depth: STEP_NQ_HEADS,
        };
        self.sink_set_pipeline(pipe);
        self.sink_set_buffer(
            &self.bufs.arena,
            self.arena().attnq_off() as usize + qo_row,
            0,
        );
        self.sink_set_buffer(&self.bufs.kvcache, l.kv_region as usize, 1);
        self.sink_set_buffer(
            &self.bufs.arena,
            self.arena().attno_off() as usize + qo_row,
            2,
        );
        self.sink_set_bytes(&dims, 3);
        self.sink_dispatch(grid, tg);
        self.sink_memory_barrier();
        Ok(())
    }

    fn encode_layer_attention_dispatch(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        let layer_off = layer_byte_offset(layer);
        let l = &layout.layers[layer];
        // E18: sliding-layer prefill flash (opt-in, quality-gated). Takes
        // precedence over mma2 when enabled.
        if self.prefill_causal && l.is_full == 0 && self.ps.attn_flash_sliding.is_some() {
            return self.encode_attn_flash_sliding(layer, layout);
        }
        // E20: top-k sparse attention for full-layer prefill (quality-gated).
        // Takes precedence over E17 when enabled. The pipelines can also be
        // compiled for the decode arm alone, so gate on the prefill flag.
        if self.prefill_causal
            && l.is_full == 1
            && crate::flags::attn_topk_enabled()
            && self.ps.attn_topk.is_some()
        {
            return self.encode_attn_topk(layer, layout);
        }
        // Decode arm (`DGQ_ATTN_TOPK_DECODE`): the same top-k pipeline on
        // full-layer DENOISE dispatches (causal=0). Full-layer denoise
        // attention in mma_full is issue-bound and linear in context — the
        // GEMM-decomp top-k runs it ~3× faster at long kv.
        if !self.prefill_causal
            && l.is_full == 1
            && crate::flags::attn_topk_decode_enabled()
            && self.ps.attn_topk.is_some()
        {
            return self.encode_attn_topk(layer, layout);
        }
        // E17: full-layer prefill attention through the GEMM decomposition.
        if self.prefill_causal && l.is_full == 1 && self.ps.attn_gemm.is_some() {
            return self.encode_attn_gemm(layer, layout);
        }
        // GQA-grouped MMA attention (`DGQ_ATTN_MMA`) handles sliding layers (hd=256)
        // via attention_mma2; full hd=512 layers use attention_mma_full
        // (`DGQ_ATTN_MMA_FULL`, register-resident O + group K/V sharing) when
        // enabled, else the scalar kernel. Identical buffer layout — only the
        // pipeline + dispatch grid differ. mma_full is non-bit-identical (quality
        // gate): default OFF.
        // mma2/mma_full honor the causal mask (AttnDims.causal) + sliding window,
        // so they run prefill too (DGQ_PREFILL_MMA=0 restores scalar prefill).
        // The old "f16 prefill hurts accuracy (11/16 vs 14/16)" verdict was from
        // the freeze era; no-freeze fixed that degeneration class, and scalar
        // prefill is O(kv_len) serial per query — unusable at long context.
        let mma_ok = !self.prefill_causal || crate::flags::prefill_mma_enabled();
        let use_mma2 = mma_ok && attn_mma_enabled() && l.is_full == 0;
        let use_mma_full = mma_ok && attn_mma_full_enabled() && l.is_full == 1;
        // E14: prefill attention reads K/V from the f32 side cache.
        let side = if self.prefill_causal && use_mma2 {
            self.bufs
                .kv_f32_side
                .as_ref()
                .zip(self.ps.attention_mma2_side.as_ref())
        } else if self.prefill_causal && use_mma_full {
            self.bufs
                .kv_f32_side
                .as_ref()
                .zip(self.ps.attention_mma_full_side.as_ref())
        } else {
            None
        };
        if let Some((_, pipe)) = &side {
            self.sink_set_pipeline(pipe);
        } else if use_mma2 {
            self.sink_set_pipeline(&self.ps.attention_mma2);
        } else if use_mma_full {
            self.sink_set_pipeline(&self.ps.attention_mma_full);
        } else {
            self.sink_set_pipeline(&self.ps.attention);
        }
        // Per-sub-chunk row offsets into the batched Q/O planes (sub_c = 0
        // outside a super-chunk).
        let qo_row = CANVAS * self.sub_c * STEP_NQ_HEADS * l.head_dim as usize * 2;
        self.sink_set_buffer(
            &self.bufs.arena,
            self.arena().attnq_off() as usize + qo_row,
            0,
        );
        self.bind_kvcache(1);
        self.sink_set_buffer(
            &self.bufs.arena,
            self.arena().attno_off() as usize + qo_row,
            2,
        );
        self.sink_set_buffer(&self.bufs.layout, layer_off as usize, 3);
        self.bind_params(4);
        // Sliding layers (is_full==0) attend a bounded window (Gemma-4
        // sliding_window=1024): denoise canvas sees only the last window-1
        // encoder positions + the canvas (MLX `_make_decoder_masks`); causal
        // prefill queries see [q-(window-1), q] (engine CausalSliding). This is
        // both the model spec for kv_len>window-1 AND keeps 25/30 layers
        // O(window) instead of O(context). No-op (bit-identical) below that.
        let window = if l.is_full == 0 && attn_window_enabled() {
            self.sliding_window
        } else {
            0
        };
        let attn_dims = crate::shaders::qk_rope_kv::AttnDims {
            canvas: self.active_canvas as u32,
            n_q_heads: STEP_NQ_HEADS as u32,
            causal: u32::from(self.prefill_causal),
            window,
        };
        self.sink_set_bytes(&attn_dims, 5);
        self.bind_debug_status(6);
        if let Some((sbuf, _)) = side {
            let idx = if use_mma_full { 9 } else { 7 };
            self.sink_set_buffer(sbuf, self.bufs.kv_f32_side_offs[layer] as usize, idx);
        }
        // Full-attention kernel takes a kv-block range + state scratch
        // (buffers 7/8); the range is (re)set per dispatch below.
        if use_mma_full {
            self.sink_set_buffer(&self.bufs.attn_state, 0, 8);
        }
        // Scalar: one threadgroup per (canvas token, Q head). MMA2: one per
        // (MT-row tile, KV head), 2 simdgroups = the 2 Q heads in the group.
        // MMA_full: one per (MT-row tile, KV head, QG-head sub-group), QG
        // simdgroups sharing K/V; (group/QG) sub-groups along z.
        let grid = if use_mma2 {
            MTLSize {
                width: self
                    .active_canvas
                    .div_ceil(crate::shaders::attention::MMA_M_TILE),
                height: l.n_kv_heads as usize,
                depth: 1,
            }
        } else if use_mma_full {
            let group = STEP_NQ_HEADS / l.n_kv_heads as usize; // 8 for full
            // One tg per (query tile, kv head, Q head); the QG simdgroups
            // split head_dim, so depth is the full GQA group.
            MTLSize {
                width: self
                    .active_canvas
                    .div_ceil(crate::shaders::attention::MMA_M_TILE),
                height: l.n_kv_heads as usize,
                depth: group,
            }
        } else {
            MTLSize {
                width: self.active_canvas,
                height: 16,
                depth: 1,
            }
        };
        // mma_full uses QG*32 lanes; scalar/mma2 use 64.
        let tg = MTLSize {
            width: if use_mma_full {
                crate::shaders::attention::MMA_FULL_QG * 32
            } else {
                64
            },
            height: 1,
            depth: 1,
        };
        if use_mma_full {
            // Flash-decode sequential kv blocks (DGQ_ATTN_KV_BLOCK, 0=off):
            // in-order dispatches over block-sized key ranges with f32
            // softmax state persisted in attn_state — bit-identical to one
            // monolithic pass, but each dispatch's threadgroups all stream
            // the same <=block key window (SLC-served instead of
            // DRAM-thrashed; the 256-consumer redundancy made the kernel
            // DRAM-bound past ~8k keys).
            let t_total = self.dispatch_kv_len() as usize + self.active_canvas;
            let block = crate::flags::attn_kv_block();
            let blocks = if block > 0 && t_total > block {
                t_total.div_ceil(block)
            } else {
                1
            };
            for b in 0..blocks {
                let t_begin = b * block;
                let t_end = if blocks == 1 {
                    t_total
                } else {
                    ((b + 1) * block).min(t_total)
                };
                let blk = [
                    t_begin as u32,
                    t_end as u32,
                    u32::from(b == 0),
                    u32::from(b + 1 == blocks),
                ];
                self.sink_set_bytes(&blk, 7);
                self.sink_dispatch(grid, tg);
            }
        } else {
            self.sink_dispatch(grid, tg);
        }
        Ok(())
    }

    /// QK-RoPE-KV write + attention (expects Q/K/V already in arena).
    pub(super) fn encode_layer_qk_rope_and_attention(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        self.encode_layer_qk_rope_kv_dispatch(layer, layout)?;
        self.encode_layer_attention_dispatch(layer, layout)
    }

    fn exec_stage(
        &mut self,
        stage: step_schedule::StepStage,
        layer: usize,
        layout: &ModelLayout,
        finish: StepFinishMode,
    ) -> Result<(), Error> {
        let _ = layout;
        use step_schedule::StepStage;
        match stage {
            StepStage::ScLogitRowstats => {
                self.encode_sc_logit_rowstats();
                Ok(())
            }
            StepStage::ScSoftembed => self.encode_sc_softembed(layout),
            StepStage::ScPreNorm => {
                self.rmsnorm(
                    self.arena().soft_off(),
                    self.arena().tmp_off(),
                    layout.sc_pre_norm,
                    HID as u32,
                    CANVAS,
                );
                Ok(())
            }
            StepStage::ScGateGemm => self.gemm_q8(
                self.arena().tmp_off(),
                self.arena().ffg_off(),
                layout.sc_gate,
                CANVAS as u32,
                DENSE_FF,
                HID as u32,
            ),
            StepStage::ScUpGemm => self.gemm_q8(
                self.arena().tmp_off(),
                self.arena().ffu_off(),
                layout.sc_up,
                CANVAS as u32,
                DENSE_FF,
                HID as u32,
            ),
            StepStage::ScGlu => {
                self.glu(
                    self.arena().ffg_off(),
                    self.arena().ffu_off(),
                    self.arena().ffg_off(),
                    CANVAS * DENSE_FF as usize,
                );
                Ok(())
            }
            StepStage::ScDownGemm => self.gemm_q8(
                self.arena().ffg_off(),
                self.arena().dense_off(),
                layout.sc_down,
                CANVAS as u32,
                HID as u32,
                DENSE_FF,
            ),
            StepStage::EmbedGather => {
                self.dispatch_embed_gather(layout.embed);
                Ok(())
            }
            StepStage::EmbedScResidual => {
                self.residual(
                    self.arena().hidden_off(),
                    self.arena().dense_off(),
                    self.arena().hidden_off(),
                    0,
                    CANVAS * HID,
                );
                Ok(())
            }
            StepStage::RmsNormHidden => {
                self.rmsnorm(
                    self.arena().hidden_off(),
                    self.arena().hidden_off(),
                    0,
                    HID as u32,
                    self.forward_m,
                );
                Ok(())
            }
            StepStage::LayerInputNormQkv => self.encode_layer_qkv_gemm(layer, layout),
            StepStage::LayerQkRopeKv => self.encode_layer_qk_rope_kv_dispatch(layer, layout),
            StepStage::LayerAttention => self.encode_layer_attention_dispatch(layer, layout),
            StepStage::LayerOProjPostAttn => self.encode_layer_o_proj_post_attn(layer, layout),
            StepStage::LayerDenseFfn => self.encode_layer_dense_ffn(layer, layout),
            StepStage::LayerRouter => self.encode_layer_router_buckets(layer, layout),
            StepStage::MoeBatchedGather => {
                // Fused gather folds the token gather into the gate_up A-load.
                if moe_fuse_gather_enabled() {
                    Ok(())
                } else {
                    self.encode_moe_batched_gather()
                }
            }
            StepStage::MoeBatchedGateUp => self.encode_moe_batched_gate_up(layer, layout),
            StepStage::MoeBatchedSwiglu => self.encode_moe_batched_swiglu(),
            StepStage::MoeBatchedDown => self.encode_moe_batched_down(layer, layout),
            StepStage::MoeBatchedScatter => self.encode_moe_batched_scatter(),
            StepStage::MoeGroupedScalar => self.encode_layer_moe_scalar(layer, layout),
            StepStage::LayerMoePostNorm => self.encode_layer_moe_post_norm(layer, layout),
            StepStage::LayerMoePostCombine => self.encode_layer_moe_post_combine(layer, layout),
            StepStage::FinalNorm => {
                self.rmsnorm(
                    self.arena().hidden_off(),
                    self.arena().tmp_off(),
                    layout.final_norm,
                    HID as u32,
                    CANVAS,
                );
                Ok(())
            }
            StepStage::LmHeadGemm => {
                let m = self.partial_lm_m;
                if self.embed_bf16 {
                    // bf16 embed: full tied lm_head via the bf16 GEMM (partial lm_head
                    // is the q8-only fast path; correctness over that optimization here).
                    self.gemm_bf16_logits(
                        self.arena().tmp_off(),
                        layout.embed,
                        CANVAS as u32,
                        VOCAB as u32,
                        HID as u32,
                        0,
                    )
                } else if partial_lm_head_enabled() && m < CANVAS as u32 {
                    self.encode_partial_lm_head(layout, m)
                } else {
                    self.gemm_q8_logits(
                        self.arena().tmp_off(),
                        layout.embed,
                        CANVAS as u32,
                        VOCAB as u32,
                        HID as u32,
                        0,
                    )
                }
            }
            StepStage::Softcap => {
                self.dispatch_softcap();
                Ok(())
            }
            StepStage::SampleRowstats
            | StepStage::SampleCommit
            | StepStage::SampleApply
            | StepStage::SampleWrite => {
                if finish == StepFinishMode::Full {
                    self.encode_step_sampler(layout)
                } else {
                    Ok(())
                }
            }
        }
    }

    fn interpret_step(
        &mut self,
        layout: &ModelLayout,
        layers: usize,
        first_step: u32,
        finish: StepFinishMode,
    ) -> Result<(), Error> {
        if arena_liveness::runtime_arena_liveness_enabled()
            && let Err(e) = arena_liveness::check_step_arena_liveness(
                &self.block_profile,
                layout,
                layers,
                first_step,
                finish,
            )
        {
            panic!("{e}");
        }
        let schedule =
            step_schedule::build_step_schedule(&self.block_profile, finish == StepFinishMode::Full);
        if first_step == 1 {
            // Deterministic first-step self-conditioning. The first denoise step has
            // no prior prediction, so the normal SC path is skipped — but the model
            // is degenerate with SC=0 (cold-start empty reply), and leaving dense_off
            // as a prior generation's residual makes reused sessions nondeterministic
            // (reset_kv carryover). Seed it deterministically: treat the initial
            // canvas as the step-0 prediction and run the SC MLP on its embedding
            // (ScPreNorm reads hidden after EmbedGather, in place of soft_off).
            use step_schedule::StepStage;
            self.exec_stage(StepStage::EmbedGather, 0, layout, finish)?;
            self.rmsnorm(
                self.arena().hidden_off(),
                self.arena().tmp_off(),
                layout.sc_pre_norm,
                HID as u32,
                CANVAS,
            );
            self.exec_stage(StepStage::ScGateGemm, 0, layout, finish)?;
            self.exec_stage(StepStage::ScUpGemm, 0, layout, finish)?;
            self.exec_stage(StepStage::ScGlu, 0, layout, finish)?;
            self.exec_stage(StepStage::ScDownGemm, 0, layout, finish)?;
            self.exec_stage(StepStage::EmbedScResidual, 0, layout, finish)?;
            self.exec_stage(StepStage::RmsNormHidden, 0, layout, finish)?;
        } else {
            for stage in step_schedule::build_preamble(first_step) {
                self.exec_stage(stage, 0, layout, finish)?;
            }
        }
        for layer in 0..layers {
            for &stage in &schedule.per_layer {
                self.exec_stage(stage, layer, layout, finish)?;
            }
        }
        let mut sampler_done = false;
        for &stage in &schedule.finish {
            if matches!(
                stage,
                step_schedule::StepStage::SampleRowstats
                    | step_schedule::StepStage::SampleCommit
                    | step_schedule::StepStage::SampleApply
                    | step_schedule::StepStage::SampleWrite
            ) {
                if !sampler_done && finish == StepFinishMode::Full {
                    self.encode_step_sampler(layout)?;
                    sampler_done = true;
                }
                continue;
            }
            self.exec_stage(stage, 0, layout, finish)?;
        }
        Ok(())
    }

    /// KV-only causal forward over one prompt chunk (the canvas holds chunk tokens).
    /// Embed + no-weight norm + the full per-layer stack (qkv → qk_rope_kv writes
    /// KV → CAUSAL attention → o_proj → dense FFN → MoE), with NO SC preamble, NO
    /// sampler, NO lm_head. The fast monolithic analog of the f32-engine prefill.
    fn encode_prefill_chunk(&mut self, layout: &ModelLayout, layers: usize) -> Result<(), Error> {
        use step_schedule::StepStage;
        self.prefill_causal = true;
        self.exec_stage(
            StepStage::EmbedGather,
            0,
            layout,
            StepFinishMode::ForwardOnly,
        )?;
        // NO RmsNormHidden here (task #64/#68 ROOT CAUSE, fixed 2026-07-10):
        // normalizing the embedded hidden is the DENOISE preamble's convention
        // (canvas stream, parity-validated); the reference ENCODER pass feeds
        // embed*sqrt(H) straight into layer 0. The norm is per-row
        // scale-invariant through input_layernorm — layer-0 K/V matched the
        // engine exactly (rel 0.0025), which hid it — but the RESIDUAL stream
        // carried a per-token rescale the model never saw in training:
        // L1 KV diverged 33% from the engine at every length, flipping MoE
        // routes systematically; short prompts tolerated the warped
        // trajectory, long ones collapsed into fluent hallucination.
        let per_layer = step_schedule::per_layer_stages(&self.block_profile);
        for layer in 0..layers {
            for &stage in &per_layer {
                self.exec_stage(stage, layer, layout, StepFinishMode::ForwardOnly)?;
            }
        }
        Ok(())
    }

    /// Batched prefill SUPER-chunk: n_subs causal 256-token sub-chunks as ONE
    /// forward. Row-independent stages (embed, norms, QKV / o_proj / dense FFN
    /// GEMMs, router, MoE) run once at M = n_subs*CANVAS — this is where the
    /// win lives (MoE expert weights streamed once per super-chunk instead of
    /// once per chunk). Rope/KV-write + attention keep their causal sequencing
    /// per sub-chunk (kv_len differs per sub -> params_sub slots; arena rows
    /// offset by sub_c*CANVAS). Bit-identical to sequential chunks: every
    /// batched stage is row-independent, and the per-sub stages are dispatched
    /// with exactly the same dims as a plain chunk.
    fn encode_prefill_super(
        &mut self,
        layout: &ModelLayout,
        layers: usize,
        n_subs: usize,
    ) -> Result<(), Error> {
        use step_schedule::StepStage;
        self.prefill_causal = true;
        self.forward_m = n_subs * CANVAS;
        self.exec_stage(
            StepStage::EmbedGather,
            0,
            layout,
            StepFinishMode::ForwardOnly,
        )?;
        // NO RmsNormHidden here (task #64/#68 ROOT CAUSE, fixed 2026-07-10):
        // normalizing the embedded hidden is the DENOISE preamble's convention
        // (canvas stream, parity-validated); the reference ENCODER pass feeds
        // embed*sqrt(H) straight into layer 0. The norm is per-row
        // scale-invariant through input_layernorm — layer-0 K/V matched the
        // engine exactly (rel 0.0025), which hid it — but the RESIDUAL stream
        // carried a per-token rescale the model never saw in training:
        // L1 KV diverged 33% from the engine at every length, flipping MoE
        // routes systematically; short prompts tolerated the warped
        // trajectory, long ones collapsed into fluent hallucination.
        let per_layer = step_schedule::per_layer_stages(&self.block_profile);
        for layer in 0..layers {
            for &stage in &per_layer {
                // Diagnostic ablation (bench_prefill_super_stages): skip a stage
                // group to measure its cost as the timing delta. Timing is
                // data-independent, so skipped stages just feed stale arena data
                // downstream. Empty in production (no-op).
                if prefill_bench_skipped(stage) {
                    continue;
                }
                match stage {
                    StepStage::LayerQkRopeKv | StepStage::LayerAttention => {
                        self.use_params_sub = true;
                        for c in 0..n_subs {
                            self.sub_c = c;
                            self.exec_stage(stage, layer, layout, StepFinishMode::ForwardOnly)?;
                        }
                        self.sub_c = 0;
                        self.use_params_sub = false;
                    }
                    _ => {
                        self.exec_stage(stage, layer, layout, StepFinishMode::ForwardOnly)?;
                    }
                }
            }
        }
        self.forward_m = CANVAS;
        Ok(())
    }

    /// Canvas token embed gather only (no no-scale RMSNorm).
    pub(super) fn encode_layer_through_attention(
        &mut self,
        layer: usize,
        layout: &ModelLayout,
    ) -> Result<(), Error> {
        self.encode_layer_qkv_gemm(layer, layout)?;
        self.encode_layer_qk_rope_and_attention(layer, layout)
    }

    fn dispatch_embed_gather(&mut self, embed_off: u64) {
        let fm = self.forward_m;
        use crate::dgq::embed_row::EMBED_SCALE;

        let ps = if self.embed_bf16 {
            &self.ps.embed_gather_bf16
        } else {
            &self.ps.embed_gather
        };
        self.sink_set_pipeline(ps);
        self.bind_blob(0);
        self.bind_state(1);
        self.sink_set_buffer(&self.bufs.arena, self.arena().hidden_off() as usize, 2);
        self.sink_set_bytes(&embed_off, 3);
        let dims = [HID as u32, fm as u32];
        self.sink_set_bytes(&dims, 4);
        self.sink_set_bytes(&EMBED_SCALE, 5);
        let vocab = VOCAB as u32;
        self.sink_set_bytes(&vocab, 6);
        self.bind_debug_status(7);
        let (grid, tg) = crate::shaders::embed_gather::dispatch_shape(HID, fm);
        self.sink_dispatch(grid, tg);
    }

    /// Canvas token embed gather only (no no-scale RMSNorm).
    pub(super) fn encode_preamble_embed_only(&mut self, layout: &ModelLayout) -> Result<(), Error> {
        self.dispatch_embed_gather(layout.embed);
        Ok(())
    }

    pub(super) fn encode_step_preamble(
        &mut self,
        layout: &ModelLayout,
        first_step: u32,
    ) -> Result<(), Error> {
        if first_step == 0 {
            self.encode_sc_logit_rowstats();
            self.encode_sc_softembed(layout)?;

            self.rmsnorm(
                self.arena().soft_off(),
                self.arena().tmp_off(),
                layout.sc_pre_norm,
                HID as u32,
                CANVAS,
            );
            self.gemm_q8(
                self.arena().tmp_off(),
                self.arena().ffg_off(),
                layout.sc_gate,
                CANVAS as u32,
                DENSE_FF,
                HID as u32,
            )?;
            self.gemm_q8(
                self.arena().tmp_off(),
                self.arena().ffu_off(),
                layout.sc_up,
                CANVAS as u32,
                DENSE_FF,
                HID as u32,
            )?;
            self.glu(
                self.arena().ffg_off(),
                self.arena().ffu_off(),
                self.arena().ffg_off(),
                CANVAS * DENSE_FF as usize,
            );
            self.gemm_q8(
                self.arena().ffg_off(),
                self.arena().dense_off(),
                layout.sc_down,
                CANVAS as u32,
                HID as u32,
                DENSE_FF,
            )?;
        }
        // first_step: self.arena().dense_off() stays zero; skip SC MLP + O(vocab) softembed.

        self.dispatch_embed_gather(layout.embed);
        self.residual(
            self.arena().hidden_off(),
            self.arena().dense_off(),
            self.arena().hidden_off(),
            0,
            CANVAS * HID,
        );
        self.rmsnorm(
            self.arena().hidden_off(),
            self.arena().hidden_off(),
            0,
            HID as u32,
            CANVAS,
        );
        Ok(())
    }

    fn encode_partial_lm_head(&mut self, layout: &ModelLayout, m: u32) -> Result<(), Error> {
        let token_list_off = std::mem::offset_of!(RouteScratch, token_list);
        let num_slots_off = std::mem::offset_of!(RouteScratch, num_slots);
        let compact_row = CANVAS as u32 - m;
        let logits_off = (compact_row as usize) * VOCAB * 2;

        self.sink_set_pipeline(&self.ps.compact_active_rows);
        self.bind_state(0);
        self.sink_set_buffer(&self.bufs.route, token_list_off, 1);
        self.sink_set_buffer(&self.bufs.route, num_slots_off, 2);
        let grid = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);

        let gather_dims = [0u32, HID as u32];
        let gather_count = (m as usize) * HID;
        self.dispatch_1d_ranged(
            &self.ps.gather_rows_bf16,
            gather_count,
            256,
            |this, base, _chunk| {
                this.sink_set_buffer(&this.bufs.arena, this.arena().tmp_off() as usize, 0);
                this.sink_set_buffer(&this.bufs.route, token_list_off, 1);
                this.sink_set_buffer(&this.bufs.arena, this.arena().dense_off() as usize, 2);
                this.sink_set_buffer(&this.bufs.dummy_dump, 0, 5);
                this.sink_set_bytes(&gather_dims, 3);
                this.sink_set_bytes(&m, 4);
                this.sink_set_bytes(&base, 6);
            },
        );

        self.gemm_q8_logits(
            self.arena().dense_off(),
            layout.embed,
            m,
            VOCAB as u32,
            HID as u32,
            logits_off,
        )?;

        let dims = [m, VOCAB as u32];
        self.sink_set_pipeline(&self.ps.scatter_logits_rows);
        self.sink_set_buffer(&self.bufs.logits, logits_off, 0);
        self.sink_set_buffer(&self.bufs.logits, 0, 1);
        self.sink_set_buffer(&self.bufs.route, token_list_off, 2);
        self.sink_set_bytes(&dims, 3);
        let grid = MTLSize {
            width: VOCAB,
            height: m as usize,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);
        Ok(())
    }

    fn encode_step_finish(
        &mut self,
        layout: &ModelLayout,
        mode: StepFinishMode,
    ) -> Result<(), Error> {
        self.rmsnorm(
            self.arena().hidden_off(),
            self.arena().tmp_off(),
            layout.final_norm,
            HID as u32,
            CANVAS,
        );
        let m = self.partial_lm_m;
        if partial_lm_head_enabled() && m < CANVAS as u32 {
            self.encode_partial_lm_head(layout, m)?;
        } else {
            self.gemm_q8_logits(
                self.arena().tmp_off(),
                layout.embed,
                CANVAS as u32,
                VOCAB as u32,
                HID as u32,
                0,
            )?;
        }
        self.dispatch_softcap();
        if mode == StepFinishMode::ForwardOnly {
            return Ok(());
        }
        self.encode_step_sampler(layout)
    }

    fn encode_step_sampler(&mut self, _layout: &ModelLayout) -> Result<(), Error> {
        let cols = VOCAB as u32;
        // Active denoise width (E3/E6 shrink-on-retry): the sampler stats
        // (mean-entropy, canvas-stable, accept-plateau) that drive early-stop
        // must cover exactly the active rows — stale rows [active..CANVAS) would
        // corrupt convergence. rowstats/apply run one threadgroup per row;
        // commit/write loop over `canvas` internally.
        let canvas = self.active_canvas as u32;
        let pad = crate::sample::PAD_TOKEN_ID;
        let filler = crate::sample::FILLER_TOKEN_ID;

        self.sink_set_pipeline(&self.ps.sample_rowstats);
        self.bind_logits(0);
        self.sink_set_buffer(&self.bufs.arena, self.arena().rs_samp_off() as usize, 1);
        self.bind_state(2);
        self.bind_params(3);
        self.sink_set_bytes(&cols, 4);
        self.sink_set_bytes(&pad, 5);
        self.sink_set_bytes(&filler, 6);
        let eos = read_struct::<StepParams>(&self.bufs.params).eos_token_id;
        self.sink_set_bytes(&eos, 7);
        self.bind_debug_status(8);
        let grid = MTLSize {
            width: self.active_canvas,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);

        self.sink_set_pipeline(&self.ps.sample_commit);
        self.bind_state(0);
        self.bind_params(1);
        self.sink_set_bytes(&canvas, 2);
        self.sink_set_bytes(&pad, 3);
        self.sink_set_bytes(&filler, 4);
        let eos = read_struct::<StepParams>(&self.bufs.params).eos_token_id;
        self.sink_set_bytes(&eos, 5);
        self.bind_debug_status(6);
        let es_ent = crate::flags::early_stop_mean_ent();
        self.sink_set_bytes(&es_ent, 7);
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let grid = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);

        self.sink_set_pipeline(&self.ps.sample_apply);
        self.bind_logits(0);
        self.sink_set_buffer(&self.bufs.arena, self.arena().rs_samp_off() as usize, 1);
        self.bind_state(2);
        self.bind_params(3);
        self.sink_set_bytes(&cols, 4);
        self.bind_debug_status(5);
        let grid = MTLSize {
            width: self.active_canvas,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);

        self.sink_set_pipeline(&self.ps.sample_write);
        self.bind_state(0);
        self.sink_set_bytes(&canvas, 1);
        self.sink_set_bytes(&cols, 2);
        self.bind_debug_status(3);
        let freeze: u32 = freeze_enabled() as u32;
        let use_argmax: u32 = denoiser_argmax_enabled() as u32;
        self.sink_set_bytes(&freeze, 4);
        self.sink_set_bytes(&use_argmax, 5);
        let grid = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        self.sink_dispatch(grid, tg);

        // MLX applies the schedule temperature before the SC soft-embed softmax.
        let st: CanvasState = read_struct(&self.bufs.state);
        let params: StepParams = read_struct(&self.bufs.params);
        let t = scheduled_temperature(st.step, &params).max(1e-6);
        self.scale_half_logits(CANVAS * VOCAB, 1.0 / t);
        Ok(())
    }
}

pub struct StepRuntime {
    /// Byte grant covering this runtime's footprint (weights blob + KV +
    /// scratch slack). Held for the runtime's lifetime; releasing IS dropping
    /// the runtime, so the budget cannot be held wrong. See `membudget`.
    _mem_permit: crate::membudget::MemPermit,
    pub(super) ctx: MetalContext,
    pipelines: &'static StepPipelines,
    /// fp16-arena pipeline set (E11, DGQ_PREFILL_F16): identical kernels
    /// compiled with K_ARENA_F16 — the fast prefill dispatches through these
    /// while denoise keeps the gate-validated bf16 set. None when off.
    pub(super) pipelines_prefill_f16: Option<&'static StepPipelines>,
    /// While true, dispatch_and_wait encodes against the fp16 set (bracketed
    /// around prefill_chunks_from's chunk loop).
    pub(super) arena_f16_mode: bool,
    pub(super) bufs: StepBuffers,
    pub(super) gpu_blob: std::sync::Arc<DgqGpuBlob>,
    weight_cache: GpuDecoderWeightCache,
    text_config: TextConfig,
    pub(super) block_profile: StepBlockProfile,
    attn_ffn_q8: bool,
    attn_ffn_bf16: bool,
    embed_bf16: bool,
    pub(super) layout: ModelLayout,
    tensor_offsets: HashMap<String, u64>,
    pub layers: usize,
    /// KV-cache capacity (positions per layer) the buffers were sized for. Every
    /// denoise block writes a CANVAS-wide canvas at [kv_len..kv_len+CANVAS], so
    /// kv_len + CANVAS must never exceed this — checked in `set_kv_len`.
    max_seq: usize,
    /// E14: KV position the f32 side ring is valid up to. A fast prefill
    /// resuming at a different offset re-hydrates the window from the
    /// monolithic cache first (rollback / restore / conversation swap all
    /// invalidate silently — the mismatch check heals every case).
    kv_f32_side_valid: usize,
    /// Active canvas width for the NEXT denoise step (E3/E6 shrink-on-retry).
    /// `CANVAS` (256) normally; the block loop narrows it (128/64) when re-
    /// rolling a degenerate reply. Clamped to [1, CANVAS].
    active_canvas: usize,
}

impl StepRuntime {
    /// KV-cache capacity (positions per layer) the buffers were sized for.
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    pub fn layout(&self) -> &ModelLayout {
        &self.layout
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
            kv_len as usize + CANVAS <= self.max_seq,
            "KV cache overflow: kv_len={kv_len} + CANVAS={CANVAS} > max_seq={}; \
             size max_seq >= prompt_len + max_new_tokens + CANVAS",
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

    /// E14: refill each sliding layer's f32 side ring from the monolithic
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
        // E11 M0 range trace (DGQ_TRACE_RANGES=1): after each chunk forward the
        // arena planes hold the LAST layer's stage outputs — sample max|x| and
        // non-finites per plane across all chunks, so position-dependence is
        // covered. Answers "do prefill activations fit fp16 (max 65504)?".
        type RangePeaks = std::collections::BTreeMap<&'static str, (f32, bool, Option<usize>)>;
        let mut range_peaks: Option<RangePeaks> =
            crate::flags::trace_ranges_enabled().then(RangePeaks::new);
        let probe_planes = |this: &Self, m_rows: usize, peaks: &mut RangePeaks| {
            let a = &this.bufs.arena_map;
            for (label, off, per_row) in [
                ("hidden", a.hidden_off(), HID),
                ("attnq(Q)", a.attnq_off(), 4096),
                ("attn_out", a.attno_off(), 4096),
                ("tmp(o_proj)", a.tmp_off(), HID),
                ("ffg(gate_up)", a.ffg_off(), DENSE_FF as usize),
                ("dense", a.dense_off(), HID),
                ("moein", a.moein_off(), HID),
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
        // would clobber the oldest live window slots (task #64).
        self.set_kv_write_end(n as u32);
        // E14: the prefill attention reads sliding K/V from the f32 side ring;
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
                (remaining / CANVAS).min(PREFILL_SUBS)
            } else {
                1
            };
            if n_subs >= 2 && pos + n_subs * CANVAS + CANVAS <= self.max_seq {
                let m = n_subs * CANVAS;
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
            let chunk_len = remaining.min(CANVAS);
            let mut ids = [0u32; CANVAS];
            ids[..chunk_len]
                .copy_from_slice(&delta_token_ids[pos - offset..pos - offset + chunk_len]);
            self.set_canvas_ids(&ids)?;
            self.set_kv_len(pos as u32);
            self.dispatch_and_wait(|enc| enc.encode_prefill_chunk(&layout, layers))?;
            if let Some(peaks) = range_peaks.as_mut() {
                probe_planes(self, CANVAS, peaks);
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

    pub fn set_canvas_ids(&mut self, ids: &[u32]) -> Result<(), Error> {
        // CANVAS for denoise / plain prefill chunks; up to PREFILL_M for a
        // batched prefill super-chunk.
        if ids.len() != CANVAS && (ids.len() > PREFILL_M || !ids.len().is_multiple_of(CANVAS)) {
            return Err(Error::Format(
                "canvas ids length must be CANVAS..=PREFILL_M",
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
            p.kv_len = base_kv_len + (c * CANVAS) as u32;
            unsafe { std::ptr::write(ptr.add(c), p) };
        }
    }

    pub fn write_canvas_state(&mut self, state: &CanvasState) {
        write_struct(&self.bufs.state, state);
    }

    /// New denoise block: fresh random canvas, reset step/stop, patch sampler params.
    /// Active denoise canvas width for subsequent steps (E3/E6 shrink-on-retry).
    /// Clamped to [1, CANVAS]. A `DGQ_FORCE_CANVAS` override still wins at the
    /// dispatch site. The block loop reads it back via `active_canvas()`.
    pub fn set_active_canvas(&mut self, w: usize) {
        self.active_canvas = w.clamp(1, CANVAS);
    }

    /// Effective active canvas width (the `DGQ_FORCE_CANVAS` test override, else
    /// the value set by `set_active_canvas`). The CPU block loop slices canvas
    /// stats/trim to this many rows.
    pub fn active_canvas(&self) -> usize {
        crate::flags::force_canvas()
            .map(|w| (w as usize).clamp(1, CANVAS))
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
        let weight_bytes = grouped_expert_blob_bytes_per_expert(self.block_profile.format);
        forward.record_expert_layers_grouped(counts, weight_bytes);
    }

    /// Host readback size for one `CanvasState` poll (shared buffer, no extra sync).
    pub const CANVAS_STATE_BYTES: usize = std::mem::size_of::<CanvasState>();

    /// P2.1 budget: host bytes touched per denoise step on the generate hot path.
    pub fn denoise_step_host_readback_bytes(check_logits: bool) -> u64 {
        // Forward reads state once on CPU to seed preamble; generate polls once after sync.
        let mut bytes = (Self::CANVAS_STATE_BYTES as u64) * 2;
        if check_logits && logits_finite_check_enabled() {
            bytes += logits_finite_sample_bytes();
        }
        bytes
    }

    /// Opt-in full-tensor logits scan (`DGQ_CHECK_LOGITS=1`). Off by default (P2.1).
    pub fn check_logits_finite(&self) -> Result<(), Error> {
        if !logits_finite_check_enabled() {
            return Ok(());
        }
        let sample = logits_finite_sample_count().min(CANVAS * VOCAB);
        let (finite, max_abs) = half_buffer_stats(&self.bufs.logits, 0, CANVAS * VOCAB, sample);
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
            block_profile: self.block_profile,
            tensor_offsets: &self.tensor_offsets,
            partial_lm_m: CANVAS as u32,
            attn_ffn_q8: self.attn_ffn_q8,
            attn_ffn_bf16: self.attn_ffn_bf16,
            embed_bf16: self.embed_bf16,
            prefill_causal: false,
            forward_m: CANVAS,
            active_canvas: CANVAS,
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
            CANVAS * HID,
        );
        let mut moe_out = vec![0.0f32; CANVAS * HID];
        let mut scratch = MoeScratch::new(CANVAS, &self.text_config);
        experts_forward_dgq_cpu(
            &mut moe_out,
            &moe_in,
            &self.weight_cache,
            layer,
            &self.text_config,
            CANVAS,
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
            CANVAS * HID,
        );
        probe(
            self,
            "preamble:dense(sc_mlp)",
            self.bufs.arena_map.dense_off(),
            CANVAS * HID,
        );
        probe(
            self,
            "preamble:hidden(embed+sc)",
            self.bufs.arena_map.hidden_off(),
            CANVAS * HID,
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
            probe(self, "layer:qkv(Q)", attnq, CANVAS * 4096);
            self.time_enc_stage(|e| e.encode_layer_qk_rope_kv_dispatch(layer, &layout))?;
            self.time_enc_stage(|e| e.encode_layer_attention_dispatch(layer, &layout))?;
            probe(self, "layer:attn_out", attno, CANVAS * 4096);
            self.time_enc_stage(|e| e.encode_layer_o_proj_gemm(layer, &layout))?;
            probe(self, "layer:o_proj", tmp, CANVAS * HID);
            self.time_enc_stage(|e| e.encode_layer_o_proj_tail(layer, &layout))?;
            probe(self, "layer:resid_pre_moe(hidden)", hidden, CANVAS * HID);
            let l = &layout.layers[layer];
            self.time_enc_stage(|e| {
                e.rmsnorm(
                    e.arena().stream_off(),
                    e.arena().tmp_off(),
                    l.pre_ff_ln,
                    HID as u32,
                    CANVAS,
                );
                Ok(())
            })?;
            self.time_enc_stage(|e| e.encode_layer_dense_gate_up(layer, &layout))?;
            probe(self, "layer:gate_up", ffg, CANVAS * DENSE_FF as usize);
            self.time_enc_stage(|e| {
                e.glu(
                    e.arena().ffg_off(),
                    e.arena().ffu_off(),
                    e.arena().ffg_off(),
                    CANVAS * DENSE_FF as usize,
                );
                Ok(())
            })?;
            probe(self, "layer:swiglu", ffg, CANVAS * DENSE_FF as usize);
            self.time_enc_stage(|e| e.encode_layer_dense_down(layer, &layout))?;
            probe(self, "layer:dense_down", dense, CANVAS * HID);
            self.time_enc_stage(|e| {
                e.rmsnorm(
                    e.arena().dense_off(),
                    e.arena().dense_off(),
                    l.post_ff_ln_1,
                    HID as u32,
                    CANVAS,
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
            probe(self, "layer:moe_norm(moein)", moein, CANVAS * HID);
            self.time_enc_stage(|e| e.encode_layer_moe_post_combine(layer, &layout))?;
            probe(self, "layer:resid_post_moe(hidden)", hidden, CANVAS * HID);
            // Per-layer residual peak (the f16-overflow suspect).
            let (_, hmx) =
                half_buffer_stats(&self.bufs.arena, hidden as usize, CANVAS * HID, SAMPLE);
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
                    HID as u32,
                    CANVAS,
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
                    CANVAS * DENSE_FF as usize,
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
                    HID as u32,
                    CANVAS,
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
    /// Holistic prefill proxy (task #87): time ONE M=1024 super-chunk forward at
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
        let m = n_subs * CANVAS;
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
            let elems = CANVAS * VOCAB;
            let sample = elems.min(8192);
            let (nf, mx) = half_buffer_stats(&self.bufs.logits, 0, elems, sample);
            eprintln!(
                "monolithic pre-sc: st.step={} logits_max_abs={:.4} non_finite_sample={}",
                st_before.step, mx, nf
            );
        }
        let first_step = if st_before.step == 0 { 1u32 } else { 0u32 };
        let partial_lm_m = partial_lm_active_rows(&st_before);
        // Denoise (Full) may run a narrowed canvas (E3/E6 shrink-on-retry);
        // ForwardOnly (prefill/dump) always uses the full CANVAS sub width.
        let active_canvas = if finish == StepFinishMode::Full {
            crate::flags::force_canvas()
                .map(|w| (w as usize).clamp(1, CANVAS))
                .unwrap_or(self.active_canvas)
                .clamp(1, CANVAS)
        } else {
            CANVAS
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

static STEP_PIPELINES_CACHE: std::sync::OnceLock<
    std::sync::Mutex<HashMap<StepPipelineKey, &'static StepPipelines>>,
> = std::sync::OnceLock::new();

/// Diagnostic stage-ablation skip set (bench_prefill_super_stages only). Empty
/// in production → `prefill_bench_skipped` is a fast empty-slice check.
static PREFILL_BENCH_SKIP: std::sync::OnceLock<std::sync::Mutex<Vec<step_schedule::StepStage>>> =
    std::sync::OnceLock::new();

fn set_prefill_bench_skip(stages: &[step_schedule::StepStage]) {
    let m = PREFILL_BENCH_SKIP.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    *m.lock().unwrap() = stages.to_vec();
}

fn prefill_bench_skipped(stage: step_schedule::StepStage) -> bool {
    match PREFILL_BENCH_SKIP.get() {
        Some(m) => m.lock().unwrap().contains(&stage),
        None => false,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct StepPipelineKey(u8);

fn step_pipeline_key(
    variant: crate::shaders::variant::KernelVariant,
    fmt: crate::shaders::kv_quant::KvFormat,
) -> StepPipelineKey {
    // KV format code occupies bits 3-4 (0=f16, 1=q8, 2=q4); bit 5 = fp16 arena.
    StepPipelineKey(
        u8::from(variant.shape_assert)
            | (u8::from(variant.debug_fast) << 1)
            | (u8::from(variant.debug_deep) << 2)
            | ((fmt.code() as u8) << 3)
            | (u8::from(variant.arena_f16) << 5),
    )
}

fn shared_step_pipelines(
    ctx: &MetalContext,
    fmt: crate::shaders::kv_quant::KvFormat,
) -> Result<&'static StepPipelines, Error> {
    let variant = crate::shaders::variant::runtime_step_variant();
    let key = step_pipeline_key(variant, fmt);
    let cache = STEP_PIPELINES_CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .map_err(|_| Error::Gpu("step pipelines cache poisoned"))?;
    if let Some(&pipelines) = guard.get(&key) {
        return Ok(pipelines);
    }

    let pipelines = StepPipelines::new(ctx, variant, fmt)?;
    let leaked: &'static StepPipelines = Box::leak(Box::new(pipelines));
    guard.insert(key, leaked);
    crate::metal::pipeline_cache::PipelineArchiveCache::flush_global();
    Ok(leaked)
}

pub fn log_step_memory_budget(blob_bytes: u64, max_seq: usize, layout: &ModelLayout) {
    let kv = kv_cache_bytes(layout, max_seq);
    let logits = (CANVAS * VOCAB * 2) as u64;
    let sc_probs = sc_probs_buffer_bytes() as u64;
    let arena = step_arena_layout().bytes();
    let (mx, mw) = gemm_scratch_bytes();
    let gemm_scratch = (mx + mw) as u64;
    let gpu_static = kv + logits + sc_probs + arena + gemm_scratch;
    let total = blob_bytes + gpu_static;
    if crate::flags::progress_enabled() {
        eprintln!("step-kernel memory budget:");
        eprintln!(
            "  blob:       {:.2} GiB",
            blob_bytes as f64 / (1024.0_f64.powi(3))
        );
        eprintln!("  arena:      {:.2} MiB", arena as f64 / (1024.0 * 1024.0));
        eprintln!(
            "  kv cache:   {:.2} MiB (max_seq={max_seq})",
            kv as f64 / (1024.0 * 1024.0)
        );
        eprintln!("  logits:     {:.2} MiB", logits as f64 / (1024.0 * 1024.0));
        eprintln!(
            "  sc_probs:   {:.2} MiB",
            sc_probs as f64 / (1024.0 * 1024.0)
        );
        eprintln!(
            "  gemm scratch:{:.2} MiB",
            gemm_scratch as f64 / (1024.0 * 1024.0)
        );
        eprintln!(
            "  gpu static: {:.2} GiB (excl. blob)",
            gpu_static as f64 / (1024.0_f64.powi(3))
        );
        eprintln!(
            "  total est:  {:.2} GiB",
            total as f64 / (1024.0_f64.powi(3))
        );
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StepRuntimeBuildTiming {
    pub compile: std::time::Duration,
    pub total: std::time::Duration,
}

pub fn build_step_runtime(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
) -> Result<(StepRuntime, StepRuntimeBuildTiming), Error> {
    let build_started = Instant::now();
    let validated = crate::metal::step_config::validate_step_model(model_dir)?;
    crate::metal::step_config::log_validated_step_model(&validated);

    let store = DgqStore::open(model_dir)?;
    let offsets = build_offsets_from_store(&store);
    let layout = build_layout(&offsets, cfg.max_seq);
    if crate::flags::progress_enabled() {
        let fmt = crate::flags::kv_format(cfg.max_seq);
        if fmt != crate::shaders::kv_quant::KvFormat::F16 {
            eprintln!(
                "step-kernel: {} KV cache (auto at long context, max_seq={}) — shrinks KV to stay under the GPU working-set cap",
                fmt.label(),
                cfg.max_seq
            );
        }
    }
    let layers = cfg.layers.min(validated.num_layers).max(1);

    // Mixed-precision .dgq stores attention + dense-FFN as bf16 (Raw) — or q8 on
    // earlier checkpoints. Detect from the actual stored kind so any vintage
    // dispatches correctly (bf16 / q8 / uniform-q4).
    let attn_ffn_kind = store
        .get_entry("model.decoder.layers.0.self_attn.q_proj.weight")
        .and_then(|e| crate::dgq::layout::parse_quant_kind(&e.meta.kind).ok());
    let attn_ffn_q8 = attn_ffn_kind == Some(crate::dgq::layout::QuantKind::Q8Row);
    let attn_ffn_bf16 = attn_ffn_kind == Some(crate::dgq::layout::QuantKind::Raw);

    // Embed (tied lm_head + SC soft-embed) is q8-per-row on most checkpoints, bf16
    // (Raw) on newer ones. Detect from the stored kind so all three embed consumers
    // (input gather, lm_head, SC softembed) dispatch the matching precision.
    let embed_bf16 = store
        .get_entry("model.decoder.embed_tokens.weight")
        .and_then(|e| crate::dgq::layout::parse_quant_kind(&e.meta.kind).ok())
        == Some(crate::dgq::layout::QuantKind::Raw);
    let block_profile = StepBlockProfile::from_store_profile(store.profile());
    if crate::flags::progress_enabled() {
        if embed_bf16 {
            eprintln!("step-kernel: bf16 embed (tied lm_head + SC)");
        }
        match block_profile.format {
            QuantFormat::NvFp4 => eprintln!("step-kernel: nvfp4 block weights"),
            QuantFormat::Q4Affine => eprintln!("step-kernel: q4 block weights"),
            _ => eprintln!("step-kernel: block weights ({:?})", block_profile.format),
        }
        match block_profile.moe_style() {
            MoeExecutionStyle::BatchedGrouped => {
                eprintln!("step-kernel: batched grouped MoE");
            }
            MoeExecutionStyle::ScalarPerExpert => {
                eprintln!("step-kernel: scalar per-expert MoE");
            }
        }
    }

    // Take the machine-wide memory grant BEFORE any model-scale allocation.
    // Blocks (FIFO, cross-process aware) until the footprint fits; see
    // `membudget` for the contract. Estimate: the weights blob (GPU upload;
    // the mmap side is clean file-backed and evictable), the KV cache at this
    // max_seq, and a fixed slack for arena/scratch/encoder-cache structures.
    let estimate = store.blob_bytes() as usize
        + crate::metal::step_kv::kv_cache_total_bytes(&layout, cfg.max_seq) as usize
        + (2usize << 30);
    let mem_permit = crate::membudget::MemBudget::global().acquire(estimate, "step-runtime");

    let ctx = MetalContext::new()?;
    let compile_started = Instant::now();
    let kv_fmt = crate::flags::kv_format(cfg.max_seq);
    // TEMP diagnostic (E11 bring-up): DGQ_ARENA_F16_ALL=1 builds the MAIN set
    // fp16 too — bisects kernel-level breakage from mode-switch wiring.
    let f16_all = crate::flags::arena_f16_all_enabled();
    if f16_all {
        crate::shaders::variant::set_arena_f16_compile(true);
    }
    let pipelines = shared_step_pipelines(&ctx, kv_fmt)?;
    // f16_all: LEAVE the compile-mode atomic on — lazy dispatch-time compiles
    // (stacked GEMM) must also build f16 for the whole session.
    let pipelines_prefill_f16 = if crate::flags::prefill_f16_enabled() {
        crate::shaders::variant::set_arena_f16_compile(true);
        let out = shared_step_pipelines(&ctx, kv_fmt);
        crate::shaders::variant::set_arena_f16_compile(false);
        Some(out?)
    } else {
        None
    };
    let compile = compile_started.elapsed();

    let gpu_blob = DgqGpuBlob::from_store(&store, &ctx.device)?;
    let gpu_blob = std::sync::Arc::clone(&gpu_blob);
    let kv_bytes = kv_cache_bytes(&layout, cfg.max_seq) as usize;
    let logits_bytes = CANVAS * VOCAB * 2;
    let sc_probs_bytes = sc_probs_buffer_bytes();

    log_step_memory_budget(store.blob_bytes(), cfg.max_seq, &layout);

    let sampler = crate::sample::sampler_for_steps(cfg.steps.max(1), cfg.no_early_stop);
    let prefill_len = cfg
        .prefill_token_ids
        .as_ref()
        .map(|t| t.len() as u32)
        .unwrap_or(cfg.kv_len);
    let model_cfg = ModelConfig::load(model_dir)?;
    let eos_token_id = model_cfg.eos_token_id_u32();
    let params = step_params_from_sampler(&sampler, prefill_len, cfg.no_early_stop, eos_token_id);
    let state = init_canvas_state(cfg.seed, VOCAB);
    let (gemm_a_bytes, gemm_b_bytes) = gemm_scratch_bytes();

    let text_config = model_cfg.text_config;
    let weight_store = WeightStore::open(model_dir)?;
    let weight_cache = GpuDecoderWeightCache::load_with_dgq_blob(
        &weight_store,
        &text_config,
        &ctx.device,
        std::sync::Arc::clone(&gpu_blob),
    )?;

    let arena_map = step_arena_layout();
    // KV cache: optionally file-backed (DGQ_KV_MMAP) so dirty pages evict to a
    // real file instead of anonymous swap. `kv_mmap_backing` must outlive
    // `kvcache` — it is declared after it in `StepBuffers` so it drops later.
    let (kvcache, kv_mmap_backing) = alloc_kv_buffer(&ctx.device, kv_bytes)?;
    // E14 f32 side K/V ring for sliding layers (DGQ_PREFILL_KV_F32).
    let mut kv_f32_side_offs = [u64::MAX; N_LAYERS];
    let kv_f32_side = if crate::flags::prefill_kv_f32_enabled() {
        let mut off = 0u64;
        for i in 0..N_LAYERS {
            let l = &layout.layers[i];
            kv_f32_side_offs[i] = off;
            off += (layer_kv_slots(l.is_full != 0, cfg.max_seq)
                * (l.n_kv_heads * l.head_dim) as usize
                * 2
                * 4) as u64;
        }
        if crate::flags::progress_enabled() {
            eprintln!(
                "step-kernel: f32 side KV ring {:.0} MiB (DGQ_PREFILL_KV_F32)",
                off as f64 / (1024.0 * 1024.0)
            );
        }
        Some(alloc_buffer(&ctx.device, off as usize)?)
    } else {
        None
    };
    let bufs = StepBuffers {
        kv_f32_side,
        kv_f32_side_offs,
        blob: gpu_blob.buffer.clone(),
        blob_experts: {
            let (b, _) = gpu_blob.expert_region();
            objc2::rc::Retained::from(b)
        },
        blob_expert_base: gpu_blob.expert_region().1,
        layout: {
            let b = alloc_buffer(&ctx.device, std::mem::size_of::<ModelLayout>())?;
            write_struct(&b, &layout);
            b
        },
        params: {
            let b = alloc_buffer(&ctx.device, std::mem::size_of::<StepParams>())?;
            write_struct(&b, &params);
            b
        },
        arena: alloc_buffer(&ctx.device, arena_map.bytes() as usize)?,
        kvcache,
        _kv_mmap: kv_mmap_backing,
        state: {
            let b = alloc_buffer(&ctx.device, std::mem::size_of::<CanvasState>())?;
            write_struct(&b, &state);
            b
        },
        logits: alloc_buffer(&ctx.device, logits_bytes)?,
        sc_probs: alloc_buffer(&ctx.device, sc_probs_bytes)?,
        route: {
            let b = alloc_buffer(&ctx.device, std::mem::size_of::<RouteScratch>())?;
            zero_buffer(&b);
            b
        },
        dummy_dump: alloc_buffer(&ctx.device, 4)?,
        debug_status: if crate::shaders::variant::runtime_kernel_debug_enabled() {
            Some(alloc_buffer(
                &ctx.device,
                crate::metal::debug_status::DEBUG_STATUS_BYTES,
            )?)
        } else {
            None
        },
        gemm_a: alloc_buffer(&ctx.device, gemm_a_bytes)?,
        gemm_b: alloc_buffer(&ctx.device, gemm_b_bytes)?,
        expert_layer_unique: alloc_buffer(&ctx.device, N_LAYERS * std::mem::size_of::<u32>())?,
        moe_grouped_indirect: alloc_buffer(&ctx.device, MOE_GROUPED_INDIRECT_BYTES)?,
        // {m,l} + O[512] per (16 heads x 256 rows), f32: ~8.4 MiB.
        attn_state: alloc_buffer(
            &ctx.device,
            STEP_NQ_HEADS * CANVAS * (2 + 512) * std::mem::size_of::<f32>(),
        )?,
        // E17 GEMM-attention prefill scratch (opt-in). Head-chunked (E17a): the
        // score matrix S/P holds only HC heads at a time —
        // [HC][CANVAS][n_pad(max_seq)]. Allocated when DGQ_GEMM_ATTN OR
        // DGQ_ATTN_TOPK is set (E20 reuses the same S plane; the two paths are
        // never both live on the same layer).
        attn_gemm_s: if crate::flags::gemm_attn_enabled()
            || crate::flags::attn_topk_enabled()
            || crate::flags::attn_topk_decode_enabled()
        {
            let np = crate::shaders::attention_gemm::n_pad(cfg.max_seq);
            // Both E17 and E20 encoders batch heads by DGQ_GEMM_ATTN_HC (task
            // #97 unified them); size from the same flag so the encoder can
            // never outrun the scratch.
            let hc = crate::flags::gemm_attn_head_chunk().min(STEP_NQ_HEADS);
            Some(alloc_buffer(
                &ctx.device,
                hc * CANVAS * np * std::mem::size_of::<f32>(),
            )?)
        } else {
            None
        },
        attn_gemm_p: if crate::flags::gemm_attn_enabled() {
            let np = crate::shaders::attention_gemm::n_pad(cfg.max_seq);
            let hc = crate::flags::gemm_attn_head_chunk().min(STEP_NQ_HEADS);
            // f32-side path stores f32 probs; size for f32 (4B) so either path fits.
            Some(alloc_buffer(&ctx.device, hc * CANVAS * np * 4)?)
        } else {
            None
        },
        attn_gemm_lrow: if crate::flags::gemm_attn_enabled() {
            let hc = crate::flags::gemm_attn_head_chunk().min(STEP_NQ_HEADS);
            Some(alloc_buffer(
                &ctx.device,
                hc * CANVAS * std::mem::size_of::<f32>(),
            )?)
        } else {
            None
        },
        // E20 top-k scratch: compressed P [HC][CANVAS][K_PAD] (f32), indices
        // Idx [HC][CANVAS][K_PAD] (u32), lrow [HC][CANVAS] (f32), and the u16
        // key plane [HC][CANVAS][n_pad] (FC32 output of QK, read by the
        // selection passes — half the bytes of the S plane). HC comes from the
        // same flag the encoder batches by (task #97). The S plane is shared
        // with attn_gemm_s (allocated above when either flag is set).
        attn_topk_p: if crate::flags::attn_topk_enabled()
            || crate::flags::attn_topk_decode_enabled()
        {
            let hc = crate::flags::gemm_attn_head_chunk().min(STEP_NQ_HEADS);
            Some(alloc_buffer(
                &ctx.device,
                hc * CANVAS * crate::flags::attn_topk_k_pad() * std::mem::size_of::<f32>(),
            )?)
        } else {
            None
        },
        attn_topk_idx: if crate::flags::attn_topk_enabled()
            || crate::flags::attn_topk_decode_enabled()
        {
            let hc = crate::flags::gemm_attn_head_chunk().min(STEP_NQ_HEADS);
            Some(alloc_buffer(
                &ctx.device,
                hc * CANVAS * crate::flags::attn_topk_k_pad() * std::mem::size_of::<u32>(),
            )?)
        } else {
            None
        },
        attn_topk_lrow: if crate::flags::attn_topk_enabled()
            || crate::flags::attn_topk_decode_enabled()
        {
            let hc = crate::flags::gemm_attn_head_chunk().min(STEP_NQ_HEADS);
            Some(alloc_buffer(
                &ctx.device,
                hc * CANVAS * std::mem::size_of::<f32>(),
            )?)
        } else {
            None
        },
        attn_topk_pat: if crate::flags::attn_topk_enabled()
            || crate::flags::attn_topk_decode_enabled()
        {
            let np = crate::shaders::attention_gemm::n_pad(cfg.max_seq);
            let hc = crate::flags::gemm_attn_head_chunk().min(STEP_NQ_HEADS);
            Some(alloc_buffer(
                &ctx.device,
                hc * CANVAS * np * std::mem::size_of::<u16>(),
            )?)
        } else {
            None
        },
        params_sub: alloc_buffer(
            &ctx.device,
            PREFILL_SUBS * std::mem::size_of::<StepParams>(),
        )?,
        arena_map,
        arena_layout_buf: {
            let b = alloc_buffer(&ctx.device, std::mem::size_of::<ArenaLayout>())?;
            write_struct(&b, &arena_map);
            b
        },
    };
    zero_buffer(&bufs.expert_layer_unique);
    zero_buffer(&bufs.moe_grouped_indirect);
    zero_buffer(&bufs.arena);
    zero_buffer(&bufs.kvcache);
    zero_buffer(&bufs.logits);

    // Fast-prefill (DGQ_FAST_PREFILL) runs on the step kernels AFTER `rt` is built
    // (prefill_chunks is a StepRuntime method); the slow f32-engine prefill runs
    // here at open time otherwise.
    if let Some(ref token_ids) = cfg.prefill_token_ids
        && !should_fast_prefill(token_ids.len())
    {
        let mut encoder = crate::metal::step_kv::MonolithicEncoderCache::open_opt(
            model_dir,
            CANVAS,
            cfg.max_seq,
            Some(std::sync::Arc::clone(&gpu_blob)),
        )?;
        let (kv_len, _) = crate::metal::step_kv::prefill_monolithic_kv_with_cache(
            &mut encoder,
            token_ids,
            &bufs.kvcache,
            &layout,
            cfg.max_seq,
            layers,
        )?;
        if kv_len as u32 != prefill_len {
            return Err(Error::Runtime("prefill kv_len mismatch"));
        }
        eprintln!("step-kernel: prefilled kv_len={kv_len} tokens");
    }

    let build = StepRuntimeBuildTiming {
        compile,
        total: build_started.elapsed(),
    };
    if crate::flags::progress_enabled() {
        eprintln!(
            "step-kernel: runtime built (total={:.2?}, compile={:.2?})",
            build.total, build.compile
        );
    }
    let mut rt = StepRuntime {
        _mem_permit: mem_permit,
        ctx,
        pipelines,
        pipelines_prefill_f16,
        arena_f16_mode: false,
        kv_f32_side_valid: 0,
        bufs,
        gpu_blob,
        weight_cache,
        text_config,
        block_profile,
        attn_ffn_q8,
        attn_ffn_bf16,
        embed_bf16,
        layout,
        tensor_offsets: offsets,
        layers,
        max_seq: cfg.max_seq,
        active_canvas: CANVAS,
    };
    if crate::flags::progress_enabled() {
        if rt.embed_bf16 && sc_sparse_enabled() {
            eprintln!(
                "step-kernel: sparse SC softembed (DGQ_SC_SPARSE=0 for the exact chunked path)"
            );
        } else {
            eprintln!("step-kernel: chunked SC softembed");
        }
    }
    if let Some(ref token_ids) = cfg.prefill_token_ids
        && should_fast_prefill(token_ids.len())
    {
        let started = Instant::now();
        let kv_len = rt.prefill_chunks(token_ids)?;
        if kv_len as u32 != prefill_len {
            return Err(Error::Runtime("fast-prefill kv_len mismatch"));
        }
        if crate::flags::progress_enabled() {
            eprintln!(
                "step-kernel: fast-prefilled kv_len={kv_len} tokens ({:.2?})",
                started.elapsed()
            );
        }
    }
    if let Some(path) = crate::flags::dump_kv_path() {
        dump_buffer_raw(&rt.bufs.kvcache, &path);
    }
    Ok((rt, build))
}
