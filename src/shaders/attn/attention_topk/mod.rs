//! Top-k sparse attention for full layers — causal PREFILL (`DGQ_ATTN_TOPK`)
//! and bidirectional DENOISE (`DGQ_ATTN_TOPK_DECODE`), both default ON.
//!
//! Sibling of dense GEMM attention. Reuses the `attn_gemm_qk` kernel verbatim
//! (the S plane is identical), then replaces softmax+PV with a top-k softmax +
//! sparse-PV that gathers V at the per-row selected key indices.
//! Quality-gated (non-bit-identical): only the top-`k` highest-scoring keys per
//! (row, head) are kept and renormalized.
//!
//! The CPU oracle (`cpu.rs`) is the authoritative numeric reference. The GPU
//! kernels (`attn_topk_softmax`, `attn_topk_pv`) live in `attention_topk.metal`.

#[cfg(target_os = "macos")]
use crate::Error;

pub mod cpu;

/// The QK pipeline is compiled from the GEMM-attention source (the kernel is identical).
pub const ENTRY_QK: &str = crate::shaders::attention_gemm::ENTRY_QK;
/// The GEMM-attention shader source (reused verbatim for the QK pipeline).
pub const SHADER_QK: &str = crate::shaders::attention_gemm::SHADER;

pub const ENTRY_TOPK_SM: &str = "attn_topk_softmax";
pub const ENTRY_TOPK_PV: &str = "attn_topk_pv";
pub const SHADER_TOPK: &str = include_str!("attention_topk.metal");

/// Default compile-time slot capacity for the compressed P/Idx planes. The
/// production K_PAD is `flags::attn_topk_k_pad()` (next-pow2 of the requested k);
/// this const is the kernel's `#ifndef` default and the value tests/benches use.
pub const K_PAD: usize = 64;

/// Slot capacity for a requested k (mirrors `flags::attn_topk_k_pad`).
#[inline]
pub fn k_pad_for(k: usize) -> usize {
    k.max(1).next_power_of_two().clamp(K_PAD, 1024)
}

/// Default tile geometry (matches the GEMM-attention defaults for the QK half).
pub const BM: usize = 64;
pub const BN: usize = 64;
pub const SOFTMAX_TPG: usize = 256;
pub const PV_BN: usize = 64;

/// Padded score-matrix width (delegates to the GEMM-attention n_pad — same S plane layout).
#[inline]
pub fn n_pad(t_total: usize) -> usize {
    crate::shaders::attention_gemm::n_pad(t_total)
}

/// Prepend the tile #defines (for tuning sweeps). The QK kernel uses the GEMM-attention
/// source directly; only the new topk kernels read these defines.
pub fn tuned_source(bm: usize, bn: usize, sm_tpg: usize, k_pad: usize) -> String {
    format!(
        "#define AG_BM {bm}\n#define AG_BN {bn}\n#define AG_SM_TPG {sm_tpg}\n#define AG_K_PAD {k_pad}\n{SHADER_TOPK}"
    )
}

#[cfg(target_os = "macos")]
pub fn pipelines(
    ctx: &crate::metal::device::MetalContext,
    variant: crate::shaders::variant::KernelVariant,
    side: bool,
    k_pad: usize,
) -> Result<[crate::metal::device::ComputePipeline; 3], Error> {
    use crate::shaders::variant::FcBool;
    let pat16 = FcBool {
        index: 32,
        value: true,
    };
    let side_fc = FcBool {
        index: 30,
        value: true,
    };
    // QK from the GEMM-attention source with FC32 (EMIT_PAT16): also writes the u16 key
    // plane the selection passes read (buffer 8). The GEMM-attention QK pipeline leaves
    // FC32 undefined — distinct cache label via the FC-value suffix.
    let qk_bools: &[FcBool] = if side { &[side_fc, pat16] } else { &[pat16] };
    let bools: &[FcBool] = if side { &[side_fc] } else { &[] };
    let label = if side { "side" } else { "default" };
    let pipe_qk = ctx.compile_subkernel_ex(SHADER_QK, ENTRY_QK, variant, label, qk_bools, &[])?;
    // topk_softmax + topk_pv with K_PAD baked to the requested capacity
    // (the k knob is live). Source-baked defines are safe in the pipeline cache —
    // the label includes source_hash for cache correctness.
    let src = tuned_source(BM, BN, SOFTMAX_TPG, k_pad);
    let pipe_sm = ctx.compile_subkernel_ex(&src, ENTRY_TOPK_SM, variant, label, bools, &[])?;
    let pipe_pv = ctx.compile_subkernel_ex(&src, ENTRY_TOPK_PV, variant, label, bools, &[])?;
    Ok([pipe_qk, pipe_sm, pipe_pv])
}

// ---- GPU oracle: run the 3-kernel top-k sequence over a Fixture. Returns
// bf16-rounded outputs to cross-check against `cpu::topk_causal`. ----
#[cfg(target_os = "macos")]
pub fn gpu(
    f: &crate::shaders::attention::Fixture,
    causal: bool,
    side: bool,
    k: usize,
) -> Result<Vec<f32>, Error> {
    use crate::shaders::attn::harness::AttnRig;
    use crate::shaders::gpu_common;
    use crate::shaders::variant::KernelVariant;
    use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

    // Standard rig: bf16 Q/out, f16 KV +8 pad key rows (GEMM-attention tile geometry),
    // f32 side-ring mirror for the FC30 path.
    let mut rig = AttnRig::from_fixture(f, 8, true)?;
    let (canvas, n_q_heads, hd) = (rig.canvas, rig.n_q_heads, rig.hd);
    let t_total = rig.t_total;
    let np = n_pad(t_total);
    let kp = k_pad_for(k);
    let k = k.max(1).min(kp);

    let [pipe_qk, pipe_sm, pipe_pv] = pipelines(&rig.ctx, KernelVariant::PRODUCTION, side, kp)?;
    let hc = 4.min(n_q_heads);
    let buf_s = rig.alloc(hc * canvas * np * 4, "alloc s")?;
    // P plane is f32 always (topk writes f32 probs regardless of side).
    let buf_p = rig.alloc(hc * canvas * kp * 4, "alloc p")?;
    let buf_idx = rig.alloc(hc * canvas * kp * 4, "alloc idx")?;
    let buf_lrow = rig.alloc(hc * canvas * 4, "alloc lrow")?;
    // u16 key plane (FC32 output of QK, read by the selection passes).
    let buf_pat = rig.alloc(hc * canvas * np * 2, "alloc pat")?;

    let mut dims = crate::shaders::attention_gemm::AttnGemmDims {
        m: canvas as u32,
        n: t_total as u32,
        k: hd as u32,
        a_row_stride: (n_q_heads * hd) as u32,
        b_row_stride: rig.kstride as u32,
        s_row_stride: np as u32,
        out_row_stride: (n_q_heads * hd) as u32,
        causal: u32::from(causal),
        kv_len: f.kv_len,
        hd: hd as u32,
        group: rig.group as u32,
        nkv: rig.nkv as u32,
        s_head_stride: (canvas * np) as u32,
        head_base: 0,
    };
    // Harness/oracle path is always FIXED k: dyn_divisor 0.
    let k_u32: [u32; 4] = [k as u32, 0, 0, 0];

    let tg128 = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    let tg_sm = MTLSize {
        width: SOFTMAX_TPG,
        height: 1,
        depth: 1,
    };
    let tg_pv = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    }; // one simdgroup

    rig.run_once(|enc| {
        let mut h0 = 0usize;
        while h0 < n_q_heads {
            let hb = (n_q_heads - h0).min(hc);
            dims.head_base = h0 as u32;
            let grid_qk = MTLSize {
                width: t_total.div_ceil(BN),
                height: canvas.div_ceil(BM),
                depth: hb,
            };
            let grid_sm = MTLSize {
                width: canvas,
                height: hb,
                depth: 1,
            };
            let grid_pv = MTLSize {
                width: hd.div_ceil(PV_BN),
                height: canvas,
                depth: hb,
            };
            // QK: same dispatch as the GEMM-attention (+ u16 key plane at 8, FC32).
            enc.setComputePipelineState(&pipe_qk.pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&rig.buf_q), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&rig.buf_kv), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&buf_s), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&buf_pat), 0, 8);
                if side {
                    enc.setBuffer_offset_atIndex(Some(rig.kvf()), 0, 9);
                }
            }
            gpu_common::set_bytes(enc, &dims, 3);
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid_qk, tg128);
            enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);

            // topk_softmax: S -> 0, P -> 1, Idx -> 2, lrow -> 3, dims -> 4, k -> 5,
            // pat16 -> 6.
            enc.setComputePipelineState(&pipe_sm.pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_s), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&buf_p), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&buf_idx), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&buf_lrow), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&buf_pat), 0, 6);
            }
            gpu_common::set_bytes(enc, &dims, 4);
            gpu_common::set_bytes(enc, &k_u32, 5);
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid_sm, tg_sm);
            enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);

            // topk_pv: buffers P -> 0, Idx -> 1, KV -> 2, out -> 3, lrow -> 4, dims -> 5
            enc.setComputePipelineState(&pipe_pv.pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_p), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&buf_idx), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&rig.buf_kv), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&rig.buf_out), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&buf_lrow), 0, 4);
                if side {
                    enc.setBuffer_offset_atIndex(Some(rig.kvf()), 0, 9);
                }
            }
            gpu_common::set_bytes(enc, &dims, 5);
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid_pv, tg_pv);
            enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
            h0 += hb;
        }
    })?;

    Ok(rig.read_out_f32(f.out_len()))
}

/// Shared state for the top-k benches: rig + pipelines + scratch, and the
/// per-head-chunk encoder both `bench_gpu` and `bench_stages` dispatch through
/// (f16 KV path, no side ring — matching the historical benches).
#[cfg(target_os = "macos")]
struct BenchRig {
    rig: crate::shaders::attn::harness::AttnRig,
    pipe_qk: crate::metal::device::ComputePipeline,
    pipe_sm: crate::metal::device::ComputePipeline,
    pipe_pv: crate::metal::device::ComputePipeline,
    buf_s: crate::shaders::attn::harness::Buf,
    buf_p: crate::shaders::attn::harness::Buf,
    buf_idx: crate::shaders::attn::harness::Buf,
    buf_lrow: crate::shaders::attn::harness::Buf,
    buf_pat: crate::shaders::attn::harness::Buf,
    dims: crate::shaders::attention_gemm::AttnGemmDims,
    /// (fixed_k, dyn_divisor, k_min, k_max) — harness is always fixed k.
    k_u32: [u32; 4],
    hc: usize,
}

#[cfg(target_os = "macos")]
impl BenchRig {
    fn new(
        f: &crate::shaders::attention::Fixture,
        hc_in: usize,
        k: usize,
        causal: bool,
    ) -> Result<Self, Error> {
        use crate::shaders::attn::harness::AttnRig;
        use crate::shaders::variant::KernelVariant;

        let mut rig = AttnRig::from_fixture(f, 8, false)?;
        let (canvas, n_q_heads) = (rig.canvas, rig.n_q_heads);
        let np = n_pad(rig.t_total);
        let kp = k_pad_for(k);
        let k = k.max(1).min(kp);

        let [pipe_qk, pipe_sm, pipe_pv] =
            pipelines(&rig.ctx, KernelVariant::PRODUCTION, false, kp)?;
        let hc = hc_in.clamp(1, n_q_heads);
        let buf_s = rig.alloc(hc * canvas * np * 4, "alloc s")?;
        let buf_p = rig.alloc(hc * canvas * kp * 4, "alloc p")?;
        let buf_idx = rig.alloc(hc * canvas * kp * 4, "alloc idx")?;
        let buf_lrow = rig.alloc(hc * canvas * 4, "alloc lrow")?;
        let buf_pat = rig.alloc(hc * canvas * np * 2, "alloc pat")?;

        let dims = crate::shaders::attention_gemm::AttnGemmDims {
            m: canvas as u32,
            n: rig.t_total as u32,
            k: rig.hd as u32,
            a_row_stride: (n_q_heads * rig.hd) as u32,
            b_row_stride: rig.kstride as u32,
            s_row_stride: np as u32,
            out_row_stride: (n_q_heads * rig.hd) as u32,
            causal: u32::from(causal),
            kv_len: f.kv_len,
            hd: rig.hd as u32,
            group: rig.group as u32,
            nkv: rig.nkv as u32,
            s_head_stride: (canvas * np) as u32,
            head_base: 0,
        };
        Ok(Self {
            rig,
            pipe_qk,
            pipe_sm,
            pipe_pv,
            buf_s,
            buf_p,
            buf_idx,
            buf_lrow,
            buf_pat,
            dims,
            k_u32: [k as u32, 0, 0, 0],
            hc,
        })
    }

    /// Encode one head-chunked layer pass. `only = None` dispatches all three
    /// stages (the production sequence); `Some(stage)` isolates one stage
    /// (0=qk, 1=sm, 2=pv) for `bench_stages`.
    fn encode(
        &self,
        enc: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        only: Option<usize>,
    ) {
        use crate::shaders::gpu_common;
        use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

        let (canvas, n_q_heads, hd) = (self.rig.canvas, self.rig.n_q_heads, self.rig.hd);
        let t_total = self.rig.t_total;
        let mut dims = self.dims;
        let tg128 = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        let tg_sm = MTLSize {
            width: SOFTMAX_TPG,
            height: 1,
            depth: 1,
        };
        let tg_pv = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };

        let mut h0 = 0usize;
        while h0 < n_q_heads {
            let hb = (n_q_heads - h0).min(self.hc);
            dims.head_base = h0 as u32;
            let grid_qk = MTLSize {
                width: t_total.div_ceil(BN),
                height: canvas.div_ceil(BM),
                depth: hb,
            };
            let grid_sm = MTLSize {
                width: canvas,
                height: hb,
                depth: 1,
            };
            let grid_pv = MTLSize {
                width: hd.div_ceil(PV_BN),
                height: canvas,
                depth: hb,
            };
            if only.is_none() || only == Some(0) {
                enc.setComputePipelineState(&self.pipe_qk.pipeline);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.rig.buf_q), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&self.rig.buf_kv), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&self.buf_s), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(&self.buf_pat), 0, 8);
                }
                gpu_common::set_bytes(enc, &dims, 3);
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid_qk, tg128);
                enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
            }
            if only.is_none() || only == Some(1) {
                enc.setComputePipelineState(&self.pipe_sm.pipeline);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.buf_s), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&self.buf_p), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&self.buf_idx), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(&self.buf_lrow), 0, 3);
                    enc.setBuffer_offset_atIndex(Some(&self.buf_pat), 0, 6);
                }
                gpu_common::set_bytes(enc, &dims, 4);
                gpu_common::set_bytes(enc, &self.k_u32, 5);
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid_sm, tg_sm);
                enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
            }
            if only.is_none() || only == Some(2) {
                enc.setComputePipelineState(&self.pipe_pv.pipeline);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.buf_p), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&self.buf_idx), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&self.rig.buf_kv), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(&self.rig.buf_out), 0, 3);
                    enc.setBuffer_offset_atIndex(Some(&self.buf_lrow), 0, 4);
                }
                gpu_common::set_bytes(enc, &dims, 5);
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid_pv, tg_pv);
                enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
            }
            h0 += hb;
        }
    }
}

/// Bench the 3-kernel top-k sequence over all heads (one full-attention layer,
/// one prefill sub-chunk) in a single command buffer; 1 warm-up + min over
/// timed rounds. Returns mean ms per layer. `causal` mirrors `gpu()` (benches
/// historically dispatched the prefill mask, causal=1).
#[cfg(target_os = "macos")]
pub fn bench_gpu(
    f: &crate::shaders::attention::Fixture,
    iters: usize,
    hc_in: usize,
    k: usize,
    causal: bool,
) -> Result<f64, Error> {
    let b = BenchRig::new(f, hc_in, k, causal)?;
    b.rig.time_rounds(iters, |enc| b.encode(enc, None))
}

/// Stage-isolated bench: returns (qk_ms, softmax_ms, pv_ms) for one layer.
/// Each stage run `iters` times in its own command buffer; min over warmed
/// rounds. `causal` mirrors `gpu()` (historically causal=1).
#[cfg(target_os = "macos")]
pub fn bench_stages(
    f: &crate::shaders::attention::Fixture,
    iters: usize,
    hc_in: usize,
    k: usize,
    causal: bool,
) -> Result<(f64, f64, f64), Error> {
    let b = BenchRig::new(f, hc_in, k, causal)?;
    let qk = b.rig.time_rounds(iters, |enc| b.encode(enc, Some(0)))?;
    let sm = b.rig.time_rounds(iters, |enc| b.encode(enc, Some(1)))?;
    let pv = b.rig.time_rounds(iters, |enc| b.encode(enc, Some(2)))?;
    Ok((qk, sm, pv))
}

/// Manifest registration; collected in common/manifest.rs::MANIFEST.
pub const SPEC: crate::shaders::manifest::KernelSpec = crate::shaders::manifest::KernelSpec {
    name: "attention_topk",
    entry: "attn_topk_softmax",
    quant_formats: &[crate::shaders::variant::QuantFormat::Q4Affine],
    fc: &[],
    variants: crate::shaders::manifest::KernelVariants::Elementwise,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shaders::test_util::{ElemFormat, assert_oracle};

    /// Parity vs the CPU oracle, causal prefill, f16 KV path. Full-layer shape,
    /// GQA group 8 (the production full-layer shape). k = K_PAD = 64.
    #[cfg(target_os = "macos")]
    #[test]
    fn topk_full_grp8_causal_vs_cpu() {
        if crate::shaders::test_util::skip_gpu_on_ci() {
            return;
        }
        let f = crate::shaders::attention::full_grp8_hd512_fixture(ElemFormat::F32);
        let got = gpu(&f, true, false, K_PAD).expect("gpu topk causal");
        let oracle = cpu::topk_causal(&f, true, K_PAD);
        // Selection is exact top-k by f32 score (4-level radix); only
        // bit-identical scores are interchangeable. Tolerance covers f16-KV /
        // bf16 rounding.
        assert_oracle(&got, &oracle, 2e-2, 0.9999);
    }

    /// Parity vs the CPU oracle, DENOISE mask (causal=0 — the decode arm's
    /// dispatch mode, `DGQ_ATTN_TOPK_DECODE`): every canvas row sees all
    /// kv_len + canvas keys. Same production full-layer shape as the causal
    /// test.
    #[cfg(target_os = "macos")]
    #[test]
    fn topk_full_grp8_denoise_vs_cpu() {
        if crate::shaders::test_util::skip_gpu_on_ci() {
            return;
        }
        let f = crate::shaders::attention::full_grp8_hd512_fixture(ElemFormat::F32);
        let got = gpu(&f, false, false, K_PAD).expect("gpu topk denoise");
        let oracle = cpu::topk_denoise(&f, true, K_PAD);
        assert_oracle(&got, &oracle, 2e-2, 0.9999);
    }

    /// Parity vs CPU oracle, f32 side-KV path (FC30).
    #[cfg(target_os = "macos")]
    #[test]
    fn topk_full_grp8_causal_side_vs_cpu() {
        if crate::shaders::test_util::skip_gpu_on_ci() {
            return;
        }
        let f = crate::shaders::attention::full_grp8_hd512_fixture(ElemFormat::F32);
        let got = gpu(&f, true, true, K_PAD).expect("gpu topk causal side");
        let oracle = cpu::topk_causal(&f, false, K_PAD);
        assert_oracle(&got, &oracle, 2e-2, 0.9999);
    }

    /// k=1 sanity: top-1 picks argmax per (row, head). Relaxed tolerance:
    /// the threshold-based selection may include ties at the threshold, so for
    /// k=1 with tied scores the GPU may pick a different (tied) key than the
    /// CPU oracle's lower-index tie-break. cos >= 0.999 is sufficient to
    /// confirm the selection is correct (output is one of the tied V rows).
    #[cfg(target_os = "macos")]
    #[test]
    fn topk_k1_matches_argmax_value() {
        if crate::shaders::test_util::skip_gpu_on_ci() {
            return;
        }
        let f = crate::shaders::attention::full_grp8_hd512_fixture(ElemFormat::F32);
        let got = gpu(&f, true, false, 1).expect("gpu topk k=1");
        let oracle = cpu::topk_causal(&f, true, 1);
        // Relaxed: ties at k=1 may pick a different (tied) key.
        assert_oracle(&got, &oracle, 2e-2, 0.999);
    }

    /// k=128 > default K_PAD=64: proves the k knob is live end-to-end (task
    /// #95) — the kernel compiles with AG_K_PAD=128 and the selection keeps
    /// 128 keys, matching the CPU oracle at the same k.
    #[cfg(target_os = "macos")]
    #[test]
    fn topk_k128_matches_cpu() {
        if crate::shaders::test_util::skip_gpu_on_ci() {
            return;
        }
        let f = crate::shaders::attention::full_grp8_hd512_fixture(ElemFormat::F32);
        let got = gpu(&f, true, false, 128).expect("gpu topk k=128");
        let oracle = cpu::topk_causal(&f, true, 128);
        assert_oracle(&got, &oracle, 2e-2, 0.9999);
    }

    /// Premise bench: top-k attention vs dense at model shape. One
    /// command buffer each, min-of-rounds. Ignored (timing).
    /// Run: `cargo test --release topk_bench -- --ignored --nocapture`
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore]
    fn topk_bench() {
        let iters = 10usize;
        println!("  kv    e17_dense   topk       ratio   | qk     sm     pv");
        for kv_len in [8192u32, 30000, 60000, 100000] {
            let f = crate::shaders::attention_gemm::model_full_fixture(kv_len);
            let e17 = crate::shaders::attention_gemm::bench_gpu(&f, iters, 4, true).unwrap();
            let topk = bench_gpu(&f, iters, 4, K_PAD, true).unwrap();
            let (qk_t, sm_t, pv_t) = bench_stages(&f, iters, 4, K_PAD, true).unwrap();
            println!(
                "{kv_len:>6}  {e17:9.3}  {topk:9.3}  {ratio:.2}x   | {qk_t:6.2} {sm_t:6.2} {pv_t:6.2}",
                ratio = e17 / topk
            );
        }
    }

    /// Premise bench for the DECODE (denoise) arm: production
    /// `attention_mma_full` (causal=0 monolithic — what denoise full layers
    /// run today) vs dense decomp vs top-k at kv-adaptive k, at the
    /// denoise full-layer shape (canvas=256, 16Q/2KV, hd=512). The dense/topk
    /// harnesses dispatch causal=1; at these kv lengths the causal mask
    /// excludes <0.5% of keys, below bench noise.
    /// Run: `cargo test --release topk_decode_bench -- --ignored --nocapture`
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore]
    fn topk_decode_bench() {
        let iters = 5usize;
        println!("DECODE shape (canvas=256, 16Q/2KV, hd=512), k_dyn = clamp(t/128, 64, 512)");
        println!("  kv    mma_full   e17_dense  topk_dyn   full/topk | qk     sm     pv");
        for kv_len in [8192u32, 30000, 60000, 100000] {
            let f = crate::shaders::attention_gemm::model_full_fixture(kv_len);
            let t_total = kv_len as usize + f.canvas;
            let k = (t_total / 128).clamp(64, 512);
            let full = crate::shaders::attention::bench_path(&f, iters, 3).unwrap();
            let e17 = crate::shaders::attention_gemm::bench_gpu(&f, iters, 16, true).unwrap();
            let topk = bench_gpu(&f, iters, 16, k, true).unwrap();
            let (qk_t, sm_t, pv_t) = bench_stages(&f, iters, 16, k, true).unwrap();
            println!(
                "{kv_len:>6}  {full:9.3}  {e17:9.3}  {topk:9.3}  {r:8.2}x | {qk_t:6.2} {sm_t:6.2} {pv_t:6.2}",
                r = full / topk
            );
        }
    }

    /// Per-kernel breakdown at the PRODUCTION prefill super-chunk shape
    /// (M=1024, n_q_heads=16, nkv=2, hd=512) — matches `bench-prefill-super`.
    /// Used to verify QK is at the half-MMA compute wall (not a kernel bug).
    /// Run: `cargo test --release topk_bench_prod -- --ignored --nocapture`
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore]
    fn topk_bench_prod() {
        let iters = 5usize;
        println!("PROD shape (M=1024, n_q_heads=16, nkv=2, hd=512)");
        println!("  kv    e17_dense   topk       ratio   | qk     sm     pv");
        for kv_len in [15000u32, 30000, 60000] {
            let f = crate::shaders::attention_gemm::model_full_fixture_prod(kv_len);
            let e17 = crate::shaders::attention_gemm::bench_gpu(&f, iters, 4, true).unwrap();
            let topk = bench_gpu(&f, iters, 4, K_PAD, true).unwrap();
            let (qk_t, sm_t, pv_t) = bench_stages(&f, iters, 4, K_PAD, true).unwrap();
            // Per-head QK FLOPs = 2 * M * t_total * hd. n_q_heads independent QK GEMMs.
            let t_total = kv_len as usize + 1024;
            let qk_flops = 2.0 * 1024.0 * t_total as f64 * 512.0 * 16.0;
            let qk_tf_s = qk_flops / (qk_t * 1e-3) / 1e12;
            println!(
                "{kv_len:>6}  {e17:9.3}  {topk:9.3}  {ratio:.2}x   | {qk_t:6.2} {sm_t:6.2} {pv_t:6.2}  (qk={qk_tf_s:.2} TF/s)",
                ratio = e17 / topk
            );
        }
    }
}

crate::register_kernel_specs!(SPEC);
