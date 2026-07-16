//! E20: top-k sparse attention for full-layer PREFILL.
//!
//! Sibling of E17 `attention_gemm`. Reuses E17's `attn_gemm_qk` kernel verbatim
//! (the S plane is identical), then replaces softmax+PV with a top-k softmax +
//! sparse-PV that gathers V at the per-row selected key indices.
//! Quality-gated (non-bit-identical): only the top-`k` highest-scoring keys per
//! (row, head) are kept and renormalized. Default OFF (`DGQ_ATTN_TOPK`).
//!
//! The CPU oracle (`cpu.rs`) is the authoritative numeric reference. The GPU
//! kernels (`attn_topk_softmax`, `attn_topk_pv`) live in `attention_topk.metal`.

#[cfg(target_os = "macos")]
use crate::Error;

pub mod cpu;

/// The QK pipeline is compiled from E17's source (the kernel is identical).
pub const ENTRY_QK: &str = crate::shaders::attention_gemm::ENTRY_QK;
/// E17's shader source (reused verbatim for the QK pipeline).
pub const SHADER_QK: &str = crate::shaders::attention_gemm::SHADER;

pub const ENTRY_TOPK_SM: &str = "attn_topk_softmax";
pub const ENTRY_TOPK_PV: &str = "attn_topk_pv";
pub const SHADER_TOPK: &str = include_str!("attention_topk.metal");

/// Default compile-time slot capacity for the compressed P/Idx planes. The
/// production K_PAD is `flags::attn_topk_k_pad()` (next-pow2 of the requested
/// k, task #95); this const is the kernel's `#ifndef` default and the value
/// tests/benches use.
pub const K_PAD: usize = 64;

/// Slot capacity for a requested k (mirrors `flags::attn_topk_k_pad`).
#[inline]
pub fn k_pad_for(k: usize) -> usize {
    k.max(1).next_power_of_two().clamp(K_PAD, 1024)
}

/// Default tile geometry (matches E17's defaults for the QK half).
pub const BM: usize = 64;
pub const BN: usize = 64;
pub const SOFTMAX_TPG: usize = 256;
pub const PV_BN: usize = 64;

/// Padded score-matrix width (delegates to E17's n_pad — same S plane layout).
#[inline]
pub fn n_pad(t_total: usize) -> usize {
    crate::shaders::attention_gemm::n_pad(t_total)
}

/// Prepend the tile #defines (for tuning sweeps). The QK kernel uses E17's
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
    // QK from E17's source with FC32 (EMIT_PAT16): also writes the u16 key
    // plane the selection passes read (buffer 8). E17's own QK pipeline leaves
    // FC32 undefined — distinct cache label via the FC-value suffix.
    let qk_bools: &[FcBool] = if side { &[side_fc, pat16] } else { &[pat16] };
    let bools: &[FcBool] = if side { &[side_fc] } else { &[] };
    let label = if side { "side" } else { "default" };
    let pipe_qk = ctx.compile_subkernel_ex(SHADER_QK, ENTRY_QK, variant, label, qk_bools, &[])?;
    // topk_softmax + topk_pv with K_PAD baked to the requested capacity (task
    // #95: the k knob is live). Source-baked defines are safe in the pipeline
    // cache — the label includes source_hash (the #99 generic fix).
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
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use crate::shaders::{bf16, gpu_common};
    use crate::shaders::variant::KernelVariant;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLSize,
    };

    let canvas = f.canvas;
    let n_q_heads = f.n_q_heads;
    let nkv = f.n_kv();
    let hd = f.head_dim();
    let group = n_q_heads / nkv;
    let kv_len = f.kv_len as usize;
    let t_total = kv_len + canvas;
    let np = n_pad(t_total);
    let kstride = nkv * hd * 2;
    let kp = k_pad_for(k);
    let k = k.max(1).min(kp);

    let ctx = MetalContext::new()?;
    let [pipe_qk, pipe_sm, pipe_pv] = pipelines(&ctx, KernelVariant::PRODUCTION, side, kp)?;
    let mut pool = BufferPool::new();
    let buf_q = pool
        .allocate(&ctx.device, canvas * n_q_heads * hd * 2)
        .ok_or(Error::Gpu("alloc q"))?;
    BufferPool::write_bf16(&buf_q, &bf16::f32_slice_to_bf16_bits(&f.q));
    let buf_out = pool
        .allocate(&ctx.device, canvas * n_q_heads * hd * 2)
        .ok_or(Error::Gpu("alloc out"))?;
    let buf_kv = pool
        .allocate(&ctx.device, (f.kvcache.len() + 8 * kstride) * 2)
        .ok_or(Error::Gpu("alloc kv"))?;
    {
        let mut bits = crate::shaders::f16::f32_slice_to_f16(&f.kvcache);
        bits.resize(bits.len() + 8 * kstride, 0);
        BufferPool::write_bf16(&buf_kv, &bits);
    }
    let buf_kvf = pool
        .allocate(&ctx.device, (f.kvcache.len() + 8 * kstride) * 4)
        .ok_or(Error::Gpu("alloc kvf"))?;
    {
        let mut fs = f.kvcache.clone();
        fs.resize(fs.len() + 8 * kstride, 0.0);
        BufferPool::write_f32(&buf_kvf, &fs);
    }

    let hc = 4.min(n_q_heads);
    let buf_s = pool
        .allocate(&ctx.device, hc * canvas * np * 4)
        .ok_or(Error::Gpu("alloc s"))?;
    // P plane is f32 always (topk writes f32 probs regardless of side).
    let buf_p = pool
        .allocate(&ctx.device, hc * canvas * kp * 4)
        .ok_or(Error::Gpu("alloc p"))?;
    let buf_idx = pool
        .allocate(&ctx.device, hc * canvas * kp * 4)
        .ok_or(Error::Gpu("alloc idx"))?;
    let buf_lrow = pool
        .allocate(&ctx.device, hc * canvas * 4)
        .ok_or(Error::Gpu("alloc lrow"))?;
    // u16 key plane (FC32 output of QK, read by the selection passes).
    let buf_pat = pool
        .allocate(&ctx.device, hc * canvas * np * 2)
        .ok_or(Error::Gpu("alloc pat"))?;

    let mut dims = crate::shaders::attention_gemm::AttnGemmDims {
        m: canvas as u32,
        n: t_total as u32,
        k: hd as u32,
        a_row_stride: (n_q_heads * hd) as u32,
        b_row_stride: kstride as u32,
        s_row_stride: np as u32,
        out_row_stride: (n_q_heads * hd) as u32,
        causal: u32::from(causal),
        kv_len: f.kv_len,
        hd: hd as u32,
        group: group as u32,
        nkv: nkv as u32,
        s_head_stride: (canvas * np) as u32,
        head_base: 0,
    };
    let k_u32 = k as u32;

    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
    let tg128 = MTLSize { width: 128, height: 1, depth: 1 };
    let tg_sm = MTLSize { width: SOFTMAX_TPG, height: 1, depth: 1 };
    let tg_pv = MTLSize { width: 32, height: 1, depth: 1 };  // one simdgroup

    let mut h0 = 0usize;
    while h0 < n_q_heads {
        let hb = (n_q_heads - h0).min(hc);
        dims.head_base = h0 as u32;
        let grid_qk = MTLSize {
            width: t_total.div_ceil(BN),
            height: canvas.div_ceil(BM),
            depth: hb,
        };
        let grid_sm = MTLSize { width: canvas, height: hb, depth: 1 };
        let grid_pv = MTLSize {
            width: hd.div_ceil(PV_BN),
            height: canvas,
            depth: hb,
        };
        // QK: same dispatch as E17 (+ u16 key plane at 8, FC32).
        enc.setComputePipelineState(&pipe_qk.pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&buf_q), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&buf_kv), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&buf_s), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&buf_pat), 0, 8);
            if side { enc.setBuffer_offset_atIndex(Some(&buf_kvf), 0, 9); }
        }
        gpu_common::set_bytes(&enc, &dims, 3);
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
        gpu_common::set_bytes(&enc, &dims, 4);
        gpu_common::set_bytes(&enc, &k_u32, 5);
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid_sm, tg_sm);
        enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);

        // topk_pv: buffers P -> 0, Idx -> 1, KV -> 2, out -> 3, lrow -> 4, dims -> 5
        enc.setComputePipelineState(&pipe_pv.pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&buf_p), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&buf_idx), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&buf_kv), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&buf_out), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&buf_lrow), 0, 4);
            if side { enc.setBuffer_offset_atIndex(Some(&buf_kvf), 0, 9); }
        }
        gpu_common::set_bytes(&enc, &dims, 5);
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid_pv, tg_pv);
        enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
        h0 += hb;
    }
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut out = vec![0.0f32; f.out_len()];
    let ptr = buf_out.contents().as_ptr() as *const u16;
    for (i, o) in out.iter_mut().enumerate() {
        *o = bf16::bf16_bits_to_f32(unsafe { *ptr.add(i) });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shaders::test_util::{ElemFormat, assert_oracle};

    /// Parity vs the CPU oracle, causal prefill, f16 KV path. Full-layer shape,
    /// GQA group 8 (the production full-layer shape). k = K_PAD = 64.
    #[cfg(target_os = "macos")]
    #[test]
    fn topk_full_grp8_causal_vs_cpu() {
        let f = crate::shaders::attention::full_grp8_hd512_fixture(ElemFormat::F32);
        let got = gpu(&f, true, false, K_PAD).expect("gpu topk causal");
        let oracle = cpu::topk_causal(&f, true, K_PAD);
        // Selection is exact top-k by f32 score (4-level radix); only
        // bit-identical scores are interchangeable. Tolerance covers f16-KV /
        // bf16 rounding.
        assert_oracle(&got, &oracle, 2e-2, 0.9999);
    }

    /// Parity vs CPU oracle, f32 side-KV path (FC30).
    #[cfg(target_os = "macos")]
    #[test]
    fn topk_full_grp8_causal_side_vs_cpu() {
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
        let f = crate::shaders::attention::full_grp8_hd512_fixture(ElemFormat::F32);
        let got = gpu(&f, true, false, 128).expect("gpu topk k=128");
        let oracle = cpu::topk_causal(&f, true, 128);
        assert_oracle(&got, &oracle, 2e-2, 0.9999);
    }

    /// Premise bench: top-k attention vs E17 (dense) at model shape. One
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
            let e17 = crate::shaders::attention_gemm::bench_gpu(&f, iters, 4).unwrap();
            let topk = bench_gpu(&f, iters, 4, K_PAD).unwrap();
            let (qk_t, sm_t, pv_t) = bench_stages(&f, iters, 4, K_PAD).unwrap();
            println!(
                "{kv_len:>6}  {e17:9.3}  {topk:9.3}  {ratio:.2}x   | {qk_t:6.2} {sm_t:6.2} {pv_t:6.2}",
                ratio = e17 / topk
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
            let e17 = crate::shaders::attention_gemm::bench_gpu(&f, iters, 4).unwrap();
            let topk = bench_gpu(&f, iters, 4, K_PAD).unwrap();
            let (qk_t, sm_t, pv_t) = bench_stages(&f, iters, 4, K_PAD).unwrap();
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

/// Bench the 3-kernel top-k sequence over all heads (one full-attention layer,
/// one prefill sub-chunk) in a single command buffer; 1 warm-up + min over
/// timed rounds. Returns mean ms per layer.
#[cfg(target_os = "macos")]
pub fn bench_gpu(
    f: &crate::shaders::attention::Fixture,
    iters: usize,
    hc_in: usize,
    k: usize,
) -> Result<f64, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use crate::shaders::{bf16, gpu_common};
    use crate::shaders::variant::KernelVariant;
    use objc2_metal::{
        MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
    };
    use std::time::Instant;

    let canvas = f.canvas;
    let n_q_heads = f.n_q_heads;
    let nkv = f.n_kv();
    let hd = f.head_dim();
    let group = n_q_heads / nkv;
    let kv_len = f.kv_len as usize;
    let t_total = kv_len + canvas;
    let np = n_pad(t_total);
    let kstride = nkv * hd * 2;
    let kp = k_pad_for(k);
    let k = k.max(1).min(kp);

    let ctx = MetalContext::new()?;
    let [pipe_qk, pipe_sm, pipe_pv] = pipelines(&ctx, KernelVariant::PRODUCTION, false, kp)?;
    let mut pool = BufferPool::new();
    let buf_q = pool
        .allocate(&ctx.device, canvas * n_q_heads * hd * 2)
        .ok_or(Error::Gpu("alloc q"))?;
    BufferPool::write_bf16(&buf_q, &bf16::f32_slice_to_bf16_bits(&f.q));
    let buf_out = pool
        .allocate(&ctx.device, canvas * n_q_heads * hd * 2)
        .ok_or(Error::Gpu("alloc out"))?;
    let buf_kv = pool
        .allocate(&ctx.device, (f.kvcache.len() + 8 * kstride) * 2)
        .ok_or(Error::Gpu("alloc kv"))?;
    {
        let mut bits = crate::shaders::f16::f32_slice_to_f16(&f.kvcache);
        bits.resize(bits.len() + 8 * kstride, 0);
        BufferPool::write_bf16(&buf_kv, &bits);
    }
    let hc = hc_in.clamp(1, n_q_heads);
    let buf_s = pool
        .allocate(&ctx.device, hc * canvas * np * 4)
        .ok_or(Error::Gpu("alloc s"))?;
    let buf_p = pool
        .allocate(&ctx.device, hc * canvas * kp * 4)
        .ok_or(Error::Gpu("alloc p"))?;
    let buf_idx = pool
        .allocate(&ctx.device, hc * canvas * kp * 4)
        .ok_or(Error::Gpu("alloc idx"))?;
    let buf_lrow = pool
        .allocate(&ctx.device, hc * canvas * 4)
        .ok_or(Error::Gpu("alloc lrow"))?;
    let buf_pat = pool
        .allocate(&ctx.device, hc * canvas * np * 2)
        .ok_or(Error::Gpu("alloc pat"))?;

    let mut dims = crate::shaders::attention_gemm::AttnGemmDims {
        m: canvas as u32,
        n: t_total as u32,
        k: hd as u32,
        a_row_stride: (n_q_heads * hd) as u32,
        b_row_stride: kstride as u32,
        s_row_stride: np as u32,
        out_row_stride: (n_q_heads * hd) as u32,
        causal: 1,
        kv_len: f.kv_len,
        hd: hd as u32,
        group: group as u32,
        nkv: nkv as u32,
        s_head_stride: (canvas * np) as u32,
        head_base: 0,
    };
    let k_u32 = k as u32;
    let tg128 = MTLSize { width: 128, height: 1, depth: 1 };
    let tg_sm = MTLSize { width: SOFTMAX_TPG, height: 1, depth: 1 };
    let tg_pv = MTLSize { width: 32, height: 1, depth: 1 };

    let mut best = f64::INFINITY;
    for round in 0..6 {
        let t = Instant::now();
        let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
        let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
        for _ in 0..iters {
            let mut h0 = 0usize;
            while h0 < n_q_heads {
                let hb = (n_q_heads - h0).min(hc);
                dims.head_base = h0 as u32;
                let grid_qk = MTLSize {
                    width: t_total.div_ceil(BN),
                    height: canvas.div_ceil(BM),
                    depth: hb,
                };
                let grid_sm = MTLSize { width: canvas, height: hb, depth: 1 };
                let grid_pv = MTLSize {
                    width: hd.div_ceil(PV_BN),
                    height: canvas,
                    depth: hb,
                };
                enc.setComputePipelineState(&pipe_qk.pipeline);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&buf_q), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&buf_kv), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&buf_s), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(&buf_pat), 0, 8);
                }
                gpu_common::set_bytes(&enc, &dims, 3);
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid_qk, tg128);
                enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);

                enc.setComputePipelineState(&pipe_sm.pipeline);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&buf_s), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&buf_p), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&buf_idx), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(&buf_lrow), 0, 3);
                    enc.setBuffer_offset_atIndex(Some(&buf_pat), 0, 6);
                }
                gpu_common::set_bytes(&enc, &dims, 4);
                gpu_common::set_bytes(&enc, &k_u32, 5);
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid_sm, tg_sm);
                enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);

                enc.setComputePipelineState(&pipe_pv.pipeline);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&buf_p), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&buf_idx), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&buf_kv), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(&buf_out), 0, 3);
                    enc.setBuffer_offset_atIndex(Some(&buf_lrow), 0, 4);
                }
                gpu_common::set_bytes(&enc, &dims, 5);
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid_pv, tg_pv);
                enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
                h0 += hb;
            }
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if round > 0 {
            best = best.min(t.elapsed().as_secs_f64() * 1e3 / iters as f64);
        }
    }
    Ok(best)
}

/// Stage-isolated bench: returns (qk_ms, softmax_ms, pv_ms) for one layer.
/// Each stage run `iters` times in its own command buffer; min over warmed rounds.
#[cfg(target_os = "macos")]
pub fn bench_stages(
    f: &crate::shaders::attention::Fixture,
    iters: usize,
    hc_in: usize,
    k: usize,
) -> Result<(f64, f64, f64), Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use crate::shaders::{bf16, gpu_common};
    use crate::shaders::variant::KernelVariant;
    use objc2_metal::{
        MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
    };
    use std::time::Instant;

    let canvas = f.canvas;
    let n_q_heads = f.n_q_heads;
    let nkv = f.n_kv();
    let hd = f.head_dim();
    let group = n_q_heads / nkv;
    let kv_len = f.kv_len as usize;
    let t_total = kv_len + canvas;
    let np = n_pad(t_total);
    let kstride = nkv * hd * 2;
    let kp = k_pad_for(k);
    let k = k.max(1).min(kp);

    let ctx = MetalContext::new()?;
    let [pipe_qk, pipe_sm, pipe_pv] = pipelines(&ctx, KernelVariant::PRODUCTION, false, kp)?;
    let mut pool = BufferPool::new();
    let buf_q = pool
        .allocate(&ctx.device, canvas * n_q_heads * hd * 2)
        .ok_or(Error::Gpu("alloc q"))?;
    BufferPool::write_bf16(&buf_q, &bf16::f32_slice_to_bf16_bits(&f.q));
    let buf_out = pool
        .allocate(&ctx.device, canvas * n_q_heads * hd * 2)
        .ok_or(Error::Gpu("alloc out"))?;
    let buf_kv = pool
        .allocate(&ctx.device, (f.kvcache.len() + 8 * kstride) * 2)
        .ok_or(Error::Gpu("alloc kv"))?;
    {
        let mut bits = crate::shaders::f16::f32_slice_to_f16(&f.kvcache);
        bits.resize(bits.len() + 8 * kstride, 0);
        BufferPool::write_bf16(&buf_kv, &bits);
    }
    let hc = hc_in.clamp(1, n_q_heads);
    let buf_s = pool
        .allocate(&ctx.device, hc * canvas * np * 4)
        .ok_or(Error::Gpu("alloc s"))?;
    let buf_p = pool
        .allocate(&ctx.device, hc * canvas * kp * 4)
        .ok_or(Error::Gpu("alloc p"))?;
    let buf_idx = pool
        .allocate(&ctx.device, hc * canvas * kp * 4)
        .ok_or(Error::Gpu("alloc idx"))?;
    let buf_lrow = pool
        .allocate(&ctx.device, hc * canvas * 4)
        .ok_or(Error::Gpu("alloc lrow"))?;
    let buf_pat = pool
        .allocate(&ctx.device, hc * canvas * np * 2)
        .ok_or(Error::Gpu("alloc pat"))?;

    let mut dims = crate::shaders::attention_gemm::AttnGemmDims {
        m: canvas as u32,
        n: t_total as u32,
        k: hd as u32,
        a_row_stride: (n_q_heads * hd) as u32,
        b_row_stride: kstride as u32,
        s_row_stride: np as u32,
        out_row_stride: (n_q_heads * hd) as u32,
        causal: 1,
        kv_len: f.kv_len,
        hd: hd as u32,
        group: group as u32,
        nkv: nkv as u32,
        s_head_stride: (canvas * np) as u32,
        head_base: 0,
    };
    let k_u32 = k as u32;
    let tg128 = MTLSize { width: 128, height: 1, depth: 1 };
    let tg_sm = MTLSize { width: SOFTMAX_TPG, height: 1, depth: 1 };
    let tg_pv = MTLSize { width: 32, height: 1, depth: 1 };

    // Helper: time `stage` (0=qk, 1=sm, 2=pv) for `iters` per command buffer.
    let mut time_stage = |stage: usize| -> Result<f64, Error> {
        let mut best = f64::INFINITY;
        for round in 0..6 {
            let t = Instant::now();
            let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
            let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
            for _ in 0..iters {
                let mut h0 = 0usize;
                while h0 < n_q_heads {
                    let hb = (n_q_heads - h0).min(hc);
                    dims.head_base = h0 as u32;
                    let grid_qk = MTLSize {
                        width: t_total.div_ceil(BN),
                        height: canvas.div_ceil(BM),
                        depth: hb,
                    };
                    let grid_sm = MTLSize { width: canvas, height: hb, depth: 1 };
                    let grid_pv = MTLSize {
                        width: hd.div_ceil(PV_BN),
                        height: canvas,
                        depth: hb,
                    };
                    if stage == 0 {
                        enc.setComputePipelineState(&pipe_qk.pipeline);
                        unsafe {
                            enc.setBuffer_offset_atIndex(Some(&buf_q), 0, 0);
                            enc.setBuffer_offset_atIndex(Some(&buf_kv), 0, 1);
                            enc.setBuffer_offset_atIndex(Some(&buf_s), 0, 2);
                            enc.setBuffer_offset_atIndex(Some(&buf_pat), 0, 8);
                        }
                        gpu_common::set_bytes(&enc, &dims, 3);
                        enc.dispatchThreadgroups_threadsPerThreadgroup(grid_qk, tg128);
                        enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
                    } else if stage == 1 {
                        enc.setComputePipelineState(&pipe_sm.pipeline);
                        unsafe {
                            enc.setBuffer_offset_atIndex(Some(&buf_s), 0, 0);
                            enc.setBuffer_offset_atIndex(Some(&buf_p), 0, 1);
                            enc.setBuffer_offset_atIndex(Some(&buf_idx), 0, 2);
                            enc.setBuffer_offset_atIndex(Some(&buf_lrow), 0, 3);
                            enc.setBuffer_offset_atIndex(Some(&buf_pat), 0, 6);
                        }
                        gpu_common::set_bytes(&enc, &dims, 4);
                        gpu_common::set_bytes(&enc, &k_u32, 5);
                        enc.dispatchThreadgroups_threadsPerThreadgroup(grid_sm, tg_sm);
                        enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
                    } else {
                        enc.setComputePipelineState(&pipe_pv.pipeline);
                        unsafe {
                            enc.setBuffer_offset_atIndex(Some(&buf_p), 0, 0);
                            enc.setBuffer_offset_atIndex(Some(&buf_idx), 0, 1);
                            enc.setBuffer_offset_atIndex(Some(&buf_kv), 0, 2);
                            enc.setBuffer_offset_atIndex(Some(&buf_out), 0, 3);
                            enc.setBuffer_offset_atIndex(Some(&buf_lrow), 0, 4);
                        }
                        gpu_common::set_bytes(&enc, &dims, 5);
                        enc.dispatchThreadgroups_threadsPerThreadgroup(grid_pv, tg_pv);
                        enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
                    }
                    h0 += hb;
                }
            }
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            if round > 0 {
                best = best.min(t.elapsed().as_secs_f64() * 1e3 / iters as f64);
            }
        }
        Ok(best)
    };

    Ok((time_stage(0)?, time_stage(1)?, time_stage(2)?))
}

/// Manifest registration; collected in common/manifest.rs::MANIFEST.
pub const SPEC: crate::shaders::manifest::KernelSpec = crate::shaders::manifest::KernelSpec {
    name: "attention_topk",
    entry: "attn_topk_softmax",
    quant_formats: &[crate::shaders::variant::QuantFormat::Q4Affine],
    fc: &[],
    variants: crate::shaders::manifest::KernelVariants::Elementwise,
};
