//! E17: GEMM-attention for full-layer PREFILL.
//!
//! `attention_mma_full` is pinned at ~1.25 TF/s by the hd=512 / 8x8-fragment
//! occupancy shape. During prefill the score matrix `S = Q.K^T` fits in memory
//! per 256-row sub-chunk, so full attention decomposes into two big GEMMs at
//! the tunable-GEMM rate (2.3-4 TF/s) plus a rowwise softmax:
//!   1. `attn_gemm_qk`      NT-GEMM  S[i,t] = <Q_i, K_t>
//!   2. `attn_gemm_softmax` rowwise  P = exp(S - rowmax) (masked), denom L
//!   3. `attn_gemm_pv`      NN-GEMM  O[i,d] = sum_t P[i,t] V[t,d] / L_i
//! No 1/sqrt(d) (folded into QK-norm upstream). P is left unnormalized; PV
//! divides by L at store — mirroring `attention_mma_full`'s final divide so the
//! two share numerics (f16 K/V, f32 accumulate). Not bit-identical; prefill
//! only (denoise keeps `attention_mma_full`).

#[cfg(target_os = "macos")]
use crate::Error;

pub const ENTRY_QK: &str = "attn_gemm_qk";
pub const ENTRY_SOFTMAX: &str = "attn_gemm_softmax";
pub const ENTRY_PV: &str = "attn_gemm_pv";

pub const SHADER: &str = include_str!("attention_gemm.metal");

/// Fragment-tile geometry (must match the shader constants).
pub const BM: usize = 64;
pub const BN: usize = 64;
pub const SOFTMAX_TPG: usize = 256;

/// Host mirror of the shader `AttnGemmDims`. All buffer base offsets (per head,
/// per KV head) are applied host-side via buffer offset args; this struct
/// carries shapes, row strides, and the softmax mask parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AttnGemmDims {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a_row_stride: u32,
    pub b_row_stride: u32,
    pub s_row_stride: u32,
    pub out_row_stride: u32,
    pub causal: u32,
    pub kv_len: u32,
    pub hd: u32,
    pub group: u32,
    pub nkv: u32,
    pub s_head_stride: u32,
}

/// Padded score-matrix width: keys rounded up to the 64-wide N-tile so the QK
/// grid, the softmax pad region, and the PV BK=32 contraction all align.
#[inline]
pub fn n_pad(t_total: usize) -> usize {
    t_total.next_multiple_of(BN)
}

#[cfg(target_os = "macos")]
pub fn pipelines(
    ctx: &crate::metal::device::MetalContext,
    variant: crate::shaders::variant::KernelVariant,
) -> Result<[crate::metal::device::ComputePipeline; 3], Error> {
    Ok([
        ctx.compile_subkernel(SHADER, ENTRY_QK, variant)?,
        ctx.compile_subkernel(SHADER, ENTRY_SOFTMAX, variant)?,
        ctx.compile_subkernel(SHADER, ENTRY_PV, variant)?,
    ])
}

// --------------------------------------------------------------------------
// GPU oracle: run the 3-kernel decomposition over an `attention::Fixture` and
// return bf16-rounded outputs, to cross-check against the CPU attention oracle
// (same target as `attention::gpu_mma_full`). Full layers (hd=512) only.
// --------------------------------------------------------------------------
#[cfg(target_os = "macos")]
pub fn gpu(f: &crate::shaders::attention::Fixture, causal: bool) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use crate::shaders::bf16;
    use crate::shaders::gpu_common;
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
    let kstride = nkv * hd * 2; // elements between per-token KV rows (K then V)

    let ctx = MetalContext::new()?;
    let [pipe_qk, pipe_sm, pipe_pv] = pipelines(&ctx, KernelVariant::PRODUCTION)?;
    let mut pool = BufferPool::new();

    // Q / out: [canvas][n_q_heads][hd] bf16 (arena default).
    let buf_q = pool
        .allocate(&ctx.device, canvas * n_q_heads * hd * 2)
        .ok_or(Error::Gpu("alloc q"))?;
    BufferPool::write_bf16(&buf_q, &bf16::f32_slice_to_bf16_bits(&f.q));
    let buf_out = pool
        .allocate(&ctx.device, canvas * n_q_heads * hd * 2)
        .ok_or(Error::Gpu("alloc out"))?;
    // KV: f16, +8 pad key rows (direct-load tiles read whole 8-key spans).
    let buf_kv = pool
        .allocate(&ctx.device, (f.kvcache.len() + 8 * kstride) * 2)
        .ok_or(Error::Gpu("alloc kv"))?;
    {
        let mut bits = crate::shaders::f16::f32_slice_to_f16(&f.kvcache);
        bits.resize(bits.len() + 8 * kstride, 0);
        BufferPool::write_bf16(&buf_kv, &bits);
    }
    // Scratch: S (f32), P (half), lrow (f32) — one slice per head.
    let buf_s = pool
        .allocate(&ctx.device, n_q_heads * canvas * np * 4)
        .ok_or(Error::Gpu("alloc s"))?;
    let buf_p = pool
        .allocate(&ctx.device, n_q_heads * canvas * np * 2)
        .ok_or(Error::Gpu("alloc p"))?;
    let buf_lrow = pool
        .allocate(&ctx.device, n_q_heads * canvas * 4)
        .ok_or(Error::Gpu("alloc lrow"))?;

    let dims = AttnGemmDims {
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
    };
    let dims_pv = AttnGemmDims {
        n: hd as u32,
        k: t_total as u32,
        a_row_stride: np as u32,
        ..dims
    };

    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;

    let tg128 = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    let heads = n_q_heads;
    let grid_qk = MTLSize {
        width: t_total.div_ceil(BN),
        height: canvas.div_ceil(BM),
        depth: heads,
    };
    let grid_sm = MTLSize {
        width: canvas,
        height: heads,
        depth: 1,
    };
    let tg_sm = MTLSize {
        width: SOFTMAX_TPG,
        height: 1,
        depth: 1,
    };
    let grid_pv = MTLSize {
        width: hd.div_ceil(BN),
        height: canvas.div_ceil(BM),
        depth: heads,
    };

    // QK: S = Q.K^T (all heads).
    enc.setComputePipelineState(&pipe_qk.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_q), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&buf_kv), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&buf_s), 0, 2);
    }
    gpu_common::set_bytes(&enc, &dims, 3);
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid_qk, tg128);
    enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);

    // Softmax: P = exp(S - rowmax), masked; denom -> lrow (all heads).
    enc.setComputePipelineState(&pipe_sm.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_s), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&buf_p), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&buf_lrow), 0, 2);
    }
    gpu_common::set_bytes(&enc, &dims, 3);
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid_sm, tg_sm);
    enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);

    // PV: O = (P.V) / L (all heads).
    enc.setComputePipelineState(&pipe_pv.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_p), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&buf_kv), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&buf_out), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&buf_lrow), 0, 3);
    }
    gpu_common::set_bytes(&enc, &dims_pv, 4);
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid_pv, tg128);

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

/// Causal CPU reference (prefill mask): query row `tok` sits at absolute
/// position `kv_len + tok` and attends keys `[0, kv_len + tok]`. Same input
/// rounding as `attention::cpu` (Q -> bf16, KV -> f16, out -> bf16) so the
/// tolerance matches the all-valid oracle. Prefill-only — the production
/// non-causal path is covered by the `attention::cpu` oracle.
#[cfg(all(test, target_os = "macos"))]
pub fn cpu_causal(f: &crate::shaders::attention::Fixture) -> Vec<f32> {
    use crate::shaders::bf16;
    use crate::shaders::f16;
    let hd = f.head_dim();
    let nkv = f.n_kv();
    let n_q_heads = f.n_q_heads;
    let canvas = f.canvas;
    let kv_len = f.kv_len as usize;
    let group = n_q_heads / nkv;
    let q = bf16::bf16_slice_to_f32(&bf16::f32_slice_to_bf16_bits(&f.q));
    let kv: Vec<f32> = f
        .kvcache
        .iter()
        .map(|&v| f16::f16_bits_to_f32(f16::f32_to_f16_bits(v)))
        .collect();
    let mut out = vec![0.0f32; f.out_len()];
    for tok in 0..canvas {
        for qh in 0..n_q_heads {
            let kvh = qh / group;
            let q_off = (tok * n_q_heads + qh) * hd;
            let qv = &q[q_off..q_off + hd];
            let t_stop = kv_len + tok; // inclusive causal cutoff
            let mut m = f32::NEG_INFINITY;
            let mut l = 0.0f32;
            let mut acc = vec![0.0f32; hd];
            for t in 0..=t_stop {
                let k_off = t * nkv * hd * 2 + kvh * hd;
                let d: f32 = qv
                    .iter()
                    .zip(kv[k_off..k_off + hd].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let mn = m.max(d);
                let corr = (m - mn).exp();
                let p = (d - mn).exp();
                let v_off = t * nkv * hd * 2 + nkv * hd + kvh * hd;
                for (a, &vv) in acc.iter_mut().zip(kv[v_off..v_off + hd].iter()) {
                    *a = *a * corr + p * vv;
                }
                l = l * corr + p;
                m = mn;
            }
            let o_off = (tok * n_q_heads + qh) * hd;
            for (o, a) in out[o_off..o_off + hd].iter_mut().zip(acc.iter()) {
                *o = bf16::store_bf16_round_half(a / l);
            }
        }
    }
    out
}

/// Model-shaped full-layer fixture (canvas=256, 16 Q / 2 KV, hd=512) for
/// benching E17 against `attention_mma_full` at realistic KV lengths.
#[cfg(target_os = "macos")]
pub fn model_full_fixture(kv_len: u32) -> crate::shaders::attention::Fixture {
    use crate::shaders::cpu::attention::LayerAttnParams;
    let (canvas, n_q_heads, nkv, hd) = (256usize, 16usize, 2usize, 512usize);
    let t_total = kv_len as usize + canvas;
    let mut kvcache = vec![0.0f32; t_total * nkv * hd * 2];
    for (i, x) in kvcache.iter_mut().enumerate() {
        *x = ((i % 4096) as f32 * 0.0007).sin() * 0.35;
    }
    crate::shaders::attention::Fixture {
        q: (0..canvas * n_q_heads * hd)
            .map(|i| (i as f32 * 0.013).sin() * 0.4)
            .collect(),
        kvcache,
        layer: LayerAttnParams {
            head_dim: hd as u32,
            n_kv_heads: nkv as u32,
            is_full: true,
            v_proj: 0,
            kv_region: 0,
            q_norm_off: 0,
            k_norm_off: 0,
        },
        canvas,
        n_q_heads,
        kv_len,
    }
}

/// Time `iters` back-to-back runs of the full E17 3-kernel sequence over all
/// heads (one full-attention layer, one prefill sub-chunk) in a single command
/// buffer; 1 warm-up round + min over timed rounds (factors out clock ramp).
/// Returns mean ms per layer.
#[cfg(target_os = "macos")]
pub fn bench_gpu(f: &crate::shaders::attention::Fixture, iters: usize) -> Result<f64, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use crate::shaders::bf16;
    use crate::shaders::gpu_common;
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
    let t_total = f.kv_len as usize + canvas;
    let np = n_pad(t_total);
    let kstride = nkv * hd * 2;

    let ctx = MetalContext::new()?;
    let [pipe_qk, pipe_sm, pipe_pv] = pipelines(&ctx, KernelVariant::PRODUCTION)?;
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
    let buf_s = pool
        .allocate(&ctx.device, n_q_heads * canvas * np * 4)
        .ok_or(Error::Gpu("alloc s"))?;
    let buf_p = pool
        .allocate(&ctx.device, n_q_heads * canvas * np * 2)
        .ok_or(Error::Gpu("alloc p"))?;
    let buf_lrow = pool
        .allocate(&ctx.device, n_q_heads * canvas * 4)
        .ok_or(Error::Gpu("alloc lrow"))?;

    let dims = AttnGemmDims {
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
    };
    let dims_pv = AttnGemmDims {
        n: hd as u32,
        k: t_total as u32,
        a_row_stride: np as u32,
        ..dims
    };
    let tg128 = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    let grid_qk = MTLSize {
        width: t_total.div_ceil(BN),
        height: canvas.div_ceil(BM),
        depth: n_q_heads,
    };
    let grid_sm = MTLSize {
        width: canvas,
        height: n_q_heads,
        depth: 1,
    };
    let tg_sm = MTLSize {
        width: SOFTMAX_TPG,
        height: 1,
        depth: 1,
    };
    let grid_pv = MTLSize {
        width: hd.div_ceil(BN),
        height: canvas.div_ceil(BM),
        depth: n_q_heads,
    };

    let mut best = f64::INFINITY;
    for round in 0..6 {
        let t = Instant::now();
        let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
        let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
        for _ in 0..iters {
            enc.setComputePipelineState(&pipe_qk.pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_q), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&buf_kv), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&buf_s), 0, 2);
            }
            gpu_common::set_bytes(&enc, &dims, 3);
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid_qk, tg128);
            enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
            enc.setComputePipelineState(&pipe_sm.pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_s), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&buf_p), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&buf_lrow), 0, 2);
            }
            gpu_common::set_bytes(&enc, &dims, 3);
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid_sm, tg_sm);
            enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
            enc.setComputePipelineState(&pipe_pv.pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_p), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&buf_kv), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&buf_out), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&buf_lrow), 0, 3);
            }
            gpu_common::set_bytes(&enc, &dims_pv, 4);
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid_pv, tg128);
            enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// GEMM-attention decomposition vs the CPU attention oracle (the target
    /// `attention::gpu_mma_full` uses). Full-layer shape (hd=512, GQA group 8),
    /// non-causal (denoise-equivalent, all-valid).
    #[cfg(target_os = "macos")]
    #[test]
    fn attn_gemm_full_grp8_vs_cpu() {
        use crate::shaders::test_util::{ElemFormat, assert_oracle};
        let f = crate::shaders::attention::full_grp8_hd512_fixture(ElemFormat::F32);
        let got = gpu(&f, false).expect("gpu attn_gemm");
        let oracle = crate::shaders::attention::cpu(&f);
        assert_oracle(&got, &oracle, 2e-2, 0.9999);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn attn_gemm_full_grp2_vs_cpu() {
        use crate::shaders::test_util::{ElemFormat, assert_oracle};
        let f = crate::shaders::attention::full_hd512_fixture(ElemFormat::F32);
        let got = gpu(&f, false).expect("gpu attn_gemm");
        let oracle = crate::shaders::attention::cpu(&f);
        assert_oracle(&got, &oracle, 2e-2, 0.9999);
    }

    /// Causal prefill mask vs the causal CPU reference. Full-layer shape,
    /// GQA group 8; kv_len=28, canvas=16 exercises per-row cutoffs across
    /// ragged 64-key N-tiles.
    #[cfg(target_os = "macos")]
    #[test]
    fn attn_gemm_full_grp8_causal_vs_cpu() {
        use crate::shaders::test_util::{ElemFormat, assert_oracle};
        let f = crate::shaders::attention::full_grp8_hd512_fixture(ElemFormat::F32);
        let got = gpu(&f, true).expect("gpu attn_gemm causal");
        let oracle = cpu_causal(&f);
        assert_oracle(&got, &oracle, 2e-2, 0.9999);
    }

    /// Premise check: E17 GEMM-attention vs `attention_mma_full` at real
    /// full-layer shape (canvas=256, 16 Q / 2 KV, hd=512). One command buffer
    /// each, min-of-rounds. Ignored (timing).
    /// Run: `cargo test --release attn_gemm_bench -- --ignored --nocapture`
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore]
    fn attn_gemm_bench() {
        let iters = 10usize;
        println!("  kv    mma_full(ms)   e17(ms)   speedup");
        for kv_len in [8192u32, 30000, 60000] {
            let f = model_full_fixture(kv_len);
            let mma_full = crate::shaders::attention::bench_path(&f, iters, 3).unwrap();
            let e17 = bench_gpu(&f, iters).unwrap();
            println!(
                "{kv_len:>6}   {mma_full:9.3}   {e17:9.3}   {:.2}x",
                mma_full / e17
            );
        }
    }
}

/// Manifest registration; collected in common/manifest.rs::MANIFEST.
pub const SPEC: crate::shaders::manifest::KernelSpec = crate::shaders::manifest::KernelSpec {
    name: "attention_gemm",
    entry: "attn_gemm_qk",
    quant_formats: &[crate::shaders::variant::QuantFormat::Q4Affine],
    fc: &[],
    variants: crate::shaders::manifest::KernelVariants::Elementwise,
};
