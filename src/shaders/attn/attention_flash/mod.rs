//! E18: fused flash prefill attention (online softmax, register-resident O, no
//! device S/P traffic) — the steel-SDPA dataflow, in contrast to E17's
//! `attention_gemm` decomposition which materializes S=Q·Kᵀ and P to device.
//!
//! The Metal kernel (`attention_flash.metal`) streams the keys in BK-wide blocks
//! and keeps the O accumulator resident across the stream, combining each block
//! into the running (max, denom, O) with the standard flash rescale. This module
//! currently holds the **CPU block-streaming reference** used to pin the exact
//! blocking/masking/GQA scheme the kernel mirrors, validated against the E17
//! `attention_gemm::cpu_causal` batch reference (both compute the same softmax;
//! flash just blocks it). GPU pipeline + oracle land with the kernel.

/// CPU block-streaming flash reference. Mirrors the planned kernel exactly: per
/// query row, stream keys in `bk`-wide blocks, online-softmax combine each block
/// into (m, l, acc), divide at the end. Output layout matches `cpu_causal`
/// (`[canvas][n_q_heads][hd]`, bf16-rounded at store) so the two are directly
/// comparable. `round_kv_f16` rounds the KV to f16 (main-cache path) vs raw f32
/// (side-ring path) — same knob as `cpu_causal`.
pub fn cpu_flash_blocked(
    f: &crate::shaders::attention::Fixture,
    bk: usize,
    round_kv_f16: bool,
) -> Vec<f32> {
    use crate::shaders::bf16;
    use crate::shaders::f16;
    let hd = f.head_dim();
    let nkv = f.n_kv();
    let n_q_heads = f.n_q_heads;
    let canvas = f.canvas;
    let kv_len = f.kv_len as usize;
    let group = n_q_heads / nkv;
    let q = bf16::bf16_slice_to_f32(&bf16::f32_slice_to_bf16_bits(&f.q));
    let kv: Vec<f32> = if round_kv_f16 {
        f.kvcache
            .iter()
            .map(|&v| f16::f16_bits_to_f32(f16::f32_to_f16_bits(v)))
            .collect()
    } else {
        f.kvcache.clone()
    };
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
            // Stream keys in BK-wide blocks (the kernel's inner loop).
            let mut t0 = 0usize;
            while t0 <= t_stop {
                let t1 = (t0 + bk).min(t_stop + 1);
                // Block scores, then block max over valid columns.
                let mut m_block = f32::NEG_INFINITY;
                let mut sblock = Vec::with_capacity(t1 - t0);
                for t in t0..t1 {
                    let k_off = t * nkv * hd * 2 + kvh * hd;
                    let d: f32 = qv
                        .iter()
                        .zip(kv[k_off..k_off + hd].iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    m_block = m_block.max(d);
                    sblock.push(d);
                }
                // Online combine this block into (m, l, acc).
                let m_new = m.max(m_block);
                let corr = (m - m_new).exp();
                for a in acc.iter_mut() {
                    *a *= corr;
                }
                l *= corr;
                for (j, t) in (t0..t1).enumerate() {
                    let p = (sblock[j] - m_new).exp();
                    let v_off = t * nkv * hd * 2 + nkv * hd + kvh * hd;
                    for (a, &vv) in acc.iter_mut().zip(kv[v_off..v_off + hd].iter()) {
                        *a += p * vv;
                    }
                    l += p;
                }
                m = m_new;
                t0 = t1;
            }
            let o_off = (tok * n_q_heads + qh) * hd;
            for (o, a) in out[o_off..o_off + hd].iter_mut().zip(acc.iter()) {
                *o = bf16::store_bf16_round_half(a / l);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// GPU kernel harness
// ---------------------------------------------------------------------------
#[cfg(target_os = "macos")]
use crate::Error;

pub const SHADER: &str = include_str!("attention_flash.metal");
pub const ENTRY: &str = "attn_flash";

/// Default tile geometry (must match the shader constants FL_BQ/FL_BK).
pub const BQ: usize = 16;
pub const BK: usize = 64;

/// Prepend the tile #defines to specialize the kernel. Un-prepended (default)
/// source reproduces the shipped 16/64/hd512 kernel. `hd` must equal the
/// dispatch head_dim (sizes the resident Q stage + O accumulator).
pub fn tuned_source(bq: usize, bk: usize, hd: usize) -> String {
    format!("#define FL_BQ {bq}\n#define FL_BK {bk}\n#define FL_HD {hd}\n{SHADER}")
}

/// Host mirror of the shader `FlashDims`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlashDims {
    pub m: u32,
    pub t_total: u32,
    pub hd: u32,
    pub a_row_stride: u32,
    pub b_row_stride: u32,
    pub out_row_stride: u32,
    pub kv_len: u32,
    pub group: u32,
    pub nkv: u32,
    pub head_base: u32,
    pub causal: u32,
    pub window: u32,
    pub kv_ring_mask: u32,
}

#[cfg(target_os = "macos")]
pub fn pipeline_flash(
    ctx: &crate::metal::device::MetalContext,
    variant: crate::shaders::variant::KernelVariant,
    bq: usize,
    bk: usize,
    hd: usize,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    let src = tuned_source(bq, bk, hd);
    ctx.compile_subkernel_ex(&src, ENTRY, variant, "flash", &[], &[])
}

/// GPU oracle: run the fused flash kernel over an `attention::Fixture`, return
/// bf16-rounded O — same target as `attention_gemm::gpu` / `cpu_causal`. f16 KV
/// only. Full layers (hd=512) and sliding (hd=256); hd a multiple of NSG*8=64.
#[cfg(target_os = "macos")]
pub fn gpu_flash(f: &crate::shaders::attention::Fixture, causal: bool) -> Result<Vec<f32>, Error> {
    use crate::shaders::attn::harness::AttnRig;
    use crate::shaders::gpu_common;
    use crate::shaders::variant::KernelVariant;
    use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

    // Standard rig; +BK pad key rows (device tile-loads read up to key
    // kt+BK-1 past T). f16 KV only — no side ring.
    let rig = AttnRig::from_fixture(f, BK, false)?;
    let (canvas, n_q_heads, hd) = (rig.canvas, rig.n_q_heads, rig.hd);

    let pipe = pipeline_flash(&rig.ctx, KernelVariant::PRODUCTION, BQ, BK, hd)?;

    let mut dims = FlashDims {
        m: canvas as u32,
        t_total: rig.t_total as u32,
        hd: hd as u32,
        a_row_stride: (n_q_heads * hd) as u32,
        b_row_stride: rig.kstride as u32,
        out_row_stride: (n_q_heads * hd) as u32,
        kv_len: f.kv_len,
        group: rig.group as u32,
        nkv: rig.nkv as u32,
        head_base: 0,
        causal: u32::from(causal),
        window: 0,
        kv_ring_mask: 0,
    };

    let tg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    let grid = MTLSize {
        width: canvas.div_ceil(BQ),
        height: 1,
        depth: n_q_heads,
    };
    rig.run_once(|enc| {
        dims.head_base = 0;
        enc.setComputePipelineState(&pipe.pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&rig.buf_q), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&rig.buf_kv), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&rig.buf_out), 0, 2);
        }
        gpu_common::set_bytes(enc, &dims, 3);
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    })?;

    Ok(rig.read_out_f32(f.out_len()))
}

/// Isolated timing: full-layer flash prefill (all heads, one command buffer),
/// min-of-warmed-rounds ms/layer. Mirrors `attention_gemm::bench_gpu` so the two
/// are directly comparable. `causal` mirrors `gpu_flash()` (benches historically
/// dispatched causal=1).
#[cfg(target_os = "macos")]
pub fn bench_flash(
    f: &crate::shaders::attention::Fixture,
    iters: usize,
    bq: usize,
    causal: bool,
) -> Result<f64, Error> {
    bench_flash_window(f, iters, bq, 0, causal)
}

/// Same as `bench_flash` with an explicit sliding window (`0` = full causal).
#[cfg(target_os = "macos")]
pub fn bench_flash_window(
    f: &crate::shaders::attention::Fixture,
    iters: usize,
    bq: usize,
    window: u32,
    causal: bool,
) -> Result<f64, Error> {
    use crate::shaders::attn::harness::AttnRig;
    use crate::shaders::gpu_common;
    use crate::shaders::variant::KernelVariant;
    use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

    let rig = AttnRig::from_fixture(f, BK, false)?;
    let (canvas, n_q_heads, hd) = (rig.canvas, rig.n_q_heads, rig.hd);

    let pipe = pipeline_flash(&rig.ctx, KernelVariant::PRODUCTION, bq, BK, hd)?;
    let mut dims = FlashDims {
        m: canvas as u32,
        t_total: rig.t_total as u32,
        hd: hd as u32,
        a_row_stride: (n_q_heads * hd) as u32,
        b_row_stride: rig.kstride as u32,
        out_row_stride: (n_q_heads * hd) as u32,
        kv_len: f.kv_len,
        group: rig.group as u32,
        nkv: rig.nkv as u32,
        head_base: 0,
        causal: u32::from(causal),
        window,
        kv_ring_mask: 0,
    };
    let tg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    let grid = MTLSize {
        width: canvas.div_ceil(bq),
        height: 1,
        depth: n_q_heads,
    };
    rig.time_rounds(iters, |enc| {
        dims.head_base = 0;
        enc.setComputePipelineState(&pipe.pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&rig.buf_q), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&rig.buf_kv), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&rig.buf_out), 0, 2);
        }
        gpu_common::set_bytes(enc, &dims, 3);
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
    })
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod mod_tests;
