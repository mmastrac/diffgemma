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

/// Prepend the tile #defines to specialize the kernel for a sweep. Un-prepended
/// (default) source reproduces the shipped 16/64 kernel.
pub fn tuned_source(bq: usize, bk: usize) -> String {
    format!("#define FL_BQ {bq}\n#define FL_BK {bk}\n{SHADER}")
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
}

#[cfg(target_os = "macos")]
pub fn pipeline_flash(
    ctx: &crate::metal::device::MetalContext,
    variant: crate::shaders::variant::KernelVariant,
    side: bool,
    bq: usize,
    bk: usize,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    use crate::shaders::variant::FcBool;
    let bools: &[FcBool] = if side {
        &[FcBool {
            index: 30,
            value: true,
        }]
    } else {
        &[]
    };
    let src = tuned_source(bq, bk);
    let label = if side { "flash_side" } else { "flash" };
    ctx.compile_subkernel_ex(&src, ENTRY, variant, label, bools, &[])
}

/// GPU oracle: run the fused flash kernel over an `attention::Fixture`, return
/// bf16-rounded O — same target as `attention_gemm::gpu` / `cpu_causal`. Full
/// layers (hd=512) and sliding (hd=256); hd must be a multiple of NSG*8=64.
#[cfg(target_os = "macos")]
pub fn gpu_flash(
    f: &crate::shaders::attention::Fixture,
    causal: bool,
    side: bool,
) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use crate::shaders::bf16;
    use crate::shaders::gpu_common;
    use crate::shaders::variant::KernelVariant;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
        MTLComputeCommandEncoder, MTLSize,
    };

    let canvas = f.canvas;
    let n_q_heads = f.n_q_heads;
    let nkv = f.n_kv();
    let hd = f.head_dim();
    let group = n_q_heads / nkv;
    let kv_len = f.kv_len as usize;
    let t_total = kv_len + canvas;
    let kstride = nkv * hd * 2;

    let ctx = MetalContext::new()?;
    let pipe = pipeline_flash(&ctx, KernelVariant::PRODUCTION, side, BQ, BK)?;
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

    let mut dims = FlashDims {
        m: canvas as u32,
        t_total: t_total as u32,
        hd: hd as u32,
        a_row_stride: (n_q_heads * hd) as u32,
        b_row_stride: kstride as u32,
        out_row_stride: (n_q_heads * hd) as u32,
        kv_len: f.kv_len,
        group: group as u32,
        nkv: nkv as u32,
        head_base: 0,
        causal: u32::from(causal),
    };

    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
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
    dims.head_base = 0;
    enc.setComputePipelineState(&pipe.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_q), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&buf_kv), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&buf_out), 0, 2);
        if side {
            enc.setBuffer_offset_atIndex(Some(&buf_kvf), 0, 9);
        }
    }
    gpu_common::set_bytes(&enc, &dims, 3);
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
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

/// Isolated timing: full-layer flash prefill (all heads, one command buffer),
/// min-of-warmed-rounds ms/layer. Mirrors `attention_gemm::bench_gpu` so the two
/// are directly comparable.
#[cfg(target_os = "macos")]
pub fn bench_flash(f: &crate::shaders::attention::Fixture, iters: usize) -> Result<f64, Error> {
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
    let kstride = nkv * hd * 2;

    let ctx = MetalContext::new()?;
    let pipe = pipeline_flash(&ctx, KernelVariant::PRODUCTION, false, BQ, BK)?;
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
    let mut dims = FlashDims {
        m: canvas as u32,
        t_total: t_total as u32,
        hd: hd as u32,
        a_row_stride: (n_q_heads * hd) as u32,
        b_row_stride: kstride as u32,
        out_row_stride: (n_q_heads * hd) as u32,
        kv_len: f.kv_len,
        group: group as u32,
        nkv: nkv as u32,
        head_base: 0,
        causal: 1,
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
    let mut best = f64::INFINITY;
    for round in 0..6 {
        let t = Instant::now();
        let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
        let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
        for _ in 0..iters {
            dims.head_base = 0;
            enc.setComputePipelineState(&pipe.pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_q), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&buf_kv), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&buf_out), 0, 2);
            }
            gpu_common::set_bytes(&enc, &dims, 3);
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
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
#[path = "mod_tests.rs"]
mod mod_tests;
