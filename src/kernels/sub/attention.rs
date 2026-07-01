//! All-valid GQA attention over monolithic KV cache (online softmax).

use super::bf16;
use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::kernels::cpu::attention::{self, LayerAttnParams};
use crate::kernels::sub::qk_rope_kv::AttnDims;
use crate::metal::{LayerOffsets, StepParams};
use crate::safetensors::Error;

pub const ENTRY: &str = "attention";
pub const THREADGROUP_WIDTH: usize = 64;

const SHADER: &str = shader_include::include_metal!("kernels/attention.metal");

/// Flash-style matrix-unit attention (simdgroup_float8x8). Same semantics as
/// `attention`, validated against it / the CPU oracle. M=8 query rows per tile.
pub const ENTRY_MMA: &str = "attention_mma";
pub const MMA_M_TILE: usize = 8;
const SHADER_MMA: &str = shader_include::include_metal!("kernels/attention_mma.metal");

/// GQA-grouped MMA attention (2 Q heads / threadgroup, shared K/V staging).
/// Group size 2, hd <= 256 only (sliding layers). `attention` stays the oracle.
pub const ENTRY_MMA2: &str = "attention_mma2";
const SHADER_MMA2: &str = shader_include::include_metal!("kernels/attention_mma2.metal");

/// MMA attention for full/global layers (hd=512): register-resident O + QG-grouped
/// K/V sharing. `attention` stays the oracle. QG simdgroups per threadgroup, 32
/// lanes each; (group/QG) sub-groups along grid.z.
pub const ENTRY_MMA_FULL: &str = "attention_mma_full";
pub const MMA_FULL_QG: usize = 2;
const SHADER_MMA_FULL: &str = shader_include::include_metal!("kernels/attention_mma_full.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub q: Vec<f32>,
    pub kvcache: Vec<f32>,
    pub layer: LayerAttnParams,
    pub canvas: usize,
    pub n_q_heads: usize,
    pub kv_len: u32,
}

impl Fixture {
    pub fn head_dim(&self) -> usize {
        self.layer.head_dim as usize
    }

    pub fn n_kv(&self) -> usize {
        self.layer.n_kv_heads as usize
    }

    pub fn out_len(&self) -> usize {
        self.canvas * self.n_q_heads * self.head_dim()
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    let canvas = 2usize;
    let n_q_heads = 4usize;
    let n_kv = 2usize;
    let hd = 32usize;
    let kv_len = 2u32;
    let t_total = kv_len as usize + canvas;
    let mut kvcache = vec![0.0f32; t_total * n_kv * hd * 2];
    for t in 0..t_total {
        for h in 0..n_kv {
            for d in 0..hd {
                let k_i = t * n_kv * hd * 2 + h * hd + d;
                let v_i = t * n_kv * hd * 2 + n_kv * hd + h * hd + d;
                kvcache[k_i] = ((t * 17 + h * 3 + d) as f32 * 0.05).sin();
                kvcache[v_i] = ((t * 13 + h * 5 + d) as f32 * 0.04).cos();
            }
        }
    }
    Fixture {
        q: (0..canvas * n_q_heads * hd)
            .map(|i| (i as f32 * 0.11).sin() * 0.6)
            .collect(),
        kvcache,
        layer: LayerAttnParams {
            head_dim: hd as u32,
            n_kv_heads: n_kv as u32,
            is_full: false,
            v_proj: 1,
            kv_region: 0,
            q_norm_off: 0,
            k_norm_off: 0,
        },
        canvas,
        n_q_heads,
        kv_len,
    }
}

pub fn wide_fixture(_: ElemFormat) -> Fixture {
    let canvas = 4usize;
    let n_q_heads = 8usize;
    let n_kv = 2usize;
    let hd = 64usize;
    let kv_len = 4u32;
    let t_total = kv_len as usize + canvas;
    let mut kvcache = vec![0.0f32; t_total * n_kv * hd * 2];
    for i in 0..kvcache.len() {
        kvcache[i] = (i as f32 * 0.031).sin() * 0.3;
    }
    Fixture {
        q: (0..canvas * n_q_heads * hd)
            .map(|i| (i as f32 * 0.02).cos() * 0.5)
            .collect(),
        kvcache,
        layer: LayerAttnParams {
            head_dim: hd as u32,
            n_kv_heads: n_kv as u32,
            is_full: false,
            v_proj: 1,
            kv_region: 0,
            q_norm_off: 0,
            k_norm_off: 0,
        },
        canvas,
        n_q_heads,
        kv_len,
    }
}

fn model_attn_fixture(
    canvas: usize,
    n_q_heads: usize,
    n_kv: usize,
    hd: usize,
    kv_len: u32,
    is_full: bool,
) -> Fixture {
    assert_eq!(n_q_heads % n_kv, 0);
    let t_total = kv_len as usize + canvas;
    let mut kvcache = vec![0.0f32; t_total * n_kv * hd * 2];
    for t in 0..t_total {
        for h in 0..n_kv {
            for d in 0..hd {
                let k_i = t * n_kv * hd * 2 + h * hd + d;
                let v_i = k_i + n_kv * hd;
                kvcache[k_i] = ((t * 19 + h * 7 + d) as f32 * 0.0031).sin() * 0.4;
                kvcache[v_i] = ((t * 23 + h * 11 + d) as f32 * 0.0027).cos() * 0.35;
            }
        }
    }
    Fixture {
        q: (0..canvas * n_q_heads * hd)
            .map(|i| (i as f32 * 0.017).sin() * 0.5 + 0.02 * (i % hd) as f32)
            .collect(),
        kvcache,
        layer: LayerAttnParams {
            head_dim: hd as u32,
            n_kv_heads: n_kv as u32,
            is_full,
            v_proj: if is_full { 0 } else { 1 },
            kv_region: 0,
            q_norm_off: 0,
            k_norm_off: 0,
        },
        canvas,
        n_q_heads,
        kv_len,
    }
}

/// Full-attention layer: hd=512, tpg_w=64 → per=8 (exact `acc[8]` fit).
pub fn full_hd512_fixture(_: ElemFormat) -> Fixture {
    model_attn_fixture(4, 16, 8, 512, 28, true)
}

/// Real full/global layer shape: hd=512, nkv=2, GQA group 8 (16 Q / 2 KV).
/// canvas=16 → 2 MT tiles; group/QG=4 sub-groups → exercises grid.z for
/// `attention_mma_full`. kv_len=28 → T=44 spans ragged key-tile tails.
pub fn full_grp8_hd512_fixture(_: ElemFormat) -> Fixture {
    model_attn_fixture(16, 16, 2, 512, 28, true)
}

/// Sliding layer: hd=256, per=4; longer KV (kv_len=128) exercises runtime T loop.
pub fn sliding_hd256_fixture(_: ElemFormat) -> Fixture {
    model_attn_fixture(4, 16, 8, 256, 128, false)
}

fn layer_offsets(f: &Fixture) -> LayerOffsets {
    LayerOffsets {
        input_ln: 0,
        q_proj: 0,
        q_norm: 0,
        k_proj: 0,
        k_norm: 0,
        v_proj: f.layer.v_proj,
        o_proj: 0,
        post_attn_ln: 0,
        pre_ff_ln: 0,
        mlp_gate: 0,
        mlp_up: 0,
        mlp_down: 0,
        post_ff_ln_1: 0,
        router_scale: 0,
        router_proj: 0,
        per_expert_scale: 0,
        pre_ff_ln_2: 0,
        experts_gate_up: 0,
        experts_down: 0,
        post_ff_ln_2: 0,
        post_ff_ln: 0,
        layer_scalar: 0,
        kv_region: f.layer.kv_region,
        head_dim: f.layer.head_dim,
        n_kv_heads: f.layer.n_kv_heads,
        is_full: u32::from(f.layer.is_full),
        _pad: 0,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let q = bf16::bf16_slice_to_f32(&bf16::f32_slice_to_bf16_bits(&f.q));
    let kvcache = bf16::bf16_slice_to_f32(&bf16::f32_slice_to_bf16_bits(&f.kvcache));
    let mut out = vec![0.0f32; f.out_len()];
    attention::attention(
        &mut out,
        &q,
        &kvcache,
        f.layer,
        f.canvas,
        f.n_q_heads,
        f.kv_len,
    );
    for o in out.iter_mut() {
        *o = bf16::store_bf16_round_half(*o);
    }
    out
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER, ENTRY, variant)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLSize,
};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let buf_q = pool
        .allocate(&ctx.device, f.q.len() * 2)
        .ok_or(Error::Format("alloc"))?;
    let buf_kv = pool
        .allocate(&ctx.device, f.kvcache.len() * 2)
        .ok_or(Error::Format("alloc"))?;
    let buf_out = pool
        .allocate(&ctx.device, f.out_len() * 2)
        .ok_or(Error::Format("alloc"))?;
    let buf_layer = pool
        .allocate(&ctx.device, std::mem::size_of::<LayerOffsets>())
        .ok_or(Error::Format("alloc"))?;

    BufferPool::write_bf16(&buf_q, &bf16::f32_slice_to_bf16_bits(&f.q));
    BufferPool::write_bf16(&buf_kv, &bf16::f32_slice_to_bf16_bits(&f.kvcache));
    let layer = layer_offsets(f);
    let layer_bytes = unsafe {
        std::slice::from_raw_parts(
            &layer as *const LayerOffsets as *const u8,
            std::mem::size_of::<LayerOffsets>(),
        )
    };
    BufferPool::write_bytes(&buf_layer, layer_bytes);

    let params = StepParams {
        kv_len: f.kv_len,
        max_steps: 8,
        entropy_bound: 0.0,
        t_min: 0.0,
        t_max: 1.0,
        conf_threshold: 0.0,
        stability_threshold: 0,
        min_early_stop_steps: 0,
        accept_plateau_threshold: 0,
        plateau_prefix_mean_max: f32::MAX,
        eos_token_id: 1,
    };
    let dims = AttnDims {
        canvas: f.canvas as u32,
        n_q_heads: f.n_q_heads as u32,
        causal: 0,
    };

    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_q), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&buf_kv), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&buf_out), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&buf_layer), 0, 3);
    }
    gpu_common::set_bytes(&enc, &params, 4);
    gpu_common::set_bytes(&enc, &dims, 5);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: f.canvas,
            height: f.n_q_heads,
            depth: 1,
        },
        MTLSize {
            width: THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        },
    );
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

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn gpu(_: &Fixture, _: KernelVariant) -> Result<Vec<f32>, Error> {
    Err(Error::Format("Metal unavailable"))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_mma_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER_MMA, ENTRY_MMA, variant)
}

/// Matrix-unit attention path. Identical buffer layout to `gpu`; differs only in
/// pipeline + grid (one threadgroup per (MMA_M_TILE-row query tile, q_head), 32 lanes).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu_mma(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_mma_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let buf_q = pool
        .allocate(&ctx.device, f.q.len() * 2)
        .ok_or(Error::Format("alloc"))?;
    let buf_kv = pool
        .allocate(&ctx.device, f.kvcache.len() * 2)
        .ok_or(Error::Format("alloc"))?;
    let buf_out = pool
        .allocate(&ctx.device, f.out_len() * 2)
        .ok_or(Error::Format("alloc"))?;
    let buf_layer = pool
        .allocate(&ctx.device, std::mem::size_of::<LayerOffsets>())
        .ok_or(Error::Format("alloc"))?;

    BufferPool::write_bf16(&buf_q, &bf16::f32_slice_to_bf16_bits(&f.q));
    BufferPool::write_bf16(&buf_kv, &bf16::f32_slice_to_bf16_bits(&f.kvcache));
    let layer = layer_offsets(f);
    let layer_bytes = unsafe {
        std::slice::from_raw_parts(
            &layer as *const LayerOffsets as *const u8,
            std::mem::size_of::<LayerOffsets>(),
        )
    };
    BufferPool::write_bytes(&buf_layer, layer_bytes);

    let params = StepParams {
        kv_len: f.kv_len,
        max_steps: 8,
        entropy_bound: 0.0,
        t_min: 0.0,
        t_max: 1.0,
        conf_threshold: 0.0,
        stability_threshold: 0,
        min_early_stop_steps: 0,
        accept_plateau_threshold: 0,
        plateau_prefix_mean_max: f32::MAX,
        eos_token_id: 1,
    };
    let dims = AttnDims {
        canvas: f.canvas as u32,
        n_q_heads: f.n_q_heads as u32,
        causal: 0,
    };

    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_q), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&buf_kv), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&buf_out), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&buf_layer), 0, 3);
    }
    gpu_common::set_bytes(&enc, &params, 4);
    gpu_common::set_bytes(&enc, &dims, 5);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: f.canvas.div_ceil(MMA_M_TILE),
            height: f.n_q_heads,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
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

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn gpu_mma(_: &Fixture, _: KernelVariant) -> Result<Vec<f32>, Error> {
    Err(Error::Format("Metal unavailable"))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_mma2_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER_MMA2, ENTRY_MMA2, variant)
}

/// GQA-grouped MMA attention path. One threadgroup per (MMA_M_TILE-row tile, KV
/// head), 64 lanes = 2 simdgroups (one per Q head in the group). hd <= 256 only.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu_mma2(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_mma2_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let buf_q = pool.allocate(&ctx.device, f.q.len() * 2).ok_or(Error::Format("alloc"))?;
    let buf_kv = pool.allocate(&ctx.device, f.kvcache.len() * 2).ok_or(Error::Format("alloc"))?;
    let buf_out = pool.allocate(&ctx.device, f.out_len() * 2).ok_or(Error::Format("alloc"))?;
    let buf_layer = pool
        .allocate(&ctx.device, std::mem::size_of::<LayerOffsets>())
        .ok_or(Error::Format("alloc"))?;

    BufferPool::write_bf16(&buf_q, &bf16::f32_slice_to_bf16_bits(&f.q));
    BufferPool::write_bf16(&buf_kv, &bf16::f32_slice_to_bf16_bits(&f.kvcache));
    let layer = layer_offsets(f);
    let layer_bytes = unsafe {
        std::slice::from_raw_parts(
            &layer as *const LayerOffsets as *const u8,
            std::mem::size_of::<LayerOffsets>(),
        )
    };
    BufferPool::write_bytes(&buf_layer, layer_bytes);

    let params = StepParams {
        kv_len: f.kv_len,
        max_steps: 8,
        entropy_bound: 0.0,
        t_min: 0.0,
        t_max: 1.0,
        conf_threshold: 0.0,
        stability_threshold: 0,
        min_early_stop_steps: 0,
        accept_plateau_threshold: 0,
        plateau_prefix_mean_max: f32::MAX,
        eos_token_id: 1,
    };
    let dims = AttnDims::new(f.canvas as u32, f.n_q_heads as u32);

    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_q), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&buf_kv), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&buf_out), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&buf_layer), 0, 3);
    }
    gpu_common::set_bytes(&enc, &params, 4);
    gpu_common::set_bytes(&enc, &dims, 5);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: f.canvas.div_ceil(MMA_M_TILE), height: f.n_kv(), depth: 1 },
        MTLSize { width: 64, height: 1, depth: 1 },
    );
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

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn gpu_mma2(_: &Fixture, _: KernelVariant) -> Result<Vec<f32>, Error> {
    Err(Error::Format("Metal unavailable"))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_mma_full_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER_MMA_FULL, ENTRY_MMA_FULL, variant)
}

/// Full-layer MMA attention path (hd=512). One threadgroup per (MT-row tile, KV
/// head, QG-head sub-group); QG simdgroups (32 lanes each) share K/V staging,
/// O accumulator is register-resident. Identical buffer layout to `gpu_mma`.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu_mma_full(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_mma_full_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let buf_q = pool.allocate(&ctx.device, f.q.len() * 2).ok_or(Error::Format("alloc"))?;
    let buf_kv = pool.allocate(&ctx.device, f.kvcache.len() * 2).ok_or(Error::Format("alloc"))?;
    let buf_out = pool.allocate(&ctx.device, f.out_len() * 2).ok_or(Error::Format("alloc"))?;
    let buf_layer = pool
        .allocate(&ctx.device, std::mem::size_of::<LayerOffsets>())
        .ok_or(Error::Format("alloc"))?;

    BufferPool::write_bf16(&buf_q, &bf16::f32_slice_to_bf16_bits(&f.q));
    BufferPool::write_bf16(&buf_kv, &bf16::f32_slice_to_bf16_bits(&f.kvcache));
    let layer = layer_offsets(f);
    let layer_bytes = unsafe {
        std::slice::from_raw_parts(
            &layer as *const LayerOffsets as *const u8,
            std::mem::size_of::<LayerOffsets>(),
        )
    };
    BufferPool::write_bytes(&buf_layer, layer_bytes);

    let params = StepParams {
        kv_len: f.kv_len,
        max_steps: 8,
        entropy_bound: 0.0,
        t_min: 0.0,
        t_max: 1.0,
        conf_threshold: 0.0,
        stability_threshold: 0,
        min_early_stop_steps: 0,
        accept_plateau_threshold: 0,
        plateau_prefix_mean_max: f32::MAX,
        eos_token_id: 1,
    };
    let dims = AttnDims::new(f.canvas as u32, f.n_q_heads as u32);
    let group = f.n_q_heads / f.n_kv();

    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_q), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&buf_kv), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&buf_out), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&buf_layer), 0, 3);
    }
    gpu_common::set_bytes(&enc, &params, 4);
    gpu_common::set_bytes(&enc, &dims, 5);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: f.canvas.div_ceil(MMA_M_TILE),
            height: f.n_kv(),
            depth: group / MMA_FULL_QG,
        },
        MTLSize { width: MMA_FULL_QG * 32, height: 1, depth: 1 },
    );
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

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn gpu_mma_full(_: &Fixture, _: KernelVariant) -> Result<Vec<f32>, Error> {
    Err(Error::Format("Metal unavailable"))
}

/// Model-shaped attention fixture (canvas=256, 16 Q / 8 KV heads) for benching.
pub fn model_bench_fixture(hd: usize, kv_len: u32, is_full: bool) -> Fixture {
    model_attn_fixture(256, 16, 8, hd, kv_len, is_full)
}

/// Time `iters` back-to-back dispatches of one attention path in a single command
/// buffer; returns mean ms/dispatch (GPU wall, compile + alloc excluded).
/// path: 0 = scalar `attention`, 1 = `attention_mma` (1 head/tg), 2 = `attention_mma2`
/// (2 heads/tg, GQA-shared K/V).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn bench_path(f: &Fixture, iters: usize, path: u8) -> Result<f64, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use std::time::Instant;

    let ctx = MetalContext::new()?;
    let prod = KernelVariant::PRODUCTION;
    let pipeline = match path {
        1 => pipeline_mma_for(&ctx, prod)?,
        2 => pipeline_mma2_for(&ctx, prod)?,
        3 => pipeline_mma_full_for(&ctx, prod)?,
        _ => pipeline_for(&ctx, prod)?,
    };
    let mut pool = BufferPool::new();
    let buf_q = pool.allocate(&ctx.device, f.q.len() * 2).ok_or(Error::Format("alloc"))?;
    let buf_kv = pool.allocate(&ctx.device, f.kvcache.len() * 2).ok_or(Error::Format("alloc"))?;
    let buf_out = pool.allocate(&ctx.device, f.out_len() * 2).ok_or(Error::Format("alloc"))?;
    let buf_layer = pool
        .allocate(&ctx.device, std::mem::size_of::<LayerOffsets>())
        .ok_or(Error::Format("alloc"))?;
    BufferPool::write_bf16(&buf_q, &bf16::f32_slice_to_bf16_bits(&f.q));
    BufferPool::write_bf16(&buf_kv, &bf16::f32_slice_to_bf16_bits(&f.kvcache));
    let layer = layer_offsets(f);
    let layer_bytes = unsafe {
        std::slice::from_raw_parts(
            &layer as *const LayerOffsets as *const u8,
            std::mem::size_of::<LayerOffsets>(),
        )
    };
    BufferPool::write_bytes(&buf_layer, layer_bytes);
    let params = StepParams {
        kv_len: f.kv_len,
        max_steps: 8,
        entropy_bound: 0.0,
        t_min: 0.0,
        t_max: 1.0,
        conf_threshold: 0.0,
        stability_threshold: 0,
        min_early_stop_steps: 0,
        accept_plateau_threshold: 0,
        plateau_prefix_mean_max: f32::MAX,
        eos_token_id: 1,
    };
    let dims = AttnDims::new(f.canvas as u32, f.n_q_heads as u32);
    let (grid, tpg) = match path {
        1 => (
            MTLSize { width: f.canvas.div_ceil(MMA_M_TILE), height: f.n_q_heads, depth: 1 },
            MTLSize { width: 32, height: 1, depth: 1 },
        ),
        2 => (
            MTLSize { width: f.canvas.div_ceil(MMA_M_TILE), height: f.n_kv(), depth: 1 },
            MTLSize { width: 64, height: 1, depth: 1 },
        ),
        3 => (
            MTLSize {
                width: f.canvas.div_ceil(MMA_M_TILE),
                height: f.n_kv(),
                depth: (f.n_q_heads / f.n_kv()) / MMA_FULL_QG,
            },
            MTLSize { width: MMA_FULL_QG * 32, height: 1, depth: 1 },
        ),
        _ => (
            MTLSize { width: f.canvas, height: f.n_q_heads, depth: 1 },
            MTLSize { width: THREADGROUP_WIDTH, height: 1, depth: 1 },
        ),
    };

    // 1 warm-up + several timed rounds (each one command buffer with `iters`
    // dispatches); report the MIN ms/dispatch to factor out GPU clock ramp/throttle.
    let mut best = f64::INFINITY;
    for round in 0..6 {
        let t = Instant::now();
        let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
        let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
        enc.setComputePipelineState(&pipeline.pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&buf_q), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&buf_kv), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&buf_out), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&buf_layer), 0, 3);
        }
        gpu_common::set_bytes(&enc, &params, 4);
        gpu_common::set_bytes(&enc, &dims, 5);
        for _ in 0..iters {
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tpg);
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
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::kernels::sub::attention::cpu,
        cpu_oracle = crate::kernels::sub::attention::cpu_oracle,
        gpu = crate::kernels::sub::attention::gpu,
        fixture = crate::kernels::sub::attention::tiny_fixture,
        out_len = crate::kernels::sub::attention::fixture_len,
        formats: [F32],
        max_tol = 1e-2,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod wide,
        cpu = crate::kernels::sub::attention::cpu,
        cpu_oracle = crate::kernels::sub::attention::cpu_oracle,
        gpu = crate::kernels::sub::attention::gpu,
        fixture = crate::kernels::sub::attention::wide_fixture,
        out_len = crate::kernels::sub::attention::fixture_len,
        formats: [F32],
        max_tol = 2e-2,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod full_hd512,
        cpu = crate::kernels::sub::attention::cpu,
        cpu_oracle = crate::kernels::sub::attention::cpu_oracle,
        gpu = crate::kernels::sub::attention::gpu,
        fixture = crate::kernels::sub::attention::full_hd512_fixture,
        out_len = crate::kernels::sub::attention::fixture_len,
        formats: [F32],
        max_tol = 2e-2,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod sliding_hd256,
        cpu = crate::kernels::sub::attention::cpu,
        cpu_oracle = crate::kernels::sub::attention::cpu_oracle,
        gpu = crate::kernels::sub::attention::gpu,
        fixture = crate::kernels::sub::attention::sliding_hd256_fixture,
        out_len = crate::kernels::sub::attention::fixture_len,
        formats: [F32],
        max_tol = 2e-2,
        min_cos = 0.9999,
    }

    // ---- Matrix-unit (flash) path: parity vs the same CPU oracle ----

    kernel_oracle_matrix! {
        mod mma_tiny,
        cpu = crate::kernels::sub::attention::cpu,
        cpu_oracle = crate::kernels::sub::attention::cpu_oracle,
        gpu = crate::kernels::sub::attention::gpu_mma,
        fixture = crate::kernels::sub::attention::tiny_fixture,
        out_len = crate::kernels::sub::attention::fixture_len,
        formats: [F32],
        max_tol = 1e-2,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod mma_wide,
        cpu = crate::kernels::sub::attention::cpu,
        cpu_oracle = crate::kernels::sub::attention::cpu_oracle,
        gpu = crate::kernels::sub::attention::gpu_mma,
        fixture = crate::kernels::sub::attention::wide_fixture,
        out_len = crate::kernels::sub::attention::fixture_len,
        formats: [F32],
        max_tol = 2e-2,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod mma_full_hd512,
        cpu = crate::kernels::sub::attention::cpu,
        cpu_oracle = crate::kernels::sub::attention::cpu_oracle,
        gpu = crate::kernels::sub::attention::gpu_mma,
        fixture = crate::kernels::sub::attention::full_hd512_fixture,
        out_len = crate::kernels::sub::attention::fixture_len,
        formats: [F32],
        max_tol = 2e-2,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod mma_sliding_hd256,
        cpu = crate::kernels::sub::attention::cpu,
        cpu_oracle = crate::kernels::sub::attention::cpu_oracle,
        gpu = crate::kernels::sub::attention::gpu_mma,
        fixture = crate::kernels::sub::attention::sliding_hd256_fixture,
        out_len = crate::kernels::sub::attention::fixture_len,
        formats: [F32],
        max_tol = 2e-2,
        min_cos = 0.9999,
    }

    // ---- Full-layer MMA path (register-O, QG-grouped K/V): parity vs oracle ----

    kernel_oracle_matrix! {
        mod mma_full_grp2,
        cpu = crate::kernels::sub::attention::cpu,
        cpu_oracle = crate::kernels::sub::attention::cpu_oracle,
        gpu = crate::kernels::sub::attention::gpu_mma_full,
        fixture = crate::kernels::sub::attention::full_hd512_fixture,
        out_len = crate::kernels::sub::attention::fixture_len,
        formats: [F32],
        max_tol = 2e-2,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod mma_full_grp8,
        cpu = crate::kernels::sub::attention::cpu,
        cpu_oracle = crate::kernels::sub::attention::cpu_oracle,
        gpu = crate::kernels::sub::attention::gpu_mma_full,
        fixture = crate::kernels::sub::attention::full_grp8_hd512_fixture,
        out_len = crate::kernels::sub::attention::fixture_len,
        formats: [F32],
        max_tol = 2e-2,
        min_cos = 0.9999,
    }

    // ---- GQA-grouped MMA path (2 heads/tg): parity on group-size-2 fixtures ----

    kernel_oracle_matrix! {
        mod mma2_tiny,
        cpu = crate::kernels::sub::attention::cpu,
        cpu_oracle = crate::kernels::sub::attention::cpu_oracle,
        gpu = crate::kernels::sub::attention::gpu_mma2,
        fixture = crate::kernels::sub::attention::tiny_fixture,
        out_len = crate::kernels::sub::attention::fixture_len,
        formats: [F32],
        max_tol = 1e-2,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod mma2_sliding_hd256,
        cpu = crate::kernels::sub::attention::cpu,
        cpu_oracle = crate::kernels::sub::attention::cpu_oracle,
        gpu = crate::kernels::sub::attention::gpu_mma2,
        fixture = crate::kernels::sub::attention::sliding_hd256_fixture,
        out_len = crate::kernels::sub::attention::fixture_len,
        formats: [F32],
        max_tol = 2e-2,
        min_cos = 0.9999,
    }

    /// Microbench: scalar vs matrix-unit attention at model shape. Ignored (timing).
    /// Run: `cargo test --features metal --bin diffgemma-mps attn_mma_bench -- --ignored --nocapture`
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore]
    fn attn_mma_bench() {
        use crate::kernels::sub::attention::{bench_path, model_bench_fixture};
        let iters = 200usize;
        for (name, hd, kv_len, is_full) in
            [("sliding hd256", 256usize, 64u32, false), ("full hd512", 512usize, 64u32, true)]
        {
            let f = model_bench_fixture(hd, kv_len, is_full);
            let scalar = bench_path(&f, iters, 0).unwrap();
            let mma = bench_path(&f, iters, 1).unwrap();
            // mma2 (2 heads/tg) only valid for hd<=256 (sliding layers).
            if hd <= 256 {
                let mma2 = bench_path(&f, iters, 2).unwrap();
                println!(
                    "{name:>14}  scalar {scalar:7.3}  mma {mma:7.3} ({:.2}x)  mma2 {mma2:7.3} ({:.2}x)",
                    scalar / mma,
                    scalar / mma2
                );
            } else {
                println!(
                    "{name:>14}  scalar {scalar:7.3}  mma {mma:7.3} ({:.2}x)  mma2 n/a (hd>256)",
                    scalar / mma
                );
            }
        }
    }
}
