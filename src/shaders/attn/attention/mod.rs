//! All-valid GQA attention over monolithic KV cache (online softmax).

use crate::Error;
use crate::metal::{LayerOffsets, StepParams};
use crate::shaders::bf16;
use crate::shaders::cpu::attention::{self, LayerAttnParams};
use crate::shaders::gpu_common;
use crate::shaders::qk_rope_kv::AttnDims;
use crate::shaders::test_util::ElemFormat;
use crate::shaders::variant::KernelVariant;

pub const ENTRY: &str = "attention";
pub const THREADGROUP_WIDTH: usize = 64;

pub const SHADER: &str = include_str!("attention.metal");

pub const MMA_M_TILE: usize = 8;

/// GQA-grouped MMA attention (2 Q heads / threadgroup, shared K/V staging).
/// Group size 2, hd <= 256 only (sliding layers). `attention` stays the oracle.
pub const ENTRY_MMA2: &str = "attention_mma2";
pub const SHADER_MMA2: &str = include_str!("attention_mma2.metal");

/// MMA attention for full/global layers (hd=512): register-resident O + QG-grouped
/// K/V sharing. `attention` stays the oracle. QG simdgroups per threadgroup, 32
/// lanes each; (group/QG) sub-groups along grid.z.
pub const ENTRY_MMA_FULL: &str = "attention_mma_full";
pub const MMA_FULL_QG: usize = 2;
pub const SHADER_MMA_FULL: &str = include_str!("attention_mma_full.metal");

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
    for (i, v) in kvcache.iter_mut().enumerate() {
        *v = (i as f32 * 0.031).sin() * 0.3;
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
        kv_ring_mask: 0,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let q = bf16::bf16_slice_to_f32(&bf16::f32_slice_to_bf16_bits(&f.q));
    // KV cache stores f16 (see attention_device.metal kv_store).
    let kvcache: Vec<f32> = f
        .kvcache
        .iter()
        .map(|&v| crate::shaders::f16::f16_bits_to_f32(crate::shaders::f16::f32_to_f16_bits(v)))
        .collect();
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

#[cfg(target_os = "macos")]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER, ENTRY, variant)
}

/// Compile with the session KV storage format (uint function constant 4).
#[cfg(target_os = "macos")]
fn kv_fc(
    fmt: crate::shaders::kv_quant::KvFormat,
) -> ([crate::shaders::variant::FcUInt; 1], &'static str) {
    (
        [crate::shaders::variant::FcUInt {
            index: 4,
            value: fmt.code(),
        }],
        fmt.label(),
    )
}

#[cfg(target_os = "macos")]
pub fn pipeline_for_kv(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
    fmt: crate::shaders::kv_quant::KvFormat,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    let (uints, label) = kv_fc(fmt);
    ctx.compile_subkernel_ex(SHADER, ENTRY, variant, label, &[], &uints)
}

#[cfg(target_os = "macos")]
pub fn pipeline_mma2_for_kv(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
    fmt: crate::shaders::kv_quant::KvFormat,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    let (uints, label) = kv_fc(fmt);
    ctx.compile_subkernel_ex(SHADER_MMA2, ENTRY_MMA2, variant, label, &[], &uints)
}

/// Prefill variant (FC30): sliding K/V read from the f32 side ring,
/// all-float MMA. See attention_mma2.metal.
#[cfg(target_os = "macos")]
pub fn pipeline_mma2_for_kv_side(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
    fmt: crate::shaders::kv_quant::KvFormat,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    let (uints, label) = kv_fc(fmt);
    let label = format!("{label}_side");
    let bools = [crate::shaders::variant::FcBool {
        index: 30,
        value: true,
    }];
    ctx.compile_subkernel_ex(SHADER_MMA2, ENTRY_MMA2, variant, &label, &bools, &uints)
}

/// Prefill variant (FC30) for FULL layers: linear f32 side K/V,
/// all-float MMA. See attention_mma_full.metal. FC31 (QK_ILP2) is set when
/// `DGQ_ATTN_MMA_FULL_QK_ILP2` is on.
#[cfg(target_os = "macos")]
pub fn pipeline_mma_full_for_kv_side(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
    fmt: crate::shaders::kv_quant::KvFormat,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    let (uints, label) = kv_fc(fmt);
    let mut label = format!("{label}_side");
    let ilp2 = crate::flags::attn_mma_full_qk_ilp2();
    if ilp2 {
        label.push_str("_ilp2");
    }
    let mut bools = vec![crate::shaders::variant::FcBool {
        index: 30,
        value: true,
    }];
    if ilp2 {
        bools.push(crate::shaders::variant::FcBool {
            index: 31,
            value: true,
        });
    }
    ctx.compile_subkernel_ex(
        SHADER_MMA_FULL,
        ENTRY_MMA_FULL,
        variant,
        &label,
        &bools,
        &uints,
    )
}

#[cfg(target_os = "macos")]
pub fn pipeline_mma_full_for_kv(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
    fmt: crate::shaders::kv_quant::KvFormat,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    let (uints, label) = kv_fc(fmt);
    let mut label = label.to_string();
    let ilp2 = crate::flags::attn_mma_full_qk_ilp2();
    if ilp2 {
        label.push_str("_ilp2");
    }
    let bools: Vec<crate::shaders::variant::FcBool> = if ilp2 {
        vec![crate::shaders::variant::FcBool {
            index: 31,
            value: true,
        }]
    } else {
        vec![]
    };
    ctx.compile_subkernel_ex(
        SHADER_MMA_FULL,
        ENTRY_MMA_FULL,
        variant,
        &label,
        &bools,
        &uints,
    )
}

#[cfg(target_os = "macos")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLSize,
};

#[cfg(target_os = "macos")]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let buf_q = pool
        .allocate(&ctx.device, f.q.len() * 2)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_kv = pool
        .allocate(
            &ctx.device,
            (f.kvcache.len() + 8 * f.n_kv() * f.head_dim() * 2) * 2,
        )
        .ok_or(Error::Gpu("alloc"))?;
    let buf_out = pool
        .allocate(&ctx.device, f.out_len() * 2)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_layer = pool
        .allocate(&ctx.device, std::mem::size_of::<LayerOffsets>())
        .ok_or(Error::Gpu("alloc"))?;

    BufferPool::write_bf16(&buf_q, &bf16::f32_slice_to_bf16_bits(&f.q));
    {
        // f16 KV + 8 zero pad rows (direct-load kernels read whole 8-key tiles;
        // the softmax masks the tail, but the bytes must be in-bounds + finite).
        let mut bits = crate::shaders::f16::f32_slice_to_f16(&f.kvcache);
        bits.resize(bits.len() + 8 * f.n_kv() * f.head_dim() * 2, 0);
        BufferPool::write_bf16(&buf_kv, &bits);
    }
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
        kv_write_end: u32::MAX,
    };
    let dims = AttnDims {
        canvas: f.canvas as u32,
        n_q_heads: f.n_q_heads as u32,
        causal: 0,
        window: 0,
    };

    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
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

#[cfg(target_os = "macos")]
pub fn pipeline_mma2_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER_MMA2, ENTRY_MMA2, variant)
}

/// GQA-grouped MMA attention path. One threadgroup per (MMA_M_TILE-row tile, KV
/// head), 64 lanes = 2 simdgroups (one per Q head in the group). hd <= 256 only.
#[cfg(target_os = "macos")]
pub fn gpu_mma2(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_mma2_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let buf_q = pool
        .allocate(&ctx.device, f.q.len() * 2)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_kv = pool
        .allocate(
            &ctx.device,
            (f.kvcache.len() + 8 * f.n_kv() * f.head_dim() * 2) * 2,
        )
        .ok_or(Error::Gpu("alloc"))?;
    let buf_out = pool
        .allocate(&ctx.device, f.out_len() * 2)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_layer = pool
        .allocate(&ctx.device, std::mem::size_of::<LayerOffsets>())
        .ok_or(Error::Gpu("alloc"))?;

    BufferPool::write_bf16(&buf_q, &bf16::f32_slice_to_bf16_bits(&f.q));
    {
        // f16 KV + 8 zero pad rows (direct-load kernels read whole 8-key tiles;
        // the softmax masks the tail, but the bytes must be in-bounds + finite).
        let mut bits = crate::shaders::f16::f32_slice_to_f16(&f.kvcache);
        bits.resize(bits.len() + 8 * f.n_kv() * f.head_dim() * 2, 0);
        BufferPool::write_bf16(&buf_kv, &bits);
    }
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
        kv_write_end: u32::MAX,
    };
    let dims = AttnDims::new(f.canvas as u32, f.n_q_heads as u32);

    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
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
            depth: 1,
        },
        MTLSize {
            width: 64,
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

#[cfg(target_os = "macos")]
pub fn pipeline_mma_full_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER_MMA_FULL, ENTRY_MMA_FULL, variant)
}

/// Full-layer MMA attention path (hd=512). One threadgroup per (MT-row tile, KV
/// head, QG-head sub-group); QG simdgroups (32 lanes each) share K/V staging,
/// O accumulator is register-resident. Identical buffer layout to `gpu_mma`.
#[cfg(target_os = "macos")]
pub fn gpu_mma_full(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_mma_full_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let buf_q = pool
        .allocate(&ctx.device, f.q.len() * 2)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_kv = pool
        .allocate(
            &ctx.device,
            (f.kvcache.len() + 8 * f.n_kv() * f.head_dim() * 2) * 2,
        )
        .ok_or(Error::Gpu("alloc"))?;
    let buf_out = pool
        .allocate(&ctx.device, f.out_len() * 2)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_layer = pool
        .allocate(&ctx.device, std::mem::size_of::<LayerOffsets>())
        .ok_or(Error::Gpu("alloc"))?;

    BufferPool::write_bf16(&buf_q, &bf16::f32_slice_to_bf16_bits(&f.q));
    {
        // f16 KV + 8 zero pad rows (direct-load kernels read whole 8-key tiles;
        // the softmax masks the tail, but the bytes must be in-bounds + finite).
        let mut bits = crate::shaders::f16::f32_slice_to_f16(&f.kvcache);
        bits.resize(bits.len() + 8 * f.n_kv() * f.head_dim() * 2, 0);
        BufferPool::write_bf16(&buf_kv, &bits);
    }
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
        kv_write_end: u32::MAX,
    };
    let dims = AttnDims::new(f.canvas as u32, f.n_q_heads as u32);
    let group = f.n_q_heads / f.n_kv();
    // kv-block state scratch (single-block dispatch here: first+last).
    let buf_state = pool
        .allocate(&ctx.device, f.n_q_heads * f.canvas * (2 + 512) * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let blk = [0u32, f.kv_len + f.canvas as u32, 1, 1];

    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_q), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&buf_kv), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&buf_out), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&buf_layer), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&buf_state), 0, 8);
    }
    gpu_common::set_bytes(&enc, &params, 4);
    gpu_common::set_bytes(&enc, &dims, 5);
    gpu_common::set_bytes(&enc, &blk, 7);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        // One tg per (query tile, kv head, Q head); the QG simdgroups split
        // head_dim, so depth is the full GQA group.
        MTLSize {
            width: f.canvas.div_ceil(MMA_M_TILE),
            height: f.n_kv(),
            depth: group,
        },
        MTLSize {
            width: MMA_FULL_QG * 32,
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

/// Model-shaped attention fixture (canvas=256, 16 Q / 8 KV heads) for benching.
pub fn model_bench_fixture(hd: usize, kv_len: u32, is_full: bool) -> Fixture {
    model_attn_fixture(256, 16, 8, hd, kv_len, is_full)
}

/// Time `iters` back-to-back dispatches of one attention path in a single command
/// buffer; returns mean ms/dispatch (GPU wall, compile + alloc excluded).
/// path: 0 = scalar `attention`, 1 = `attention_mma` (1 head/tg), 2 = `attention_mma2`
/// (2 heads/tg, GQA-shared K/V).
#[cfg(target_os = "macos")]
pub fn bench_path(f: &Fixture, iters: usize, path: u8) -> Result<f64, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use std::time::Instant;

    let ctx = MetalContext::new()?;
    let prod = KernelVariant::PRODUCTION;
    let pipeline = match path {
        2 => pipeline_mma2_for(&ctx, prod)?,
        3 => pipeline_mma_full_for(&ctx, prod)?,
        _ => pipeline_for(&ctx, prod)?,
    };
    let mut pool = BufferPool::new();
    let buf_q = pool
        .allocate(&ctx.device, f.q.len() * 2)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_kv = pool
        .allocate(
            &ctx.device,
            (f.kvcache.len() + 8 * f.n_kv() * f.head_dim() * 2) * 2,
        )
        .ok_or(Error::Gpu("alloc"))?;
    let buf_out = pool
        .allocate(&ctx.device, f.out_len() * 2)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_layer = pool
        .allocate(&ctx.device, std::mem::size_of::<LayerOffsets>())
        .ok_or(Error::Gpu("alloc"))?;
    BufferPool::write_bf16(&buf_q, &bf16::f32_slice_to_bf16_bits(&f.q));
    {
        // f16 KV + 8 zero pad rows (direct-load kernels read whole 8-key tiles;
        // the softmax masks the tail, but the bytes must be in-bounds + finite).
        let mut bits = crate::shaders::f16::f32_slice_to_f16(&f.kvcache);
        bits.resize(bits.len() + 8 * f.n_kv() * f.head_dim() * 2, 0);
        BufferPool::write_bf16(&buf_kv, &bits);
    }
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
        kv_write_end: u32::MAX,
    };
    let dims = AttnDims::new(f.canvas as u32, f.n_q_heads as u32);
    let (grid, tpg) = match path {
        2 => (
            MTLSize {
                width: f.canvas.div_ceil(MMA_M_TILE),
                height: f.n_kv(),
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        ),
        3 => (
            MTLSize {
                width: f.canvas.div_ceil(MMA_M_TILE),
                height: f.n_kv(),
                depth: f.n_q_heads / f.n_kv(),
            },
            MTLSize {
                width: MMA_FULL_QG * 32,
                height: 1,
                depth: 1,
            },
        ),
        _ => (
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
        ),
    };

    // kv-block state scratch for the mma_full path (single-block: first+last).
    let buf_state = pool
        .allocate(&ctx.device, f.n_q_heads * f.canvas * (2 + 512) * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let blk = [0u32, f.kv_len + f.canvas as u32, 1, 1];

    // 1 warm-up + several timed rounds (each one command buffer with `iters`
    // dispatches); report the MIN ms/dispatch to factor out GPU clock ramp/throttle.
    let mut best = f64::INFINITY;
    for round in 0..6 {
        let t = Instant::now();
        let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
        let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
        enc.setComputePipelineState(&pipeline.pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&buf_q), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&buf_kv), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&buf_out), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&buf_layer), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&buf_state), 0, 8);
        }
        gpu_common::set_bytes(&enc, &params, 4);
        gpu_common::set_bytes(&enc, &dims, 5);
        gpu_common::set_bytes(&enc, &blk, 7);
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

/// Manifest registration; collected in common/manifest.rs::MANIFEST.
pub const SPEC: crate::shaders::manifest::KernelSpec = crate::shaders::manifest::KernelSpec {
    name: "attention",
    entry: "attention",
    quant_formats: &[crate::shaders::variant::QuantFormat::Q4Affine],
    fc: &[],
    variants: crate::shaders::manifest::KernelVariants::Elementwise,
};

#[cfg(test)]
mod tests {
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::shaders::attention::cpu,
        cpu_oracle = crate::shaders::attention::cpu_oracle,
        gpu = crate::shaders::attention::gpu,
        fixture = crate::shaders::attention::tiny_fixture,
        out_len = crate::shaders::attention::fixture_len,
        formats: [F32],
        max_tol = 1e-2,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod wide,
        cpu = crate::shaders::attention::cpu,
        cpu_oracle = crate::shaders::attention::cpu_oracle,
        gpu = crate::shaders::attention::gpu,
        fixture = crate::shaders::attention::wide_fixture,
        out_len = crate::shaders::attention::fixture_len,
        formats: [F32],
        max_tol = 2e-2,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod full_hd512,
        cpu = crate::shaders::attention::cpu,
        cpu_oracle = crate::shaders::attention::cpu_oracle,
        gpu = crate::shaders::attention::gpu,
        fixture = crate::shaders::attention::full_hd512_fixture,
        out_len = crate::shaders::attention::fixture_len,
        formats: [F32],
        max_tol = 2e-2,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod sliding_hd256,
        cpu = crate::shaders::attention::cpu,
        cpu_oracle = crate::shaders::attention::cpu_oracle,
        gpu = crate::shaders::attention::gpu,
        fixture = crate::shaders::attention::sliding_hd256_fixture,
        out_len = crate::shaders::attention::fixture_len,
        formats: [F32],
        max_tol = 2e-2,
        min_cos = 0.9999,
    }

    // ---- Matrix-unit (flash) paths: parity vs the same CPU oracle ----

    // ---- Full-layer MMA path (register-O, QG-grouped K/V): parity vs oracle ----

    kernel_oracle_matrix! {
        mod mma_full_grp2,
        cpu = crate::shaders::attention::cpu,
        cpu_oracle = crate::shaders::attention::cpu_oracle,
        gpu = crate::shaders::attention::gpu_mma_full,
        fixture = crate::shaders::attention::full_hd512_fixture,
        out_len = crate::shaders::attention::fixture_len,
        formats: [F32],
        max_tol = 2e-2,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod mma_full_grp8,
        cpu = crate::shaders::attention::cpu,
        cpu_oracle = crate::shaders::attention::cpu_oracle,
        gpu = crate::shaders::attention::gpu_mma_full,
        fixture = crate::shaders::attention::full_grp8_hd512_fixture,
        out_len = crate::shaders::attention::fixture_len,
        formats: [F32],
        max_tol = 2e-2,
        min_cos = 0.9999,
    }

    // ---- GQA-grouped MMA path (2 heads/tg): parity on group-size-2 fixtures ----

    kernel_oracle_matrix! {
        mod mma2_tiny,
        cpu = crate::shaders::attention::cpu,
        cpu_oracle = crate::shaders::attention::cpu_oracle,
        gpu = crate::shaders::attention::gpu_mma2,
        fixture = crate::shaders::attention::tiny_fixture,
        out_len = crate::shaders::attention::fixture_len,
        formats: [F32],
        max_tol = 1e-2,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod mma2_sliding_hd256,
        cpu = crate::shaders::attention::cpu,
        cpu_oracle = crate::shaders::attention::cpu_oracle,
        gpu = crate::shaders::attention::gpu_mma2,
        fixture = crate::shaders::attention::sliding_hd256_fixture,
        out_len = crate::shaders::attention::fixture_len,
        formats: [F32],
        max_tol = 2e-2,
        min_cos = 0.9999,
    }

    /// Microbench: scalar vs matrix-unit attention at model shape. Ignored (timing).
    /// Run: `cargo test --bin diffgemma attn_mma_bench -- --ignored --nocapture`
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore]
    fn attn_mma_bench() {
        use crate::shaders::attention::{bench_path, model_bench_fixture};
        let iters = 50usize;
        for kv_len in [64u32, 512, 1024, 8192, 32768] {
            for (name, hd, is_full) in [
                ("sliding hd256", 256usize, false),
                ("full hd512", 512usize, true),
            ] {
                // Sliding layers are window-clamped (1024) in production —
                // benching them unwindowed at long kv is meaningless.
                if !is_full && kv_len > 1024 {
                    continue;
                }
                let f = model_bench_fixture(hd, kv_len, is_full);
                // Scalar at long kv is minutes per measurement and adds no
                // information (linear serial loop); report the MMA paths only.
                let scalar = if kv_len <= 8192 {
                    bench_path(&f, iters, 0).unwrap()
                } else {
                    f64::NAN
                };
                if hd <= 256 {
                    // mma2 (2 heads/tg) only valid for hd<=256 (sliding layers).
                    let mma2 = bench_path(&f, iters, 2).unwrap();
                    println!(
                        "kv{kv_len:>5} {name:>14}  scalar {scalar:7.3}  mma2 {mma2:7.3} ({:.2}x)",
                        scalar / mma2
                    );
                } else {
                    let mma_full = bench_path(&f, iters, 3).unwrap();
                    println!(
                        "kv{kv_len:>5} {name:>14}  scalar {scalar:7.3}  mma_full {mma_full:7.3} ({:.2}x)",
                        scalar / mma_full
                    );
                }
            }
        }
    }

    /// Production-shape mma2 vs flash bench (M=1024, hd=256, 16 q_heads,
    /// 8 kv_heads) at the sliding window (kv=1024). Backs out the TF/s to
    /// check whether mma2 is at the half-MMA wall (~3.8 TF/s) or leaving
    /// compute on the table — and whether flash's barrier-free PV loop wins
    /// once both paths honor the same window.
    /// Run: `cargo test --release mma2_prod_bench -- --ignored --nocapture`
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore]
    fn mma2_prod_bench() {
        use crate::shaders::attention::{bench_path, model_attn_fixture};
        use crate::shaders::attn::attention_flash::{bench_flash, bench_flash_window};
        let iters = 50usize;
        let canvas = 1024usize;
        let nq = 16usize;
        let nkv = 8usize;
        let hd = 256usize;
        let kv_len = 1024u32; // sliding window
        let f = model_attn_fixture(canvas, nq, nkv, hd, kv_len, false);
        let mma2 = bench_path(&f, iters, 2).expect("mma2 bench");
        // mma2 scans window=1024 keys/row (so W=kv_len here). flash without
        // window scans t_total=kv_len+canvas=2048. flash with window=1024
        // should match mma2's work.
        let t = kv_len as usize + canvas;
        let mma2_flops = 2.0 * (nq * canvas * kv_len as usize * hd * 2) as f64;
        let mma2_tf_s = mma2_flops / (mma2 * 1e-3) / 1e12;
        println!(
            "mma2 prod (M={canvas}, hd={hd}, nq={nq}, nkv={nkv}, kv={kv_len}): {mma2:.3} ms, {mma2_tf_s:.2} TF/s ({:.1}% of 3.8 wall)",
            100.0 * mma2_tf_s / 3.8
        );
        let flash_nowin = bench_flash(&f, iters, 16, true).expect("flash no-window");
        let flash_nowin_flops = 2.0 * (nq * canvas * t * hd * 2) as f64;
        let flash_nowin_tf = flash_nowin_flops / (flash_nowin * 1e-3) / 1e12;
        println!(
            "flash bq16 (no window, scans t={t}): {flash_nowin:.3} ms, {flash_nowin_tf:.2} TF/s ({:.1}% wall)  — vs mma2 {:.2}x",
            100.0 * flash_nowin_tf / 3.8,
            mma2 / flash_nowin,
        );
        let flash_win = bench_flash_window(&f, iters, 16, kv_len, true).expect("flash window");
        let flash_win_flops = 2.0 * (nq * canvas * kv_len as usize * hd * 2) as f64;
        let flash_win_tf = flash_win_flops / (flash_win * 1e-3) / 1e12;
        println!(
            "flash bq16 (window={kv_len}):            {flash_win:.3} ms, {flash_win_tf:.2} TF/s ({:.1}% wall)  — vs mma2 {:.2}x",
            100.0 * flash_win_tf / 3.8,
            mma2 / flash_win,
        );
    }
}
