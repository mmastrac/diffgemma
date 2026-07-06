//! Q/K RMSNorm + RoPE + KV cache write (monolithic step path).

use super::bf16;
use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::kernels::cpu::attention::{self, LayerAttnParams};
use crate::metal::{LayerOffsets, StepParams};
use crate::safetensors::Error;

pub const ENTRY: &str = "qk_rope_kv";

const SHADER: &str = shader_include::include_metal!("kernels/qk_rope_kv.metal");

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AttnDims {
    pub canvas: u32,
    pub n_q_heads: u32,
    /// 0 = bidirectional all-valid (denoise); 1 = causal (prefill).
    pub causal: u32,
    /// Sliding-window size (1024 for sliding layers, 0 = unwindowed/full).
    /// Denoise (causal=0): canvas attends encoder keys [max(0, kv_len-(window-1)), kv_len)
    /// + all canvas (MLX `_make_decoder_masks` semantics). Causal (prefill):
    /// query at pos q attends [max(0, q-(window-1)), q] (engine CausalSliding).
    /// No-op while kv_len+tok < window, so short contexts are bit-identical.
    pub window: u32,
}

impl AttnDims {
    pub fn new(canvas: u32, n_q_heads: u32) -> Self {
        Self { canvas, n_q_heads, causal: 0, window: 0 }
    }
}

#[derive(Debug, Clone)]
pub struct Fixture {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub kvcache: Vec<f32>,
    pub q_norm_w: Vec<f32>,
    pub k_norm_w: Vec<f32>,
    pub layer: LayerAttnParams,
    pub canvas: usize,
    pub n_q_heads: usize,
    pub kv_len: u32,
    pub v_proj: u64,
}

impl Fixture {
    pub fn head_dim(&self) -> usize {
        self.layer.head_dim as usize
    }

    pub fn n_kv(&self) -> usize {
        self.layer.n_kv_heads as usize
    }

    pub fn qk_grid_y(&self) -> usize {
        self.n_q_heads + 2 * self.n_kv()
    }

    pub fn kv_slots(&self) -> usize {
        (self.kv_len as usize + self.canvas) * self.n_kv() * self.head_dim() * 2
    }

    pub fn out_len(&self) -> usize {
        self.q.len() + self.k.len() + self.canvas * self.n_kv() * self.head_dim() * 2
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
    let q_len = canvas * n_q_heads * hd;
    let kv_len_flat = canvas * n_kv * hd;
    Fixture {
        q: (0..q_len)
            .map(|i| (i as f32 * 0.13).sin() * 0.5)
            .collect(),
        k: (0..kv_len_flat)
            .map(|i| (i as f32 * 0.17).cos() * 0.4)
            .collect(),
        v: (0..kv_len_flat)
            .map(|i| (i as f32 * 0.11).sin() * 0.3)
            .collect(),
        kvcache: vec![0.0; (kv_len as usize + canvas) * n_kv * hd * 2],
        q_norm_w: vec![1.0; hd],
        k_norm_w: vec![1.0; hd],
        layer: LayerAttnParams {
            head_dim: hd as u32,
            n_kv_heads: n_kv as u32,
            is_full: false,
            v_proj: 1,
            kv_region: 0,
            q_norm_off: 0,
            k_norm_off: (hd * 2) as u64,
        },
        canvas,
        n_q_heads,
        kv_len,
        v_proj: 1,
    }
}

pub fn full_layer_fixture(_: ElemFormat) -> Fixture {
    let mut f = tiny_fixture(ElemFormat::F32);
    f.layer.is_full = true;
    f.layer.head_dim = 64;
    f.v_proj = 1;
    f.layer.v_proj = 1;
    let hd = f.head_dim();
    let canvas = f.canvas;
    let n_q = f.n_q_heads;
    let n_kv = f.n_kv();
    f.q_norm_w = vec![1.0; hd];
    f.k_norm_w = vec![1.0; hd];
    f.layer.k_norm_off = (hd * 2) as u64;
    f.q = (0..canvas * n_q * hd)
        .map(|i| (i as f32 * 0.07).sin() * 0.6)
        .collect();
    f.k = (0..canvas * n_kv * hd)
        .map(|i| (i as f32 * 0.09).cos() * 0.5)
        .collect();
    f.v = (0..canvas * n_kv * hd)
        .map(|i| (i as f32 * 0.08).sin() * 0.45)
        .collect();
    f.kvcache = vec![0.0; (f.kv_len as usize + canvas) * n_kv * hd * 2];
    f
}

/// Full-attention layer with no `v_proj` (V aliases raw k_proj, per MLX).
pub fn full_attn_v_alias_fixture(fmt: ElemFormat) -> Fixture {
    let mut f = full_layer_fixture(fmt);
    f.v_proj = 0;
    f.layer.v_proj = 0;
    f
}

/// Full-attention hd=512 (production full layers) with V-alias.
pub fn full_hd512_v_alias_fixture(_fmt: ElemFormat) -> Fixture {
    let canvas = 1usize;
    let n_q_heads = 16usize;
    let n_kv = 2usize;
    let hd = 512usize;
    let kv_len = 22u32;
    let q_len = canvas * n_q_heads * hd;
    let kv_len_flat = canvas * n_kv * hd;
    Fixture {
        q: (0..q_len)
            .map(|i| (i as f32 * 0.031).sin() * 0.55)
            .collect(),
        k: (0..kv_len_flat)
            .map(|i| (i as f32 * 0.029).cos() * 0.48)
            .collect(),
        v: vec![0.0; kv_len_flat],
        kvcache: vec![0.0; (kv_len as usize + canvas) * n_kv * hd * 2],
        q_norm_w: (0..hd).map(|i| 1.0 + (i as f32 * 0.0001)).collect(),
        k_norm_w: (0..hd).map(|i| 1.0 - (i as f32 * 0.00005)).collect(),
        layer: LayerAttnParams {
            head_dim: hd as u32,
            n_kv_heads: n_kv as u32,
            is_full: true,
            v_proj: 0,
            kv_region: 0,
            q_norm_off: 0,
            k_norm_off: (hd * 2) as u64,
        },
        canvas,
        n_q_heads,
        kv_len,
        v_proj: 0,
    }
}

fn layer_offsets(f: &Fixture) -> LayerOffsets {
    LayerOffsets {
        input_ln: 0,
        q_proj: 0,
        q_norm: f.layer.q_norm_off,
        k_proj: 0,
        k_norm: f.layer.k_norm_off,
        v_proj: f.v_proj,
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

fn pack_out(q: &[f32], k: &[f32], kvcache: &[f32], f: &Fixture) -> Vec<f32> {
    let hd = f.head_dim();
    let nkv = f.n_kv();
    let mut out = Vec::with_capacity(f.out_len());
    out.extend_from_slice(q);
    out.extend_from_slice(k);
    let base = f.layer.kv_region as usize / 2;
    for tok in 0..f.canvas {
        let pos = f.kv_len as usize + tok;
        let off = base + pos * nkv * hd * 2;
        out.extend_from_slice(&kvcache[off..off + nkv * hd * 2]);
    }
    out
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let mut q = bf16::bf16_slice_to_f32(&bf16::f32_slice_to_bf16_bits(&f.q));
    let mut k = bf16::bf16_slice_to_f32(&bf16::f32_slice_to_bf16_bits(&f.k));
    let mut v = bf16::bf16_slice_to_f32(&bf16::f32_slice_to_bf16_bits(&f.v));
    let mut kvcache = f.kvcache.clone();
    attention::qk_rope_kv(
        &mut q,
        &mut k,
        &mut v,
        &mut kvcache,
        &f.q_norm_w,
        &f.k_norm_w,
        f.layer,
        f.canvas,
        f.n_q_heads,
        f.kv_len,
    );
    for v in q.iter_mut().chain(k.iter_mut()) {
        *v = bf16::store_bf16_round_half(*v);
    }
    // KV cache stores f16 (attention_device.metal kv_store).
    for v in kvcache.iter_mut() {
        *v = crate::kernels::sub::f16::f16_bits_to_f32(crate::kernels::sub::f16::f32_to_f16_bits(*v));
    }
    pack_out(&q, &k, &kvcache, f)
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
pub fn pipeline_for_kv(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
    kv_q8: bool,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    let bools = [crate::kernels::sub::variant::FcBool { index: 4, value: kv_q8 }];
    ctx.compile_subkernel_ex(
        SHADER,
        ENTRY,
        variant,
        if kv_q8 { "kvq8" } else { "kvf16" },
        &bools,
        &[],
    )
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
    let _hd = f.head_dim();
    let buf_q = pool
        .allocate(&ctx.device, f.q.len() * 2)
        .ok_or(Error::Format("alloc"))?;
    let buf_k = pool
        .allocate(&ctx.device, f.k.len() * 2)
        .ok_or(Error::Format("alloc"))?;
    let buf_v = pool
        .allocate(&ctx.device, f.v.len() * 2)
        .ok_or(Error::Format("alloc"))?;
    let buf_kv = pool
        .allocate(&ctx.device, f.kvcache.len() * 2)
        .ok_or(Error::Format("alloc"))?;
    let blob = [
        bf16::pack_bf16_slice(&f.q_norm_w),
        bf16::pack_bf16_slice(&f.k_norm_w),
    ]
    .concat();
    let buf_blob = pool
        .allocate(&ctx.device, blob.len())
        .ok_or(Error::Format("alloc"))?;
    let buf_layer = pool
        .allocate(&ctx.device, std::mem::size_of::<LayerOffsets>())
        .ok_or(Error::Format("alloc"))?;

    BufferPool::write_bf16(&buf_q, &bf16::f32_slice_to_bf16_bits(&f.q));
    BufferPool::write_bf16(&buf_k, &bf16::f32_slice_to_bf16_bits(&f.k));
    BufferPool::write_bf16(&buf_v, &bf16::f32_slice_to_bf16_bits(&f.v));
    // KV cache stores f16 (attention_device.metal kv_store); q/k/v stay bf16 (arena).
    BufferPool::write_bf16(&buf_kv, &crate::kernels::sub::f16::f32_slice_to_f16(&f.kvcache));
    BufferPool::write_bytes(&buf_blob, &blob);
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
        window: 0,
    };

    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_q), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&buf_k), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&buf_v), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&buf_kv), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&buf_blob), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&buf_layer), 0, 5);
    }
    gpu_common::set_bytes(&enc, &params, 6);
    gpu_common::set_bytes(&enc, &dims, 7);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: f.canvas,
            height: f.qk_grid_y(),
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut q = vec![0.0f32; f.q.len()];
    let mut k = vec![0.0f32; f.k.len()];
    let mut kvcache = vec![0.0f32; f.kvcache.len()];
    let read_half = |buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>, out: &mut [f32]| {
        let ptr = buf.contents().as_ptr() as *const u16;
        for (i, o) in out.iter_mut().enumerate() {
            *o = bf16::bf16_bits_to_f32(unsafe { *ptr.add(i) });
        }
    };
    read_half(&buf_q, &mut q);
    read_half(&buf_k, &mut k);
    // kvcache is f16, not bf16.
    {
        let ptr = buf_kv.contents().as_ptr() as *const u16;
        for (i, o) in kvcache.iter_mut().enumerate() {
            *o = crate::kernels::sub::f16::f16_bits_to_f32(unsafe { *ptr.add(i) });
        }
    }
    Ok(pack_out(&q, &k, &kvcache, f))
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn gpu(_: &Fixture, _: KernelVariant) -> Result<Vec<f32>, Error> {
    Err(Error::Format("Metal unavailable"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::kernels::sub::qk_rope_kv::cpu,
        cpu_oracle = crate::kernels::sub::qk_rope_kv::cpu_oracle,
        gpu = crate::kernels::sub::qk_rope_kv::gpu,
        fixture = crate::kernels::sub::qk_rope_kv::tiny_fixture,
        out_len = crate::kernels::sub::qk_rope_kv::fixture_len,
        formats: [F32],
        max_tol = 1e-2,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod full_layer,
        cpu = crate::kernels::sub::qk_rope_kv::cpu,
        cpu_oracle = crate::kernels::sub::qk_rope_kv::cpu_oracle,
        gpu = crate::kernels::sub::qk_rope_kv::gpu,
        fixture = crate::kernels::sub::qk_rope_kv::full_layer_fixture,
        out_len = crate::kernels::sub::qk_rope_kv::fixture_len,
        formats: [F32],
        max_tol = 1e-2,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod full_attn_v_alias,
        cpu = crate::kernels::sub::qk_rope_kv::cpu,
        cpu_oracle = crate::kernels::sub::qk_rope_kv::cpu_oracle,
        gpu = crate::kernels::sub::qk_rope_kv::gpu,
        fixture = crate::kernels::sub::qk_rope_kv::full_attn_v_alias_fixture,
        out_len = crate::kernels::sub::qk_rope_kv::fixture_len,
        formats: [F32],
        max_tol = 1e-2,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod full_hd512_v_alias,
        cpu = crate::kernels::sub::qk_rope_kv::cpu,
        cpu_oracle = crate::kernels::sub::qk_rope_kv::cpu_oracle,
        gpu = crate::kernels::sub::qk_rope_kv::gpu,
        fixture = crate::kernels::sub::qk_rope_kv::full_hd512_v_alias_fixture,
        out_len = crate::kernels::sub::qk_rope_kv::fixture_len,
        formats: [F32],
        max_tol = 2e-2,
        min_cos = 0.999,
    }
}
