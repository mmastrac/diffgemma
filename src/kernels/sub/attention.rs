//! All-valid GQA attention over monolithic KV cache (online softmax).

use super::f16;
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
            .map(|i| ((i as f32 * 0.11).sin() * 0.6))
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
    let q = f16::f16_slice_to_f32(&f16::f32_slice_to_f16(&f.q));
    let kvcache = f16::f16_slice_to_f32(&f16::f32_slice_to_f16(&f.kvcache));
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

    BufferPool::write_bf16(&buf_q, &f16::f32_slice_to_f16(&f.q));
    BufferPool::write_bf16(&buf_kv, &f16::f32_slice_to_f16(&f.kvcache));
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
    };
    let dims = AttnDims {
        canvas: f.canvas as u32,
        n_q_heads: f.n_q_heads as u32,
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
    let ptr = unsafe { buf_out.contents().as_ptr() as *const u16 };
    for (i, o) in out.iter_mut().enumerate() {
        *o = f16::f16_bits_to_f32(unsafe { *ptr.add(i) });
    }
    Ok(out)
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
}
