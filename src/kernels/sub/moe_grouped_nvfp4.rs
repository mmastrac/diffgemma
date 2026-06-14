//! Grouped MoE NVFP4: per-expert gate||up → GELU×up → down, weighted scatter.

use super::f16;
use super::gpu_common;
use super::moe_grouped::{tiny_fixture as q4_tiny, wide_fixture as q4_wide, Fixture, THREADGROUP_WIDTH};
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::dgq::layout::nvfp4_matrix_bytes;
use crate::dgq::nvfp4::quantize_f32_matrix_nvfp4;
use crate::kernels::cpu::moe_grouped::moe_grouped_nvfp4;
use crate::metal::{LayerOffsets, RouteScratch};
use crate::safetensors::Error;

pub const ENTRY: &str = "moe_grouped";

const SHADER: &str = concat!(
    include_str!("../../../shaders/include/fc_axes.metal"),
    include_str!("../../../shaders/include/common.metal"),
    include_str!("../../../shaders/include/dequant.metal"),
    include_str!("../../../shaders/include/activations.metal"),
    include_str!("../../../shaders/include/attention_device.metal"),
    include_str!("../../../shaders/include/moe_router_device.metal"),
    include_str!("../../../shaders/include/moe_grouped_device.metal"),
    include_str!("../../../shaders/kernels/moe_grouped.metal"),
);

impl Fixture {
    fn gate_up_nvfp4(&self) -> Vec<u8> {
        quantize_stack_nvfp4(&self.gate_up_f32, self.n_experts, self.moe_ff * 2, self.hidden)
    }

    fn down_nvfp4(&self) -> Vec<u8> {
        quantize_stack_nvfp4(&self.down_f32, self.n_experts, self.hidden, self.moe_ff)
    }

    pub fn nvfp4_blob(&self) -> Vec<u8> {
        let mut blob = self.gate_up_nvfp4();
        blob.extend_from_slice(&self.down_nvfp4());
        blob
    }

    pub fn nvfp4_layer_offsets(&self) -> LayerOffsets {
        let gu = 0u64;
        let dn = self.gate_up_nvfp4().len() as u64;
        LayerOffsets {
            input_ln: 0,
            q_proj: 0,
            q_norm: 0,
            k_proj: 0,
            k_norm: 0,
            v_proj: 0,
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
            experts_gate_up: gu,
            experts_down: dn,
            post_ff_ln_2: 0,
            post_ff_ln: 0,
            layer_scalar: 0,
            kv_region: 0,
            head_dim: 0,
            n_kv_heads: 0,
            is_full: 0,
            _pad: 0,
        }
    }
}

fn quantize_stack_nvfp4(rows: &[f32], experts: usize, out_dim: usize, in_dim: usize) -> Vec<u8> {
    let per = nvfp4_matrix_bytes(out_dim, in_dim);
    let mut dst = vec![0u8; experts * per];
    for e in 0..experts {
        let src_off = e * out_dim * in_dim;
        quantize_f32_matrix_nvfp4(
            &rows[src_off..src_off + out_dim * in_dim],
            out_dim,
            in_dim,
            &mut dst[e * per..(e + 1) * per],
        );
    }
    dst
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

pub fn tiny_fixture(fmt: ElemFormat) -> Fixture {
    q4_tiny(fmt)
}

pub fn wide_fixture(fmt: ElemFormat) -> Fixture {
    q4_wide(fmt)
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let moe_in: Vec<f32> = f16::f16_slice_to_f32(&f16::f32_slice_to_f16(&f.moe_in));
    let mut out = vec![0.0f32; f.out_len()];
    moe_grouped_nvfp4(
        &mut out,
        &moe_in,
        &f.gate_up_nvfp4(),
        &f.down_nvfp4(),
        &f.grouped_route(),
        f.dims().into(),
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
    let mut v = variant;
    v.quant_format = super::QuantFormat::NvFp4;
    super::moe_grouped::pipeline_for(ctx, v)
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
    let moe_in_f16 = f16::f32_slice_to_f16(&f.moe_in);
    let buf_in = pool
        .allocate(&ctx.device, moe_in_f16.len() * 2)
        .ok_or(Error::Format("alloc"))?;
    let buf_out = pool
        .allocate(&ctx.device, f.out_len() * 4)
        .ok_or(Error::Format("alloc"))?;
    let blob = f.nvfp4_blob();
    let buf_blob = pool
        .allocate(&ctx.device, blob.len())
        .ok_or(Error::Format("alloc"))?;
    let buf_layer = pool
        .allocate(&ctx.device, std::mem::size_of::<LayerOffsets>())
        .ok_or(Error::Format("alloc"))?;
    let buf_route = pool
        .allocate(&ctx.device, std::mem::size_of::<RouteScratch>())
        .ok_or(Error::Format("alloc"))?;

    BufferPool::write_bf16(&buf_in, &moe_in_f16);
    BufferPool::write_f32(&buf_out, &vec![0.0f32; f.out_len()]);
    BufferPool::write_bytes(&buf_blob, &blob);
    let layer = f.nvfp4_layer_offsets();
    BufferPool::write_bytes(
        &buf_layer,
        unsafe {
            std::slice::from_raw_parts(
                &layer as *const LayerOffsets as *const u8,
                std::mem::size_of::<LayerOffsets>(),
            )
        },
    );
    let route = f.route_scratch();
    BufferPool::write_bytes(
        &buf_route,
        unsafe {
            std::slice::from_raw_parts(
                &route as *const RouteScratch as *const u8,
                std::mem::size_of::<RouteScratch>(),
            )
        },
    );

    let dims = f.dims();
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_in), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&buf_out), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&buf_blob), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&buf_layer), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&buf_route), 0, 4);
    }
    gpu_common::set_bytes(&enc, &dims, 5);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: f.canvas,
            height: f.n_experts,
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
    BufferPool::read_f32(&buf_out, &mut out);
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
        cpu = crate::kernels::sub::moe_grouped_nvfp4::cpu,
        cpu_oracle = crate::kernels::sub::moe_grouped_nvfp4::cpu_oracle,
        gpu = crate::kernels::sub::moe_grouped_nvfp4::gpu,
        fixture = crate::kernels::sub::moe_grouped_nvfp4::tiny_fixture,
        out_len = crate::kernels::sub::moe_grouped_nvfp4::fixture_len,
        formats: [F32],
        max_tol = 5e-2,
        min_cos = 0.999,
    }

    kernel_oracle_matrix! {
        mod wide,
        cpu = crate::kernels::sub::moe_grouped_nvfp4::cpu,
        cpu_oracle = crate::kernels::sub::moe_grouped_nvfp4::cpu_oracle,
        gpu = crate::kernels::sub::moe_grouped_nvfp4::gpu,
        fixture = crate::kernels::sub::moe_grouped_nvfp4::wide_fixture,
        out_len = crate::kernels::sub::moe_grouped_nvfp4::fixture_len,
        formats: [F32],
        max_tol = 8e-2,
        min_cos = 0.999,
    }
}
