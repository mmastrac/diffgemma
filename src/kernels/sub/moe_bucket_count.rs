//! Zero expert counts before MoE bucketing.

use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::metal::{RouteScratch, N_EXPERTS};
use crate::safetensors::Error;

pub const ENTRY: &str = "moe_bucket_count";

const SHADER: &str = shader_include::include_metal!("kernels/moe_bucket_count.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub n_experts: usize,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.n_experts
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    Fixture { n_experts: 8 }
}

pub fn production_fixture(_: ElemFormat) -> Fixture {
    Fixture { n_experts: N_EXPERTS }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    vec![0.0; f.n_experts]
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
    let buf_route = pool
        .allocate(&ctx.device, std::mem::size_of::<RouteScratch>())
        .ok_or(Error::Format("alloc"))?;
    let mut scratch = RouteScratch {
        weight: [[0; crate::metal::TOP_K]; crate::metal::CANVAS],
        expert: [[0; crate::metal::TOP_K]; crate::metal::CANVAS],
        count: [1; N_EXPERTS],
        row_start: [0; N_EXPERTS + 1],
        num_slots: 99,
        num_active_experts: 0,
        active_expert: [0; crate::metal::N_EXPERTS],
        token_list: [0; crate::metal::CANVAS * crate::metal::TOP_K],
        slot_list: [0; crate::metal::CANVAS * crate::metal::TOP_K],
        token_slot: [[0; crate::metal::TOP_K]; crate::metal::CANVAS],
        block_expert: [0; crate::metal::MOE_MAX_BLOCKS],
        block_row0: [0; crate::metal::MOE_MAX_BLOCKS],
        num_blocks: 0,
    };
    BufferPool::write_bytes(
        &buf_route,
        unsafe {
            std::slice::from_raw_parts(
                &scratch as *const RouteScratch as *const u8,
                std::mem::size_of::<RouteScratch>(),
            )
        },
    );

    let n_experts = f.n_experts as u32;
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_route), 0, 0);
    }
    gpu_common::set_bytes(&enc, &n_experts, 1);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: f.n_experts,
            height: 1,
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

    scratch = unsafe { std::ptr::read(buf_route.contents().as_ptr() as *const RouteScratch) };
    Ok(scratch.count[..f.n_experts].iter().map(|&v| v as f32).collect())
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
        cpu = crate::kernels::sub::moe_bucket_count::cpu,
        cpu_oracle = crate::kernels::sub::moe_bucket_count::cpu_oracle,
        gpu = crate::kernels::sub::moe_bucket_count::gpu,
        fixture = crate::kernels::sub::moe_bucket_count::tiny_fixture,
        out_len = crate::kernels::sub::moe_bucket_count::fixture_len,
        formats: [F32],
        max_tol = 0.0,
        min_cos = 1.0,
    }
}
