//! Weighted scatter of batched MoE expert outputs to canvas rows.

use super::bf16;
use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::metal::RouteScratch;
use crate::safetensors::Error;

pub const ENTRY: &str = "moe_scatter_weighted";

const SHADER: &str = shader_include::include_metal!("kernels/moe_scatter_weighted.metal");

#[derive(Clone)]
pub struct Fixture {
    pub expert_out: Vec<f32>,
    pub route: RouteScratch,
    pub hidden: usize,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        crate::metal::CANVAS * self.hidden
    }
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    let hidden = 4usize;
    let mut route = RouteScratch {
        weight: [[0; crate::metal::TOP_K]; crate::metal::CANVAS],
        expert: [[0; crate::metal::TOP_K]; crate::metal::CANVAS],
        count: [0; crate::metal::N_EXPERTS],
        row_start: [0; crate::metal::N_EXPERTS + 1],
        num_slots: 2,
        num_active_experts: 0,
        active_expert: [0; crate::metal::N_EXPERTS],
        token_list: [0; crate::metal::CANVAS * crate::metal::TOP_K],
        slot_list: [0; crate::metal::CANVAS * crate::metal::TOP_K],
        token_slot: [[0; crate::metal::TOP_K]; crate::metal::CANVAS],
        block_expert: [0; crate::metal::MOE_MAX_BLOCKS],
        block_row0: [0; crate::metal::MOE_MAX_BLOCKS],
        num_blocks: 0,
    };
    route.weight[0][0] = bf16::f32_to_bf16_bits(0.5);
    route.weight[1][0] = bf16::f32_to_bf16_bits(1.0);
    route.token_list[0] = 0;
    route.token_list[1] = 1;
    route.row_start[0] = 0;
    route.row_start[1] = 2;
    route.num_slots = 2;
    route.slot_list[0] = 0;
    route.slot_list[1] = 0;
    crate::metal::fill_token_slot(&mut route);
    Fixture {
        expert_out: vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        route,
        hidden,
    }
}

/// Overlapping token scatter: same canvas row from multiple expert slots/weights.
pub fn moe_routing_fixture(_: ElemFormat) -> Fixture {
    let hidden = 8usize;
    let mut route = RouteScratch {
        weight: [[0; crate::metal::TOP_K]; crate::metal::CANVAS],
        expert: [[0; crate::metal::TOP_K]; crate::metal::CANVAS],
        count: [0; crate::metal::N_EXPERTS],
        row_start: [0; crate::metal::N_EXPERTS + 1],
        num_slots: 8,
        num_active_experts: 0,
        active_expert: [0; crate::metal::N_EXPERTS],
        token_list: [0; crate::metal::CANVAS * crate::metal::TOP_K],
        slot_list: [0; crate::metal::CANVAS * crate::metal::TOP_K],
        token_slot: [[0; crate::metal::TOP_K]; crate::metal::CANVAS],
        block_expert: [0; crate::metal::MOE_MAX_BLOCKS],
        block_row0: [0; crate::metal::MOE_MAX_BLOCKS],
        num_blocks: 0,
    };
    route.token_list[..8].copy_from_slice(&[5, 5, 17, 33, 17, 99, 12, 44]);
    route.slot_list[..8].copy_from_slice(&[0, 1, 0, 0, 1, 0, 0, 0]);
    route.weight[5][0] = bf16::f32_to_bf16_bits(0.6);
    route.weight[5][1] = bf16::f32_to_bf16_bits(0.4);
    route.weight[17][0] = bf16::f32_to_bf16_bits(0.75);
    route.weight[17][1] = bf16::f32_to_bf16_bits(0.25);
    route.weight[33][0] = bf16::f32_to_bf16_bits(1.0);
    route.weight[99][0] = bf16::f32_to_bf16_bits(0.5);
    route.weight[12][0] = bf16::f32_to_bf16_bits(0.8);
    route.weight[44][0] = bf16::f32_to_bf16_bits(0.3);
    crate::metal::fill_token_slot(&mut route);
    let expert_out: Vec<f32> = (0..8 * hidden)
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    Fixture {
        expert_out,
        route,
        hidden,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    crate::kernels::cpu::moe_scatter_weighted::moe_scatter_weighted(
        &f.expert_out,
        &f.route,
        f.hidden,
    )
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
use objc2::runtime::ProtocolObject;
#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn bind_gpu_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    expert_out: &ProtocolObject<dyn MTLBuffer>,
    moe_out: &ProtocolObject<dyn MTLBuffer>,
    route: &ProtocolObject<dyn MTLBuffer>,
    hidden: u32,
    canvas: u32,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(expert_out), 0, 0);
        enc.setBuffer_offset_atIndex(Some(moe_out), 0, 1);
        enc.setBuffer_offset_atIndex(Some(route), 0, 2);
    }
    gpu_common::set_bytes(enc, &hidden, 3);
    gpu_common::set_bytes(enc, &canvas, 4);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let out_len = f.out_len();
    let buf_ex = pool
        .allocate(&ctx.device, f.expert_out.len() * 4)
        .ok_or(Error::Format("alloc"))?;
    let buf_out = pool
        .allocate(&ctx.device, out_len * 4)
        .ok_or(Error::Format("alloc"))?;
    let buf_route = pool
        .allocate(&ctx.device, std::mem::size_of::<RouteScratch>())
        .ok_or(Error::Format("alloc"))?;
    BufferPool::write_f32(&buf_ex, &f.expert_out);
    BufferPool::write_f32(&buf_out, &vec![0.0f32; out_len]);
    BufferPool::write_bytes(
        &buf_route,
        unsafe {
            std::slice::from_raw_parts(
                &f.route as *const RouteScratch as *const u8,
                std::mem::size_of::<RouteScratch>(),
            )
        },
    );
    gpu_common::dispatch_grid(
        &ctx.queue,
        &pipeline.pipeline,
        f.hidden,
        crate::metal::CANVAS,
        8,
        |enc| {
            bind_gpu_buffers(
                enc,
                &buf_ex,
                &buf_out,
                &buf_route,
                f.hidden as u32,
                crate::metal::CANVAS as u32,
            );
        },
    )?;
    let mut out = vec![0.0f32; out_len];
    BufferPool::read_f32(&buf_out, &mut out);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::kernels::sub::moe_scatter_weighted::cpu,
        cpu_oracle = crate::kernels::sub::moe_scatter_weighted::cpu_oracle,
        gpu = crate::kernels::sub::moe_scatter_weighted::gpu,
        fixture = crate::kernels::sub::moe_scatter_weighted::tiny_fixture,
        out_len = |f: &crate::kernels::sub::moe_scatter_weighted::Fixture| f.out_len(),
        formats: [F32],
        max_tol = 1e-5,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod moe_routing,
        cpu = crate::kernels::sub::moe_scatter_weighted::cpu,
        cpu_oracle = crate::kernels::sub::moe_scatter_weighted::cpu_oracle,
        gpu = crate::kernels::sub::moe_scatter_weighted::gpu,
        fixture = crate::kernels::sub::moe_scatter_weighted::moe_routing_fixture,
        out_len = |f: &crate::kernels::sub::moe_scatter_weighted::Fixture| f.out_len(),
        formats: [F32],
        max_tol = 1e-5,
        min_cos = 0.9999,
    }
}
