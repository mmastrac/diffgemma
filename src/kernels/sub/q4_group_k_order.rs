//! Q4 group K-order decode parity (dequant_q4_group vs q4_at_col).

use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::dgq::block::quantize_row_q4;
use crate::dgq::layout::GROUP_SIZE;
use crate::kernels::cpu::dequant::q4_group_k_order_row;
use crate::safetensors::Error;

pub const ENTRY: &str = "q4_group_k_order";

const SHADER: &str = concat!(
    include_str!("../../../shaders/kernels/common.metal"),
    include_str!("../../../shaders/include/common.metal"),
    include_str!("../../../shaders/include/dequant.metal"),
    include_str!("../../../shaders/kernels/q4_group_k_order.metal"),
);

#[derive(Debug, Clone)]
pub struct Fixture {
    pub row_f32: Vec<f32>,
    pub in_dim: usize,
    pub k0: usize,
}

impl Fixture {
    pub fn row_q4(&self) -> Vec<u8> {
        let mut row = vec![0u8; (self.in_dim / 32) * 20];
        for g in 0..self.in_dim / 32 {
            let off = g * GROUP_SIZE;
            quantize_row_q4(
                &self.row_f32[off..off + GROUP_SIZE],
                GROUP_SIZE,
                &mut row[g * 20..(g + 1) * 20],
            );
        }
        row
    }
}

pub fn fixture_len(_: &Fixture) -> usize {
    64
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    let in_dim = 64usize;
    Fixture {
        row_f32: (0..in_dim)
            .map(|i| ((i as f32) * 0.09).sin() * 0.4)
            .collect(),
        in_dim,
        k0: 0,
    }
}

pub fn wide_fixture(_: ElemFormat) -> Fixture {
    let in_dim = 128usize;
    Fixture {
        row_f32: (0..in_dim)
            .map(|i| ((i as f32) * 0.04).cos() * 0.25)
            .collect(),
        in_dim,
        k0: 32,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    q4_group_k_order_row(&f.row_q4(), f.k0, f.in_dim)
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
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let mut run_variant = variant;
    if run_variant.dump_stage == 0 {
        run_variant.dump_stage = 1;
    }
    let pipeline = pipeline_for(&ctx, run_variant)?;
    let row = f.row_q4();
    let mut pool = BufferPool::new();
    let buf_row = pool
        .allocate(&ctx.device, row.len())
        .ok_or(Error::Format("alloc"))?;
    let dump_bytes = 64 * 4;
    let buf_dump = pool
        .allocate(&ctx.device, dump_bytes)
        .ok_or(Error::Format("alloc"))?;
    BufferPool::write_bytes(&buf_row, &row);

    let k0 = f.k0 as u32;
    let in_dim = f.in_dim as u32;
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_row), 0, 0);
    }
    super::gpu_common::set_bytes(&enc, &k0, 1);
    super::gpu_common::set_bytes(&enc, &in_dim, 2);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_dump), 0, 3);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: 1,
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

    let mut out = vec![0.0f32; 64];
    BufferPool::read_f32(&buf_dump, &mut out);
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
        cpu = crate::kernels::sub::q4_group_k_order::cpu,
        cpu_oracle = crate::kernels::sub::q4_group_k_order::cpu_oracle,
        gpu = crate::kernels::sub::q4_group_k_order::gpu,
        fixture = crate::kernels::sub::q4_group_k_order::tiny_fixture,
        out_len = crate::kernels::sub::q4_group_k_order::fixture_len,
        formats: [F32],
        max_tol = 1e-5,
        min_cos = 0.99999,
    }

    kernel_oracle_matrix! {
        mod wide,
        cpu = crate::kernels::sub::q4_group_k_order::cpu,
        cpu_oracle = crate::kernels::sub::q4_group_k_order::cpu_oracle,
        gpu = crate::kernels::sub::q4_group_k_order::gpu,
        fixture = crate::kernels::sub::q4_group_k_order::wide_fixture,
        out_len = crate::kernels::sub::q4_group_k_order::fixture_len,
        formats: [F32],
        max_tol = 1e-5,
        min_cos = 0.99999,
    }
}
