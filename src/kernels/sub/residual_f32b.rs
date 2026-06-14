//! Residual add: half + f32 -> half (MoE scatter path).

use super::f16;
use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::safetensors::Error;

pub const ENTRY: &str = "residual_f32b";

const SHADER: &str = concat!(
    include_str!("../../../shaders/kernels/common.metal"),
    include_str!("../../../shaders/kernels/residual_f32b.metal"),
);

#[derive(Debug, Clone)]
pub struct Fixture {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.a.len()
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        a: vec![1.0, -0.5, 2.0, 0.0],
        b: vec![0.25, 1.5, -1.0, 3.0],
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    f.a
        .iter()
        .zip(f.b.iter())
        .map(|(&a, &b)| f16::round_half(a + b))
        .collect()
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
    a: &ProtocolObject<dyn MTLBuffer>,
    b: &ProtocolObject<dyn MTLBuffer>,
    y: &ProtocolObject<dyn MTLBuffer>,
    dump: &ProtocolObject<dyn MTLBuffer>,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(a), 0, 0);
        enc.setBuffer_offset_atIndex(Some(b), 0, 1);
        enc.setBuffer_offset_atIndex(Some(y), 0, 2);
        enc.setBuffer_offset_atIndex(Some(dump), 0, 3);
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let len = f.len();
    let buf_a = pool.allocate(&ctx.device, len * 2).ok_or(Error::Format("alloc"))?;
    let buf_b = pool.allocate(&ctx.device, len * 4).ok_or(Error::Format("alloc"))?;
    let buf_y = pool.allocate(&ctx.device, len * 2).ok_or(Error::Format("alloc"))?;
    let dump_bytes = if variant.dump_stage > 0 { len * 4 } else { 4 };
    let buf_d = pool.allocate(&ctx.device, dump_bytes).ok_or(Error::Format("alloc"))?;
    BufferPool::write_bf16(&buf_a, &f16::f32_slice_to_f16(&f.a));
    BufferPool::write_f32(&buf_b, &f.b);
    gpu_common::dispatch_1d(&ctx.queue, &pipeline.pipeline, len, |enc| {
        bind_gpu_buffers(enc, &buf_a, &buf_b, &buf_y, &buf_d);
    })?;
    let ptr = unsafe { buf_y.contents().as_ptr() as *const u16 };
    Ok((0..len)
        .map(|i| f16::f16_bits_to_f32(unsafe { *ptr.add(i) }))
        .collect())
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
        cpu = crate::kernels::sub::residual_f32b::cpu,
        cpu_oracle = crate::kernels::sub::residual_f32b::cpu_oracle,
        gpu = crate::kernels::sub::residual_f32b::gpu,
        fixture = crate::kernels::sub::residual_f32b::tiny_fixture,
        out_len = crate::kernels::sub::residual_f32b::fixture_len,
        formats: [F32],
        max_tol = 1e-3,
        min_cos = 0.9999,
    }
}
