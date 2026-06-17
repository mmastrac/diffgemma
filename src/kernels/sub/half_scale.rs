//! In-place bf16 arena scale (token embed path).

use super::bf16;
use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::safetensors::Error;

pub const ENTRY: &str = "half_scale";

const SHADER: &str = shader_include::include_metal!("kernels/half_scale.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub y: Vec<f32>,
    pub scale: f32,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.y.len()
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        y: vec![1.0, -2.0, 0.5, 4.0, -0.125],
        scale: 0.018844940515378,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    f.y
        .iter()
        .map(|&v| bf16::round_bf16_f32(v * f.scale))
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
    y: &ProtocolObject<dyn MTLBuffer>,
    dump: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
    scale: f32,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(y), 0, 0);
        enc.setBuffer_offset_atIndex(Some(dump), 0, 3);
    }
    gpu_common::set_bytes(enc, &n, 1);
    gpu_common::set_bytes(enc, &scale, 2);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let len = f.len();
    let buf_y = pool.allocate(&ctx.device, len * 2).ok_or(Error::Format("alloc"))?;
    let dump_bytes = if variant.dump_stage > 0 { len * 4 } else { 4 };
    let buf_d = pool.allocate(&ctx.device, dump_bytes).ok_or(Error::Format("alloc"))?;
    BufferPool::write_bf16(&buf_y, &bf16::f32_slice_to_bf16_bits(&f.y));
    gpu_common::dispatch_1d(&ctx.queue, &pipeline.pipeline, len, |enc| {
        bind_gpu_buffers(enc, &buf_y, &buf_d, len as u32, f.scale);
    })?;
    let ptr = buf_y.contents().as_ptr() as *const u16;
    Ok((0..len)
        .map(|i| bf16::bf16_bits_to_f32(unsafe { *ptr.add(i) }))
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
        cpu = crate::kernels::sub::half_scale::cpu,
        cpu_oracle = crate::kernels::sub::half_scale::cpu_oracle,
        gpu = crate::kernels::sub::half_scale::gpu,
        fixture = crate::kernels::sub::half_scale::tiny_fixture,
        out_len = crate::kernels::sub::half_scale::fixture_len,
        formats: [F32],
        max_tol = 1e-3,
        min_cos = 0.9999,
    }
}
