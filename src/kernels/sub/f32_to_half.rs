//! Convert f32 buffer slice to fp16 arena layout (monolith MPS-Q4 path).

use super::f16;
use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::safetensors::Error;

pub const ENTRY: &str = "f32_to_half";

const SHADER: &str = concat!(
    include_str!("../../../shaders/include/fc_axes.metal"),
    include_str!("../../../shaders/kernels/f32_to_half.metal"),
);

#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
    pub base: u32,
    pub len: u32,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.len as usize
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        x: vec![0.0, 1.5, -2.0, 0.125, 4.0],
        base: 0,
        len: 4,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let slice = &f.x[f.base as usize..(f.base + f.len) as usize];
    f16::f16_slice_to_f32(&f16::f32_slice_to_f16(slice))
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
    x: &ProtocolObject<dyn MTLBuffer>,
    y: &ProtocolObject<dyn MTLBuffer>,
    dump: &ProtocolObject<dyn MTLBuffer>,
    base: u32,
    len: u32,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(x), 0, 0);
        enc.setBuffer_offset_atIndex(Some(y), 0, 1);
        enc.setBuffer_offset_atIndex(Some(dump), 0, 4);
    }
    gpu_common::set_bytes(enc, &base, 2);
    gpu_common::set_bytes(enc, &len, 3);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let buf_x = pool
        .allocate(&ctx.device, f.x.len() * 4)
        .ok_or(Error::Format("alloc"))?;
    let out_elems = f.base as usize + f.len as usize;
    let buf_y = pool
        .allocate(&ctx.device, out_elems * 2)
        .ok_or(Error::Format("alloc"))?;
    let dump_bytes = if variant.dump_stage > 0 {
        f.len as usize * 4
    } else {
        4
    };
    let buf_d = pool.allocate(&ctx.device, dump_bytes).ok_or(Error::Format("alloc"))?;
    BufferPool::write_f32(&buf_x, &f.x);
    gpu_common::dispatch_1d(&ctx.queue, &pipeline.pipeline, f.len as usize, |enc| {
        bind_gpu_buffers(enc, &buf_x, &buf_y, &buf_d, f.base, f.len);
    })?;
    let mut y_bits = vec![0u16; out_elems];
    let ptr = unsafe { buf_y.contents().as_ptr() as *const u16 };
    for (i, slot) in y_bits.iter_mut().enumerate() {
        *slot = unsafe { *ptr.add(i) };
    }
    Ok(f16::f16_slice_to_f32(
        &y_bits[f.base as usize..(f.base + f.len) as usize],
    ))
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
        cpu = crate::kernels::sub::f32_to_half::cpu,
        cpu_oracle = crate::kernels::sub::f32_to_half::cpu_oracle,
        gpu = crate::kernels::sub::f32_to_half::gpu,
        fixture = crate::kernels::sub::f32_to_half::tiny_fixture,
        out_len = crate::kernels::sub::f32_to_half::fixture_len,
        formats: [F32],
        max_tol = 1e-3,
        min_cos = 0.9999,
    }
}
