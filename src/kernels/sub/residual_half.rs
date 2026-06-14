//! Half residual add with optional bf16 layer scalar from blob.

use super::bf16;
use super::f16;
use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::safetensors::Error;

pub const ENTRY: &str = "residual_half";

const SHADER: &str = concat!(
    include_str!("../../../shaders/kernels/common.metal"),
    include_str!("../../../shaders/kernels/bf16.metal"),
    include_str!("../../../shaders/kernels/residual_half.metal"),
);

#[derive(Debug, Clone)]
pub struct Fixture {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub scale: Option<f32>,
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
        a: vec![1.0, -1.0, 0.5, 2.0],
        b: vec![0.5, 1.0, -0.25, -1.0],
        scale: Some(0.75),
    }
}

pub fn no_scale_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        a: vec![1.0, 2.0, -3.0],
        b: vec![0.25, -0.5, 1.0],
        scale: None,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let s = f.scale.unwrap_or(1.0);
    f.a
        .iter()
        .zip(f.b.iter())
        .map(|(&a, &b)| f16::round_half((a + b) * s))
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
    blob: &ProtocolObject<dyn MTLBuffer>,
    dump: &ProtocolObject<dyn MTLBuffer>,
    scal_off: u64,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(a), 0, 0);
        enc.setBuffer_offset_atIndex(Some(b), 0, 1);
        enc.setBuffer_offset_atIndex(Some(y), 0, 2);
        enc.setBuffer_offset_atIndex(Some(blob), 0, 3);
        enc.setBuffer_offset_atIndex(Some(dump), 0, 5);
    }
    gpu_common::set_bytes(enc, &scal_off, 4);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let len = f.len();
    let a_f16 = f16::f32_slice_to_f16(&f.a);
    let b_f16 = f16::f32_slice_to_f16(&f.b);
    let buf_a = pool.allocate(&ctx.device, len * 2).ok_or(Error::Format("alloc"))?;
    let buf_b = pool.allocate(&ctx.device, len * 2).ok_or(Error::Format("alloc"))?;
    let buf_y = pool.allocate(&ctx.device, len * 2).ok_or(Error::Format("alloc"))?;
    let (blob, scal_off) = match f.scale {
        Some(v) => {
            let mut b = vec![0u8; 2];
            b.extend(bf16::pack_bf16_scalar(v));
            (b, 2u64)
        }
        None => (vec![0u8; 2], 0u64),
    };
    let buf_blob = pool
        .allocate(&ctx.device, blob.len())
        .ok_or(Error::Format("alloc"))?;
    let dump_bytes = if variant.dump_stage > 0 { len * 4 } else { 4 };
    let buf_d = pool.allocate(&ctx.device, dump_bytes).ok_or(Error::Format("alloc"))?;
    BufferPool::write_bf16(&buf_a, &a_f16);
    BufferPool::write_bf16(&buf_b, &b_f16);
    BufferPool::write_bytes(&buf_blob, &blob);
    gpu_common::dispatch_1d(&ctx.queue, &pipeline.pipeline, len, |enc| {
        bind_gpu_buffers(enc, &buf_a, &buf_b, &buf_y, &buf_blob, &buf_d, scal_off);
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
        cpu = crate::kernels::sub::residual_half::cpu,
        cpu_oracle = crate::kernels::sub::residual_half::cpu_oracle,
        gpu = crate::kernels::sub::residual_half::gpu,
        fixture = crate::kernels::sub::residual_half::tiny_fixture,
        out_len = crate::kernels::sub::residual_half::fixture_len,
        formats: [F32],
        max_tol = 1e-3,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod no_scale,
        cpu = crate::kernels::sub::residual_half::cpu,
        cpu_oracle = crate::kernels::sub::residual_half::cpu_oracle,
        gpu = crate::kernels::sub::residual_half::gpu,
        fixture = crate::kernels::sub::residual_half::no_scale_fixture,
        out_len = crate::kernels::sub::residual_half::fixture_len,
        formats: [F32],
        max_tol = 1e-3,
        min_cos = 0.9999,
    }
}
