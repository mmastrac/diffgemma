//! SwiGLU gate: `half(gelu(gate) * up)` (dense MLP path).

use super::f16;
use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::kernels::cpu::{gelu_pytorch_tanh_f32, gelu_pytorch_tanh};
use crate::safetensors::Error;

pub const ENTRY: &str = "glu_half";

const SHADER: &str = concat!(
    include_str!("../../../shaders/kernels/common.metal"),
    include_str!("../../../shaders/kernels/activations.metal"),
    include_str!("../../../shaders/kernels/glu_half.metal"),
);

#[derive(Debug, Clone)]
pub struct Fixture {
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.gate.len()
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        gate: vec![-1.0, 0.0, 1.0, 2.0],
        up: vec![1.0, -0.5, 0.25, 3.0],
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    f.gate
        .iter()
        .zip(f.up.iter())
        .map(|(&g, &u)| f16::round_half(gelu_pytorch_tanh_f32(g) * u))
        .collect()
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    let mut gate = f.gate.clone();
    gelu_pytorch_tanh(&mut gate);
    gate.iter()
        .zip(f.up.iter())
        .map(|(&g, &u)| f16::round_half(g * u))
        .collect()
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
    gate: &ProtocolObject<dyn MTLBuffer>,
    up: &ProtocolObject<dyn MTLBuffer>,
    y: &ProtocolObject<dyn MTLBuffer>,
    dump: &ProtocolObject<dyn MTLBuffer>,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(gate), 0, 0);
        enc.setBuffer_offset_atIndex(Some(up), 0, 1);
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
    let buf_g = pool.allocate(&ctx.device, len * 2).ok_or(Error::Format("alloc"))?;
    let buf_u = pool.allocate(&ctx.device, len * 2).ok_or(Error::Format("alloc"))?;
    let buf_y = pool.allocate(&ctx.device, len * 2).ok_or(Error::Format("alloc"))?;
    let dump_bytes = if variant.dump_stage > 0 { len * 4 } else { 4 };
    let buf_d = pool.allocate(&ctx.device, dump_bytes).ok_or(Error::Format("alloc"))?;
    BufferPool::write_bf16(&buf_g, &f16::f32_slice_to_f16(&f.gate));
    BufferPool::write_bf16(&buf_u, &f16::f32_slice_to_f16(&f.up));
    gpu_common::dispatch_1d(&ctx.queue, &pipeline.pipeline, len, |enc| {
        bind_gpu_buffers(enc, &buf_g, &buf_u, &buf_y, &buf_d);
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
        cpu = crate::kernels::sub::glu_half::cpu,
        cpu_oracle = crate::kernels::sub::glu_half::cpu_oracle,
        gpu = crate::kernels::sub::glu_half::gpu,
        fixture = crate::kernels::sub::glu_half::tiny_fixture,
        out_len = crate::kernels::sub::glu_half::fixture_len,
        formats: [F32],
        max_tol = 1e-3,
        min_cos = 0.9999,
    }
}
