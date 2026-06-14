//! Logit softcap: `tanh(v/30)*30` in fp16.

use super::f16;
use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::safetensors::Error;

pub const ENTRY: &str = "softcap_half";

const SHADER: &str = shader_include::include_metal!("kernels/softcap_half.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub logits: Vec<f32>,
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
        logits: vec![-50.0, -5.0, 0.0, 10.0, 100.0, 0.5],
        base: 1,
        len: 4,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    f.logits[f.base as usize..(f.base + f.len) as usize]
        .iter()
        .map(|&v| f16::softcap_f32(v))
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
    logits: &ProtocolObject<dyn MTLBuffer>,
    dump: &ProtocolObject<dyn MTLBuffer>,
    base: u32,
    len: u32,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(logits), 0, 0);
        enc.setBuffer_offset_atIndex(Some(dump), 0, 3);
    }
    gpu_common::set_bytes(enc, &base, 1);
    gpu_common::set_bytes(enc, &len, 2);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let buf = pool
        .allocate(&ctx.device, f.logits.len() * 2)
        .ok_or(Error::Format("alloc"))?;
    let dump_bytes = if variant.dump_stage > 0 {
        f.len as usize * 4
    } else {
        4
    };
    let buf_d = pool.allocate(&ctx.device, dump_bytes).ok_or(Error::Format("alloc"))?;
    BufferPool::write_bf16(&buf, &f16::f32_slice_to_f16(&f.logits));
    gpu_common::dispatch_1d(&ctx.queue, &pipeline.pipeline, f.len as usize, |enc| {
        bind_gpu_buffers(enc, &buf, &buf_d, f.base, f.len);
    })?;
    let ptr = unsafe { buf.contents().as_ptr() as *const u16 };
    Ok((f.base..f.base + f.len)
        .map(|i| f16::f16_bits_to_f32(unsafe { *ptr.add(i as usize) }))
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
        cpu = crate::kernels::sub::softcap_half::cpu,
        cpu_oracle = crate::kernels::sub::softcap_half::cpu_oracle,
        gpu = crate::kernels::sub::softcap_half::gpu,
        fixture = crate::kernels::sub::softcap_half::tiny_fixture,
        out_len = crate::kernels::sub::softcap_half::fixture_len,
        formats: [F32],
        max_tol = 1e-2,
        min_cos = 0.9999,
    }
}
