//! Zero memory in 16-byte `uchar4` chunks (monolith arena clears).

use crate::safetensors::Error;
use crate::shaders::gpu_common;
use crate::shaders::test_util::ElemFormat;
use crate::shaders::variant::KernelVariant;

pub const ENTRY: &str = "memzero_bytes";

pub const SHADER: &str = include_str!("memzero_bytes.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub bytes: Vec<u8>,
    pub zero_off: usize,
    pub zero_len: usize,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.zero_len
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        bytes: vec![0xFF; 32],
        zero_off: 0,
        zero_len: 32,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let mut b = f.bytes.clone();
    for byte in &mut b[f.zero_off..f.zero_off + f.zero_len] {
        *byte = 0;
    }
    b[f.zero_off..f.zero_off + f.zero_len]
        .iter()
        .map(|&v| v as f32)
        .collect()
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(target_os = "macos")]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER, ENTRY, variant)
}

#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};

#[cfg(target_os = "macos")]
pub fn bind_gpu_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buf: &ProtocolObject<dyn MTLBuffer>,
    dump: &ProtocolObject<dyn MTLBuffer>,
    byte_off: usize,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf), byte_off, 0);
        enc.setBuffer_offset_atIndex(Some(dump), 0, 1);
    }
}

#[cfg(target_os = "macos")]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let buf = pool
        .allocate(&ctx.device, f.bytes.len())
        .ok_or(Error::Format("alloc"))?;
    let count = gpu_common::div_up(f.zero_len, 4);
    let dump_bytes = if variant.dump_stage > 0 { count * 4 } else { 4 };
    let buf_d = pool
        .allocate(&ctx.device, dump_bytes)
        .ok_or(Error::Format("alloc"))?;
    BufferPool::write_bytes(&buf, &f.bytes);
    gpu_common::dispatch_1d(&ctx.queue, &pipeline.pipeline, count, |enc| {
        bind_gpu_buffers(enc, &buf, &buf_d, f.zero_off);
    })?;
    let ptr = buf.contents().as_ptr() as *const u8;
    let out: Vec<f32> = (0..f.zero_len)
        .map(|i| unsafe { *ptr.add(f.zero_off + i) } as f32)
        .collect();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::shaders::memzero_bytes::cpu,
        cpu_oracle = crate::shaders::memzero_bytes::cpu_oracle,
        gpu = crate::shaders::memzero_bytes::gpu,
        fixture = crate::shaders::memzero_bytes::tiny_fixture,
        out_len = crate::shaders::memzero_bytes::fixture_len,
        formats: [F32],
        max_tol = 0.0,
        min_cos = 1.0,
    }
}
