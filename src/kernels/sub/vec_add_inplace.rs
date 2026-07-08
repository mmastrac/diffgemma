//! `out += addend` elementwise.

use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::safetensors::Error;

pub const ENTRY: &str = "vec_add_inplace";

const SHADER: &str = shader_include::include_metal!("kernels/vec_add_inplace.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub out: Vec<f32>,
    pub addend: Vec<f32>,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.out.len()
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        out: vec![1.0, 2.0, 3.0, 4.0],
        addend: vec![0.5, -1.0, 2.0, 0.0],
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let mut out = f.out.clone();
    for (o, a) in out.iter_mut().zip(f.addend.iter()) {
        *o += *a;
    }
    out
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
    out: &ProtocolObject<dyn MTLBuffer>,
    addend: &ProtocolObject<dyn MTLBuffer>,
    dump: &ProtocolObject<dyn MTLBuffer>,
    len: u32,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(out), 0, 0);
        enc.setBuffer_offset_atIndex(Some(addend), 0, 1);
        enc.setBuffer_offset_atIndex(Some(dump), 0, 3);
    }
    gpu_common::set_bytes(enc, &len, 2);
}

#[cfg(target_os = "macos")]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let len = f.len();
    let buf_o = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Format("alloc"))?;
    let buf_a = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Format("alloc"))?;
    let dump_bytes = if variant.dump_stage > 0 { len * 4 } else { 4 };
    let buf_d = pool
        .allocate(&ctx.device, dump_bytes)
        .ok_or(Error::Format("alloc"))?;
    BufferPool::write_f32(&buf_o, &f.out);
    BufferPool::write_f32(&buf_a, &f.addend);
    gpu_common::dispatch_1d(&ctx.queue, &pipeline.pipeline, len, |enc| {
        bind_gpu_buffers(enc, &buf_o, &buf_a, &buf_d, len as u32);
    })?;
    let mut out = vec![0.0f32; len];
    BufferPool::read_f32(&buf_o, &mut out);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::kernels::sub::vec_add_inplace::cpu,
        cpu_oracle = crate::kernels::sub::vec_add_inplace::cpu_oracle,
        gpu = crate::kernels::sub::vec_add_inplace::gpu,
        fixture = crate::kernels::sub::vec_add_inplace::tiny_fixture,
        out_len = crate::kernels::sub::vec_add_inplace::fixture_len,
        formats: [F32],
        max_tol = 1e-6,
        min_cos = 0.9999,
    }
}
