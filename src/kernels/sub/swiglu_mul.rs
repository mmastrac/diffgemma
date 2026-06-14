//! SwiGLU multiply step: `gate *= up`.

use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::safetensors::Error;

pub const ENTRY: &str = "swiglu_mul";

const SHADER: &str = concat!(
    include_str!("../../../shaders/kernels/common.metal"),
    include_str!("../../../shaders/kernels/swiglu_mul.metal"),
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

pub fn fixture_len(fix: &Fixture) -> usize {
    fix.len()
}

pub fn tiny_fixture(_fmt: ElemFormat) -> Fixture {
    Fixture {
        gate: vec![1.0, -0.5, 2.0, 0.0],
        up: vec![2.0, 4.0, -1.0, 3.0],
    }
}

pub fn moe_inter_fixture(_fmt: ElemFormat) -> Fixture {
    let len = 8 * 704;
    Fixture {
        gate: (0..len).map(|i| ((i as f32) * 0.01).sin()).collect(),
        up: (0..len).map(|i| ((i as f32) * 0.013).cos()).collect(),
    }
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    let mut out = fix.gate.clone();
    for (g, &u) in out.iter_mut().zip(fix.up.iter()) {
        *g *= u;
    }
    out
}

pub fn cpu_oracle(fix: &Fixture) -> Vec<f32> {
    cpu(fix)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(fix: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize};

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let len = fix.len();
    let buf_g = pool.allocate(&ctx.device, len * 4).ok_or(Error::Format("alloc"))?;
    let buf_u = pool.allocate(&ctx.device, len * 4).ok_or(Error::Format("alloc"))?;
    let dump_bytes = if variant.dump_stage > 0 { len * 4 } else { 4 };
    let buf_dump = pool.allocate(&ctx.device, dump_bytes).ok_or(Error::Format("alloc"))?;

    BufferPool::write_f32(&buf_g, &fix.gate);
    BufferPool::write_f32(&buf_u, &fix.up);

    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    bind_gpu_buffers(&enc, &buf_g, &buf_u, &buf_dump, len as u32);
    let tg = 256usize.min(len);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: div_up(len, tg), height: 1, depth: 1 },
        MTLSize { width: tg, height: 1, depth: 1 },
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut out = vec![0.0f32; len];
    BufferPool::read_f32(&buf_g, &mut out);
    pool.release(len * 4, buf_g);
    pool.release(len * 4, buf_u);
    pool.release(dump_bytes, buf_dump);
    Ok(out)
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn gpu(_fix: &Fixture, _variant: KernelVariant) -> Result<Vec<f32>, Error> {
    Err(Error::Format("Metal unavailable"))
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
    dump: &ProtocolObject<dyn MTLBuffer>,
    len: u32,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(gate), 0, 0);
        enc.setBuffer_offset_atIndex(Some(up), 0, 1);
        enc.setBuffer_offset_atIndex(Some(dump), 0, 3);
    }
    set_bytes(enc, &len, 2);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn set_bytes<T>(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, value: &T, index: usize) {
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from_ref(value).cast(),
            std::mem::size_of_val(value),
            index,
        );
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn div_up(value: usize, group: usize) -> usize {
    (value + group - 1) / group
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::kernels::sub::swiglu_mul::cpu,
        cpu_oracle = crate::kernels::sub::swiglu_mul::cpu_oracle,
        gpu = crate::kernels::sub::swiglu_mul::gpu,
        fixture = crate::kernels::sub::swiglu_mul::tiny_fixture,
        out_len = crate::kernels::sub::swiglu_mul::fixture_len,
        formats: [F32],
        max_tol = 1e-5,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod moe_inter,
        cpu = crate::kernels::sub::swiglu_mul::cpu,
        cpu_oracle = crate::kernels::sub::swiglu_mul::cpu_oracle,
        gpu = crate::kernels::sub::swiglu_mul::gpu,
        fixture = crate::kernels::sub::swiglu_mul::moe_inter_fixture,
        out_len = crate::kernels::sub::swiglu_mul::fixture_len,
        formats: [F32],
        max_tol = 1e-5,
        min_cos = 0.9999,
    }
}
