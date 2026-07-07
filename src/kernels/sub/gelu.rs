//! GELU (PyTorch tanh) — CPU oracle, GPU dispatch, tier-1 tests.

use super::manifest;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::kernels::cpu;
use crate::safetensors::Error;

pub const ENTRY: &str = "gelu";

const SHADER: &str = shader_include::include_metal!("kernels/gelu.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.x.len()
    }
}

pub fn fixture_len(fix: &Fixture) -> usize {
    fix.len()
}

pub fn tiny_fixture(_fmt: ElemFormat) -> Fixture {
    Fixture {
        x: vec![-2.0, -1.0, 0.0, 0.5, 1.5, 3.0, 10.229641],
    }
}

pub fn mlp_shape_fixture(_fmt: ElemFormat) -> Fixture {
    let len = 16 * 2112;
    Fixture {
        x: (0..len).map(|i| ((i as f32) * 0.001).sin()).collect(),
    }
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    let mut out = fix.x.clone();
    cpu::gelu_pytorch_tanh(&mut out);
    out
}

pub fn cpu_oracle(fix: &Fixture) -> Vec<f32> {
    cpu(fix)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(fix: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    use objc2_metal::{
        MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
    };

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let len = fix.len();
    let buf = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Format("buffer alloc failed"))?;
    let dump_bytes = if variant.dump_stage > 0 { len * 4 } else { 4 };
    let buf_dump = pool
        .allocate(&ctx.device, dump_bytes)
        .ok_or(Error::Format("buffer alloc failed"))?;

    BufferPool::write_f32(&buf, &fix.x);

    let cmd = ctx
        .queue
        .commandBuffer()
        .ok_or(Error::Format("cmd buffer"))?;
    let enc = cmd
        .computeCommandEncoder()
        .ok_or(Error::Format("encoder"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    let len_u = len as u32;
    bind_gpu_in_place(&enc, &buf, &buf_dump, len_u);
    let tg = 256usize.min(len);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: div_up(len, tg),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut out = vec![0.0f32; len];
    BufferPool::read_f32(&buf, &mut out);
    pool.release(len * 4, buf);
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
    manifest::validate_shared(ENTRY, variant)?;
    ctx.compile_subkernel(SHADER, ENTRY, variant)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2::runtime::ProtocolObject;
#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn bind_gpu_in_place(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buf: &ProtocolObject<dyn MTLBuffer>,
    buf_dump: &ProtocolObject<dyn MTLBuffer>,
    len: u32,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(buf_dump), 0, 2);
    }
    set_bytes(enc, &len, 1);
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
        cpu = crate::kernels::sub::gelu::cpu,
        cpu_oracle = crate::kernels::sub::gelu::cpu_oracle,
        gpu = crate::kernels::sub::gelu::gpu,
        fixture = crate::kernels::sub::gelu::tiny_fixture,
        out_len = crate::kernels::sub::gelu::fixture_len,
        formats: [F32],
        max_tol = 1e-4,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod mlp_shape,
        cpu = crate::kernels::sub::gelu::cpu,
        cpu_oracle = crate::kernels::sub::gelu::cpu_oracle,
        gpu = crate::kernels::sub::gelu::gpu,
        fixture = crate::kernels::sub::gelu::mlp_shape_fixture,
        out_len = crate::kernels::sub::gelu::fixture_len,
        formats: [F32],
        max_tol = 1e-4,
        min_cos = 0.9999,
    }
}
