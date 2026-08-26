//! GELU (PyTorch tanh) — CPU oracle, GPU dispatch, tier-1 tests.

use crate::Error;
use crate::shaders::cpu;
use crate::shaders::manifest;
use crate::shaders::test_util::ElemFormat;
use crate::shaders::variant::KernelVariant;

pub const ENTRY: &str = "gelu";

pub const SHADER: &str = include_str!("gelu.metal");

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

#[cfg(target_os = "macos")]
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
        .ok_or(Error::Gpu("buffer alloc failed"))?;
    let dump_bytes = if variant.dump_stage > 0 { len * 4 } else { 4 };
    let buf_dump = pool
        .allocate(&ctx.device, dump_bytes)
        .ok_or(Error::Gpu("buffer alloc failed"))?;

    BufferPool::write_f32(&buf, &fix.x);

    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd buffer"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("encoder"))?;
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

#[cfg(target_os = "macos")]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    manifest::validate_shared(ENTRY, variant)?;
    ctx.compile_subkernel(SHADER, ENTRY, variant)
}

#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
fn set_bytes<T>(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, value: &T, index: usize) {
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from_ref(value).cast(),
            std::mem::size_of_val(value),
            index,
        );
    }
}

#[cfg(target_os = "macos")]
use crate::shaders::gpu_common::div_up;

crate::kernel_spec! {
    pub const SPEC {
        name: "gelu",
        entry: "gelu",
        source: SHADER,
        quant_formats: &[QuantFormat::Q4Affine],
        fc: &[],
        variants: KernelVariants::Gelu,
    }
}

#[cfg(test)]
mod tests {
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::shaders::gelu::cpu,
        cpu_oracle = crate::shaders::gelu::cpu_oracle,
        gpu = crate::shaders::gelu::gpu,
        fixture = crate::shaders::gelu::tiny_fixture,
        out_len = crate::shaders::gelu::fixture_len,
        formats: [F32],
        max_tol = 1e-4,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod mlp_shape,
        cpu = crate::shaders::gelu::cpu,
        cpu_oracle = crate::shaders::gelu::cpu_oracle,
        gpu = crate::shaders::gelu::gpu,
        fixture = crate::shaders::gelu::mlp_shape_fixture,
        out_len = crate::shaders::gelu::fixture_len,
        formats: [F32],
        max_tol = 1e-4,
        min_cos = 0.9999,
    }
}
