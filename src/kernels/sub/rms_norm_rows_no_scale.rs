//! Per-row RMSNorm (no scale) — CPU oracle, GPU dispatch, tier-1 tests.

use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::kernels::cpu;

use crate::safetensors::Error;

pub const ENTRY: &str = "rms_norm_rows_no_scale";

const SHADER: &str = concat!(
    include_str!("../../../shaders/kernels/common.metal"),
    include_str!("../../../shaders/kernels/rms_norm_rows_no_scale.metal"),
);

#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
    pub seq_len: usize,
    pub hidden: usize,
    pub eps: f32,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.seq_len * self.hidden
    }
}

pub fn fixture_len(fix: &Fixture) -> usize {
    fix.len()
}

pub fn tiny_fixture(_fmt: ElemFormat) -> Fixture {
    Fixture {
        x: vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -0.5],
        seq_len: 2,
        hidden: 4,
        eps: 1e-6,
    }
}

pub fn mlp_shape_fixture(_fmt: ElemFormat) -> Fixture {
    let seq_len = 3;
    let hidden = 64;
    let len = seq_len * hidden;
    Fixture {
        x: (0..len)
            .map(|i| ((i as f32) * 0.017).sin() * 0.5)
            .collect(),
        seq_len,
        hidden,
        eps: 1e-6,
    }
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    let mut out = vec![0.0f32; fix.len()];
    cpu::rms_norm_rows_no_scale(&mut out, &fix.x, fix.seq_len, fix.hidden, fix.eps);
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
    use objc2_metal::{
        MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
    };

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let len = fix.len();
    let buf_x = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Format("buffer alloc failed"))?;
    let buf_o = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Format("buffer alloc failed"))?;
    let dump_bytes = if variant.dump_stage > 0 {
        fix.seq_len * 4
    } else {
        4
    };
    let buf_dump = pool
        .allocate(&ctx.device, dump_bytes)
        .ok_or(Error::Format("buffer alloc failed"))?;

    BufferPool::write_f32(&buf_x, &fix.x);
    BufferPool::write_f32(&buf_o, &vec![0.0f32; len]);

    let cmd = ctx
        .queue
        .commandBuffer()
        .ok_or(Error::Format("command buffer failed"))?;
    let enc = cmd
        .computeCommandEncoder()
        .ok_or(Error::Format("encoder failed"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    let dims = [fix.seq_len as u32, fix.hidden as u32];
    bind_gpu_buffers(&enc, &buf_x, &buf_o, &buf_dump, &dims, fix.eps);
    let tg_width = 256usize.min(fix.seq_len);
    let grid = MTLSize {
        width: div_up(fix.seq_len, tg_width),
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut out = vec![0.0f32; len];
    BufferPool::read_f32(&buf_o, &mut out);
    pool.release(len * 4, buf_x);
    pool.release(len * 4, buf_o);
    pool.release(dump_bytes, buf_dump);
    Ok(out)
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn gpu(_fix: &Fixture, _variant: KernelVariant) -> Result<Vec<f32>, Error> {
    Err(Error::Format("Metal unavailable"))
}

pub fn shader_source() -> &'static str {
    SHADER
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
    buf_x: &ProtocolObject<dyn MTLBuffer>,
    buf_o: &ProtocolObject<dyn MTLBuffer>,
    buf_dump: &ProtocolObject<dyn MTLBuffer>,
    dims: &[u32; 2],
    eps: f32,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf_x), 0, 0);
        enc.setBuffer_offset_atIndex(Some(buf_o), 0, 1);
        enc.setBuffer_offset_atIndex(Some(buf_dump), 0, 4);
    }
    set_bytes(enc, dims, 2);
    set_bytes(enc, &eps, 3);
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
        cpu = crate::kernels::sub::rms_norm_rows_no_scale::cpu,
        cpu_oracle = crate::kernels::sub::rms_norm_rows_no_scale::cpu_oracle,
        gpu = crate::kernels::sub::rms_norm_rows_no_scale::gpu,
        fixture = crate::kernels::sub::rms_norm_rows_no_scale::tiny_fixture,
        out_len = crate::kernels::sub::rms_norm_rows_no_scale::fixture_len,
        formats: [F32],
        max_tol = 1e-5,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod mlp_shape,
        cpu = crate::kernels::sub::rms_norm_rows_no_scale::cpu,
        cpu_oracle = crate::kernels::sub::rms_norm_rows_no_scale::cpu_oracle,
        gpu = crate::kernels::sub::rms_norm_rows_no_scale::gpu,
        fixture = crate::kernels::sub::rms_norm_rows_no_scale::mlp_shape_fixture,
        out_len = crate::kernels::sub::rms_norm_rows_no_scale::fixture_len,
        formats: [F32],
        max_tol = 1e-5,
        min_cos = 0.9999,
    }
}
