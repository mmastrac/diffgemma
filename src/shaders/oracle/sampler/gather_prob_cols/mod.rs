//! Copy a vocab column slice from row-major softmax probs.

use crate::safetensors::Error;
use crate::shaders::gpu_common;
use crate::shaders::test_util::ElemFormat;
use crate::shaders::variant::KernelVariant;

pub const ENTRY: &str = "gather_prob_cols";
pub const TILE: usize = 16;

pub const SHADER: &str = include_str!("gather_prob_cols.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub probs: Vec<f32>,
    pub rows: usize,
    pub vocab: usize,
    pub v0: usize,
    pub chunk: usize,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.rows * self.chunk
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        probs: (0..24).map(|i| i as f32 * 0.01).collect(),
        rows: 3,
        vocab: 8,
        v0: 2,
        chunk: 4,
    }
}

pub fn lm_head_chunk_fixture(_: ElemFormat) -> Fixture {
    let rows = 4;
    let vocab = 512;
    let chunk = 64;
    let len = rows * vocab;
    Fixture {
        probs: (0..len).map(|i| (i as f32 * 0.0001).sin().abs()).collect(),
        rows,
        vocab,
        v0: 128,
        chunk,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let mut out = vec![0.0f32; f.out_len()];
    for r in 0..f.rows {
        for c in 0..f.chunk {
            out[r * f.chunk + c] = f.probs[r * f.vocab + f.v0 + c];
        }
    }
    out
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(target_os = "macos")]
pub fn dispatch_shape(rows: usize, chunk: usize) -> (objc2_metal::MTLSize, objc2_metal::MTLSize) {
    use objc2_metal::MTLSize;
    (
        MTLSize {
            width: gpu_common::div_up(chunk, TILE),
            height: rows,
            depth: 1,
        },
        MTLSize {
            width: TILE,
            height: TILE,
            depth: 1,
        },
    )
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
    probs: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    dump: &ProtocolObject<dyn MTLBuffer>,
    params: &[u32; 4],
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(probs), 0, 0);
        enc.setBuffer_offset_atIndex(Some(out), 0, 1);
        enc.setBuffer_offset_atIndex(Some(dump), 0, 3);
    }
    gpu_common::set_bytes(enc, params, 2);
}

#[cfg(target_os = "macos")]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use objc2_metal::{
        MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    };

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let out_len = f.out_len();
    let buf_p = pool
        .allocate(&ctx.device, f.probs.len() * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_o = pool
        .allocate(&ctx.device, out_len * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let dump_bytes = if variant.dump_stage > 0 {
        out_len * 4
    } else {
        4
    };
    let buf_d = pool
        .allocate(&ctx.device, dump_bytes)
        .ok_or(Error::Gpu("alloc"))?;
    BufferPool::write_f32(&buf_p, &f.probs);
    let params = [f.rows as u32, f.vocab as u32, f.v0 as u32, f.chunk as u32];
    let (grid, tg) = dispatch_shape(f.rows, f.chunk);
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    bind_gpu_buffers(&enc, &buf_p, &buf_o, &buf_d, &params);
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let mut out = vec![0.0f32; out_len];
    BufferPool::read_f32(&buf_o, &mut out);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::shaders::gather_prob_cols::cpu,
        cpu_oracle = crate::shaders::gather_prob_cols::cpu_oracle,
        gpu = crate::shaders::gather_prob_cols::gpu,
        fixture = crate::shaders::gather_prob_cols::tiny_fixture,
        out_len = crate::shaders::gather_prob_cols::fixture_len,
        formats: [F32],
        max_tol = 1e-6,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod lm_head_chunk,
        cpu = crate::shaders::gather_prob_cols::cpu,
        cpu_oracle = crate::shaders::gather_prob_cols::cpu_oracle,
        gpu = crate::shaders::gather_prob_cols::gpu,
        fixture = crate::shaders::gather_prob_cols::lm_head_chunk_fixture,
        out_len = crate::shaders::gather_prob_cols::fixture_len,
        formats: [F32],
        max_tol = 1e-6,
        min_cos = 0.9999,
    }
}
