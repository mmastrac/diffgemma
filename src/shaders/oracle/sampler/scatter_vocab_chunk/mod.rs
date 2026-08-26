//! Scatter lm_head GEMM chunk into `[seq, vocab]` logits.

use crate::Error;
use crate::model::embed::LM_HEAD_CHUNK;
use crate::shaders::gpu_common;
use crate::shaders::test_util::ElemFormat;

pub const ENTRY: &str = "scatter_vocab_chunk";

pub const SHADER: &str = include_str!("scatter_vocab_chunk.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub chunk: Vec<f32>,
    pub logits: Vec<f32>,
    pub seq_len: usize,
    pub chunk_cols: usize,
    pub v0: usize,
    pub vocab: usize,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.logits.len()
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

fn fill(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32) * seed).sin() * 0.5 + 0.1)
        .collect()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    let seq_len = 2usize;
    let chunk_cols = 4usize;
    let v0 = 1usize;
    let vocab = 8usize;
    Fixture {
        chunk: fill(seq_len * chunk_cols, 0.11),
        logits: vec![0.0; seq_len * vocab],
        seq_len,
        chunk_cols,
        v0,
        vocab,
    }
}

/// One lm_head chunk at production offset (4096-wide slice into 262k vocab).
pub fn lm_head_chunk_fixture(_: ElemFormat) -> Fixture {
    let seq_len = 256usize;
    let chunk_cols = LM_HEAD_CHUNK;
    let v0 = 8192usize;
    let vocab = 262_144usize;
    Fixture {
        chunk: fill(seq_len * chunk_cols, 0.000021),
        logits: vec![0.0; seq_len * vocab],
        seq_len,
        chunk_cols,
        v0,
        vocab,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let mut out = f.logits.clone();
    for row in 0..f.seq_len {
        for col in 0..f.chunk_cols {
            let dst = row * f.vocab + f.v0 + col;
            let src = row * f.chunk_cols + col;
            out[dst] = f.chunk[src];
        }
    }
    out
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(target_os = "macos")]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    Ok(ctx.compile_kernel(SHADER, ENTRY)?)
}

#[cfg(target_os = "macos")]
pub fn gpu(f: &Fixture) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use objc2_metal::{
        MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    };

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx)?;
    let mut pool = BufferPool::new();

    let chunk_bytes = f.chunk.len() * 4;
    let logits_bytes = f.logits.len() * 4;
    let buf_chunk = pool
        .allocate(&ctx.device, chunk_bytes)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_logits = pool
        .allocate(&ctx.device, logits_bytes)
        .ok_or(Error::Gpu("alloc"))?;
    BufferPool::write_f32(&buf_chunk, &f.chunk);
    BufferPool::write_f32(&buf_logits, &f.logits);

    let params = [
        f.seq_len as u32,
        f.chunk_cols as u32,
        f.v0 as u32,
        f.vocab as u32,
    ];
    let (grid, tg) = gpu_common::scatter_vocab_grid(f.seq_len, f.chunk_cols);

    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_chunk), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&buf_logits), 0, 1);
    }
    gpu_common::set_bytes(&enc, &params, 2);
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut out = vec![0.0f32; f.logits.len()];
    BufferPool::read_f32(&buf_logits, &mut out);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shaders::test_util::{ElemFormat, assert_oracle};

    fn run_matrix(fixture_fn: fn(ElemFormat) -> Fixture, max_tol: f32) {
        let fix = fixture_fn(ElemFormat::F32);
        let cpu = cpu(&fix);
        assert!(cpu.iter().all(|v| v.is_finite()));
        assert_eq!(cpu.len(), fixture_len(&fix));

        #[cfg(target_os = "macos")]
        {
            let gpu = gpu(&fix).expect("gpu");
            assert_oracle(&gpu, &cpu, max_tol, 0.9999);
        }
    }

    #[test]
    fn cpu_tiny() {
        if crate::shaders::test_util::skip_gpu_on_ci() {
            return;
        }
        run_matrix(tiny_fixture, 0.0);
    }

    #[test]
    fn cpu_lm_head_chunk() {
        if crate::shaders::test_util::skip_gpu_on_ci() {
            return;
        }
        run_matrix(lm_head_chunk_fixture, 0.0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn gpu_tiny() {
        if crate::shaders::test_util::skip_gpu_on_ci() {
            return;
        }
        run_matrix(tiny_fixture, 0.0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn gpu_lm_head_chunk() {
        if crate::shaders::test_util::skip_gpu_on_ci() {
            return;
        }
        run_matrix(lm_head_chunk_fixture, 0.0);
    }
}

crate::kernel_spec! {
    pub const SPEC {
        name: "scatter_vocab_chunk",
        entry: "scatter_vocab_chunk",
        source: SHADER,
        quant_formats: &[QuantFormat::Q4Affine],
        fc: &[],
        variants: KernelVariants::Elementwise,
    }
}
