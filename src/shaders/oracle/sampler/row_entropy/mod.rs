//! Per-row Shannon entropy from logits `[rows, cols]`.

use crate::Error;
use crate::sample::token_entropy;
use crate::shaders::gpu_common;
use crate::shaders::sampler_ranged::{CANVAS_LEN, VOCAB};
use crate::shaders::test_util::ElemFormat;

crate::shader_kernel! {
    name = "row_entropy",
    metal = "row_entropy.metal",
    spec = {
            quant_formats: &[QuantFormat::Q4Affine],
            fc: &[],
            variants: KernelVariants::Elementwise,
    },
    tests = [
        tiny => tiny_fixture => no_variant => (1e-4, 0.9999),
        canvas_vocab => canvas_vocab_fixture => no_variant => (2.5e-2, 0.9999),
    ],
}

#[derive(Debug, Clone)]
pub struct Fixture {
    pub logits: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.rows
    }
}

fn fill_logits(rows: usize, cols: usize, seed: f32) -> Vec<f32> {
    let len = rows * cols;
    (0..len)
        .map(|i| ((i as f32) * seed).sin() * 0.4 + ((i as f32) * seed * 0.19).cos() * 0.2)
        .collect()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    let rows = 4usize;
    let cols = 512usize;
    Fixture {
        logits: fill_logits(rows, cols, 0.05),
        rows,
        cols,
    }
}

/// Production canvas × vocab.
pub fn canvas_vocab_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        logits: fill_logits(CANVAS_LEN, VOCAB, 0.000008),
        rows: CANVAS_LEN,
        cols: VOCAB,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    token_entropy(&f.logits, f.rows, f.cols)
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
    use objc2_metal::MTLComputeCommandEncoder;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx)?;
    let mut pool = BufferPool::new();
    let in_bytes = f.logits.len() * 4;
    let out_bytes = f.rows * 4;
    let buf_in = pool
        .allocate(&ctx.device, in_bytes)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_out = pool
        .allocate(&ctx.device, out_bytes)
        .ok_or(Error::Gpu("alloc"))?;
    BufferPool::write_f32(&buf_in, &f.logits);

    let dims = [f.rows as u32, f.cols as u32];
    gpu_common::dispatch_rows(&ctx.queue, &pipeline.pipeline, f.rows, |enc| {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&buf_in), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&buf_out), 0, 1);
        }
        gpu_common::set_bytes(enc, &dims, 2);
    })?;

    let mut out = vec![0.0f32; f.rows];
    BufferPool::read_f32(&buf_out, &mut out);
    Ok(out)
}
