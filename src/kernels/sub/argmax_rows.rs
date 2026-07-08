//! Per-row argmax over logits `[rows, cols]`.

use super::gpu_common;
use super::sampler_ranged::{CANVAS_LEN, VOCAB};
use super::test_util::ElemFormat;
use crate::safetensors::Error;
use crate::sample::argmax_canvas;

pub const ENTRY: &str = "argmax_rows";

const SHADER: &str = shader_include::include_metal!("kernels/argmax_rows.metal");

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

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

fn fill_logits(rows: usize, cols: usize, seed: f32) -> Vec<f32> {
    let len = rows * cols;
    (0..len)
        .map(|i| {
            let spike = if i % cols == (i / cols) * 17 % cols.max(1) {
                3.0
            } else {
                0.0
            };
            spike + ((i as f32) * seed).sin() * 0.02
        })
        .collect()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    let rows = 4usize;
    let cols = 512usize;
    Fixture {
        logits: fill_logits(rows, cols, 0.07),
        rows,
        cols,
    }
}

/// Production canvas × vocab (cols > TG width 256).
pub fn canvas_vocab_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        logits: fill_logits(CANVAS_LEN, VOCAB, 0.000009),
        rows: CANVAS_LEN,
        cols: VOCAB,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    argmax_canvas(&f.logits, f.rows, f.cols)
        .into_iter()
        .map(|v| v as f32)
        .collect()
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(target_os = "macos")]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_kernel(SHADER, ENTRY)
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
        .ok_or(Error::Format("alloc"))?;
    let buf_out = pool
        .allocate(&ctx.device, out_bytes)
        .ok_or(Error::Format("alloc"))?;
    BufferPool::write_f32(&buf_in, &f.logits);

    let dims = [f.rows as u32, f.cols as u32];
    gpu_common::dispatch_rows(&ctx.queue, &pipeline.pipeline, f.rows, |enc| {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&buf_in), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&buf_out), 0, 1);
        }
        gpu_common::set_bytes(enc, &dims, 2);
    })?;

    let mut out_u32 = vec![0u32; f.rows];
    BufferPool::read_u32(&buf_out, &mut out_u32);
    Ok(out_u32.into_iter().map(|v| v as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::sub::test_util::{ElemFormat, assert_oracle};

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
        run_matrix(tiny_fixture, 0.0);
    }

    #[test]
    fn cpu_canvas_vocab() {
        run_matrix(canvas_vocab_fixture, 0.0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn gpu_tiny() {
        run_matrix(tiny_fixture, 0.0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn gpu_canvas_vocab() {
        run_matrix(canvas_vocab_fixture, 0.0);
    }
}
