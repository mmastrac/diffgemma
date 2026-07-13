//! Stable row softmax — CPU oracle, GPU dispatch, tier-1 tests + invariants.

use crate::shaders::cpu;
use crate::shaders::sampler_ranged::{CANVAS_LEN, VOCAB};
use crate::shaders::test_util::ElemFormat;
use crate::shaders::variant::KernelVariant;

use crate::safetensors::Error;

pub const ENTRY: &str = "softmax_rows";
pub const THREADGROUP_WIDTH: usize = 256;

pub const SHADER: &str = include_str!("softmax_rows.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub logits: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub fn fixture_len(fix: &Fixture) -> usize {
    fix.len()
}

pub fn tiny_fixture(_fmt: ElemFormat) -> Fixture {
    Fixture {
        logits: vec![1.0, 2.0, 3.0, 0.0, -1.0, 0.5],
        rows: 2,
        cols: 3,
    }
}

/// MoE router shape: few tokens × 128 experts.
pub fn router_fixture(_fmt: ElemFormat) -> Fixture {
    let rows = 4;
    let cols = 128;
    let len = rows * cols;
    Fixture {
        logits: (0..len).map(|i| ((i as f32) * 0.03).sin() * 2.0).collect(),
        rows,
        cols,
    }
}

/// Exercises the strided column loop (cols > threadgroup width).
pub fn wide_cols_fixture(_fmt: ElemFormat) -> Fixture {
    let rows = 2;
    let cols = 512;
    let len = rows * cols;
    Fixture {
        logits: (0..len).map(|i| (i as f32 % 17.0) - 8.0).collect(),
        rows,
        cols,
    }
}

/// Production canvas × vocab softmax (256 rows, 262k cols).
pub fn canvas_vocab_fixture(_fmt: ElemFormat) -> Fixture {
    let rows = CANVAS_LEN;
    let cols = VOCAB;
    let len = rows * cols;
    Fixture {
        logits: (0..len)
            .map(|i| ((i as f32) * 0.000007).sin() * 0.6 + ((i as f32) * 0.000003).cos() * 0.3)
            .collect(),
        rows,
        cols,
    }
}

pub fn cpu(fix: &Fixture) -> Vec<f32> {
    let mut out = fix.logits.clone();
    cpu::softmax_rows(&mut out, fix.rows, fix.cols);
    out
}

pub fn cpu_oracle(fix: &Fixture) -> Vec<f32> {
    cpu(fix)
}

pub fn assert_row_invariants(out: &[f32], rows: usize, cols: usize) {
    assert_eq!(out.len(), rows * cols);
    for r in 0..rows {
        let row = &out[r * cols..(r + 1) * cols];
        assert!(row.iter().all(|v| v.is_finite() && *v >= 0.0));
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "row {r} sum={sum} (expected 1.0)");
        let max_p = row.iter().copied().fold(0.0f32, f32::max);
        if max_p > 0.0 {
            let ent: f32 = -row
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| p * p.ln())
                .sum::<f32>();
            let ln_n = (cols as f32).ln();
            assert!(
                ent <= ln_n + 1e-4,
                "row {r} entropy {ent} > ln({cols})={ln_n}"
            );
        }
    }
}

#[cfg(target_os = "macos")]
pub fn gpu(fix: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let len = fix.len();
    let buf = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Gpu("buffer alloc failed"))?;
    let dump_bytes = if variant.dump_stage > 0 {
        fix.rows * 8
    } else {
        4
    };
    let buf_dump = pool
        .allocate(&ctx.device, dump_bytes)
        .ok_or(Error::Gpu("buffer alloc failed"))?;

    BufferPool::write_f32(&buf, &fix.logits);

    let cmd = ctx
        .queue
        .commandBuffer()
        .ok_or(Error::Gpu("command buffer failed"))?;
    let enc = cmd
        .computeCommandEncoder()
        .ok_or(Error::Gpu("encoder failed"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    let dims = [fix.rows as u32, fix.cols as u32];
    bind_gpu_in_place(&enc, &buf, &buf_dump, &dims);
    let (grid, tg) = dispatch_shape(fix.rows);
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut out = vec![0.0f32; len];
    BufferPool::read_f32(&buf, &mut out);
    pool.release(len * 4, buf);
    pool.release(dump_bytes, buf_dump);
    Ok(out)
}

pub fn shader_source() -> &'static str {
    SHADER
}

#[cfg(target_os = "macos")]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER, ENTRY, variant)
}

#[cfg(target_os = "macos")]
pub fn dispatch_shape(rows: usize) -> (objc2_metal::MTLSize, objc2_metal::MTLSize) {
    use objc2_metal::MTLSize;
    let grid = MTLSize {
        width: 1,
        height: rows,
        depth: 1,
    };
    let tg = MTLSize {
        width: THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    (grid, tg)
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
    dims: &[u32; 2],
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(buf_dump), 0, 2);
    }
    set_bytes(enc, dims, 1);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::shaders::softmax_rows::cpu,
        cpu_oracle = crate::shaders::softmax_rows::cpu_oracle,
        gpu = crate::shaders::softmax_rows::gpu,
        fixture = crate::shaders::softmax_rows::tiny_fixture,
        out_len = crate::shaders::softmax_rows::fixture_len,
        formats: [F32],
        max_tol = 1e-5,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod router,
        cpu = crate::shaders::softmax_rows::cpu,
        cpu_oracle = crate::shaders::softmax_rows::cpu_oracle,
        gpu = crate::shaders::softmax_rows::gpu,
        fixture = crate::shaders::softmax_rows::router_fixture,
        out_len = crate::shaders::softmax_rows::fixture_len,
        formats: [F32],
        max_tol = 1e-5,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod wide_cols,
        cpu = crate::shaders::softmax_rows::cpu,
        cpu_oracle = crate::shaders::softmax_rows::cpu_oracle,
        gpu = crate::shaders::softmax_rows::gpu,
        fixture = crate::shaders::softmax_rows::wide_cols_fixture,
        out_len = crate::shaders::softmax_rows::fixture_len,
        formats: [F32],
        max_tol = 1e-5,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod canvas_vocab,
        cpu = crate::shaders::softmax_rows::cpu,
        cpu_oracle = crate::shaders::softmax_rows::cpu_oracle,
        gpu = crate::shaders::softmax_rows::gpu,
        fixture = crate::shaders::softmax_rows::canvas_vocab_fixture,
        out_len = crate::shaders::softmax_rows::fixture_len,
        formats: [F32],
        max_tol = 1e-4,
        min_cos = 0.9999,
    }

    #[test]
    fn cpu_row_invariants() {
        for fix in [
            tiny_fixture(ElemFormat::F32),
            router_fixture(ElemFormat::F32),
            wide_cols_fixture(ElemFormat::F32),
        ] {
            let out = cpu(&fix);
            assert_row_invariants(&out, fix.rows, fix.cols);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn gpu_row_invariants() {
        let fix = router_fixture(ElemFormat::F32);
        let out = gpu(&fix, KernelVariant::PRODUCTION).expect("gpu");
        assert_row_invariants(&out, fix.rows, fix.cols);
    }
}
