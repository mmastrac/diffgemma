//! Materialize softmax rows from logits + precomputed row stats (SC GEMM path).

use super::bf16;
use super::f16;
use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::safetensors::Error;

pub const ENTRY: &str = "sc_probs";
pub const THREADGROUP_WIDTH: usize = 256;
/// Prob up-scale before the fp16 GEMM tiles; must match `SC_PROB_GEMM_SCALE` in
/// shaders/include/sc_prob_scale.metal and src/metal/step_kernel.rs.
const SC_PROB_GEMM_SCALE: f32 = 4096.0;

const SHADER: &str = shader_include::include_metal!("kernels/sc_probs.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub logits: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Fixture {
    pub fn rowstat(&self) -> Vec<f32> {
        crate::kernels::sub::logit_rowstats::cpu(&crate::kernels::sub::logit_rowstats::Fixture {
            logits: self.logits.clone(),
            rows: self.rows,
            cols: self.cols,
        })
    }

    pub fn out_len(&self) -> usize {
        self.rows * self.cols
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        logits: vec![1.0, 2.0, 3.0, 0.0, -1.0, 0.5],
        rows: 2,
        cols: 3,
    }
}

pub fn wide_fixture(_: ElemFormat) -> Fixture {
    let rows = 2usize;
    let cols = 512usize;
    let logits: Vec<f32> = (0..rows * cols).map(|i| (i as f32 % 17.0) - 8.0).collect();
    Fixture { logits, rows, cols }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let rowstat = f.rowstat();
    let mut out = vec![0.0f32; f.out_len()];
    for row in 0..f.rows {
        let mx = rowstat[row * 2];
        let sum = rowstat[row * 2 + 1];
        for v in 0..f.cols {
            let x = f.logits[row * f.cols + v];
            // Kernel stores fp16(prob * SCALE); recover the prob (caller divides
            // SCALE back out) so the test compares probs at the kernel's precision.
            let prob = ((x - mx).exp()) / sum;
            out[row * f.cols + v] = f16::round_half(prob * SC_PROB_GEMM_SCALE) / SC_PROB_GEMM_SCALE;
        }
    }
    out
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER, ENTRY, variant)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn dispatch_shape(rows: usize) -> (objc2_metal::MTLSize, objc2_metal::MTLSize) {
    crate::kernels::sub::logit_rowstats::dispatch_shape(rows)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2::runtime::ProtocolObject;
#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn bind_gpu_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    logits: &ProtocolObject<dyn MTLBuffer>,
    rowstat: &ProtocolObject<dyn MTLBuffer>,
    probs: &ProtocolObject<dyn MTLBuffer>,
    dims: &[u32; 2],
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(logits), 0, 0);
        enc.setBuffer_offset_atIndex(Some(rowstat), 0, 1);
        enc.setBuffer_offset_atIndex(Some(probs), 0, 2);
    }
    gpu_common::set_bytes(enc, dims, 3);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let rowstat = f.rowstat();
    let buf_logits = pool
        .allocate(&ctx.device, f.logits.len() * 2)
        .ok_or(Error::Format("alloc"))?;
    let buf_rowstat = pool
        .allocate(&ctx.device, rowstat.len() * 4)
        .ok_or(Error::Format("alloc"))?;
    let buf_probs = pool
        .allocate(&ctx.device, f.out_len() * 2)
        .ok_or(Error::Format("alloc"))?;
    BufferPool::write_bf16(&buf_logits, &bf16::f32_slice_to_bf16_bits(&f.logits));
    BufferPool::write_f32(&buf_rowstat, &rowstat);
    let dims = [f.rows as u32, f.cols as u32];
    let (grid, tg) = dispatch_shape(f.rows);
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    bind_gpu_buffers(&enc, &buf_logits, &buf_rowstat, &buf_probs, &dims);
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    // Kernel writes fp16 probs scaled by SC_PROB_GEMM_SCALE; read fp16 and divide
    // the scale back out to recover the prob (matches the cpu reference).
    let ptr = buf_probs.contents().as_ptr() as *const u16;
    Ok((0..f.out_len())
        .map(|i| f16::f16_bits_to_f32(unsafe { *ptr.add(i) }) / SC_PROB_GEMM_SCALE)
        .collect())
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn gpu(_: &Fixture, _: KernelVariant) -> Result<Vec<f32>, Error> {
    Err(Error::Format("Metal unavailable"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::kernels::sub::sc_probs::cpu,
        cpu_oracle = crate::kernels::sub::sc_probs::cpu_oracle,
        gpu = crate::kernels::sub::sc_probs::gpu,
        fixture = crate::kernels::sub::sc_probs::tiny_fixture,
        out_len = crate::kernels::sub::sc_probs::fixture_len,
        formats: [F32],
        max_tol = 1e-3,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod wide,
        cpu = crate::kernels::sub::sc_probs::cpu,
        cpu_oracle = crate::kernels::sub::sc_probs::cpu_oracle,
        gpu = crate::kernels::sub::sc_probs::gpu,
        fixture = crate::kernels::sub::sc_probs::wide_fixture,
        out_len = crate::kernels::sub::sc_probs::fixture_len,
        formats: [F32],
        max_tol = 1e-2,
        min_cos = 0.9999,
    }

    #[test]
    fn cpu_rows_sum_to_one() {
        for fix in [tiny_fixture(ElemFormat::F32), wide_fixture(ElemFormat::F32)] {
            let out = cpu(&fix);
            for row in 0..fix.rows {
                let s: f32 = out[row * fix.cols..(row + 1) * fix.cols].iter().sum();
                assert!((s - 1.0).abs() < 0.01, "row {row} sum={s}");
            }
        }
    }
}
