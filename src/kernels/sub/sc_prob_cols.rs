//! Chunked SC softmax columns from logits + row stats (avoids full vocab prob matrix).

use super::bf16;
use super::f16;
use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::safetensors::Error;

pub const ENTRY: &str = "sc_prob_cols";
const TILE: usize = 16;
/// Prob up-scale before the fp16 GEMM tiles; must match `SC_PROB_GEMM_SCALE` in
/// shaders/include/sc_prob_scale.metal and src/metal/step_kernel.rs.
const SC_PROB_GEMM_SCALE: f32 = 4096.0;

const SHADER: &str = shader_include::include_metal!("kernels/sc_prob_cols.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub logits: Vec<f32>,
    pub rows: usize,
    pub vocab: usize,
    pub v0: usize,
    pub chunk: usize,
}

impl Fixture {
    pub fn rowstat(&self) -> Vec<f32> {
        crate::kernels::sub::logit_rowstats::cpu(&crate::kernels::sub::logit_rowstats::Fixture {
            logits: self.logits.clone(),
            rows: self.rows,
            cols: self.vocab,
        })
    }

    pub fn out_len(&self) -> usize {
        self.rows * self.chunk
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        logits: vec![1.0, 2.0, 3.0, 0.0, -1.0, 0.5],
        rows: 2,
        vocab: 3,
        v0: 0,
        chunk: 3,
    }
}

pub fn chunk_fixture(_: ElemFormat) -> Fixture {
    let rows = 2usize;
    let vocab = 512usize;
    let v0 = 128usize;
    let chunk = 64usize;
    let logits: Vec<f32> = (0..rows * vocab)
        .map(|i| (i as f32 % 17.0) - 8.0)
        .collect();
    Fixture {
        logits,
        rows,
        vocab,
        v0,
        chunk,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let rowstat = f.rowstat();
    let mut out = vec![0.0f32; f.out_len()];
    for row in 0..f.rows {
        let mx = rowstat[row * 2];
        let sum = rowstat[row * 2 + 1];
        for col in 0..f.chunk {
            let v = f.v0 + col;
            if v >= f.vocab {
                break;
            }
            let x = f.logits[row * f.vocab + v];
            // Kernel stores fp16(prob * SCALE); recover the prob (caller divides
            // SCALE back out) so the test compares probs at the kernel's precision.
            let prob = ((x - mx).exp()) / sum;
            out[row * f.chunk + col] = f16::round_half(prob * SC_PROB_GEMM_SCALE) / SC_PROB_GEMM_SCALE;
        }
    }
    out
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    let full = crate::kernels::sub::sc_probs::cpu(&crate::kernels::sub::sc_probs::Fixture {
        logits: f.logits.clone(),
        rows: f.rows,
        cols: f.vocab,
    });
    let mut out = vec![0.0f32; f.out_len()];
    for row in 0..f.rows {
        for col in 0..f.chunk {
            let v = f.v0 + col;
            if v >= f.vocab {
                break;
            }
            out[row * f.chunk + col] = full[row * f.vocab + v];
        }
    }
    out
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER, ENTRY, variant)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn dispatch_shape(rows: usize, chunk: usize) -> (objc2_metal::MTLSize, objc2_metal::MTLSize) {
    use objc2_metal::MTLSize;
    (
        MTLSize {
            width: gpu_common::div_up(chunk, TILE),
            height: gpu_common::div_up(rows, TILE),
            depth: 1,
        },
        MTLSize {
            width: TILE,
            height: TILE,
            depth: 1,
        },
    )
}

#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2::runtime::ProtocolObject;
#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn bind_gpu_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    logits: &ProtocolObject<dyn MTLBuffer>,
    rowstat: &ProtocolObject<dyn MTLBuffer>,
    probs: &ProtocolObject<dyn MTLBuffer>,
    params: &[u32; 4],
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(logits), 0, 0);
        enc.setBuffer_offset_atIndex(Some(rowstat), 0, 1);
        enc.setBuffer_offset_atIndex(Some(probs), 0, 2);
    }
    gpu_common::set_bytes(enc, params, 3);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let rowstat = f.rowstat();
    let buf_l = pool
        .allocate(&ctx.device, f.logits.len() * 2)
        .ok_or(Error::Format("alloc"))?;
    let buf_rs = pool
        .allocate(&ctx.device, rowstat.len() * 4)
        .ok_or(Error::Format("alloc"))?;
    let buf_o = pool
        .allocate(&ctx.device, f.out_len() * 2)
        .ok_or(Error::Format("alloc"))?;
    BufferPool::write_bf16(&buf_l, &bf16::f32_slice_to_bf16_bits(&f.logits));
    BufferPool::write_f32(&buf_rs, &rowstat);
    let params = [
        f.rows as u32,
        f.vocab as u32,
        f.v0 as u32,
        f.chunk as u32,
    ];
    let (grid, tg) = dispatch_shape(f.rows, f.chunk);
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    bind_gpu_buffers(&enc, &buf_l, &buf_rs, &buf_o, &params);
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    // Kernel writes fp16 probs scaled by SC_PROB_GEMM_SCALE; read fp16 and divide
    // the scale back out to recover the prob (matches the cpu reference).
    let ptr = buf_o.contents().as_ptr() as *const u16;
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
        cpu = crate::kernels::sub::sc_prob_cols::cpu,
        cpu_oracle = crate::kernels::sub::sc_prob_cols::cpu_oracle,
        gpu = crate::kernels::sub::sc_prob_cols::gpu,
        fixture = crate::kernels::sub::sc_prob_cols::tiny_fixture,
        out_len = crate::kernels::sub::sc_prob_cols::fixture_len,
        formats: [F32],
        max_tol = 1e-3,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod chunk,
        cpu = crate::kernels::sub::sc_prob_cols::cpu,
        cpu_oracle = crate::kernels::sub::sc_prob_cols::cpu_oracle,
        gpu = crate::kernels::sub::sc_prob_cols::gpu,
        fixture = crate::kernels::sub::sc_prob_cols::chunk_fixture,
        out_len = crate::kernels::sub::sc_prob_cols::fixture_len,
        formats: [F32],
        max_tol = 1e-2,
        min_cos = 0.999,
    }
}
