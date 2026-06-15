//! O(vocab*hidden) SC softembed fallback: weighted Q8 embed rows by softmax(logits).

use super::f16;
use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::dgq::block::{q8_weight_at, quantize_row_q8};
use crate::dgq::embed_row::EMBED_SCALE;
use crate::dgq::layout::q8_row_bytes;
use crate::safetensors::Error;

pub const ENTRY: &str = "sc_softembed";
pub const DIM_TILE: usize = 64;

const SHADER: &str = shader_include::include_metal!("kernels/sc_softembed.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub logits: Vec<f32>,
    pub embed_f32: Vec<f32>,
    pub rows: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub embed_scale: f32,
    pub first_step: bool,
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
        self.rows * self.hidden
    }

    pub fn embed_q8(&self) -> Vec<u8> {
        let row_bytes = q8_row_bytes(self.hidden);
        let mut dst = vec![0u8; self.vocab * row_bytes];
        for v in 0..self.vocab {
            let off = v * self.hidden;
            quantize_row_q8(
                &self.embed_f32[off..off + self.hidden],
                self.hidden,
                &mut dst[v * row_bytes..],
            );
        }
        dst
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    let rows = 2usize;
    let vocab = 8usize;
    let hidden = 16usize;
    Fixture {
        logits: (0..rows * vocab)
            .map(|i| (i as f32 * 0.11).sin() * 2.0)
            .collect(),
        embed_f32: (0..vocab * hidden)
            .map(|i| (i as f32 * 0.05).cos() * 0.04)
            .collect(),
        rows,
        hidden,
        vocab,
        embed_scale: EMBED_SCALE,
        first_step: false,
    }
}

pub fn first_step_fixture(_: ElemFormat) -> Fixture {
    let mut f = tiny_fixture(ElemFormat::F32);
    f.first_step = true;
    f
}

pub fn sc_softembed_cpu(f: &Fixture) -> Vec<f32> {
    let mut out = vec![0.0f32; f.out_len()];
    if f.first_step {
        return out;
    }
    let rowstat = f.rowstat();
    let embed_q8 = f.embed_q8();
    for tok in 0..f.rows {
        let mx = rowstat[tok * 2];
        let sum = rowstat[tok * 2 + 1];
        for d in 0..f.hidden {
            let mut acc = 0.0f32;
            for v in 0..f.vocab {
                let p = ((f.logits[tok * f.vocab + v] - mx).exp()) / sum;
                acc += p * q8_weight_at(&embed_q8, v, d, f.hidden);
            }
            out[tok * f.hidden + d] = f16::round_half(acc * f.embed_scale);
        }
    }
    out
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    sc_softembed_cpu(f)
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
pub fn dispatch_shape(hidden: usize, num_tokens: usize) -> (objc2_metal::MTLSize, objc2_metal::MTLSize) {
    use objc2_metal::MTLSize;
    (
        MTLSize {
            width: hidden.div_ceil(DIM_TILE),
            height: num_tokens,
            depth: 1,
        },
        MTLSize {
            width: DIM_TILE,
            height: 1,
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
    blob: &ProtocolObject<dyn MTLBuffer>,
    soft: &ProtocolObject<dyn MTLBuffer>,
    w_off: u64,
    first_step: u32,
    hidden: u32,
    num_tokens: u32,
    vocab: u32,
    embed_scale: f32,
) {
    let dims = [hidden, num_tokens, vocab];
    unsafe {
        enc.setBuffer_offset_atIndex(Some(logits), 0, 0);
        enc.setBuffer_offset_atIndex(Some(rowstat), 0, 1);
        enc.setBuffer_offset_atIndex(Some(blob), 0, 2);
        enc.setBuffer_offset_atIndex(Some(soft), 0, 3);
    }
    gpu_common::set_bytes(enc, &w_off, 4);
    gpu_common::set_bytes(enc, &first_step, 5);
    gpu_common::set_bytes(enc, &dims, 6);
    gpu_common::set_bytes(enc, &embed_scale, 7);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let rowstat = f.rowstat();
    let embed_q8 = f.embed_q8();
    let buf_logits = pool
        .allocate(&ctx.device, f.logits.len() * 2)
        .ok_or(Error::Format("alloc"))?;
    let buf_rowstat = pool
        .allocate(&ctx.device, rowstat.len() * 4)
        .ok_or(Error::Format("alloc"))?;
    let buf_blob = pool
        .allocate(&ctx.device, embed_q8.len())
        .ok_or(Error::Format("alloc"))?;
    let buf_out = pool
        .allocate(&ctx.device, f.out_len() * 2)
        .ok_or(Error::Format("alloc"))?;
    BufferPool::write_bf16(&buf_logits, &f16::f32_slice_to_f16(&f.logits));
    BufferPool::write_f32(&buf_rowstat, &rowstat);
    BufferPool::write_bytes(&buf_blob, &embed_q8);
    let first_step = u32::from(f.first_step);
    let (grid, tg) = dispatch_shape(f.hidden, f.rows);
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    bind_gpu_buffers(
        &enc,
        &buf_logits,
        &buf_rowstat,
        &buf_blob,
        &buf_out,
        0,
        first_step,
        f.hidden as u32,
        f.rows as u32,
        f.vocab as u32,
        f.embed_scale,
    );
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let ptr = buf_out.contents().as_ptr() as *const u16;
    Ok((0..f.out_len())
        .map(|i| f16::f16_bits_to_f32(unsafe { *ptr.add(i) }))
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
        cpu = crate::kernels::sub::sc_softembed::cpu,
        cpu_oracle = crate::kernels::sub::sc_softembed::cpu_oracle,
        gpu = crate::kernels::sub::sc_softembed::gpu,
        fixture = crate::kernels::sub::sc_softembed::tiny_fixture,
        out_len = crate::kernels::sub::sc_softembed::fixture_len,
        formats: [F32],
        max_tol = 0.01,
        min_cos = 0.999,
    }

    kernel_oracle_matrix! {
        mod first_step,
        cpu = crate::kernels::sub::sc_softembed::cpu,
        cpu_oracle = crate::kernels::sub::sc_softembed::cpu_oracle,
        gpu = crate::kernels::sub::sc_softembed::gpu,
        fixture = crate::kernels::sub::sc_softembed::first_step_fixture,
        out_len = crate::kernels::sub::sc_softembed::fixture_len,
        formats: [F32],
        max_tol = 1e-6,
        min_cos = 0.9999,
    }
}
