//! Tied lm_head on GPU (q8 embed weights from `.dgq` blob).

use crate::metal::batch::{GpuBatch, set_bytes};
use crate::metal::device::ComputePipeline;
use crate::metal::dgq_gpu::Q8LinearGpu;
use crate::metal::linear::f32_q8_linear_gpu_bufs;
use crate::metal::sampler::GpuLogitsBuf;
use crate::metal::sampler_kernels::GpuSamplerKernels;
use crate::model::embed::LM_HEAD_CHUNK;
use crate::safetensors::Error;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};

/// `logits = hidden @ embed^T` in vocab chunks; q8 weights read directly from the blob.
pub fn lm_head_tied_q8_gpu(
    batch: &mut GpuBatch<'_>,
    q8_pipeline: &ComputePipeline,
    hidden: &[f32],
    embed: &Q8LinearGpu,
    seq_len: usize,
    hidden_dim: usize,
    vocab_size: usize,
    logits: &mut [f32],
) -> Result<(), Error> {
    assert_eq!(hidden.len(), seq_len * hidden_dim);
    assert_eq!(logits.len(), seq_len * vocab_size);
    assert_eq!(embed.in_dim, hidden_dim);
    assert_eq!(embed.out_dim, vocab_size);

    let buf_a = batch.alloc_f32(hidden)?;
    logits.fill(0.0);

    for v0 in (0..vocab_size).step_by(LM_HEAD_CHUNK) {
        let v1 = (v0 + LM_HEAD_CHUNK).min(vocab_size);
        let chunk = v1 - v0;
        let w_chunk = embed.row_slice(v0, chunk);
        let buf_c = f32_q8_linear_gpu_bufs(
            batch,
            q8_pipeline,
            &buf_a,
            &w_chunk,
            seq_len,
            hidden_dim,
            chunk,
        )?;
        for s in 0..seq_len {
            let dst = &mut logits[s * vocab_size + v0..s * vocab_size + v1];
            batch.register_read_offset(buf_c.clone(), s * chunk, dst);
        }
    }
    Ok(())
}

/// Same as [`lm_head_tied_q8_gpu`] but accumulates into a GPU logits buffer (no CPU readback).
pub fn lm_head_tied_q8_gpu_buf(
    batch: &mut GpuBatch<'_>,
    q8_pipeline: &ComputePipeline,
    sampler_kernels: &GpuSamplerKernels,
    kernels: &crate::metal::kernels::GpuKernels,
    hidden: &[f32],
    embed: &Q8LinearGpu,
    logits: &mut GpuLogitsBuf,
    seq_len: usize,
    hidden_dim: usize,
    vocab_size: usize,
) -> Result<(), Error> {
    assert_eq!(hidden.len(), seq_len * hidden_dim);
    assert_eq!(embed.in_dim, hidden_dim);
    assert_eq!(embed.out_dim, vocab_size);

    let total = seq_len * vocab_size;
    let buf_logits = logits
        .as_buf()
        .ok_or(Error::Format("gpu logits buffer not allocated"))?;
    zero_logits_buf(batch, kernels, buf_logits, total)?;
    let buf_a = batch.alloc_f32(hidden)?;

    for v0 in (0..vocab_size).step_by(LM_HEAD_CHUNK) {
        let v1 = (v0 + LM_HEAD_CHUNK).min(vocab_size);
        let chunk = v1 - v0;
        let w_chunk = embed.row_slice(v0, chunk);
        let buf_c = f32_q8_linear_gpu_bufs(
            batch,
            q8_pipeline,
            &buf_a,
            &w_chunk,
            seq_len,
            hidden_dim,
            chunk,
        )?;
        let params = [seq_len as u32, chunk as u32, v0 as u32, vocab_size as u32];
        let tg_w = 16usize;
        let tg_h = 16usize;
        let grid = objc2_metal::MTLSize {
            width: crate::metal::batch::div_up(chunk, tg_w),
            height: crate::metal::batch::div_up(seq_len, tg_h),
            depth: 1,
        };
        let tg = objc2_metal::MTLSize {
            width: tg_w,
            height: tg_h,
            depth: 1,
        };
        batch.dispatch_with_grid(
            &sampler_kernels.scatter_vocab_chunk.pipeline,
            grid,
            tg,
            |enc| {
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&buf_c), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(buf_logits), 0, 1);
                }
                set_bytes(enc, &params, 2);
            },
        );
    }
    Ok(())
}

/// Zero-fill logits buffer before first chunk write (call once per forward).
pub fn zero_logits_buf(
    batch: &mut GpuBatch<'_>,
    kernels: &crate::metal::kernels::GpuKernels,
    buf: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
) -> Result<(), Error> {
    crate::metal::embed::vec_fill_zero_gpu_buf(batch, kernels, buf, len);
    Ok(())
}
