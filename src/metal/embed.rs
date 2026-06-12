//! Soft embeddings and related GPU embed ops (`.dgq` q8 table).

use crate::metal::batched_kernels as bk;
use crate::metal::batch::{begin_engine_batch, GpuBatch};
use crate::metal::dgq_gpu::Q8LinearGpu;
use crate::metal::engine::GpuDecoderEngine;
use crate::metal::kernels::GpuKernels;
use crate::metal::linear::f32_q8_linear_gpu_bufs;
use crate::model::embed::LM_HEAD_CHUNK;
use crate::safetensors::Error;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLSize};

pub fn softmax_rows_gpu_buf(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuKernels,
    buf: &ProtocolObject<dyn MTLBuffer>,
    seq_len: usize,
    cols: usize,
) {
    let dims = [seq_len as u32, cols as u32];
    let tg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    let grid = MTLSize {
        width: 1,
        height: seq_len,
        depth: 1,
    };
    batch.dispatch_with_grid(&kernels.softmax_rows.pipeline, grid, tg, |enc| {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf), 0, 0);
        }
        crate::metal::batch::set_bytes(enc, &dims, 1);
    });
}

pub fn gather_prob_cols_gpu_buf(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuKernels,
    probs: &ProtocolObject<dyn MTLBuffer>,
    seq_len: usize,
    vocab: usize,
    v0: usize,
    chunk: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    let buf_out = batch.alloc_f32_out(seq_len * chunk)?;
    let params = [
        seq_len as u32,
        vocab as u32,
        v0 as u32,
        chunk as u32,
    ];
    let tg = MTLSize {
        width: 16,
        height: 16,
        depth: 1,
    };
    let grid = MTLSize {
        width: (chunk + 15) / 16,
        height: seq_len,
        depth: 1,
    };
    batch.dispatch_with_grid(&kernels.gather_prob_cols.pipeline, grid, tg, |enc| {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(probs), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&buf_out), 0, 1);
        }
        crate::metal::batch::set_bytes(enc, &params, 2);
    });
    Ok(buf_out)
}

pub fn vec_fill_zero_gpu_buf(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuKernels,
    buf: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
) {
    let len_u = len as u32;
    batch.dispatch_1d(&kernels.vec_fill_zero.pipeline, len, |enc| {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf), 0, 0);
        }
        crate::metal::batch::set_bytes(enc, &len_u, 1);
    });
}

pub fn vec_scale_gpu_buf(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuKernels,
    buf: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
    scale: f32,
) {
    let len_u = len as u32;
    batch.dispatch_1d(&kernels.vec_scale.pipeline, len, |enc| {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf), 0, 0);
        }
        crate::metal::batch::set_bytes(enc, &scale, 1);
        crate::metal::batch::set_bytes(enc, &len_u, 2);
    });
}

/// `out[s,h] = scale * sum_v softmax(logits[s,v]) * embed[v,h]` via chunked q8 GEMMs.
pub fn soft_embeddings_q8_gpu(
    engine: &mut GpuDecoderEngine,
    logits: &[f32],
    embed: &Q8LinearGpu,
    seq_len: usize,
    vocab: usize,
    hidden: usize,
    scale: f32,
    out: &mut [f32],
) -> Result<(), Error> {
    assert_eq!(logits.len(), seq_len * vocab);
    assert_eq!(out.len(), seq_len * hidden);
    assert_eq!(embed.in_dim, hidden);
    assert_eq!(embed.out_dim, vocab);

    let out_len = seq_len * hidden;
    let telemetry = engine.batch_telemetry();
    let mut batch = begin_engine_batch(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        telemetry,
    )?;
    let buf_logits = batch.alloc_f32(logits)?;
    softmax_rows_gpu_buf(&mut batch, &engine.kernels, &buf_logits, seq_len, vocab);

    let buf_out = batch.alloc_f32_out(out_len)?;
    vec_fill_zero_gpu_buf(&mut batch, &engine.kernels, &buf_out, out_len);

    for v0 in (0..vocab).step_by(LM_HEAD_CHUNK) {
        let v1 = (v0 + LM_HEAD_CHUNK).min(vocab);
        let chunk = v1 - v0;
        let buf_probs = gather_prob_cols_gpu_buf(
            &mut batch,
            &engine.kernels,
            &buf_logits,
            seq_len,
            vocab,
            v0,
            chunk,
        )?;
        let w_chunk = embed.row_slice(v0, chunk);
        let buf_partial = f32_q8_linear_gpu_bufs(
            &mut batch,
            &engine.f32_q8_linear_pipeline,
            &buf_probs,
            &w_chunk,
            seq_len,
            chunk,
            hidden,
        )?;
        bk::vec_add_gpu_bufs(&mut batch, &engine.kernels, &buf_out, &buf_partial, out_len)?;
    }

    if scale != 1.0 {
        vec_scale_gpu_buf(&mut batch, &engine.kernels, &buf_out, out_len, scale);
    }
    batch.register_read(buf_out, out);
    batch.end()
}

/// Soft embeddings from logits already on GPU (temperature-scaled path for self-conditioning).
pub fn soft_embeddings_q8_gpu_from_buf(
    engine: &mut GpuDecoderEngine,
    logits_buf: &ProtocolObject<dyn MTLBuffer>,
    embed: &Q8LinearGpu,
    seq_len: usize,
    vocab: usize,
    hidden: usize,
    scale: f32,
    out: &mut [f32],
) -> Result<(), Error> {
    assert_eq!(out.len(), seq_len * hidden);
    assert_eq!(embed.in_dim, hidden);
    assert_eq!(embed.out_dim, vocab);

    let out_len = seq_len * hidden;
    let telemetry = engine.batch_telemetry();
    let mut batch = begin_engine_batch(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        telemetry,
    )?;
    softmax_rows_gpu_buf(&mut batch, &engine.kernels, logits_buf, seq_len, vocab);

    let buf_out = batch.alloc_f32_out(out_len)?;
    vec_fill_zero_gpu_buf(&mut batch, &engine.kernels, &buf_out, out_len);

    for v0 in (0..vocab).step_by(LM_HEAD_CHUNK) {
        let v1 = (v0 + LM_HEAD_CHUNK).min(vocab);
        let chunk = v1 - v0;
        let buf_probs = gather_prob_cols_gpu_buf(
            &mut batch,
            &engine.kernels,
            logits_buf,
            seq_len,
            vocab,
            v0,
            chunk,
        )?;
        let w_chunk = embed.row_slice(v0, chunk);
        let buf_partial = f32_q8_linear_gpu_bufs(
            &mut batch,
            &engine.f32_q8_linear_pipeline,
            &buf_probs,
            &w_chunk,
            seq_len,
            chunk,
            hidden,
        )?;
        bk::vec_add_gpu_bufs(&mut batch, &engine.kernels, &buf_out, &buf_partial, out_len)?;
    }

    if scale != 1.0 {
        vec_scale_gpu_buf(&mut batch, &engine.kernels, &buf_out, out_len, scale);
    }
    batch.register_read(buf_out, out);
    batch.end()
}
