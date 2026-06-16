//! Soft embeddings and related GPU embed ops (`.dgq` q8 table).

use crate::kernels::sub::{gather_prob_cols, softmax_rows, vec_fill_zero, vec_scale_inplace};
use crate::metal::batched_kernels as bk;
use crate::metal::batch::{begin_engine_batch, GpuBatch};
use crate::metal::dgq_gpu::Q8LinearGpu;
use crate::metal::engine::GpuDecoderEngine;
use crate::metal::kernels::GpuKernels;
use crate::metal::linear::f32_q8_linear_kxn_gpu_bufs;
use crate::model::embed::LM_HEAD_CHUNK;
use crate::safetensors::Error;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

pub fn sc_log_enabled() -> bool {
    std::env::var("DGQ_LOG_SC").ok().as_deref() == Some("1")
}

#[derive(Debug, Clone, Copy)]
pub struct F32RowStats {
    pub min: f32,
    pub max: f32,
    pub n_nan: u32,
    pub n_inf: u32,
    pub sum: f32,
}

/// Sample one row of a shared f32 buffer (e.g. logits row 0).
pub fn f32_row_stats(buf: &ProtocolObject<dyn MTLBuffer>, row: usize, cols: usize) -> F32RowStats {
    let byte_off = row * cols * 4;
    let ptr = unsafe { buf.contents().as_ptr().add(byte_off) } as *const f32;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut n_nan = 0u32;
    let mut n_inf = 0u32;
    let mut sum = 0.0f32;
    for c in 0..cols {
        let v = unsafe { *ptr.add(c) };
        if v.is_nan() {
            n_nan += 1;
        } else if !v.is_finite() {
            n_inf += 1;
        } else {
            min = min.min(v);
            max = max.max(v);
            sum += v;
        }
    }
    if !min.is_finite() {
        min = 0.0;
    }
    if !max.is_finite() {
        max = 0.0;
    }
    F32RowStats {
        min,
        max,
        n_nan,
        n_inf,
        sum,
    }
}

fn log_engine_sc_softembed(label: &str, pre: F32RowStats, post_z: f32, post_max: f32, out: &[f32]) {
    let out_max = out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let out_finite = out.iter().filter(|v| v.is_finite()).count();
    eprintln!(
        "engine sc-softembed {label}: pre_exp min={:.4} max={:.4} nan={} inf={} | post_softmax Z={:.6} max_p={:.6} | out_max_abs={:.4} finite={}/{}",
        pre.min,
        pre.max,
        pre.n_nan,
        pre.n_inf,
        post_z,
        post_max,
        out_max,
        out_finite,
        out.len()
    );
}

#[derive(Debug, Clone, Copy)]
pub struct F32BufStats {
    pub finite: usize,
    pub total: usize,
    pub max_abs: f32,
}

pub fn f32_buf_stats(buf: &ProtocolObject<dyn MTLBuffer>, len: usize) -> F32BufStats {
    let ptr = buf.contents().as_ptr() as *const f32;
    let mut finite = 0usize;
    let mut max_abs = 0.0f32;
    for i in 0..len {
        let v = unsafe { *ptr.add(i) };
        if v.is_finite() {
            finite += 1;
            max_abs = max_abs.max(v.abs());
        }
    }
    F32BufStats {
        finite,
        total: len,
        max_abs,
    }
}

fn f32_row_sum(buf: &ProtocolObject<dyn MTLBuffer>, row: usize, cols: usize) -> f32 {
    let byte_off = row * cols * 4;
    let ptr = unsafe { buf.contents().as_ptr().add(byte_off) } as *const f32;
    let mut sum = 0.0f32;
    for c in 0..cols {
        let v = unsafe { *ptr.add(c) };
        if v.is_finite() {
            sum += v;
        }
    }
    sum
}

fn log_chunk0_bisect(
    chunk: usize,
    buf_probs: &ProtocolObject<dyn MTLBuffer>,
    buf_partial: &ProtocolObject<dyn MTLBuffer>,
    buf_out: &ProtocolObject<dyn MTLBuffer>,
    seq_len: usize,
    hidden: usize,
) {
    let probs_len = seq_len * chunk;
    let out_len = seq_len * hidden;
    let probs_row0 = f32_row_sum(buf_probs, 0, chunk);
    let p = f32_buf_stats(buf_probs, probs_len);
    let partial = f32_buf_stats(buf_partial, out_len);
    let out = f32_buf_stats(buf_out, out_len);
    eprintln!(
        "engine sc chunk-0 bisect: chunk={chunk} probs_row0_sum={probs_row0:.6} probs_finite={}/{} partial_finite={}/{} partial_max={:.4} out_finite={}/{} out_max={:.4}",
        p.finite,
        p.total,
        partial.finite,
        partial.total,
        partial.max_abs,
        out.finite,
        out.total,
        out.max_abs,
    );
}

pub fn softmax_rows_gpu_buf(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuKernels,
    buf: &ProtocolObject<dyn MTLBuffer>,
    seq_len: usize,
    cols: usize,
) {
    let buf_dump = batch.alloc_f32_out(1).expect("dump buf");
    let dims = [seq_len as u32, cols as u32];
    let (grid, tg) = softmax_rows::dispatch_shape(seq_len);
    batch.dispatch_with_grid(&kernels.softmax_rows.pipeline, grid, tg, |enc| {
        softmax_rows::bind_gpu_in_place(&enc, buf, &buf_dump, &dims);
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
    let buf_dump = batch.alloc_f32_out(1)?;
    let params = [
        seq_len as u32,
        vocab as u32,
        v0 as u32,
        chunk as u32,
    ];
    let (grid, tg) = gather_prob_cols::dispatch_shape(seq_len, chunk);
    batch.dispatch_with_grid(&kernels.gather_prob_cols.pipeline, grid, tg, |enc| {
        gather_prob_cols::bind_gpu_buffers(&enc, probs, &buf_out, &buf_dump, &params);
    });
    Ok(buf_out)
}

pub fn vec_fill_zero_gpu_buf(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuKernels,
    buf: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
) {
    let buf_dump = batch.alloc_f32_out(1).expect("dump buf");
    batch.dispatch_1d_ranged(&kernels.vec_fill_zero.pipeline, len, |enc, base, chunk| {
        vec_fill_zero::bind_gpu_buffers(&enc, buf, &buf_dump, base, chunk);
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
    let buf_dump = batch.alloc_f32_out(1).expect("dump buf");
    batch.dispatch_1d(&kernels.vec_scale.pipeline, len, |enc| {
        vec_scale_inplace::bind_gpu_buffers(&enc, buf, &buf_dump, scale, len_u);
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
        let buf_partial = f32_q8_linear_kxn_gpu_bufs(
            &mut batch,
            &engine.f32_q8_linear_kxn_pipeline,
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
    let pre_stats = if sc_log_enabled() {
        Some(f32_row_stats(logits_buf, 0, vocab))
    } else {
        None
    };
    let telemetry = engine.batch_telemetry();
    let mut batch = begin_engine_batch(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        telemetry,
    )?;
    softmax_rows_gpu_buf(&mut batch, &engine.kernels, logits_buf, seq_len, vocab);

    if sc_log_enabled() {
        batch.end()?;
        let row0_z = f32_row_stats(logits_buf, 0, vocab).sum;
        eprintln!(
            "engine sc post-softmax: row0_Z={row0_z:.6} (expect ~1.0; vocab={vocab} LM_HEAD_CHUNK={LM_HEAD_CHUNK} n_chunks={})",
            vocab.div_ceil(LM_HEAD_CHUNK)
        );
        let telemetry = engine.batch_telemetry();
        batch = begin_engine_batch(
            &engine.ctx.queue,
            &mut engine.pool,
            &engine.ctx.device,
            telemetry,
        )?;
    }

    let buf_out = batch.alloc_f32_out(out_len)?;
    vec_fill_zero_gpu_buf(&mut batch, &engine.kernels, &buf_out, out_len);

    let mut v0 = 0usize;
    while v0 < vocab {
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
        if sc_log_enabled() && v0 == 0 {
            crate::metal::dgq_gpu::log_q8_chunk_scale_histogram(&w_chunk, chunk, hidden);
        }
        let buf_partial = f32_q8_linear_kxn_gpu_bufs(
            &mut batch,
            &engine.f32_q8_linear_kxn_pipeline,
            &buf_probs,
            &w_chunk,
            seq_len,
            chunk,
            hidden,
        )?;
        bk::vec_add_gpu_bufs(&mut batch, &engine.kernels, &buf_out, &buf_partial, out_len)?;

        if sc_log_enabled() && v0 == 0 {
            batch.end()?;
            log_chunk0_bisect(chunk, &buf_probs, &buf_partial, &buf_out, seq_len, hidden);
            let telemetry = engine.batch_telemetry();
            batch = begin_engine_batch(
                &engine.ctx.queue,
                &mut engine.pool,
                &engine.ctx.device,
                telemetry,
            )?;
        }

        v0 = v1;
    }

    if scale != 1.0 {
        vec_scale_gpu_buf(&mut batch, &engine.kernels, &buf_out, out_len, scale);
    }
    batch.register_read(buf_out, out);
    batch.end()?;
    if let Some(pre) = pre_stats {
        let post = f32_row_stats(logits_buf, 0, vocab);
        log_engine_sc_softembed("gpu_buf", pre, post.sum, post.max, out);
    }
    Ok(())
}

/// Q8 embed-table gather for encoder prefill (`.dgq` only).
pub fn embed_token_ids_q8_gpu(
    engine: &mut GpuDecoderEngine,
    embed: &Q8LinearGpu,
    blob: &ProtocolObject<dyn MTLBuffer>,
    token_ids: &[u32],
    seq_len: usize,
    hidden: usize,
    out: &mut [f32],
) -> Result<(), Error> {
    let len = seq_len * hidden;
    if token_ids.len() != seq_len || out.len() < len {
        return Err(Error::Format("embed_token_ids_q8_gpu shape mismatch"));
    }
    use crate::dgq::embed_row::EMBED_SCALE;
    use crate::kernels::sub::{embed_gather, half_to_f32};

    let embed_pipeline = engine.kernels.embed_gather.pipeline.clone();
    let half_pipeline = engine.kernels.half_to_f32.pipeline.clone();
    let telemetry = engine.batch_telemetry();
    let mut batch = begin_engine_batch(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        telemetry,
    )?;
    let buf_ids = batch.alloc_bytes(unsafe {
        std::slice::from_raw_parts(
            token_ids.as_ptr() as *const u8,
            token_ids.len() * 4,
        )
    })?;
    let half_len = len * 2;
    let buf_half = batch
        .alloc_bytes(&vec![0u8; half_len])
        .map_err(|_| Error::Format("embed half alloc failed"))?;
    let buf_f32 = batch.alloc_f32_out(len)?;
    let dump = batch.alloc_f32_out(1)?;

    let w_off = embed.byte_offset;
    let (grid, tg) = embed_gather::dispatch_shape(hidden, seq_len);
    batch.dispatch_with_grid(
        &embed_pipeline,
        grid,
        tg,
        |enc| {
            embed_gather::bind_gpu_buffers(
                enc,
                blob,
                &buf_ids,
                &buf_half,
                w_off,
                hidden as u32,
                seq_len as u32,
                EMBED_SCALE,
                embed.out_dim as u32,
                None,
            );
        },
    );

    let base = 0u32;
    let len_u32 = len as u32;
    batch.dispatch_1d(&half_pipeline, len, |enc| {
        half_to_f32::bind_gpu_buffers(enc, &buf_half, &buf_f32, &dump, base, len_u32);
    });

    batch.register_read(buf_f32, &mut out[..len]);
    batch.end()
}
