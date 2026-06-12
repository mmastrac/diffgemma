//! Self-conditioning on GPU (`.dgq` q8 projections).

use crate::buffer::Buffer;
use crate::metal::batched_kernels as bk;
use crate::metal::batch::begin_engine_batch;
use crate::metal::engine::GpuDecoderEngine;
use crate::metal::dgq_gpu::Q8LinearGpu;
use crate::metal::linear::f32_q8_linear_gpu_bufs;
use crate::safetensors::Error;

pub struct GpuSelfConditioningWeights {
    pub pre_norm: Buffer<f32>,
    pub gate_proj: Q8LinearGpu,
    pub up_proj: Q8LinearGpu,
    pub down_proj: Q8LinearGpu,
}

pub fn apply_gpu(
    engine: &mut GpuDecoderEngine,
    weights: &GpuSelfConditioningWeights,
    hidden_a: &mut [f32],
    embed_buf: &[f32],
    soft_signal: &[f32],
    seq_len: usize,
    hidden: usize,
    inter: usize,
    eps: f32,
) -> Result<(), Error> {
    let len = seq_len * hidden;
    let act_len = seq_len * inter;
    if hidden_a.len() != len || embed_buf.len() != len || soft_signal.len() != len {
        return Err(Error::Format("self_conditioning gpu shape mismatch"));
    }

    let telemetry = engine.batch_telemetry();
    let mut batch = begin_engine_batch(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        telemetry,
    )?;
    let buf_soft = batch.alloc_f32(soft_signal)?;
    let buf_embed = batch.alloc_f32(embed_buf)?;
    let buf_normed = bk::rms_norm_rows_gpu_buf(
        &mut batch,
        &engine.kernels,
        &buf_soft,
        weights.pre_norm.as_slice(),
        seq_len,
        hidden,
        eps,
    )?;
    let buf_gate = f32_q8_linear_gpu_bufs(
        &mut batch,
        &engine.f32_q8_linear_pipeline,
        &buf_normed,
        &weights.gate_proj,
        seq_len,
        hidden,
        inter,
    )?;
    let buf_up = f32_q8_linear_gpu_bufs(
        &mut batch,
        &engine.f32_q8_linear_pipeline,
        &buf_normed,
        &weights.up_proj,
        seq_len,
        hidden,
        inter,
    )?;
    bk::gelu_pytorch_tanh_gpu_buf(&mut batch, &engine.kernels, &buf_gate, act_len)?;
    bk::swiglu_mul_gpu_bufs(&mut batch, &engine.kernels, &buf_gate, &buf_up, act_len)?;
    let buf_signal = f32_q8_linear_gpu_bufs(
        &mut batch,
        &engine.f32_q8_linear_pipeline,
        &buf_gate,
        &weights.down_proj,
        seq_len,
        inter,
        hidden,
    )?;
    let buf_sum = batch.alloc_f32(embed_buf)?;
    bk::vec_add_gpu_bufs(&mut batch, &engine.kernels, &buf_sum, &buf_signal, len)?;
    let buf_out = bk::rms_norm_rows_no_scale_gpu_buf(
        &mut batch,
        &engine.kernels,
        &buf_sum,
        seq_len,
        hidden,
        eps,
    )?;
    batch.register_read(buf_out, hidden_a);
    batch.end()
}
