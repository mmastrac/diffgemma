use crate::config::TextConfig;
use crate::kernels::cpu::gelu_pytorch_tanh;
use crate::metal::batched_kernels::{self as bk};
use crate::metal::batch::GpuBatch;
use crate::metal::engine::GpuDecoderEngine;
use crate::metal::linear::{linear_cached_batched_in_buf, linear_cached_batched_in_cpu_out};
use crate::metal::moe::experts_forward_gpu_batched;
use crate::metal::weights::GpuLayerWeightCache;
use crate::metal::decoder_attention::{
    forward_decoder_attention, forward_encoder_extend_attention, forward_encoder_prefill_attention,
};
use crate::metal::kv_cache::GpuKvCache;
use crate::model::attention::{AttentionParams, AttentionScratch};
use crate::model::decoder_layer::{forward_decoder as cpu_forward_decoder, DecoderLayerScratch};
use crate::model::kv_cache::LayerKvView;
use crate::model::layer_weights::DecoderLayerWeights;
use crate::model::mask::DecoderAttnMask;
use crate::safetensors::Error;

pub struct GpuDecoderLayerScratch {
    pub cpu: DecoderLayerScratch,
    pub attn: AttentionScratch,
    pub attn_out: Vec<f32>,
    pub residual: Vec<f32>,
    pub normed: Vec<f32>,
    pub gate: Vec<f32>,
    pub mlp_hidden: Vec<f32>,
    pub dense_out: Vec<f32>,
    pub moe_branch: Vec<f32>,
    pub moe_input: Vec<f32>,
    pub moe_batch_input: Vec<f32>,
    pub moe_batch_gate_up: Vec<f32>,
    pub moe_batch_gate_act: Vec<f32>,
    pub moe_batch_out: Vec<f32>,
}

impl GpuDecoderLayerScratch {
    pub fn with_kv_len(
        seq_len: usize,
        cfg: &TextConfig,
        layer: usize,
        kv_cache_len: usize,
    ) -> Result<Self, Error> {
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let attn_params = AttentionParams::for_layer(cfg, layer)?;
        Ok(Self {
            cpu: DecoderLayerScratch::with_kv_len(seq_len, cfg, layer, kv_cache_len)?,
            attn: AttentionScratch::with_kv_len_gpu(seq_len, hidden, &attn_params, kv_cache_len),
            attn_out: vec![0.0; seq_len * hidden],
            residual: vec![0.0; seq_len * hidden],
            normed: vec![0.0; seq_len * hidden],
            gate: vec![0.0; seq_len * inter],
            mlp_hidden: vec![0.0; seq_len * inter],
            dense_out: vec![0.0; seq_len * hidden],
            moe_branch: vec![0.0; seq_len * hidden],
            moe_input: vec![0.0; seq_len * hidden],
            moe_batch_input: Vec::new(),
            moe_batch_gate_up: Vec::new(),
            moe_batch_gate_act: Vec::new(),
            moe_batch_out: Vec::new(),
        })
    }
}

fn rms_norm_batch(
    engine: &mut GpuDecoderEngine,
    out: &mut [f32],
    x: &[f32],
    weight: &[f32],
    seq_len: usize,
    hidden: usize,
    eps: f32,
) -> Result<(), Error> {
    let mut batch = GpuBatch::begin(&engine.ctx.queue, &mut engine.pool, &engine.ctx.device)?;
    bk::rms_norm_rows(
        &mut batch,
        &engine.kernels,
        out,
        x,
        weight,
        seq_len,
        hidden,
        eps,
    )?;
    batch.end()
}

pub fn forward_decoder(
    out: &mut [f32],
    hidden_states: &[f32],
    weights: &DecoderLayerWeights<'_>,
    cached: &GpuLayerWeightCache,
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    positions: &[i64],
    kv: LayerKvView<'_>,
    mask: Option<&DecoderAttnMask>,
    scratch: &mut GpuDecoderLayerScratch,
    engine: &mut GpuDecoderEngine,
    gpu_kv: Option<&GpuKvCache>,
) -> Result<(), Error> {
    let _hidden = cfg.hidden_size;

    forward_decoder_attention(
        &mut scratch.attn_out,
        hidden_states,
        cached,
        cfg,
        layer,
        seq_len,
        positions,
        kv,
        mask,
        &mut scratch.attn,
        engine,
        gpu_kv,
    )?;

    forward_layer_ff(
        out,
        hidden_states,
        weights,
        cached,
        cfg,
        seq_len,
        scratch,
        engine,
    )
}

fn forward_layer_ff(
    out: &mut [f32],
    hidden_states: &[f32],
    weights: &DecoderLayerWeights<'_>,
    cached: &GpuLayerWeightCache,
    cfg: &TextConfig,
    seq_len: usize,
    scratch: &mut GpuDecoderLayerScratch,
    engine: &mut GpuDecoderEngine,
) -> Result<(), Error> {
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps as f32;

    scratch.residual.copy_from_slice(hidden_states);

    rms_norm_batch(
        engine,
        &mut scratch.normed,
        &scratch.attn_out,
        cached.post_attn_norm.as_slice(),
        seq_len,
        hidden,
        eps,
    )?;
    for i in 0..scratch.normed.len() {
        scratch.normed[i] += scratch.residual[i];
    }
    scratch.residual.copy_from_slice(&scratch.normed);

    {
        let mut batch = GpuBatch::begin(&engine.ctx.queue, &mut engine.pool, &engine.ctx.device)?;
        let buf_normed = bk::rms_norm_rows_gpu(
            &mut batch,
            &engine.kernels,
            &scratch.residual,
            cached.pre_ff_norm.as_slice(),
            seq_len,
            hidden,
            eps,
        )?;
        linear_cached_batched_in_cpu_out(
            &mut batch,
            &engine.f32_bf16_gemm_pipeline,
            &mut scratch.gate,
            &buf_normed,
            &cached.mlp_gate,
            seq_len,
        )?;
        linear_cached_batched_in_cpu_out(
            &mut batch,
            &engine.f32_bf16_gemm_pipeline,
            &mut scratch.mlp_hidden,
            &buf_normed,
            &cached.mlp_up,
            seq_len,
        )?;
        batch.end()?;
    }

    gelu_pytorch_tanh(&mut scratch.gate);
    for i in 0..scratch.mlp_hidden.len() {
        scratch.mlp_hidden[i] *= scratch.gate[i];
    }

    {
        let mut batch = GpuBatch::begin(&engine.ctx.queue, &mut engine.pool, &engine.ctx.device)?;
        let buf_mlp = batch.alloc_f32(&scratch.mlp_hidden)?;
        let buf_down = linear_cached_batched_in_buf(
            &mut batch,
            &engine.f32_bf16_gemm_pipeline,
            &buf_mlp,
            &cached.mlp_down,
            seq_len,
        )?;
        let buf_normed = bk::rms_norm_rows_gpu_buf(
            &mut batch,
            &engine.kernels,
            &buf_down,
            cached.post_ff_norm_1.as_slice(),
            seq_len,
            hidden,
            eps,
        )?;
        batch.register_read(buf_normed, &mut scratch.normed);
        batch.end()?;
    }

    let routes = crate::model::moe::route_with_cached_weights(
        &scratch.residual,
        cached.router_proj.as_slice(),
        cached.router_scale.as_slice(),
        cached.per_expert_scale.as_slice(),
        cfg,
        seq_len,
        &mut scratch.cpu.moe,
    )?;

    rms_norm_batch(
        engine,
        &mut scratch.moe_input,
        &scratch.residual,
        cached.pre_ff_norm_2.as_slice(),
        seq_len,
        hidden,
        eps,
    )?;

    experts_forward_gpu_batched(
        &mut scratch.moe_branch,
        &scratch.moe_input,
        weights,
        cached,
        cfg,
        seq_len,
        &routes,
        &mut scratch.moe_batch_input,
        &mut scratch.moe_batch_gate_up,
        &mut scratch.moe_batch_gate_act,
        &mut scratch.moe_batch_out,
        &engine.ctx,
        &mut engine.pool,
        &engine.f32_bf16_gemm_pipeline,
    )?;

    scratch.dense_out.copy_from_slice(&scratch.moe_branch);
    rms_norm_batch(
        engine,
        &mut scratch.moe_branch,
        &scratch.dense_out,
        cached.post_ff_norm_2.as_slice(),
        seq_len,
        hidden,
        eps,
    )?;

    for i in 0..scratch.normed.len() {
        scratch.normed[i] += scratch.moe_branch[i];
    }

    rms_norm_batch(
        engine,
        out,
        &scratch.normed,
        cached.post_ff_norm.as_slice(),
        seq_len,
        hidden,
        eps,
    )?;
    for i in 0..out.len() {
        out[i] += scratch.residual[i];
    }
    for v in out.iter_mut() {
        *v *= cached.layer_scalar;
    }

    Ok(())
}

pub fn forward_encoder_prefill(
    out: &mut [f32],
    hidden_states: &[f32],
    weights: &DecoderLayerWeights<'_>,
    cached: &GpuLayerWeightCache,
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    positions: &[i64],
    scratch: &mut GpuDecoderLayerScratch,
    engine: &mut GpuDecoderEngine,
    gpu_kv: &GpuKvCache,
) -> Result<(), Error> {
    forward_encoder_prefill_attention(
        &mut scratch.attn_out,
        hidden_states,
        cached,
        cfg,
        layer,
        seq_len,
        positions,
        &mut scratch.attn,
        engine,
        gpu_kv,
    )?;

    forward_layer_ff(
        out,
        hidden_states,
        weights,
        cached,
        cfg,
        seq_len,
        scratch,
        engine,
    )
}

pub fn forward_encoder_extend(
    out: &mut [f32],
    hidden_states: &[f32],
    weights: &DecoderLayerWeights<'_>,
    cached: &GpuLayerWeightCache,
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    positions: &[i64],
    kv_cache_len: usize,
    scratch: &mut GpuDecoderLayerScratch,
    engine: &mut GpuDecoderEngine,
    gpu_kv: &GpuKvCache,
) -> Result<(), Error> {
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps as f32;

    forward_encoder_extend_attention(
        &mut scratch.attn_out,
        hidden_states,
        cached,
        cfg,
        layer,
        seq_len,
        positions,
        kv_cache_len,
        &mut scratch.attn,
        engine,
        gpu_kv,
    )?;

    forward_layer_ff(
        out,
        hidden_states,
        weights,
        cached,
        cfg,
        seq_len,
        scratch,
        engine,
    )
}

#[allow(dead_code)]
pub fn forward_decoder_cpu(
    out: &mut [f32],
    hidden_states: &[f32],
    weights: &DecoderLayerWeights<'_>,
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    positions: &[i64],
    kv: LayerKvView<'_>,
    mask: Option<&DecoderAttnMask>,
    scratch: &mut GpuDecoderLayerScratch,
    _engine: &mut GpuDecoderEngine,
) -> Result<(), Error> {
    cpu_forward_decoder(
        out,
        hidden_states,
        weights,
        cfg,
        layer,
        seq_len,
        positions,
        kv,
        mask,
        &mut scratch.cpu,
    )
}
