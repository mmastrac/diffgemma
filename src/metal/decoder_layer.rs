use crate::config::TextConfig;
use crate::kernels::cpu::gelu_pytorch_tanh;
use crate::metal::batched_kernels::{self as bk};
use crate::metal::batch::GpuBatch;
use crate::metal::engine::GpuDecoderEngine;
use crate::metal::linear::linear_cached_batched;
use crate::metal::moe::experts_forward_gpu_batched;
use crate::metal::weights::GpuLayerWeightCache;
use crate::model::attention::{
    forward_decoder as attention_forward_decoder, AttentionParams, AttentionScratch,
};
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
            attn: AttentionScratch::with_kv_len(seq_len, hidden, &attn_params, kv_cache_len),
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
) -> Result<(), Error> {
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let eps = cfg.rms_norm_eps as f32;

    attention_forward_decoder(
        &mut scratch.attn_out,
        hidden_states,
        weights,
        cfg,
        layer,
        seq_len,
        positions,
        kv,
        mask,
        &mut scratch.attn,
    )?;

    scratch.residual.copy_from_slice(hidden_states);

    // Batch 1: post-attn norm → dense gate/up linears
    {
        let mut batch = GpuBatch::begin(&engine.ctx.queue, &mut engine.pool, &engine.ctx.device)?;
        bk::rms_norm_rows(
            &mut batch,
            &engine.kernels,
            &mut scratch.normed,
            &scratch.attn_out,
            &cached.post_attn_norm,
            seq_len,
            hidden,
            eps,
        )?;
        bk::vec_add_inplace(&mut batch, &engine.kernels, &mut scratch.normed, &scratch.residual)?;
        scratch.residual.copy_from_slice(&scratch.normed);
        bk::rms_norm_rows(
            &mut batch,
            &engine.kernels,
            &mut scratch.normed,
            &scratch.residual,
            &cached.pre_ff_norm,
            seq_len,
            hidden,
            eps,
        )?;
        linear_cached_batched(
            &mut batch,
            &engine.f32_bf16_gemm_pipeline,
            &mut scratch.gate,
            &scratch.normed,
            &cached.mlp_gate,
            seq_len,
        )?;
        linear_cached_batched(
            &mut batch,
            &engine.f32_bf16_gemm_pipeline,
            &mut scratch.mlp_hidden,
            &scratch.normed,
            &cached.mlp_up,
            seq_len,
        )?;
        batch.end()?;
    }

    gelu_pytorch_tanh(&mut scratch.gate);
    for i in 0..scratch.mlp_hidden.len() {
        scratch.mlp_hidden[i] *= scratch.gate[i];
    }

    // Batch 2: down linear → post_ff_1 norm
    {
        let mut batch = GpuBatch::begin(&engine.ctx.queue, &mut engine.pool, &engine.ctx.device)?;
        linear_cached_batched(
            &mut batch,
            &engine.f32_bf16_gemm_pipeline,
            &mut scratch.dense_out,
            &scratch.mlp_hidden,
            &cached.mlp_down,
            seq_len,
        )?;
        bk::rms_norm_rows(
            &mut batch,
            &engine.kernels,
            &mut scratch.normed,
            &scratch.dense_out,
            &cached.post_ff_norm_1,
            seq_len,
            hidden,
            eps,
        )?;
        batch.end()?;
    }

    let routes = crate::model::moe::route(
        &scratch.residual,
        weights,
        cfg,
        seq_len,
        &mut scratch.cpu.moe,
    )?;

    // Batch 3: MoE input norm → expert GEMMs → final norms/add/scale
    {
        let mut batch = GpuBatch::begin(&engine.ctx.queue, &mut engine.pool, &engine.ctx.device)?;
        bk::rms_norm_rows(
            &mut batch,
            &engine.kernels,
            &mut scratch.moe_input,
            &scratch.residual,
            &cached.pre_ff_norm_2,
            seq_len,
            hidden,
            eps,
        )?;
        batch.end()?;
    }

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

    {
        let mut batch = GpuBatch::begin(&engine.ctx.queue, &mut engine.pool, &engine.ctx.device)?;
        scratch.dense_out.copy_from_slice(&scratch.moe_branch);
        bk::rms_norm_rows(
            &mut batch,
            &engine.kernels,
            &mut scratch.moe_branch,
            &scratch.dense_out,
            &cached.post_ff_norm_2,
            seq_len,
            hidden,
            eps,
        )?;
        bk::vec_add_inplace(
            &mut batch,
            &engine.kernels,
            &mut scratch.normed,
            &scratch.moe_branch,
        )?;
        bk::rms_norm_rows(
            &mut batch,
            &engine.kernels,
            out,
            &scratch.normed,
            &cached.post_ff_norm,
            seq_len,
            hidden,
            eps,
        )?;
        bk::vec_add_inplace(&mut batch, &engine.kernels, out, &scratch.residual)?;
        bk::vec_scale_inplace(&mut batch, &engine.kernels, out, cached.layer_scalar)?;
        batch.end()?;
    }

    Ok(())
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
