use crate::config::TextConfig;
use crate::kernels::cpu::gelu_pytorch_tanh;
use crate::metal::engine::GpuDecoderEngine;
use crate::metal::linear::linear_bf16;
use crate::metal::moe::{experts_forward_gpu, GpuMoeScratch};
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
    pub moe: GpuMoeScratch,
    pub attn_out: Vec<f32>,
    pub residual: Vec<f32>,
    pub normed: Vec<f32>,
    pub gate: Vec<f32>,
    pub mlp_hidden: Vec<f32>,
    pub dense_out: Vec<f32>,
    pub moe_branch: Vec<f32>,
    pub moe_input: Vec<f32>,
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
            moe: GpuMoeScratch::new(seq_len, cfg),
            attn_out: vec![0.0; seq_len * hidden],
            residual: vec![0.0; seq_len * hidden],
            normed: vec![0.0; seq_len * hidden],
            gate: vec![0.0; seq_len * inter],
            mlp_hidden: vec![0.0; seq_len * inter],
            dense_out: vec![0.0; seq_len * hidden],
            moe_branch: vec![0.0; seq_len * hidden],
            moe_input: vec![0.0; seq_len * hidden],
        })
    }
}

pub fn forward_decoder(
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

    let post_attn_w = weights.post_attention_layernorm.bf16()?.to_f32_vec();
    scratch.residual.copy_from_slice(hidden_states);
    engine.kernels.rms_norm_rows(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        &mut scratch.normed,
        &scratch.attn_out,
        &post_attn_w,
        seq_len,
        hidden,
        eps,
    )?;
    engine.kernels.vec_add_inplace(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        &mut scratch.normed,
        &scratch.residual,
    )?;

    scratch.residual.copy_from_slice(&scratch.normed);
    let pre_ff_w = weights.pre_feedforward_layernorm.bf16()?.to_f32_vec();
    engine.kernels.rms_norm_rows(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        &mut scratch.normed,
        &scratch.residual,
        &pre_ff_w,
        seq_len,
        hidden,
        eps,
    )?;

    linear_bf16(
        &mut engine.gemm,
        &mut scratch.gate,
        &scratch.normed,
        weights.mlp_gate.bf16()?,
        seq_len,
        hidden,
        inter,
    )?;
    linear_bf16(
        &mut engine.gemm,
        &mut scratch.mlp_hidden,
        &scratch.normed,
        weights.mlp_up.bf16()?,
        seq_len,
        hidden,
        inter,
    )?;
    gelu_pytorch_tanh(&mut scratch.gate);
    for i in 0..scratch.mlp_hidden.len() {
        scratch.mlp_hidden[i] *= scratch.gate[i];
    }
    linear_bf16(
        &mut engine.gemm,
        &mut scratch.dense_out,
        &scratch.mlp_hidden,
        weights.mlp_down.bf16()?,
        seq_len,
        inter,
        hidden,
    )?;

    let post_ff_1_w = weights.post_feedforward_layernorm_1.bf16()?.to_f32_vec();
    engine.kernels.rms_norm_rows(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        &mut scratch.normed,
        &scratch.dense_out,
        &post_ff_1_w,
        seq_len,
        hidden,
        eps,
    )?;

    // CPU router for now: GPU router logits differ enough to change top-k vs CPU.
    let routes = crate::model::moe::route(
        &scratch.residual,
        weights,
        cfg,
        seq_len,
        &mut scratch.cpu.moe,
    )?;

    let pre_ff_2_w = weights.pre_feedforward_layernorm_2.bf16()?.to_f32_vec();
    engine.kernels.rms_norm_rows(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        &mut scratch.moe_input,
        &scratch.residual,
        &pre_ff_2_w,
        seq_len,
        hidden,
        eps,
    )?;
    experts_forward_gpu(
        &mut scratch.moe_branch,
        &scratch.moe_input,
        weights,
        cfg,
        seq_len,
        &routes,
        &mut scratch.moe,
        &engine.ctx,
        &mut engine.pool,
        &engine.kernels,
        &engine.f32_bf16_gemm_pipeline,
    )?;

    scratch.dense_out.copy_from_slice(&scratch.moe_branch);
    let post_ff_2_w = weights.post_feedforward_layernorm_2.bf16()?.to_f32_vec();
    engine.kernels.rms_norm_rows(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        &mut scratch.moe_branch,
        &scratch.dense_out,
        &post_ff_2_w,
        seq_len,
        hidden,
        eps,
    )?;
    engine.kernels.vec_add_inplace(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        &mut scratch.normed,
        &scratch.moe_branch,
    )?;

    let post_ff_w = weights.post_feedforward_layernorm.bf16()?.to_f32_vec();
    engine.kernels.rms_norm_rows(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        out,
        &scratch.normed,
        &post_ff_w,
        seq_len,
        hidden,
        eps,
    )?;
    engine.kernels.vec_add_inplace(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        out,
        &scratch.residual,
    )?;

    let layer_scalar = weights.layer_scalar.bf16_scalar()?;
    engine.kernels.vec_scale_inplace(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        out,
        layer_scalar,
    )?;
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
