use crate::config::TextConfig;
use crate::kernels::cpu::{gelu_pytorch_tanh, linear, rms_norm_rows};
use crate::model::attention::{
    forward as attention_forward, forward_decoder as attention_forward_decoder, AttentionScratch,
};
use crate::model::kv_cache::LayerKvView;
use crate::model::mask::DecoderAttnMask;
use crate::model::layer_weights::DecoderLayerWeights;
use crate::model::moe::{experts_forward, prepare_expert_input, route, MoeScratch};
use crate::safetensors::Error;

pub struct DecoderLayerScratch {
    pub attn: AttentionScratch,
    pub moe: MoeScratch,
    pub attn_out: Vec<f32>,
    pub residual: Vec<f32>,
    pub normed: Vec<f32>,
    pub mlp_hidden: Vec<f32>,
    pub gate: Vec<f32>,
    pub dense_out: Vec<f32>,
    pub moe_branch: Vec<f32>,
    pub moe_input: Vec<f32>,
    pub mlp_gate_w: Vec<f32>,
    pub mlp_up_w: Vec<f32>,
    pub mlp_down_w: Vec<f32>,
    pub post_attn_norm_w: Vec<f32>,
    pub pre_ff_norm_w: Vec<f32>,
    pub post_ff_norm_w: Vec<f32>,
    pub post_ff_norm_1_w: Vec<f32>,
    pub pre_ff_norm_2_w: Vec<f32>,
    pub post_ff_norm_2_w: Vec<f32>,
}

impl DecoderLayerScratch {
    pub fn new(seq_len: usize, cfg: &TextConfig, layer: usize) -> Result<Self, Error> {
        Self::with_kv_len(seq_len, cfg, layer, 0)
    }

    pub fn with_kv_len(
        seq_len: usize,
        cfg: &TextConfig,
        layer: usize,
        kv_cache_len: usize,
    ) -> Result<Self, Error> {
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let attn_params = crate::model::attention::AttentionParams::for_layer(cfg, layer)?;
        Ok(Self {
            attn: AttentionScratch::with_kv_len(seq_len, hidden, &attn_params, kv_cache_len),
            moe: MoeScratch::new(seq_len, cfg),
            attn_out: vec![0.0; seq_len * hidden],
            residual: vec![0.0; seq_len * hidden],
            normed: vec![0.0; seq_len * hidden],
            mlp_hidden: vec![0.0; seq_len * inter],
            gate: vec![0.0; seq_len * inter],
            dense_out: vec![0.0; seq_len * hidden],
            moe_branch: vec![0.0; seq_len * hidden],
            moe_input: vec![0.0; seq_len * hidden],
            mlp_gate_w: Vec::new(),
            mlp_up_w: Vec::new(),
            mlp_down_w: Vec::new(),
            post_attn_norm_w: Vec::new(),
            pre_ff_norm_w: Vec::new(),
            post_ff_norm_w: Vec::new(),
            post_ff_norm_1_w: Vec::new(),
            pre_ff_norm_2_w: Vec::new(),
            post_ff_norm_2_w: Vec::new(),
        })
    }

    fn load_norm_and_mlp(&mut self, weights: &DecoderLayerWeights<'_>) -> Result<(), Error> {
        self.post_attn_norm_w = weights.post_attention_layernorm.bf16()?.to_f32_vec();
        self.pre_ff_norm_w = weights.pre_feedforward_layernorm.bf16()?.to_f32_vec();
        self.post_ff_norm_w = weights.post_feedforward_layernorm.bf16()?.to_f32_vec();
        self.post_ff_norm_1_w = weights.post_feedforward_layernorm_1.bf16()?.to_f32_vec();
        self.pre_ff_norm_2_w = weights.pre_feedforward_layernorm_2.bf16()?.to_f32_vec();
        self.post_ff_norm_2_w = weights.post_feedforward_layernorm_2.bf16()?.to_f32_vec();
        self.mlp_gate_w = weights.mlp_gate.bf16()?.to_f32_vec();
        self.mlp_up_w = weights.mlp_up.bf16()?.to_f32_vec();
        self.mlp_down_w = weights.mlp_down.bf16()?.to_f32_vec();
        Ok(())
    }
}

pub fn forward(
    out: &mut [f32],
    hidden_states: &[f32],
    weights: &DecoderLayerWeights<'_>,
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    positions: &[i64],
    scratch: &mut DecoderLayerScratch,
) -> Result<(), Error> {
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let eps = cfg.rms_norm_eps as f32;

    assert_eq!(hidden_states.len(), seq_len * hidden);
    assert_eq!(out.len(), seq_len * hidden);
    assert_eq!(positions.len(), seq_len);

    scratch.load_norm_and_mlp(weights)?;

    // Attention sub-layer
    scratch.residual.copy_from_slice(hidden_states);
    attention_forward(
        &mut scratch.attn_out,
        hidden_states,
        weights,
        cfg,
        layer,
        seq_len,
        positions,
        &mut scratch.attn,
    )?;
    rms_norm_rows(
        &mut scratch.normed,
        &scratch.attn_out,
        &scratch.post_attn_norm_w,
        seq_len,
        hidden,
        eps,
    );
    for i in 0..scratch.normed.len() {
        scratch.normed[i] += scratch.residual[i];
    }

    // Dense MLP + MoE (HF Gemma4TextDecoderLayer)
    scratch.residual.copy_from_slice(&scratch.normed);
    rms_norm_rows(
        &mut scratch.normed,
        &scratch.residual,
        &scratch.pre_ff_norm_w,
        seq_len,
        hidden,
        eps,
    );
    linear(
        &mut scratch.gate,
        &scratch.normed,
        &scratch.mlp_gate_w,
        None,
        seq_len,
        hidden,
        inter,
    );
    linear(
        &mut scratch.mlp_hidden,
        &scratch.normed,
        &scratch.mlp_up_w,
        None,
        seq_len,
        hidden,
        inter,
    );
    gelu_pytorch_tanh(&mut scratch.gate);
    for i in 0..scratch.mlp_hidden.len() {
        scratch.mlp_hidden[i] *= scratch.gate[i];
    }
    linear(
        &mut scratch.dense_out,
        &scratch.mlp_hidden,
        &scratch.mlp_down_w,
        None,
        seq_len,
        inter,
        hidden,
    );

    rms_norm_rows(
        &mut scratch.normed,
        &scratch.dense_out,
        &scratch.post_ff_norm_1_w,
        seq_len,
        hidden,
        eps,
    );

    let routes = route(&scratch.residual, weights, cfg, seq_len, &mut scratch.moe)?;
    prepare_expert_input(
        &mut scratch.moe_input,
        &scratch.residual,
        &scratch.pre_ff_norm_2_w,
        seq_len,
        hidden,
        eps,
    );
    experts_forward(
        &mut scratch.moe_branch,
        &scratch.moe_input,
        weights,
        cfg,
        seq_len,
        &routes,
        &mut scratch.moe,
    )?;
    scratch.dense_out.copy_from_slice(&scratch.moe_branch);
    rms_norm_rows(
        &mut scratch.moe_branch,
        &scratch.dense_out,
        &scratch.post_ff_norm_2_w,
        seq_len,
        hidden,
        eps,
    );

    for i in 0..scratch.normed.len() {
        scratch.normed[i] += scratch.moe_branch[i];
    }

    rms_norm_rows(
        out,
        &scratch.normed,
        &scratch.post_ff_norm_w,
        seq_len,
        hidden,
        eps,
    );
    for i in 0..out.len() {
        out[i] += scratch.residual[i];
    }

    let layer_scalar = weights.layer_scalar.bf16_scalar()?;
    for v in out.iter_mut() {
        *v *= layer_scalar;
    }
    Ok(())
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
    scratch: &mut DecoderLayerScratch,
) -> Result<(), Error> {
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let eps = cfg.rms_norm_eps as f32;

    assert_eq!(hidden_states.len(), seq_len * hidden);
    assert_eq!(out.len(), seq_len * hidden);
    assert_eq!(positions.len(), seq_len);

    scratch.load_norm_and_mlp(weights)?;

    scratch.residual.copy_from_slice(hidden_states);
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
    rms_norm_rows(
        &mut scratch.normed,
        &scratch.attn_out,
        &scratch.post_attn_norm_w,
        seq_len,
        hidden,
        eps,
    );
    for i in 0..scratch.normed.len() {
        scratch.normed[i] += scratch.residual[i];
    }

    scratch.residual.copy_from_slice(&scratch.normed);
    rms_norm_rows(
        &mut scratch.normed,
        &scratch.residual,
        &scratch.pre_ff_norm_w,
        seq_len,
        hidden,
        eps,
    );
    linear(
        &mut scratch.gate,
        &scratch.normed,
        &scratch.mlp_gate_w,
        None,
        seq_len,
        hidden,
        inter,
    );
    linear(
        &mut scratch.mlp_hidden,
        &scratch.normed,
        &scratch.mlp_up_w,
        None,
        seq_len,
        hidden,
        inter,
    );
    gelu_pytorch_tanh(&mut scratch.gate);
    for i in 0..scratch.mlp_hidden.len() {
        scratch.mlp_hidden[i] *= scratch.gate[i];
    }
    linear(
        &mut scratch.dense_out,
        &scratch.mlp_hidden,
        &scratch.mlp_down_w,
        None,
        seq_len,
        inter,
        hidden,
    );

    rms_norm_rows(
        &mut scratch.normed,
        &scratch.dense_out,
        &scratch.post_ff_norm_1_w,
        seq_len,
        hidden,
        eps,
    );

    let routes = route(&scratch.residual, weights, cfg, seq_len, &mut scratch.moe)?;
    prepare_expert_input(
        &mut scratch.moe_input,
        &scratch.residual,
        &scratch.pre_ff_norm_2_w,
        seq_len,
        hidden,
        eps,
    );
    experts_forward(
        &mut scratch.moe_branch,
        &scratch.moe_input,
        weights,
        cfg,
        seq_len,
        &routes,
        &mut scratch.moe,
    )?;
    scratch.dense_out.copy_from_slice(&scratch.moe_branch);
    rms_norm_rows(
        &mut scratch.moe_branch,
        &scratch.dense_out,
        &scratch.post_ff_norm_2_w,
        seq_len,
        hidden,
        eps,
    );

    for i in 0..scratch.normed.len() {
        scratch.normed[i] += scratch.moe_branch[i];
    }

    rms_norm_rows(
        out,
        &scratch.normed,
        &scratch.post_ff_norm_w,
        seq_len,
        hidden,
        eps,
    );
    for i in 0..out.len() {
        out[i] += scratch.residual[i];
    }

    let layer_scalar = weights.layer_scalar.bf16_scalar()?;
    for v in out.iter_mut() {
        *v *= layer_scalar;
    }
    Ok(())
}
