use crate::Error;
use crate::config::{LayerType, TextConfig};
use crate::model::kv_cache::{LayerKv, LayerKvView};
use crate::model::layer_weights::DecoderLayerWeights;
use crate::model::mask::DecoderAttnMask;
use crate::shaders::cpu::{
    apply_rope_tensor, compute_rope_freqs, linear, rms_norm, rms_norm_no_scale, rms_norm_rows,
    rope_kind_for_layer, softmax_rows,
};

pub const MASK_NEG: f32 = -1e9;

/// Mask mode for shared CPU/GPU GQA attention (`gqa_attention`).
pub enum GqaMask<'a> {
    CausalSliding,
    EncoderExtend {
        kv_cache_len: usize,
        positions: &'a [i64],
    },
    DecoderBitmap(&'a DecoderAttnMask),
}

#[derive(Debug, Clone)]
pub struct AttentionParams {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub n_groups: usize,
    pub sliding_window: Option<usize>,
}

impl AttentionParams {
    pub fn for_layer(cfg: &TextConfig, layer: usize) -> Result<Self, Error> {
        let layer_type = cfg
            .layer_types
            .get(layer)
            .ok_or(Error::Runtime("invalid layer index"))?;
        let (n_kv_heads, head_dim) = match layer_type {
            LayerType::SlidingAttention => (cfg.num_key_value_heads, cfg.head_dim),
            LayerType::FullAttention => (cfg.num_global_key_value_heads, cfg.global_head_dim),
        };
        Ok(Self {
            n_heads: cfg.num_attention_heads,
            n_kv_heads,
            head_dim,
            rotary_dim: cfg.rotary_dim_for_layer(layer).unwrap_or(head_dim),
            n_groups: cfg.num_attention_heads / n_kv_heads,
            sliding_window: match layer_type {
                LayerType::SlidingAttention => Some(cfg.sliding_window),
                LayerType::FullAttention => None,
            },
        })
    }
}

pub struct AttentionScratch {
    pub normed: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub k_full: Vec<f32>,
    pub v_full: Vec<f32>,
    pub rope_freqs: Vec<f32>,
    pub scores: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub head_buf: Vec<f32>,
    pub q_w: Vec<f32>,
    pub k_w: Vec<f32>,
    pub v_w: Option<Vec<f32>>,
    pub o_w: Vec<f32>,
    pub q_norm_w: Vec<f32>,
    pub k_norm_w: Vec<f32>,
    pub total_kv_len: usize,
}

impl AttentionScratch {
    pub fn new(seq_len: usize, hidden: usize, params: &AttentionParams) -> Self {
        Self::with_kv_len(seq_len, hidden, params, 0)
    }

    pub fn with_kv_len(
        seq_len: usize,
        hidden: usize,
        params: &AttentionParams,
        kv_cache_len: usize,
    ) -> Self {
        Self::with_kv_len_inner(seq_len, hidden, params, kv_cache_len, true)
    }

    /// GPU decoder/encoder extend: skip `k_full`/`v_full`/`scores` (KV concat on GPU).
    pub fn with_kv_len_gpu(
        seq_len: usize,
        hidden: usize,
        params: &AttentionParams,
        kv_cache_len: usize,
    ) -> Self {
        Self::with_kv_len_inner(seq_len, hidden, params, kv_cache_len, false)
    }

    fn with_kv_len_inner(
        seq_len: usize,
        hidden: usize,
        params: &AttentionParams,
        kv_cache_len: usize,
        cpu_kv_buffers: bool,
    ) -> Self {
        let q_dim = params.n_heads * params.head_dim;
        let kv_dim = params.n_kv_heads * params.head_dim;
        let total_kv_len = kv_cache_len + seq_len;
        let (k_full, v_full, scores) = if cpu_kv_buffers {
            (
                vec![0.0; total_kv_len * kv_dim],
                vec![0.0; total_kv_len * kv_dim],
                vec![0.0; seq_len * params.n_heads * total_kv_len],
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        Self {
            normed: vec![0.0; seq_len * hidden],
            q: vec![0.0; seq_len * q_dim],
            k: vec![0.0; seq_len * kv_dim],
            v: vec![0.0; seq_len * kv_dim],
            k_full,
            v_full,
            rope_freqs: vec![0.0; seq_len * params.rotary_dim],
            scores,
            attn_out: vec![0.0; seq_len * q_dim],
            head_buf: vec![0.0; params.head_dim],
            q_w: Vec::new(),
            k_w: Vec::new(),
            v_w: None,
            o_w: Vec::new(),
            q_norm_w: Vec::new(),
            k_norm_w: Vec::new(),
            total_kv_len,
        }
    }

    /// Lazily allocate CPU concat/scores buffers (GPU KV fallback in `decoder_attention`).
    pub fn ensure_cpu_kv_buffers(&mut self, params: &AttentionParams) {
        if !self.k_full.is_empty() {
            return;
        }
        let kv_dim = params.n_kv_heads * params.head_dim;
        let total_kv = self.total_kv_len;
        let seq_len = self.q.len() / (params.n_heads * params.head_dim);
        self.k_full.resize(total_kv * kv_dim, 0.0);
        self.v_full.resize(total_kv * kv_dim, 0.0);
        self.scores.resize(seq_len * params.n_heads * total_kv, 0.0);
    }

    fn load_weights(&mut self, weights: &DecoderLayerWeights<'_>) -> Result<(), Error> {
        self.q_w = weights.q_proj.bf16()?.to_f32_vec();
        self.k_w = weights.k_proj.bf16()?.to_f32_vec();
        self.v_w = match &weights.v_proj {
            Some(v) => Some(v.bf16()?.to_f32_vec()),
            None => None,
        };
        self.o_w = weights.o_proj.bf16()?.to_f32_vec();
        self.q_norm_w = weights.q_norm.bf16()?.to_f32_vec();
        self.k_norm_w = weights.k_norm.bf16()?.to_f32_vec();
        Ok(())
    }
}

pub fn forward(
    out: &mut [f32],
    hidden: &[f32],
    weights: &DecoderLayerWeights<'_>,
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    positions: &[i64],
    scratch: &mut AttentionScratch,
) -> Result<(), Error> {
    let hidden_size = cfg.hidden_size;
    let params = AttentionParams::for_layer(cfg, layer)?;
    let q_dim = params.n_heads * params.head_dim;

    assert_eq!(hidden.len(), seq_len * hidden_size);
    assert_eq!(out.len(), seq_len * hidden_size);

    prepare_qkv_pre_rope(hidden, weights, cfg, layer, seq_len, positions, scratch)?;
    apply_rope_to_scratch_qkv(cfg, layer, seq_len, positions, scratch)?;
    gqa_attention(
        &mut scratch.attn_out,
        &mut scratch.scores,
        &scratch.q,
        &scratch.k,
        &scratch.v,
        seq_len,
        seq_len,
        &params,
        GqaMask::CausalSliding,
    );

    linear(
        out,
        &scratch.attn_out,
        &scratch.o_w,
        None,
        seq_len,
        q_dim,
        hidden_size,
    );
    Ok(())
}

/// Q/K/V projections and per-head norm; RoPE freqs computed but not applied.
pub fn prepare_qkv_pre_rope(
    hidden: &[f32],
    weights: &DecoderLayerWeights<'_>,
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    positions: &[i64],
    scratch: &mut AttentionScratch,
) -> Result<(), Error> {
    let hidden_size = cfg.hidden_size;
    let eps = cfg.rms_norm_eps as f32;
    let params = AttentionParams::for_layer(cfg, layer)?;
    let q_dim = params.n_heads * params.head_dim;
    let kv_dim = params.n_kv_heads * params.head_dim;

    assert_eq!(hidden.len(), seq_len * hidden_size);
    assert_eq!(positions.len(), seq_len);

    scratch.load_weights(weights)?;

    let input_norm_w = weights.input_layernorm.bf16()?.to_f32_vec();
    rms_norm_rows(
        &mut scratch.normed,
        hidden,
        &input_norm_w,
        seq_len,
        hidden_size,
        eps,
    );

    linear(
        &mut scratch.q,
        &scratch.normed,
        &scratch.q_w,
        None,
        seq_len,
        hidden_size,
        q_dim,
    );
    linear(
        &mut scratch.k,
        &scratch.normed,
        &scratch.k_w,
        None,
        seq_len,
        hidden_size,
        kv_dim,
    );
    if let Some(v_w) = &scratch.v_w {
        linear(
            &mut scratch.v,
            &scratch.normed,
            v_w,
            None,
            seq_len,
            hidden_size,
            kv_dim,
        );
    } else {
        scratch.v.copy_from_slice(&scratch.k);
    }

    {
        let q_norm_w = scratch.q_norm_w.clone();
        let k_norm_w = scratch.k_norm_w.clone();
        normalize_qkv_heads(
            &mut scratch.q,
            &mut scratch.k,
            &mut scratch.v,
            seq_len,
            &params,
            eps,
            &mut scratch.head_buf,
            &q_norm_w,
            &k_norm_w,
        );
    }

    let rope_kind = rope_kind_for_layer(cfg, layer).ok_or(Error::Format("rope kind"))?;
    compute_rope_freqs(&mut scratch.rope_freqs, positions, rope_kind);
    Ok(())
}

fn apply_rope_to_scratch_qkv(
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    _positions: &[i64],
    scratch: &mut AttentionScratch,
) -> Result<(), Error> {
    let params = AttentionParams::for_layer(cfg, layer)?;
    apply_rope_tensor(
        &mut scratch.q,
        seq_len,
        params.n_heads,
        params.head_dim,
        &scratch.rope_freqs,
        params.rotary_dim,
    );
    apply_rope_tensor(
        &mut scratch.k,
        seq_len,
        params.n_kv_heads,
        params.head_dim,
        &scratch.rope_freqs,
        params.rotary_dim,
    );
    Ok(())
}

/// Attention through RoPE and GQA softmax; stops before `o_proj`.
pub fn forward_to_attn_out(
    attn_out: &mut [f32],
    hidden: &[f32],
    weights: &DecoderLayerWeights<'_>,
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    positions: &[i64],
    scratch: &mut AttentionScratch,
) -> Result<(), Error> {
    let params = AttentionParams::for_layer(cfg, layer)?;
    let q_dim = params.n_heads * params.head_dim;
    assert_eq!(attn_out.len(), seq_len * q_dim);

    prepare_qkv_pre_rope(hidden, weights, cfg, layer, seq_len, positions, scratch)?;
    apply_rope_to_scratch_qkv(cfg, layer, seq_len, positions, scratch)?;
    gqa_attention(
        attn_out,
        &mut scratch.scores,
        &scratch.q,
        &scratch.k,
        &scratch.v,
        seq_len,
        seq_len,
        &params,
        GqaMask::CausalSliding,
    );
    Ok(())
}

/// GQA attention from post-RoPE Q/K/V (CPU reference for Metal parity).
pub fn gqa_attention(
    attn_out: &mut [f32],
    scores: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    total_kv: usize,
    params: &AttentionParams,
    mask: GqaMask<'_>,
) {
    assert_eq!(q.len(), seq_len * params.n_heads * params.head_dim);
    assert_eq!(scores.len(), seq_len * params.n_heads * total_kv);
    assert_eq!(attn_out.len(), seq_len * params.n_heads * params.head_dim);

    if total_kv == seq_len && matches!(mask, GqaMask::CausalSliding) {
        attention_scores_gqa(scores, q, k, seq_len, params);
        apply_attention_mask(scores, seq_len, params);
    } else {
        attention_scores_gqa_ext(scores, q, k, seq_len, total_kv, params);
        match mask {
            GqaMask::CausalSliding => apply_attention_mask(scores, seq_len, params),
            GqaMask::EncoderExtend {
                kv_cache_len,
                positions,
            } => apply_encoder_extend_mask(
                scores,
                seq_len,
                total_kv,
                kv_cache_len,
                positions,
                params,
            ),
            GqaMask::DecoderBitmap(decoder_mask) => {
                apply_decoder_mask(scores, seq_len, total_kv, params, decoder_mask)
            }
        }
    }
    softmax_rows(scores, seq_len * params.n_heads, total_kv);
    attention_output_gqa_ext(attn_out, scores, v, seq_len, total_kv, params);
}

/// Incremental encoder prefill: causal attention over cached KV + new tokens; appends K/V.
#[allow(dead_code)]
pub fn forward_encoder_extend(
    out: &mut [f32],
    hidden: &[f32],
    weights: &DecoderLayerWeights<'_>,
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    positions: &[i64],
    kv: LayerKvView<'_>,
    scratch: &mut AttentionScratch,
) -> Result<(), Error> {
    let hidden_size = cfg.hidden_size;
    let eps = cfg.rms_norm_eps as f32;
    let params = AttentionParams::for_layer(cfg, layer)?;
    let q_dim = params.n_heads * params.head_dim;
    let kv_dim = params.n_kv_heads * params.head_dim;
    let total_kv = scratch.total_kv_len;
    let kv_cache_len = kv.kv_len;

    assert_eq!(hidden.len(), seq_len * hidden_size);
    assert_eq!(out.len(), seq_len * hidden_size);
    assert_eq!(positions.len(), seq_len);
    assert_eq!(kv_cache_len + seq_len, total_kv);

    scratch.load_weights(weights)?;

    let input_norm_w = weights.input_layernorm.bf16()?.to_f32_vec();
    rms_norm_rows(
        &mut scratch.normed,
        hidden,
        &input_norm_w,
        seq_len,
        hidden_size,
        eps,
    );

    linear(
        &mut scratch.q,
        &scratch.normed,
        &scratch.q_w,
        None,
        seq_len,
        hidden_size,
        q_dim,
    );
    linear(
        &mut scratch.k,
        &scratch.normed,
        &scratch.k_w,
        None,
        seq_len,
        hidden_size,
        kv_dim,
    );
    if let Some(v_w) = &scratch.v_w {
        linear(
            &mut scratch.v,
            &scratch.normed,
            v_w,
            None,
            seq_len,
            hidden_size,
            kv_dim,
        );
    } else {
        scratch.v.copy_from_slice(&scratch.k);
    }

    {
        let q_norm_w = scratch.q_norm_w.clone();
        let k_norm_w = scratch.k_norm_w.clone();
        normalize_qkv_heads(
            &mut scratch.q,
            &mut scratch.k,
            &mut scratch.v,
            seq_len,
            &params,
            eps,
            &mut scratch.head_buf,
            &q_norm_w,
            &k_norm_w,
        );
    }

    let rope_kind = rope_kind_for_layer(cfg, layer).ok_or(Error::Format("rope kind"))?;
    compute_rope_freqs(&mut scratch.rope_freqs, positions, rope_kind);
    apply_rope_tensor(
        &mut scratch.q,
        seq_len,
        params.n_heads,
        params.head_dim,
        &scratch.rope_freqs,
        params.rotary_dim,
    );
    apply_rope_tensor(
        &mut scratch.k,
        seq_len,
        params.n_kv_heads,
        params.head_dim,
        &scratch.rope_freqs,
        params.rotary_dim,
    );

    concat_kv_cache(
        &mut scratch.k_full,
        &mut scratch.v_full,
        kv,
        &scratch.k,
        &scratch.v,
        &params,
    );

    attention_scores_gqa_ext(
        &mut scratch.scores,
        &scratch.q,
        &scratch.k_full,
        seq_len,
        total_kv,
        &params,
    );
    apply_encoder_extend_mask(
        &mut scratch.scores,
        seq_len,
        total_kv,
        kv_cache_len,
        positions,
        &params,
    );
    softmax_rows(&mut scratch.scores, seq_len * params.n_heads, total_kv);

    attention_output_gqa_ext(
        &mut scratch.attn_out,
        &scratch.scores,
        &scratch.v_full,
        seq_len,
        total_kv,
        &params,
    );

    linear(
        out,
        &scratch.attn_out,
        &scratch.o_w,
        None,
        seq_len,
        q_dim,
        hidden_size,
    );
    Ok(())
}

/// Causal encoder attention; writes post-RoPE K/V into `kv_out` for decoder cross-attn.
pub fn forward_encoder(
    out: &mut [f32],
    hidden: &[f32],
    weights: &DecoderLayerWeights<'_>,
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    positions: &[i64],
    kv_out: &mut LayerKv,
    scratch: &mut AttentionScratch,
) -> Result<(), Error> {
    forward(
        out, hidden, weights, cfg, layer, seq_len, positions, scratch,
    )?;
    let params = AttentionParams::for_layer(cfg, layer)?;
    kv_out.n_kv_heads = params.n_kv_heads;
    kv_out.head_dim = params.head_dim;
    kv_out.set_kv(&scratch.k, &scratch.v, seq_len)?;
    Ok(())
}

/// Decoder attention: bidirectional canvas queries over encoder KV cache + canvas KV.
pub fn forward_decoder(
    out: &mut [f32],
    hidden: &[f32],
    weights: &DecoderLayerWeights<'_>,
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    positions: &[i64],
    kv: LayerKvView<'_>,
    mask: Option<&DecoderAttnMask>,
    scratch: &mut AttentionScratch,
) -> Result<(), Error> {
    let hidden_size = cfg.hidden_size;
    let eps = cfg.rms_norm_eps as f32;
    let params = AttentionParams::for_layer(cfg, layer)?;
    let q_dim = params.n_heads * params.head_dim;
    let kv_dim = params.n_kv_heads * params.head_dim;
    let total_kv = scratch.total_kv_len;

    assert_eq!(hidden.len(), seq_len * hidden_size);
    assert_eq!(out.len(), seq_len * hidden_size);
    assert_eq!(positions.len(), seq_len);
    assert_eq!(kv.kv_len + seq_len, total_kv);

    scratch.load_weights(weights)?;

    let input_norm_w = weights.input_layernorm.bf16()?.to_f32_vec();
    rms_norm_rows(
        &mut scratch.normed,
        hidden,
        &input_norm_w,
        seq_len,
        hidden_size,
        eps,
    );

    linear(
        &mut scratch.q,
        &scratch.normed,
        &scratch.q_w,
        None,
        seq_len,
        hidden_size,
        q_dim,
    );
    linear(
        &mut scratch.k,
        &scratch.normed,
        &scratch.k_w,
        None,
        seq_len,
        hidden_size,
        kv_dim,
    );
    if let Some(v_w) = &scratch.v_w {
        linear(
            &mut scratch.v,
            &scratch.normed,
            v_w,
            None,
            seq_len,
            hidden_size,
            kv_dim,
        );
    } else {
        scratch.v.copy_from_slice(&scratch.k);
    }

    {
        let q_norm_w = scratch.q_norm_w.clone();
        let k_norm_w = scratch.k_norm_w.clone();
        normalize_qkv_heads(
            &mut scratch.q,
            &mut scratch.k,
            &mut scratch.v,
            seq_len,
            &params,
            eps,
            &mut scratch.head_buf,
            &q_norm_w,
            &k_norm_w,
        );
    }

    let rope_kind = rope_kind_for_layer(cfg, layer).ok_or(Error::Format("rope kind"))?;
    compute_rope_freqs(&mut scratch.rope_freqs, positions, rope_kind);
    apply_rope_tensor(
        &mut scratch.q,
        seq_len,
        params.n_heads,
        params.head_dim,
        &scratch.rope_freqs,
        params.rotary_dim,
    );
    apply_rope_tensor(
        &mut scratch.k,
        seq_len,
        params.n_kv_heads,
        params.head_dim,
        &scratch.rope_freqs,
        params.rotary_dim,
    );

    concat_kv_cache(
        &mut scratch.k_full,
        &mut scratch.v_full,
        kv,
        &scratch.k,
        &scratch.v,
        &params,
    );

    attention_scores_gqa_ext(
        &mut scratch.scores,
        &scratch.q,
        &scratch.k_full,
        seq_len,
        total_kv,
        &params,
    );
    if let Some(mask) = mask {
        apply_decoder_mask(&mut scratch.scores, seq_len, total_kv, &params, mask);
    }
    softmax_rows(&mut scratch.scores, seq_len * params.n_heads, total_kv);

    attention_output_gqa_ext(
        &mut scratch.attn_out,
        &scratch.scores,
        &scratch.v_full,
        seq_len,
        total_kv,
        &params,
    );

    linear(
        out,
        &scratch.attn_out,
        &scratch.o_w,
        None,
        seq_len,
        q_dim,
        hidden_size,
    );
    Ok(())
}

pub fn normalize_qkv_heads(
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    seq_len: usize,
    params: &AttentionParams,
    eps: f32,
    head_buf: &mut [f32],
    q_norm_w: &[f32],
    k_norm_w: &[f32],
) {
    for s in 0..seq_len {
        for h in 0..params.n_heads {
            let off = (s * params.n_heads + h) * params.head_dim;
            head_buf.copy_from_slice(&q[off..off + params.head_dim]);
            rms_norm(&mut q[off..off + params.head_dim], head_buf, q_norm_w, eps);
        }
        for h in 0..params.n_kv_heads {
            let off = (s * params.n_kv_heads + h) * params.head_dim;
            head_buf.copy_from_slice(&k[off..off + params.head_dim]);
            rms_norm(&mut k[off..off + params.head_dim], head_buf, k_norm_w, eps);
            head_buf.copy_from_slice(&v[off..off + params.head_dim]);
            rms_norm_no_scale(&mut v[off..off + params.head_dim], head_buf, eps);
        }
    }
}

pub fn concat_kv_for_decoder(
    k_full: &mut [f32],
    v_full: &mut [f32],
    kv: &LayerKvView<'_>,
    k_canvas: &[f32],
    v_canvas: &[f32],
    params: &AttentionParams,
) {
    concat_kv_cache(k_full, v_full, *kv, k_canvas, v_canvas, params);
}

fn concat_kv_cache(
    k_full: &mut [f32],
    v_full: &mut [f32],
    kv: LayerKvView<'_>,
    k_canvas: &[f32],
    v_canvas: &[f32],
    params: &AttentionParams,
) {
    let kv_dim = params.n_kv_heads * params.head_dim;
    let canvas_len = k_canvas.len() / kv_dim;
    assert_eq!(kv.kv_len + canvas_len, k_full.len() / kv_dim);

    for t in 0..kv.kv_len {
        let dst = t * kv_dim;
        let src = t * kv_dim;
        k_full[dst..dst + kv_dim].copy_from_slice(&kv.keys[src..src + kv_dim]);
        v_full[dst..dst + kv_dim].copy_from_slice(&kv.values[src..src + kv_dim]);
    }
    let off = kv.kv_len * kv_dim;
    k_full[off..].copy_from_slice(k_canvas);
    v_full[off..].copy_from_slice(v_canvas);
}

fn attention_scores_gqa(
    scores: &mut [f32],
    q: &[f32],
    k: &[f32],
    seq_len: usize,
    params: &AttentionParams,
) {
    let hd = params.head_dim;
    for qi in 0..seq_len {
        for h in 0..params.n_heads {
            let kv_h = h / params.n_groups;
            let q_off = (qi * params.n_heads + h) * hd;
            let row = &mut scores
                [(qi * params.n_heads + h) * seq_len..(qi * params.n_heads + h + 1) * seq_len];
            for (ki, row_ki) in row.iter_mut().enumerate() {
                let k_off = (ki * params.n_kv_heads + kv_h) * hd;
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += q[q_off + d] * k[k_off + d];
                }
                *row_ki = dot;
            }
        }
    }
}

fn apply_encoder_extend_mask(
    scores: &mut [f32],
    seq_len: usize,
    total_kv: usize,
    kv_cache_len: usize,
    positions: &[i64],
    params: &AttentionParams,
) {
    for qi in 0..seq_len {
        let pos_q = positions[qi];
        for h in 0..params.n_heads {
            let row = &mut scores
                [(qi * params.n_heads + h) * total_kv..(qi * params.n_heads + h + 1) * total_kv];
            for ki in 0..total_kv {
                let pos_k = if ki < kv_cache_len {
                    ki as i64
                } else {
                    positions[ki - kv_cache_len]
                };
                let mut masked = pos_k > pos_q;
                if let Some(window) = params.sliding_window
                    && pos_k + window as i64 <= pos_q
                {
                    masked = true;
                }
                if masked {
                    row[ki] = MASK_NEG;
                }
            }
        }
    }
}

fn apply_attention_mask(scores: &mut [f32], seq_len: usize, params: &AttentionParams) {
    for qi in 0..seq_len {
        for h in 0..params.n_heads {
            let row = &mut scores
                [(qi * params.n_heads + h) * seq_len..(qi * params.n_heads + h + 1) * seq_len];
            for (ki, row_ki) in row.iter_mut().enumerate() {
                let mut masked = false;
                if ki > qi {
                    masked = true;
                }
                if let Some(window) = params.sliding_window
                    && ki + window <= qi
                {
                    masked = true;
                }
                if masked {
                    *row_ki = MASK_NEG;
                }
            }
        }
    }
}

fn attention_scores_gqa_ext(
    scores: &mut [f32],
    q: &[f32],
    k: &[f32],
    seq_len: usize,
    total_kv: usize,
    params: &AttentionParams,
) {
    let hd = params.head_dim;
    let kv_dim = params.n_kv_heads * hd;
    for qi in 0..seq_len {
        for h in 0..params.n_heads {
            let kv_h = h / params.n_groups;
            let q_off = (qi * params.n_heads + h) * hd;
            let row = &mut scores
                [(qi * params.n_heads + h) * total_kv..(qi * params.n_heads + h + 1) * total_kv];
            for (ki, row_ki) in row.iter_mut().enumerate() {
                let k_off = ki * kv_dim + kv_h * hd;
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += q[q_off + d] * k[k_off + d];
                }
                *row_ki = dot;
            }
        }
    }
}

fn apply_decoder_mask(
    scores: &mut [f32],
    seq_len: usize,
    total_kv: usize,
    params: &AttentionParams,
    mask: &DecoderAttnMask,
) {
    for qi in 0..seq_len {
        for h in 0..params.n_heads {
            let row = &mut scores
                [(qi * params.n_heads + h) * total_kv..(qi * params.n_heads + h + 1) * total_kv];
            for (ki, row_ki) in row.iter_mut().enumerate() {
                if !mask.can_attend(qi, ki) {
                    *row_ki = MASK_NEG;
                }
            }
        }
    }
}

fn attention_output_gqa_ext(
    out: &mut [f32],
    scores: &[f32],
    v: &[f32],
    seq_len: usize,
    total_kv: usize,
    params: &AttentionParams,
) {
    let hd = params.head_dim;
    let kv_dim = params.n_kv_heads * hd;
    out.fill(0.0);
    for qi in 0..seq_len {
        for h in 0..params.n_heads {
            let kv_h = h / params.n_groups;
            let score_row = &scores
                [(qi * params.n_heads + h) * total_kv..(qi * params.n_heads + h + 1) * total_kv];
            let o_off = (qi * params.n_heads + h) * hd;
            for (ki, &w) in score_row.iter().enumerate() {
                let v_off = ki * kv_dim + kv_h * hd;
                for d in 0..hd {
                    out[o_off + d] += w * v[v_off + d];
                }
            }
        }
    }
}

/// Diagnostics for raw Q·K magnitudes and resulting softmax concentration (CPU path).
#[derive(Debug, Clone, Copy)]
pub struct AttnScoreStats {
    pub raw_dot_max: f32,
    pub raw_dot_min: f32,
    pub mean_row_softmax_entropy: f32,
    pub mean_row_max_prob: f32,
    /// Unmasked key slots per row (must match softmax support size).
    pub keys_per_row: usize,
    /// Mean over rows of Σ_t p_t after normalization (must be 1.0).
    pub mean_weight_sum: f32,
}

pub fn attn_score_stats_decoder(
    q: &[f32],
    k: &[f32],
    seq_len: usize,
    total_kv: usize,
    params: &AttentionParams,
    mask: &DecoderAttnMask,
) -> AttnScoreStats {
    let mut scores = vec![0.0f32; seq_len * params.n_heads * total_kv];
    attention_scores_gqa_ext(&mut scores, q, k, seq_len, total_kv, params);
    apply_decoder_mask(&mut scores, seq_len, total_kv, params, mask);

    let mut raw_max = f32::NEG_INFINITY;
    let mut raw_min = f32::INFINITY;
    let mut ent_sum = 0.0f32;
    let mut max_prob_sum = 0.0f32;
    let mut weight_sum_acc = 0.0f32;
    let mut active_keys = 0usize;
    let mut rows = 0usize;

    for qi in 0..seq_len {
        for h in 0..params.n_heads {
            let row = &scores
                [(qi * params.n_heads + h) * total_kv..(qi * params.n_heads + h + 1) * total_kv];
            let mut row_max = f32::NEG_INFINITY;
            let mut row_active = 0usize;
            for &s in row {
                if s > MASK_NEG / 2.0 {
                    raw_max = raw_max.max(s);
                    raw_min = raw_min.min(s);
                    row_max = row_max.max(s);
                    row_active += 1;
                }
            }
            if !row_max.is_finite() {
                continue;
            }
            if rows == 0 {
                active_keys = row_active;
            }
            let mut probs = row.to_vec();
            for v in &mut probs {
                if *v <= MASK_NEG / 2.0 {
                    *v = f32::NEG_INFINITY;
                } else {
                    *v = (*v - row_max).exp();
                }
            }
            let sum: f32 = probs.iter().filter(|&&p| p.is_finite()).sum();
            if sum <= 0.0 {
                continue;
            }
            let mut ent = 0.0f32;
            let mut max_p = 0.0f32;
            let mut row_weight_sum = 0.0f32;
            for &p in &probs {
                if p.is_finite() && p > 0.0 {
                    let pn = p / sum;
                    row_weight_sum += pn;
                    ent -= pn * pn.ln();
                    max_p = max_p.max(pn);
                }
            }
            ent_sum += ent;
            max_prob_sum += max_p;
            weight_sum_acc += row_weight_sum;
            rows += 1;
        }
    }
    let n = rows.max(1) as f32;
    AttnScoreStats {
        raw_dot_max: raw_max,
        raw_dot_min: raw_min,
        mean_row_softmax_entropy: ent_sum / n,
        mean_row_max_prob: max_prob_sum / n,
        keys_per_row: active_keys,
        mean_weight_sum: weight_sum_acc / n,
    }
}

pub fn qk_norm_weight_stats(weight: &[f32]) -> (f32, f32) {
    if weight.is_empty() {
        return (0.0, 0.0);
    }
    let mean_abs = weight.iter().map(|w| w.abs()).sum::<f32>() / weight.len() as f32;
    let rms = (weight.iter().map(|w| w * w).sum::<f32>() / weight.len() as f32).sqrt();
    (mean_abs, rms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_mask_blocks_future_and_distant_past() {
        let params = AttentionParams {
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 4,
            rotary_dim: 4,
            n_groups: 1,
            sliding_window: Some(2),
        };
        let mut scores = vec![0.0f32; 9];
        apply_attention_mask(&mut scores, 3, &params);
        // query 2: can attend 1,2 only (window=2)
        assert_eq!(scores[2 * 3], MASK_NEG);
        assert_eq!(scores[2 * 3 + 1], 0.0);
        assert_eq!(scores[2 * 3 + 2], 0.0);
    }
}
