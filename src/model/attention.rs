use crate::config::{LayerType, TextConfig};
use crate::kernels::cpu::{
    apply_rope_tensor, compute_rope_freqs, linear, rms_norm, rms_norm_no_scale, rms_norm_rows,
    rope_kind_for_layer, softmax_rows,
};
use crate::model::layer_weights::DecoderLayerWeights;
use crate::safetensors::Error;

const MASK_NEG: f32 = -1e9;

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
            .ok_or(Error::Format("invalid layer index"))?;
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
    pub rope_freqs: Vec<f32>,
    pub scores: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub head_buf: Vec<f32>,
    pub q_w: Vec<f32>,
    pub k_w: Vec<f32>,
    pub v_w: Vec<f32>,
    pub o_w: Vec<f32>,
    pub q_norm_w: Vec<f32>,
    pub k_norm_w: Vec<f32>,
}

impl AttentionScratch {
    pub fn new(seq_len: usize, hidden: usize, params: &AttentionParams) -> Self {
        let q_dim = params.n_heads * params.head_dim;
        let kv_dim = params.n_kv_heads * params.head_dim;
        Self {
            normed: vec![0.0; seq_len * hidden],
            q: vec![0.0; seq_len * q_dim],
            k: vec![0.0; seq_len * kv_dim],
            v: vec![0.0; seq_len * kv_dim],
            rope_freqs: vec![0.0; seq_len * params.rotary_dim],
            scores: vec![0.0; seq_len * params.n_heads * seq_len],
            attn_out: vec![0.0; seq_len * q_dim],
            head_buf: vec![0.0; params.head_dim],
            q_w: Vec::new(),
            k_w: Vec::new(),
            v_w: Vec::new(),
            o_w: Vec::new(),
            q_norm_w: Vec::new(),
            k_norm_w: Vec::new(),
        }
    }

    fn load_weights(&mut self, weights: &DecoderLayerWeights<'_>) -> Result<(), Error> {
        self.q_w = weights.q_proj.bf16()?.to_f32_vec();
        self.k_w = weights.k_proj.bf16()?.to_f32_vec();
        self.v_w = weights.v_proj.bf16()?.to_f32_vec();
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
    let eps = cfg.rms_norm_eps as f32;
    let params = AttentionParams::for_layer(cfg, layer)?;
    let q_dim = params.n_heads * params.head_dim;
    let kv_dim = params.n_kv_heads * params.head_dim;

    assert_eq!(hidden.len(), seq_len * hidden_size);
    assert_eq!(out.len(), seq_len * hidden_size);
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
    linear(
        &mut scratch.v,
        &scratch.normed,
        &scratch.v_w,
        None,
        seq_len,
        hidden_size,
        kv_dim,
    );

    for s in 0..seq_len {
        for h in 0..params.n_heads {
            let off = (s * params.n_heads + h) * params.head_dim;
            scratch.head_buf.copy_from_slice(&scratch.q[off..off + params.head_dim]);
            rms_norm(
                &mut scratch.q[off..off + params.head_dim],
                &scratch.head_buf,
                &scratch.q_norm_w,
                eps,
            );
        }
        for h in 0..params.n_kv_heads {
            let off = (s * params.n_kv_heads + h) * params.head_dim;
            scratch.head_buf.copy_from_slice(&scratch.k[off..off + params.head_dim]);
            rms_norm(
                &mut scratch.k[off..off + params.head_dim],
                &scratch.head_buf,
                &scratch.k_norm_w,
                eps,
            );
            scratch.head_buf.copy_from_slice(&scratch.v[off..off + params.head_dim]);
            rms_norm_no_scale(
                &mut scratch.v[off..off + params.head_dim],
                &scratch.head_buf,
                eps,
            );
        }
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

    attention_scores_gqa(
        &mut scratch.scores,
        &scratch.q,
        &scratch.k,
        seq_len,
        &params,
    );
    apply_attention_mask(&mut scratch.scores, seq_len, &params);
    softmax_rows(&mut scratch.scores, seq_len * params.n_heads, seq_len);

    attention_output_gqa(
        &mut scratch.attn_out,
        &scratch.scores,
        &scratch.v,
        seq_len,
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
            let row = &mut scores[(qi * params.n_heads + h) * seq_len..(qi * params.n_heads + h + 1) * seq_len];
            for ki in 0..seq_len {
                let k_off = (ki * params.n_kv_heads + kv_h) * hd;
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += q[q_off + d] * k[k_off + d];
                }
                row[ki] = dot;
            }
        }
    }
}

fn apply_attention_mask(scores: &mut [f32], seq_len: usize, params: &AttentionParams) {
    for qi in 0..seq_len {
        for h in 0..params.n_heads {
            let row = &mut scores[(qi * params.n_heads + h) * seq_len..(qi * params.n_heads + h + 1) * seq_len];
            for ki in 0..seq_len {
                let mut masked = false;
                if ki > qi {
                    masked = true;
                }
                if let Some(window) = params.sliding_window {
                    if ki + window <= qi {
                        masked = true;
                    }
                }
                if masked {
                    row[ki] = MASK_NEG;
                }
            }
        }
    }
}

fn attention_output_gqa(
    out: &mut [f32],
    scores: &[f32],
    v: &[f32],
    seq_len: usize,
    params: &AttentionParams,
) {
    let hd = params.head_dim;
    out.fill(0.0);
    for qi in 0..seq_len {
        for h in 0..params.n_heads {
            let kv_h = h / params.n_groups;
            let score_row =
                &scores[(qi * params.n_heads + h) * seq_len..(qi * params.n_heads + h + 1) * seq_len];
            let o_off = (qi * params.n_heads + h) * hd;
            for ki in 0..seq_len {
                let w = score_row[ki];
                let v_off = (ki * params.n_kv_heads + kv_h) * hd;
                for d in 0..hd {
                    out[o_off + d] += w * v[v_off + d];
                }
            }
        }
    }
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
        assert_eq!(scores[2 * 3 + 0], MASK_NEG);
        assert_eq!(scores[2 * 3 + 1], 0.0);
        assert_eq!(scores[2 * 3 + 2], 0.0);
    }
}
