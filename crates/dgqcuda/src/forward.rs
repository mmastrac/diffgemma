//! The DiffusionGemma forward pass (f32), transliterated from the engine's
//! `src/model/*` CPU oracle.
//!
//! This is the reference every GPU stage is checked against, and it is also the
//! shape of the CUDA path: embed -> N decoder layers (attention, dense MLP, MoE)
//! -> final norm -> tied LM head -> softcap. Weights are dequantized to f32 by
//! `crate::weights`.

use crate::config::{Error, ModelConfig};
use crate::weights::{LayerKeys, Weights};

const MASK_NEG: f32 = -1.0e9;

pub struct LayerWeights {
    pub input_layernorm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub pre_feedforward_layernorm: Vec<f32>,
    pub post_feedforward_layernorm: Vec<f32>,
    pub post_feedforward_layernorm_1: Vec<f32>,
    pub post_feedforward_layernorm_2: Vec<f32>,
    pub pre_feedforward_layernorm_2: Vec<f32>,
    pub mlp_gate: Vec<f32>,
    pub mlp_up: Vec<f32>,
    pub mlp_down: Vec<f32>,
    pub router_proj: Vec<f32>,
    pub router_scale: Vec<f32>,
    pub router_per_expert_scale: Vec<f32>,
    pub experts_gate_up: Vec<f32>,
    pub experts_down: Vec<f32>,
    pub layer_scalar: f32,
}

impl LayerWeights {
    pub fn load(w: &Weights, layer: usize) -> Result<Self, Error> {
        let k = LayerKeys::new(layer);
        Ok(Self {
            input_layernorm: w.tensor_f32(&k.input_layernorm)?,
            q_norm: w.tensor_f32(&k.q_norm)?,
            k_norm: w.tensor_f32(&k.k_norm)?,
            q_proj: w.tensor_f32(&k.q_proj)?,
            k_proj: w.tensor_f32(&k.k_proj)?,
            // Full-attention layers alias V from the raw k_proj and carry no
            // v_proj tensor at all; the layer body handles the empty case.
            v_proj: if w.has(&k.v_proj) {
                w.tensor_f32(&k.v_proj)?
            } else {
                Vec::new()
            },
            o_proj: w.tensor_f32(&k.o_proj)?,
            post_attention_layernorm: w.tensor_f32(&k.post_attention_layernorm)?,
            pre_feedforward_layernorm: w.tensor_f32(&k.pre_feedforward_layernorm)?,
            post_feedforward_layernorm: w.tensor_f32(&k.post_feedforward_layernorm)?,
            post_feedforward_layernorm_1: w.tensor_f32(&k.post_feedforward_layernorm_1)?,
            post_feedforward_layernorm_2: w.tensor_f32(&k.post_feedforward_layernorm_2)?,
            pre_feedforward_layernorm_2: w.tensor_f32(&k.pre_feedforward_layernorm_2)?,
            mlp_gate: w.tensor_f32(&k.mlp_gate)?,
            mlp_up: w.tensor_f32(&k.mlp_up)?,
            mlp_down: w.tensor_f32(&k.mlp_down)?,
            router_proj: w.tensor_f32(&k.router_proj)?,
            router_scale: w.tensor_f32(&k.router_scale)?,
            router_per_expert_scale: w.tensor_f32(&k.router_per_expert_scale)?,
            experts_gate_up: w.tensor_f32(&k.experts_gate_up)?,
            experts_down: w.tensor_f32(&k.experts_down)?,
            layer_scalar: w.scalar(&k.layer_scalar)?,
        })
    }
}

pub fn rms_norm_rows(
    out: &mut [f32],
    x: &[f32],
    weight: &[f32],
    seq: usize,
    hidden: usize,
    eps: f32,
) {
    for s in 0..seq {
        let off = s * hidden;
        let row = &x[off..off + hidden];
        let sum_sq: f32 = row.iter().map(|v| v * v).sum();
        let inv = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
        for i in 0..hidden {
            out[off + i] = row[i] * inv * weight[i];
        }
    }
}

/// `y[s, o] = x[s, :] @ w[o, :]^T` (PyTorch linear layout).
pub fn linear(y: &mut [f32], x: &[f32], w: &[f32], seq: usize, in_dim: usize, out_dim: usize) {
    for s in 0..seq {
        let xr = &x[s * in_dim..(s + 1) * in_dim];
        let yr = &mut y[s * out_dim..(s + 1) * out_dim];
        for o in 0..out_dim {
            let wr = &w[o * in_dim..(o + 1) * in_dim];
            let mut acc = 0.0f32;
            for d in 0..in_dim {
                acc += xr[d] * wr[d];
            }
            yr[o] = acc;
        }
    }
}

pub fn gelu_tanh(x: f32) -> f32 {
    let x3 = x * x * x;
    let u = 0.797_884_6 * (x + 0.044_715 * x3);
    let t = if u > 8.0 {
        1.0
    } else if u < -8.0 {
        -1.0
    } else {
        u.tanh()
    };
    0.5 * x * (1.0 + t)
}

fn rms_norm_head(v: &mut [f32], weight: Option<&[f32]>, eps: f32) {
    let n = v.len();
    let ss: f32 = v.iter().map(|x| x * x).sum();
    let inv = 1.0 / (ss / n as f32 + eps).sqrt();
    for (i, x) in v.iter_mut().enumerate() {
        *x *= inv * weight.map(|w| w[i]).unwrap_or(1.0);
    }
}

/// Interleaved [cos, sin] pairs per frequency.
pub fn rope_freqs(seq: usize, rotary_dim: usize, full_head_dim: usize, theta: f32) -> Vec<f32> {
    let mut freqs = vec![0.0f32; seq * rotary_dim];
    let half = rotary_dim / 2;
    for s in 0..seq {
        let base = s * rotary_dim;
        for d in 0..half {
            let exponent = (2 * d) as f32 / full_head_dim as f32;
            let freq = 1.0 / theta.powf(exponent);
            let angle = s as f32 * freq;
            freqs[base + 2 * d] = angle.cos();
            freqs[base + 2 * d + 1] = angle.sin();
        }
    }
    freqs
}

/// Rotate one head in place; `rotary_dim < head_dim` pairs `(d, head_dim/2+d)`.
pub fn apply_rope(vec: &mut [f32], freqs: &[f32], rotary_dim: usize) {
    let head_dim = vec.len();
    let half = rotary_dim / 2;
    let half_head = head_dim / 2;
    let proportional = rotary_dim < head_dim;
    for d in 0..half {
        let cos = freqs[2 * d];
        let sin = freqs[2 * d + 1];
        let i1 = if proportional {
            half_head + d
        } else {
            d + half
        };
        let x0 = vec[d];
        let x1 = vec[i1];
        vec[d] = x0 * cos - x1 * sin;
        vec[i1] = x0 * sin + x1 * cos;
    }
}

fn softmax_row(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v *= inv;
    }
}

/// Everything a layer body borrows: the hidden state lives outside this so
/// `layer_forward` can take it and the scratch buffers at the same time.
pub struct Buffers {
    pub embed: Vec<f32>,
    pub normed: Vec<f32>,
    pub residual: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub scores: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub proj_out: Vec<f32>,
    pub mlp_gate: Vec<f32>,
    pub mlp_up: Vec<f32>,
    pub mlp_down: Vec<f32>,
    pub moe_input: Vec<f32>,
    pub moe_out: Vec<f32>,
    pub router_logits: Vec<f32>,
    pub expert_gate_up: Vec<f32>,
    pub expert_act: Vec<f32>,
    pub expert_out: Vec<f32>,
    pub norm_scratch: Vec<f32>,
}

pub struct Scratch {
    pub hidden_a: Vec<f32>,
    pub hidden_b: Vec<f32>,
    pub bufs: Buffers,
    pub logits: Vec<f32>,
}

impl Buffers {
    pub fn new(seq: usize, cfg: &ModelConfig) -> Self {
        let t = &cfg.text_config;
        let hidden = t.hidden_size;
        let max_q = t.num_attention_heads * t.global_head_dim;
        let max_kv = t.num_key_value_heads.max(t.num_global_key_value_heads) * t.global_head_dim;
        Self {
            embed: vec![0.0; seq * hidden],
            normed: vec![0.0; seq * hidden],
            residual: vec![0.0; seq * hidden],
            q: vec![0.0; seq * max_q],
            k: vec![0.0; seq * max_kv],
            v: vec![0.0; seq * max_kv],
            scores: vec![0.0; seq * t.num_attention_heads * seq],
            attn_out: vec![0.0; seq * max_q],
            proj_out: vec![0.0; seq * hidden],
            mlp_gate: vec![0.0; seq * t.intermediate_size],
            mlp_up: vec![0.0; seq * t.intermediate_size],
            mlp_down: vec![0.0; seq * hidden],
            moe_input: vec![0.0; seq * hidden],
            moe_out: vec![0.0; seq * hidden],
            router_logits: vec![0.0; seq * t.num_experts],
            expert_gate_up: vec![0.0; t.moe_intermediate_size * 2],
            expert_act: vec![0.0; t.moe_intermediate_size],
            expert_out: vec![0.0; hidden],
            norm_scratch: vec![0.0; seq * hidden],
        }
    }
}

impl Scratch {
    pub fn new(seq: usize, cfg: &ModelConfig) -> Self {
        Self {
            hidden_a: vec![0.0; seq * cfg.text_config.hidden_size],
            hidden_b: vec![0.0; seq * cfg.text_config.hidden_size],
            bufs: Buffers::new(seq, cfg),
            logits: vec![0.0; seq * cfg.text_config.vocab_size],
        }
    }
}

/// One decoder layer, matching `src/model/decoder_layer.rs::forward`.
fn layer_forward_at(
    out: &mut [f32],
    hidden_states: &[f32],
    lw: &LayerWeights,
    cfg: &ModelConfig,
    layer: usize,
    seq: usize,
    b: &mut Buffers,
    stop_at: u8,
) {
    let t = &cfg.text_config;
    let hidden = t.hidden_size;
    let eps = t.rms_norm_eps as f32;
    let (n_kv, head_dim, rotary_dim, theta, window) = t.attn_geometry(layer);
    let n_heads = t.num_attention_heads;
    let n_groups = n_heads / n_kv;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv * head_dim;

    // ---- attention sub-layer -------------------------------------------
    b.residual.copy_from_slice(hidden_states);
    rms_norm_rows(
        &mut b.normed,
        hidden_states,
        &lw.input_layernorm,
        seq,
        hidden,
        eps,
    );
    linear(&mut b.q, &b.normed, &lw.q_proj, seq, hidden, q_dim);
    linear(&mut b.k, &b.normed, &lw.k_proj, seq, hidden, kv_dim);
    if lw.v_proj.is_empty() {
        b.v[..seq * kv_dim].copy_from_slice(&b.k[..seq * kv_dim]);
    } else {
        linear(&mut b.v, &b.normed, &lw.v_proj, seq, hidden, kv_dim);
    }

    // per-head QK-norm; V is normalized without a weight
    for s in 0..seq {
        for h in 0..n_heads {
            let off = (s * n_heads + h) * head_dim;
            let mut head: Vec<f32> = b.q[off..off + head_dim].to_vec();
            rms_norm_head(&mut head, Some(&lw.q_norm), eps);
            b.q[off..off + head_dim].copy_from_slice(&head);
        }
        for h in 0..n_kv {
            let off = (s * n_kv + h) * head_dim;
            let mut head: Vec<f32> = b.k[off..off + head_dim].to_vec();
            rms_norm_head(&mut head, Some(&lw.k_norm), eps);
            b.k[off..off + head_dim].copy_from_slice(&head);
            let mut vhead: Vec<f32> = b.v[off..off + head_dim].to_vec();
            rms_norm_head(&mut vhead, None, eps);
            b.v[off..off + head_dim].copy_from_slice(&vhead);
        }
    }

    let freqs = rope_freqs(seq, rotary_dim, head_dim, theta);
    for s in 0..seq {
        for h in 0..n_heads {
            let off = (s * n_heads + h) * head_dim;
            let foff = s * rotary_dim;
            apply_rope(
                &mut b.q[off..off + head_dim],
                &freqs[foff..foff + rotary_dim],
                rotary_dim,
            );
        }
        for h in 0..n_kv {
            let off = (s * n_kv + h) * head_dim;
            let foff = s * rotary_dim;
            apply_rope(
                &mut b.k[off..off + head_dim],
                &freqs[foff..foff + rotary_dim],
                rotary_dim,
            );
        }
    }

    // GQA scores + causal (optionally windowed) mask + softmax + weighted V
    for qi in 0..seq {
        for h in 0..n_heads {
            let kv_h = h / n_groups;
            let q_off = (qi * n_heads + h) * head_dim;
            let row = &mut b.scores[(qi * n_heads + h) * seq..(qi * n_heads + h + 1) * seq];
            for ki in 0..seq {
                let k_off = (ki * n_kv + kv_h) * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += b.q[q_off + d] * b.k[k_off + d];
                }
                let masked = ki > qi || window.is_some_and(|w| ki + w <= qi);
                row[ki] = if masked { MASK_NEG } else { dot };
            }
            softmax_row(row);
            if qi == 3 && h == 7 && std::env::var_os("DGQ_DBG").is_some() {
                eprintln!(
                    "[dbg] cpu attn qi=3 h=7 row={:?} v0[0]={} v1[0]={}",
                    &row[..4],
                    b.v[(0 * n_kv + kv_h) * head_dim],
                    b.v[(1 * n_kv + kv_h) * head_dim]
                );
            }
            let o = &mut b.attn_out[q_off..q_off + head_dim];
            o.fill(0.0);
            for ki in 0..seq {
                let p = row[ki];
                if p == 0.0 {
                    continue;
                }
                let v_off = (ki * n_kv + kv_h) * head_dim;
                for d in 0..head_dim {
                    o[d] += p * b.v[v_off + d];
                }
            }
        }
    }

    linear(&mut b.proj_out, &b.attn_out, &lw.o_proj, seq, q_dim, hidden);
    rms_norm_rows(
        &mut b.normed,
        &b.proj_out,
        &lw.post_attention_layernorm,
        seq,
        hidden,
        eps,
    );
    for i in 0..b.normed.len() {
        b.normed[i] += b.residual[i];
    }

    if stop_at == 1 {
        return;
    }
    // ---- dense MLP -------------------------------------------------------
    b.residual.copy_from_slice(&b.normed);
    rms_norm_rows(
        &mut b.normed,
        &b.residual,
        &lw.pre_feedforward_layernorm,
        seq,
        hidden,
        eps,
    );
    linear(
        &mut b.mlp_gate,
        &b.normed,
        &lw.mlp_gate,
        seq,
        hidden,
        t.intermediate_size,
    );
    linear(
        &mut b.mlp_up,
        &b.normed,
        &lw.mlp_up,
        seq,
        hidden,
        t.intermediate_size,
    );
    for i in 0..b.mlp_gate.len() {
        b.mlp_up[i] *= gelu_tanh(b.mlp_gate[i]);
    }
    linear(
        &mut b.mlp_down,
        &b.mlp_up,
        &lw.mlp_down,
        seq,
        t.intermediate_size,
        hidden,
    );
    b.norm_scratch.copy_from_slice(&b.mlp_down);
    rms_norm_rows(
        &mut b.mlp_down,
        &b.norm_scratch,
        &lw.post_feedforward_layernorm_1,
        seq,
        hidden,
        eps,
    );

    if stop_at == 2 {
        return;
    }
    // ---- MoE -------------------------------------------------------------
    let root = (hidden as f32).powf(-0.5);
    for s in 0..seq {
        let off = s * hidden;
        let row = &b.residual[off..off + hidden];
        let sum_sq: f32 = row.iter().map(|v| v * v).sum();
        let inv = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
        for i in 0..hidden {
            b.normed[off + i] = row[i] * inv * lw.router_scale[i] * root;
        }
    }
    linear(
        &mut b.router_logits,
        &b.normed,
        &lw.router_proj,
        seq,
        hidden,
        t.num_experts,
    );

    rms_norm_rows(
        &mut b.moe_input,
        &b.residual,
        &lw.pre_feedforward_layernorm_2,
        seq,
        hidden,
        eps,
    );

    b.moe_out.fill(0.0);
    let moe_inter = t.moe_intermediate_size;
    let gu_stride = moe_inter * 2 * hidden;
    let down_stride = hidden * moe_inter;
    for s in 0..seq {
        let logits = &b.router_logits[s * t.num_experts..(s + 1) * t.num_experts];
        let (idx, weights) = top_k_route(logits, t.top_k_experts, &lw.router_per_expert_scale);
        if s == 0 && std::env::var_os("DGQ_DBG").is_some() {
            eprintln!("[dbg] cpu route idx={:?} w={:?}", idx, weights);
            eprintln!("[dbg] cpu router_logits[0..4]={:?}", &logits[..4]);
        }
        let x = &b.moe_input[s * hidden..(s + 1) * hidden];
        let o = &mut b.moe_out[s * hidden..(s + 1) * hidden];
        for (e, w) in idx.iter().zip(weights.iter()) {
            let gu = &lw.experts_gate_up[e * gu_stride..(e + 1) * gu_stride];
            linear(&mut b.expert_gate_up, x, gu, 1, hidden, moe_inter * 2);
            let (gate, up) = b.expert_gate_up.split_at(moe_inter);
            for i in 0..moe_inter {
                b.expert_act[i] = gelu_tanh(gate[i]) * up[i];
            }
            let dn = &lw.experts_down[e * down_stride..(e + 1) * down_stride];
            linear(&mut b.expert_out, &b.expert_act, dn, 1, moe_inter, hidden);
            if s == 0 && e == &idx[0] && std::env::var_os("DGQ_DBG").is_some() {
                eprintln!(
                    "[dbg] cpu expert_out[0..4]={:?} act[0..2]={:?}",
                    &b.expert_out[..4],
                    &b.expert_act[..2]
                );
            }
            for i in 0..hidden {
                // Engine order: the expert weight is applied after the down
                // projection (moe_scatter_weighted).
                o[i] += w * b.expert_out[i];
            }
        }
    }

    b.norm_scratch.copy_from_slice(&b.moe_out);
    rms_norm_rows(
        &mut b.moe_out,
        &b.norm_scratch,
        &lw.post_feedforward_layernorm_2,
        seq,
        hidden,
        eps,
    );
    for i in 0..b.normed.len() {
        b.normed[i] += b.moe_out[i];
    }

    if stop_at == 3 {
        return;
    }
    // ---- output ----------------------------------------------------------
    rms_norm_rows(
        out,
        &b.normed,
        &lw.post_feedforward_layernorm,
        seq,
        hidden,
        eps,
    );
    for i in 0..out.len() {
        out[i] = (out[i] + b.residual[i]) * lw.layer_scalar;
    }
}

fn layer_forward(
    out: &mut [f32],
    hidden_states: &[f32],
    lw: &LayerWeights,
    cfg: &ModelConfig,
    layer: usize,
    seq: usize,
    b: &mut Buffers,
) {
    layer_forward_at(out, hidden_states, lw, cfg, layer, seq, b, 0)
}

/// MLX/Gemma4 top-k: rank raw logits, softmax over the selected set, then
/// multiply by the per-expert scale. Ties break toward the lower index.
pub fn top_k_route(logits: &[f32], k: usize, per_expert_scale: &[f32]) -> (Vec<usize>, Vec<f32>) {
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|(ia, pa), (ib, pb)| {
        pb.partial_cmp(pa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(ia.cmp(ib))
    });
    let top = &ranked[..k];
    let indices: Vec<usize> = top.iter().map(|(i, _)| *i).collect();
    let raw: Vec<f32> = top.iter().map(|(_, s)| *s).collect();
    let mx = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = raw.iter().map(|&x| (x - mx).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut weights: Vec<f32> = if sum > 0.0 {
        exps.iter().map(|e| e / sum).collect()
    } else {
        vec![1.0 / k as f32; k]
    };
    for (w, &i) in weights.iter_mut().zip(indices.iter()) {
        *w *= per_expert_scale[i];
    }
    (indices, weights)
}

pub struct ForwardOutput {
    pub logits: Vec<f32>,
}

/// Which positions get LM-head logits. `Last` is what generation needs and
/// costs 1/seq of the head; `All` is for parity checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogitRows {
    All,
    Last,
    Only(usize),
}

/// Full forward pass: embed -> layers -> final norm -> tied LM head -> softcap.
pub fn forward(
    w: &Weights,
    cfg: &ModelConfig,
    token_ids: &[u32],
    layers: Option<usize>,
    rows: LogitRows,
    sc: &mut Scratch,
) -> Result<ForwardOutput, Error> {
    let t = &cfg.text_config;
    let seq = token_ids.len();
    let hidden = t.hidden_size;
    let embed_scale = (hidden as f32).sqrt();

    // embed gather (the pack's table is raw bf16)
    let embed_w = w.tensor_f32("model.decoder.embed_tokens.weight")?;
    for (s, &id) in token_ids.iter().enumerate() {
        let src = id as usize * hidden;
        let dst = s * hidden;
        for i in 0..hidden {
            sc.bufs.embed[dst + i] = embed_w[src + i] * embed_scale;
        }
    }
    sc.hidden_a.copy_from_slice(&sc.bufs.embed);

    let n_layers = layers
        .unwrap_or(t.num_hidden_layers)
        .min(t.num_hidden_layers);
    for layer in 0..n_layers {
        let lw = LayerWeights::load(w, layer)?;
        layer_forward(
            &mut sc.hidden_b,
            &sc.hidden_a,
            &lw,
            cfg,
            layer,
            seq,
            &mut sc.bufs,
        );
        std::mem::swap(&mut sc.hidden_a, &mut sc.hidden_b);
    }

    let norm_w = w.tensor_f32("model.decoder.norm.weight")?;
    rms_norm_rows(
        &mut sc.hidden_b,
        &sc.hidden_a,
        &norm_w,
        seq,
        hidden,
        t.rms_norm_eps as f32,
    );

    // tied LM head: logits = hidden @ embed^T, chunked over the vocabulary so
    // the inner loop walks a small resident block instead of the whole table.
    let want: Vec<usize> = match rows {
        LogitRows::All => (0..seq).collect(),
        LogitRows::Last => vec![seq - 1],
        LogitRows::Only(s) => vec![s],
    };
    const CHUNK: usize = 4096;
    for v0 in (0..t.vocab_size).step_by(CHUNK) {
        let v1 = (v0 + CHUNK).min(t.vocab_size);
        let wblock = &embed_w[v0 * hidden..v1 * hidden];
        for &s in &want {
            let h = &sc.hidden_b[s * hidden..(s + 1) * hidden];
            let row = &mut sc.logits[s * t.vocab_size + v0..s * t.vocab_size + v1];
            for (o, dst) in row.iter_mut().enumerate() {
                let wr = &wblock[o * hidden..(o + 1) * hidden];
                let mut acc = 0.0f32;
                for d in 0..hidden {
                    acc += h[d] * wr[d];
                }
                *dst = acc;
            }
        }
    }
    if t.final_logit_softcapping > 0.0 {
        let cap = t.final_logit_softcapping as f32;
        for v in sc.logits.iter_mut() {
            *v = (*v / cap).tanh() * cap;
        }
    }

    Ok(ForwardOutput {
        logits: sc.logits.clone(),
    })
}

/// Hidden state after \`layers\` decoder layers (the CPU oracle's stage output).
pub fn hidden_after(
    w: &Weights,
    cfg: &ModelConfig,
    token_ids: &[u32],
    layers: usize,
    stop_at: u8,
    sc: &mut Scratch,
) -> Result<Vec<f32>, Error> {
    let t = &cfg.text_config;
    let seq = token_ids.len();
    let hidden = t.hidden_size;
    let embed_scale = (hidden as f32).sqrt();
    let embed_w = w.tensor_f32("model.decoder.embed_tokens.weight")?;
    for (s, &id) in token_ids.iter().enumerate() {
        let src = id as usize * hidden;
        let dst = s * hidden;
        for i in 0..hidden {
            sc.bufs.embed[dst + i] = embed_w[src + i] * embed_scale;
        }
    }
    sc.hidden_a.copy_from_slice(&sc.bufs.embed);
    for layer in 0..layers {
        let lw = LayerWeights::load(w, layer)?;
        let stop = if layer + 1 == layers { stop_at } else { 0 };
        layer_forward_at(
            &mut sc.hidden_b,
            &sc.hidden_a,
            &lw,
            cfg,
            layer,
            seq,
            &mut sc.bufs,
            stop,
        );
        if stop == 1 || stop == 2 || stop == 3 {
            return Ok(sc.bufs.normed[..seq * hidden].to_vec());
        }
        std::mem::swap(&mut sc.hidden_a, &mut sc.hidden_b);
    }
    Ok(sc.hidden_a[..seq * hidden].to_vec())
}

/// Attention sub-layer output (post-attention norm + residual), the CPU oracle’s
/// intermediate stage for bisecting a layer.
pub fn attn_stage(
    w: &Weights,
    cfg: &ModelConfig,
    token_ids: &[u32],
    sc: &mut Scratch,
) -> Result<Vec<f32>, Error> {
    let t = &cfg.text_config;
    let seq = token_ids.len();
    let hidden = t.hidden_size;
    let embed_scale = (hidden as f32).sqrt();
    let embed_w = w.tensor_f32("model.decoder.embed_tokens.weight")?;
    for (s, &id) in token_ids.iter().enumerate() {
        let src = id as usize * hidden;
        let dst = s * hidden;
        for i in 0..hidden {
            sc.bufs.embed[dst + i] = embed_w[src + i] * embed_scale;
        }
    }
    sc.hidden_a.copy_from_slice(&sc.bufs.embed);
    let lw = LayerWeights::load(w, 0)?;
    let eps = t.rms_norm_eps as f32;
    let (n_kv, head_dim, rotary_dim, theta, window) = t.attn_geometry(0);
    let n_heads = t.num_attention_heads;
    let n_groups = n_heads / n_kv;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv * head_dim;
    let b = &mut sc.bufs;
    b.residual.copy_from_slice(&sc.hidden_a);
    rms_norm_rows(
        &mut b.normed,
        &sc.hidden_a,
        &lw.input_layernorm,
        seq,
        hidden,
        eps,
    );
    linear(&mut b.q, &b.normed, &lw.q_proj, seq, hidden, q_dim);
    linear(&mut b.k, &b.normed, &lw.k_proj, seq, hidden, kv_dim);
    if lw.v_proj.is_empty() {
        b.v[..seq * kv_dim].copy_from_slice(&b.k[..seq * kv_dim]);
    } else {
        linear(&mut b.v, &b.normed, &lw.v_proj, seq, hidden, kv_dim);
    }
    for s in 0..seq {
        for h in 0..n_heads {
            let off = (s * n_heads + h) * head_dim;
            let mut head: Vec<f32> = b.q[off..off + head_dim].to_vec();
            rms_norm_head(&mut head, Some(&lw.q_norm), eps);
            b.q[off..off + head_dim].copy_from_slice(&head);
        }
        for h in 0..n_kv {
            let off = (s * n_kv + h) * head_dim;
            let mut head: Vec<f32> = b.k[off..off + head_dim].to_vec();
            rms_norm_head(&mut head, Some(&lw.k_norm), eps);
            b.k[off..off + head_dim].copy_from_slice(&head);
            let mut vh: Vec<f32> = b.v[off..off + head_dim].to_vec();
            rms_norm_head(&mut vh, None, eps);
            b.v[off..off + head_dim].copy_from_slice(&vh);
        }
    }
    let freqs = rope_freqs(seq, rotary_dim, head_dim, theta);
    for s in 0..seq {
        for h in 0..n_heads {
            let off = (s * n_heads + h) * head_dim;
            let foff = s * rotary_dim;
            apply_rope(
                &mut b.q[off..off + head_dim],
                &freqs[foff..foff + rotary_dim],
                rotary_dim,
            );
        }
        for h in 0..n_kv {
            let off = (s * n_kv + h) * head_dim;
            let foff = s * rotary_dim;
            apply_rope(
                &mut b.k[off..off + head_dim],
                &freqs[foff..foff + rotary_dim],
                rotary_dim,
            );
        }
    }
    for qi in 0..seq {
        for h in 0..n_heads {
            let kv_h = h / n_groups;
            let q_off = (qi * n_heads + h) * head_dim;
            let row = &mut b.scores[(qi * n_heads + h) * seq..(qi * n_heads + h + 1) * seq];
            for ki in 0..seq {
                let k_off = (ki * n_kv + kv_h) * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += b.q[q_off + d] * b.k[k_off + d];
                }
                let masked = ki > qi || window.is_some_and(|w| ki + w <= qi);
                row[ki] = if masked { MASK_NEG } else { dot };
            }
            softmax_row(row);
            let o = &mut b.attn_out[q_off..q_off + head_dim];
            o.fill(0.0);
            for ki in 0..seq {
                let p = row[ki];
                if p == 0.0 {
                    continue;
                }
                let v_off = (ki * n_kv + kv_h) * head_dim;
                for d in 0..head_dim {
                    o[d] += p * b.v[v_off + d];
                }
            }
        }
    }
    linear(&mut b.proj_out, &b.attn_out, &lw.o_proj, seq, q_dim, hidden);
    rms_norm_rows(
        &mut b.normed,
        &b.proj_out,
        &lw.post_attention_layernorm,
        seq,
        hidden,
        eps,
    );
    for i in 0..b.normed.len() {
        b.normed[i] += b.residual[i];
    }
    Ok(b.normed[..seq * hidden].to_vec())
}

/// The attention sub-layer's Q/K/V buffers after QK-norm and RoPE, for
/// layout tests. Returns copies of the oracle's own buffers.
pub struct AttnBuffers {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

pub fn attn_buffers(
    w: &Weights,
    cfg: &ModelConfig,
    token_ids: &[u32],
    sc: &mut Scratch,
) -> Result<AttnBuffers, Error> {
    let t = &cfg.text_config;
    let seq = token_ids.len();
    let hidden = t.hidden_size;
    let (n_kv, head_dim, rotary_dim, theta, _) = t.attn_geometry(0);
    let n_heads = t.num_attention_heads;
    let eps = t.rms_norm_eps as f32;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv * head_dim;
    let embed_scale = (hidden as f32).sqrt();
    let embed_w = w.tensor_f32("model.decoder.embed_tokens.weight")?;
    for (s, &id) in token_ids.iter().enumerate() {
        let src = id as usize * hidden;
        let dst = s * hidden;
        for i in 0..hidden {
            sc.hidden_a[dst + i] = embed_w[src + i] * embed_scale;
        }
    }
    let lw = LayerWeights::load(w, 0)?;
    let b = &mut sc.bufs;
    rms_norm_rows(
        &mut b.normed,
        &sc.hidden_a,
        &lw.input_layernorm,
        seq,
        hidden,
        eps,
    );
    linear(&mut b.q, &b.normed, &lw.q_proj, seq, hidden, q_dim);
    linear(&mut b.k, &b.normed, &lw.k_proj, seq, hidden, kv_dim);
    linear(&mut b.v, &b.normed, &lw.v_proj, seq, hidden, kv_dim);
    for s in 0..seq {
        for h in 0..n_heads {
            let off = (s * n_heads + h) * head_dim;
            let mut head: Vec<f32> = b.q[off..off + head_dim].to_vec();
            rms_norm_head(&mut head, Some(&lw.q_norm), eps);
            b.q[off..off + head_dim].copy_from_slice(&head);
        }
        for h in 0..n_kv {
            let off = (s * n_kv + h) * head_dim;
            let mut head: Vec<f32> = b.k[off..off + head_dim].to_vec();
            rms_norm_head(&mut head, Some(&lw.k_norm), eps);
            b.k[off..off + head_dim].copy_from_slice(&head);
            let mut vh: Vec<f32> = b.v[off..off + head_dim].to_vec();
            rms_norm_head(&mut vh, None, eps);
            b.v[off..off + head_dim].copy_from_slice(&vh);
        }
    }
    let freqs = rope_freqs(seq, rotary_dim, head_dim, theta);
    for s in 0..seq {
        for h in 0..n_heads {
            let off = (s * n_heads + h) * head_dim;
            let foff = s * rotary_dim;
            apply_rope(
                &mut b.q[off..off + head_dim],
                &freqs[foff..foff + rotary_dim],
                rotary_dim,
            );
        }
        for h in 0..n_kv {
            let off = (s * n_kv + h) * head_dim;
            let foff = s * rotary_dim;
            apply_rope(
                &mut b.k[off..off + head_dim],
                &freqs[foff..foff + rotary_dim],
                rotary_dim,
            );
        }
    }
    Ok(AttnBuffers {
        q: b.q[..seq * q_dim].to_vec(),
        k: b.k[..seq * kv_dim].to_vec(),
        v: b.v[..seq * kv_dim].to_vec(),
    })
}
