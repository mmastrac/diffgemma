//! Gemma 4 MTP draft head, CPU f32 v0.
//!
//! Loads the `google/gemma-4-26B-A4B-it-assistant` bf16 checkpoint (4 dense
//! gemma4_text layers, no K/V projections of its own) and drafts tokens the
//! way transformers' `SinglePositionMultiTokenCandidateGenerator` drives it:
//! input = concat(target-embed of the last token, last hidden state) through
//! `pre_projection`, Q-only attention cross-attending the backbone's
//! last-sliding / last-full layer KV at a constant position, recurring on
//! `post_projection`'s output. lm_head is tied to the head's own embed table.
//!
//! CPU-only by design for v0: at M=1 a draft token is ~0.6 GFLOP
//! (lm_head-dominated), so kernel work buys nothing until the loop moves
//! on-GPU with the rest of serve.
//!
//! Experiment infra, retained after the free-run drafting line was cut
//! (see ARCHITECTURE.md Negative Knowledge): only the ignored `mtp_*` probe
//! tests and the `mtp-dump` command exercise it.
#![allow(dead_code)]

use std::path::Path;

use crate::Error;
use crate::safetensors::SafetensorsFile;
use crate::shaders::attn::cpu::{apply_proportional_rope, apply_split_half_rope, rms_norm_head};
use crate::shaders::cpu::{bf16_to_f32, gelu_pytorch_tanh, rms_norm};

pub const HEAD_HID: usize = 1024;
pub const BACKBONE_HID: usize = 2816;
const N_Q_HEADS: usize = 16;
const FFW: usize = 8192;
const EPS: f32 = 1e-6;

/// The backbone-scale embed multiplier as the HF pipeline computes it:
/// sqrt(2816) squeezed through bf16 (the head was trained downstream of that
/// rounding, so we reproduce it rather than the exact f32 value).
pub const TARGET_EMBED_SCALE: f32 = 53.0;

struct HeadLayer {
    input_ln: Vec<f32>,
    post_attn_ln: Vec<f32>,
    pre_ffw_ln: Vec<f32>,
    post_ffw_ln: Vec<f32>,
    layer_scalar: f32,
    q_proj: Vec<f32>,
    q_norm: Vec<f32>,
    o_proj: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    head_dim: usize,
    is_full: bool,
}

pub struct MtpHead {
    layers: Vec<HeadLayer>,
    final_norm: Vec<f32>,
    /// Tied embed / lm_head, dequantized to f32: [262144, 1024].
    embed: Vec<f32>,
    pre_projection: Vec<f32>,
    post_projection: Vec<f32>,
    pub vocab: usize,
}

/// Backbone KV the head cross-attends into, f32, head-major [n_kv, seq, hd].
/// swa = the backbone's LAST sliding layer (28), full = last full layer (29).
pub struct BackboneKv {
    pub k_swa: Vec<f32>,
    pub v_swa: Vec<f32>,
    pub k_full: Vec<f32>,
    pub v_full: Vec<f32>,
    pub seq: usize,
}

fn find_tensor_f32(st: &SafetensorsFile, name: &str) -> Result<Vec<f32>, Error> {
    let info = st
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or(Error::Runtime("missing MTP head tensor"))?;
    let bytes = st.data(info);
    Ok(bytes
        .chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect())
}

impl MtpHead {
    /// Load from the HF snapshot's `model.safetensors` (single shard, bf16).
    pub fn load(safetensors_path: &Path) -> Result<Self, Error> {
        let st = SafetensorsFile::open(safetensors_path)?;
        let t = |name: &str| find_tensor_f32(&st, name);
        let mut layers = Vec::with_capacity(4);
        for i in 0..4 {
            let p = format!("model.layers.{i}");
            // Layer types are [sliding x3, full]: hd 256 vs 512, mirroring the
            // backbone KV geometry each layer reads.
            let is_full = i == 3;
            let head_dim = if is_full { 512 } else { 256 };
            let layer = HeadLayer {
                input_ln: t(&format!("{p}.input_layernorm.weight"))?,
                post_attn_ln: t(&format!("{p}.post_attention_layernorm.weight"))?,
                pre_ffw_ln: t(&format!("{p}.pre_feedforward_layernorm.weight"))?,
                post_ffw_ln: t(&format!("{p}.post_feedforward_layernorm.weight"))?,
                layer_scalar: t(&format!("{p}.layer_scalar"))?[0],
                q_proj: t(&format!("{p}.self_attn.q_proj.weight"))?,
                q_norm: t(&format!("{p}.self_attn.q_norm.weight"))?,
                o_proj: t(&format!("{p}.self_attn.o_proj.weight"))?,
                gate: t(&format!("{p}.mlp.gate_proj.weight"))?,
                up: t(&format!("{p}.mlp.up_proj.weight"))?,
                down: t(&format!("{p}.mlp.down_proj.weight"))?,
                head_dim,
                is_full,
            };
            assert_eq!(layer.q_proj.len(), N_Q_HEADS * head_dim * HEAD_HID);
            assert_eq!(layer.q_norm.len(), head_dim);
            assert_eq!(layer.o_proj.len(), HEAD_HID * N_Q_HEADS * head_dim);
            layers.push(layer);
        }
        let embed = t("model.embed_tokens.weight")?;
        let vocab = embed.len() / HEAD_HID;
        let head = Self {
            layers,
            final_norm: t("model.norm.weight")?,
            embed,
            pre_projection: t("pre_projection.weight")?,
            post_projection: t("post_projection.weight")?,
            vocab,
        };
        assert_eq!(head.pre_projection.len(), HEAD_HID * 2 * BACKBONE_HID);
        assert_eq!(head.post_projection.len(), BACKBONE_HID * HEAD_HID);
        Ok(head)
    }
}

/// y = W @ x with W row-major [out_dim, x.len()].
fn matvec(w: &[f32], x: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    assert_eq!(w.len(), out_dim * in_dim);
    w.chunks_exact(in_dim)
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Two-pass softmax attention for one query head, scale 1.0 (Gemma 4 sets
/// no 1/sqrt(hd) factor). `k`/`v` are one kv head's [seq, hd] planes.
fn attend_head(out: &mut [f32], q: &[f32], k: &[f32], v: &[f32], hd: usize, kv_len: usize) {
    let mut scores = vec![0.0f32; kv_len];
    let mut max = f32::NEG_INFINITY;
    for t in 0..kv_len {
        let s = dot(q, &k[t * hd..(t + 1) * hd]);
        scores[t] = s;
        max = max.max(s);
    }
    let mut denom = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max).exp();
        denom += *s;
    }
    for o in out.iter_mut() {
        *o = 0.0;
    }
    for t in 0..kv_len {
        let w = scores[t] / denom;
        for (o, x) in out.iter_mut().zip(&v[t * hd..(t + 1) * hd]) {
            *o += w * x;
        }
    }
}

fn layer_forward(l: &HeadLayer, h: &mut [f32], kv: &BackboneKv, pos: usize, kv_len: usize) {
    let hd = l.head_dim;
    let mut hn = vec![0.0f32; HEAD_HID];
    rms_norm(&mut hn, h, &l.input_ln, EPS);

    let mut q = matvec(&l.q_proj, &hn, N_Q_HEADS * hd);
    let (k, v, n_kv) = if l.is_full {
        (&kv.k_full, &kv.v_full, 2)
    } else {
        (&kv.k_swa, &kv.v_swa, 8)
    };
    let group = N_Q_HEADS / n_kv;
    let mut attn = vec![0.0f32; N_Q_HEADS * hd];
    for qh in 0..N_Q_HEADS {
        let qs = &mut q[qh * hd..(qh + 1) * hd];
        rms_norm_head(qs, Some(&l.q_norm), EPS);
        if l.is_full {
            apply_proportional_rope(qs, (hd / 4) as u32, hd as u32, 1e6, pos as u32);
        } else {
            apply_split_half_rope(qs, hd as u32, hd as u32, 1e4, pos as u32);
        }
        let kvh = qh / group;
        let plane = kvh * kv.seq * hd;
        attend_head(
            &mut attn[qh * hd..(qh + 1) * hd],
            qs,
            &k[plane..plane + kv.seq * hd],
            &v[plane..plane + kv.seq * hd],
            hd,
            kv_len,
        );
    }
    let a = matvec(&l.o_proj, &attn, HEAD_HID);
    let mut an = vec![0.0f32; HEAD_HID];
    rms_norm(&mut an, &a, &l.post_attn_ln, EPS);
    for (hi, x) in h.iter_mut().zip(&an) {
        *hi += x;
    }

    rms_norm(&mut hn, h, &l.pre_ffw_ln, EPS);
    let mut g = matvec(&l.gate, &hn, FFW);
    gelu_pytorch_tanh(&mut g);
    let u = matvec(&l.up, &hn, FFW);
    for (gi, ui) in g.iter_mut().zip(&u) {
        *gi *= ui;
    }
    let m = matvec(&l.down, &g, HEAD_HID);
    let mut mn = vec![0.0f32; HEAD_HID];
    rms_norm(&mut mn, &m, &l.post_ffw_ln, EPS);
    for (hi, x) in h.iter_mut().zip(&mn) {
        *hi += x;
    }

    for hi in h.iter_mut() {
        *hi *= l.layer_scalar;
    }
}

fn lm_argmax(embed: &[f32], x: &[f32]) -> u32 {
    let mut best = f32::NEG_INFINITY;
    let mut arg = 0u32;
    for (i, row) in embed.chunks_exact(HEAD_HID).enumerate() {
        let s = dot(row, x);
        if s > best {
            best = s;
            arg = i as u32;
        }
    }
    arg
}

/// Draft up to `k_draft` tokens from position `pos` (0-based; the last
/// validated token's position). `init_hidden` is the backbone's pre-final-norm
/// layer-29 output at `pos`; `init_tok` the token at `pos`. `embed_fn` maps a
/// token id to its unscaled target embed row (2816); scaling happens here.
/// Returns fewer than `k_draft` only if `embed_fn` can't supply a row.
pub fn draft_tokens(
    head: &MtpHead,
    kv: &BackboneKv,
    pos: usize,
    init_hidden: &[f32],
    init_tok: u32,
    k_draft: usize,
    embed_fn: &mut dyn FnMut(u32) -> Option<Vec<f32>>,
) -> Vec<u32> {
    assert_eq!(init_hidden.len(), BACKBONE_HID);
    let kv_len = pos + 1;
    assert!(kv_len <= kv.seq);
    let mut hidden = init_hidden.to_vec();
    let mut tok = init_tok;
    let mut out = Vec::new();
    for _ in 0..k_draft {
        let Some(mut x) = embed_fn(tok) else { break };
        assert_eq!(x.len(), BACKBONE_HID);
        for v in x.iter_mut() {
            *v *= TARGET_EMBED_SCALE;
        }
        x.extend_from_slice(&hidden);
        let mut h = matvec(&head.pre_projection, &x, HEAD_HID);
        for l in &head.layers {
            layer_forward(l, &mut h, kv, pos, kv_len);
        }
        let mut hn = vec![0.0f32; HEAD_HID];
        rms_norm(&mut hn, &h, &head.final_norm, EPS);
        tok = lm_argmax(&head.embed, &hn);
        hidden = matvec(&head.post_projection, &hn, BACKBONE_HID);
        out.push(tok);
    }
    out
}
