//! Backward pass: analytic gradients for the tiny GPT, composed from the same
//! dgops kernels as the forward pass (every matrix gradient is a GEMM with the
//! appropriate transpose flags).

use crate::model::{Cache, EPS, GptConfig, Weights};
use crate::tensor as t;
use dgops::Error;

pub struct LayerGrads {
    pub ln1: Vec<f32>,
    pub wqkv: Vec<f32>,
    pub wproj: Vec<f32>,
    pub ln2: Vec<f32>,
    pub w1: Vec<f32>,
    pub w2: Vec<f32>,
}

impl LayerGrads {
    fn zeros(cfg: &GptConfig) -> Self {
        let c = cfg.n_embd;
        Self {
            ln1: vec![0.0; c],
            wqkv: vec![0.0; 3 * c * c],
            wproj: vec![0.0; c * c],
            ln2: vec![0.0; c],
            w1: vec![0.0; cfg.mlp() * c],
            w2: vec![0.0; c * cfg.mlp()],
        }
    }
}

pub struct Grads {
    pub tok_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,
    pub layers: Vec<LayerGrads>,
    pub ln_f: Vec<f32>,
    pub lm_head: Vec<f32>,
}

/// Mean cross-entropy and dLogits = (softmax - onehot) / rows.
fn cross_entropy(logits: &[f32], targets: &[u32], rows: usize, vocab: usize) -> (f32, Vec<f32>) {
    let mut loss = 0.0f32;
    let mut d = vec![0.0f32; rows * vocab];
    for i in 0..rows {
        let row = &logits[i * vocab..(i + 1) * vocab];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..vocab {
            let e = (row[j] - max).exp();
            d[i * vocab + j] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        let tgt = targets[i] as usize;
        loss += -(row[tgt] - max) + sum.ln();
        for j in 0..vocab {
            let p = d[i * vocab + j] * inv;
            d[i * vocab + j] = (p - if j == tgt { 1.0 } else { 0.0 }) / rows as f32;
        }
    }
    (loss / rows as f32, d)
}

/// Cross-entropy of a forward pass, without keeping the cache.
pub fn loss(w: &Weights, cfg: &GptConfig, tokens: &[u32], targets: &[u32]) -> Result<f32, Error> {
    let cache = crate::model::forward(w, cfg, tokens, cfg.block)?;
    let rows = cache.logits.len() / cfg.vocab;
    Ok(cross_entropy(&cache.logits, targets, rows, cfg.vocab).0)
}

/// (dX, dW) for Y = X @ W^T with X (m,k), W (n,k).
fn linear_bwd(
    x: &[f32],
    w: &[f32],
    dy: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(Vec<f32>, Vec<f32>), Error> {
    let dx = t::gemm(dy, w, vec![0.0; m * k], m, k, n, n, k, k, false, false, 0.0)?;
    let dw = t::gemm(dy, x, vec![0.0; n * k], n, k, m, n, k, k, true, false, 0.0)?;
    Ok((dx, dw))
}

/// dW for an RMSNorm scale: sum over rows of dy * x * inv.
fn rms_norm_dw(x: &[f32], dy: &[f32], rows: usize, hidden: usize, eps: f32) -> Vec<f32> {
    let mut dw = vec![0.0f32; hidden];
    for r in 0..rows {
        let off = r * hidden;
        let mut sum_sq = 0.0f32;
        for i in 0..hidden {
            sum_sq += x[off + i] * x[off + i];
        }
        let inv = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
        for i in 0..hidden {
            dw[i] += dy[off + i] * x[off + i] * inv;
        }
    }
    dw
}

pub fn backward(
    w: &Weights,
    cfg: &GptConfig,
    cache: &Cache,
    targets: &[u32],
) -> Result<(f32, Grads), Error> {
    let c = cfg.n_embd;
    let heads = cfg.n_head;
    let hd = cfg.head_dim();
    let seq = cache.seq;
    let batch = cache.batch;
    let m = batch * seq;
    let vocab = cfg.vocab;

    let (loss, d_logits) = cross_entropy(&cache.logits, targets, m, vocab);

    let (d_hf, d_lm_head) = linear_bwd(&cache.hf, &w.lm_head, &d_logits, m, vocab, c)?;
    let x_final: &[f32] = cache
        .layers
        .last()
        .map(|l| l.x_out.as_slice())
        .unwrap_or(&cache.x0);
    let d_ln_f = rms_norm_dw(x_final, &d_hf, m, c, EPS);
    let mut d_x = t::rms_norm_backward(x_final, &w.ln_f, &d_hf, m, c, EPS)?;

    let mut layer_grads: Vec<LayerGrads> =
        (0..cfg.n_layer).map(|_| LayerGrads::zeros(cfg)).collect();
    let scale = 1.0 / (hd as f32).sqrt();

    for l in (0..cfg.n_layer).rev() {
        let lc = &cache.layers[l];
        let lw = &w.layers[l];
        let x_in: &[f32] = if l == 0 {
            &cache.x0
        } else {
            &cache.layers[l - 1].x_out
        };

        // x_out = x_attn + down
        let d_down = d_x.clone();
        let mut d_x_attn = d_x.clone();

        let (d_f, d_w2) = linear_bwd(&lc.f, &lw.w2, &d_down, m, c, cfg.mlp())?;
        let d_g = t::gelu_backward(&lc.g, &d_f)?;
        let (d_h2, d_w1) = linear_bwd(&lc.h2, &lw.w1, &d_g, m, cfg.mlp(), c)?;
        layer_grads[l].w2 = d_w2;
        layer_grads[l].w1 = d_w1;
        layer_grads[l].ln2 = rms_norm_dw(&lc.x_attn, &d_h2, m, c, EPS);
        let d_ln2_in = t::rms_norm_backward(&lc.x_attn, &lw.ln2, &d_h2, m, c, EPS)?;
        for (a, b) in d_x_attn.iter_mut().zip(d_ln2_in.iter()) {
            *a += b;
        }

        // x_attn = x_in + proj
        let (d_attn, d_wproj) = linear_bwd(&lc.attn, &lw.wproj, &d_x_attn, m, c, c)?;
        layer_grads[l].wproj = d_wproj;

        let mut d_qkv = vec![0.0f32; m * 3 * c];
        for b in 0..batch {
            for h in 0..heads {
                let base = b * seq * 3 * c + h * hd;
                let pbase = (b * heads + h) * seq * seq;
                let p = &lc.probs[pbase..pbase + seq * seq];
                let da = &d_attn[b * seq * c + h * hd..];
                let q = &lc.qkv[base..];
                let k = &lc.qkv[base + c..];
                let v = &lc.qkv[base + 2 * c..];

                // attn_h = P @ V_h
                let dp = t::gemm(
                    da,
                    v,
                    vec![0.0; seq * seq],
                    seq,
                    seq,
                    hd,
                    c,
                    3 * c,
                    seq,
                    false,
                    true,
                    0.0,
                )?;
                let dv = t::gemm(
                    p,
                    da,
                    vec![0.0; seq * hd],
                    seq,
                    hd,
                    seq,
                    seq,
                    c,
                    hd,
                    true,
                    false,
                    0.0,
                )?;
                let ds = t::softmax_backward(p, &dp, seq, seq)?;
                let ds_scaled: Vec<f32> = ds.iter().map(|x| x * scale).collect();
                // scores = (Q @ K^T) * scale
                let dq = t::gemm(
                    &ds_scaled,
                    k,
                    vec![0.0; seq * hd],
                    seq,
                    hd,
                    seq,
                    seq,
                    3 * c,
                    hd,
                    false,
                    false,
                    0.0,
                )?;
                let dk = t::gemm(
                    &ds_scaled,
                    q,
                    vec![0.0; seq * hd],
                    seq,
                    hd,
                    seq,
                    seq,
                    3 * c,
                    hd,
                    true,
                    false,
                    0.0,
                )?;
                for i in 0..seq {
                    let dst = (b * seq + i) * 3 * c + h * hd;
                    d_qkv[dst..dst + hd].copy_from_slice(&dq[i * hd..(i + 1) * hd]);
                    d_qkv[dst + c..dst + c + hd].copy_from_slice(&dk[i * hd..(i + 1) * hd]);
                    d_qkv[dst + 2 * c..dst + 2 * c + hd].copy_from_slice(&dv[i * hd..(i + 1) * hd]);
                }
            }
        }

        // h1 = rms_norm(x_in, ln1); qkv = h1 @ Wqkv^T
        let (d_h1, d_wqkv) = linear_bwd(&lc.h1, &lw.wqkv, &d_qkv, m, 3 * c, c)?;
        layer_grads[l].wqkv = d_wqkv;
        layer_grads[l].ln1 = rms_norm_dw(x_in, &d_h1, m, c, EPS);
        let d_ln1_in = t::rms_norm_backward(x_in, &lw.ln1, &d_h1, m, c, EPS)?;
        let mut d_x_in = d_x_attn;
        for (a, b) in d_x_in.iter_mut().zip(d_ln1_in.iter()) {
            *a += b;
        }
        d_x = d_x_in;
    }

    // x0 = tok_emb(tokens) + pos_emb(positions)
    let d_tok = t::scatter_add_rows(&vec![0.0; vocab * c], &cache.tokens, &d_x, c)?;
    let d_pos = t::scatter_add_rows(&vec![0.0; cfg.block * c], &cache.pos_ids, &d_x, c)?;

    Ok((
        loss,
        Grads {
            tok_emb: d_tok,
            pos_emb: d_pos,
            layers: layer_grads,
            ln_f: d_ln_f,
            lm_head: d_lm_head,
        },
    ))
}

/// Parameter tensors in the fixed order the optimizer walks them.
pub fn param_list(w: &mut Weights) -> Vec<&mut Vec<f32>> {
    let mut out: Vec<&mut Vec<f32>> = vec![&mut w.tok_emb, &mut w.pos_emb];
    for l in &mut w.layers {
        out.push(&mut l.ln1);
        out.push(&mut l.wqkv);
        out.push(&mut l.wproj);
        out.push(&mut l.ln2);
        out.push(&mut l.w1);
        out.push(&mut l.w2);
    }
    out.push(&mut w.ln_f);
    out.push(&mut w.lm_head);
    out
}

/// Gradients in the same order as param_list.
pub fn grad_list(g: &mut Grads) -> Vec<&mut Vec<f32>> {
    let mut out: Vec<&mut Vec<f32>> = vec![&mut g.tok_emb, &mut g.pos_emb];
    for l in &mut g.layers {
        out.push(&mut l.ln1);
        out.push(&mut l.wqkv);
        out.push(&mut l.wproj);
        out.push(&mut l.ln2);
        out.push(&mut l.w1);
        out.push(&mut l.w2);
    }
    out.push(&mut g.ln_f);
    out.push(&mut g.lm_head);
    out
}
