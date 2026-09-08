//! Independent CPU forward pass -- the oracle the GPU composition is checked
//! against. Deliberately written as plain loops over flat arrays, sharing no
//! helper with model.rs.

use crate::model::{GptConfig, Weights};

fn linear(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += x[i * k + kk] * w[j * k + kk];
            }
            y[i * n + j] = acc;
        }
    }
    y
}

fn rms_norm(x: &[f32], weight: &[f32], rows: usize, hidden: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * hidden];
    for r in 0..rows {
        let off = r * hidden;
        let mut sum_sq = 0.0f32;
        for i in 0..hidden {
            sum_sq += x[off + i] * x[off + i];
        }
        let inv = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
        for i in 0..hidden {
            out[off + i] = x[off + i] * inv * weight[i];
        }
    }
    out
}

fn gelu(x: &[f32]) -> Vec<f32> {
    const COEF: f32 = 0.797_884_6;
    x.iter()
        .map(|&v| {
            let u = COEF * (v + 0.044_715 * v * v * v);
            let tanh_u = if u > 8.0 {
                1.0
            } else if u < -8.0 {
                -1.0
            } else {
                u.tanh()
            };
            0.5 * v * (1.0 + tanh_u)
        })
        .collect()
}

fn softmax_row(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

pub fn forward(w: &Weights, cfg: &GptConfig, tokens: &[u32], seq: usize) -> Vec<f32> {
    let c = cfg.n_embd;
    let heads = cfg.n_head;
    let hd = cfg.head_dim();
    let batch = tokens.len() / seq;
    let m = batch * seq;
    let scale = 1.0 / (hd as f32).sqrt();

    let mut x = vec![0.0f32; m * c];
    for (i, &tok) in tokens.iter().enumerate() {
        let t = i % seq;
        for d in 0..c {
            x[i * c + d] = w.tok_emb[tok as usize * c + d] + w.pos_emb[t * c + d];
        }
    }

    for lw in &w.layers {
        let h1 = rms_norm(&x, &lw.ln1, m, c, crate::model::EPS);
        let qkv = linear(&h1, &lw.wqkv, m, 3 * c, c);

        let mut attn = vec![0.0f32; m * c];
        for b in 0..batch {
            for h in 0..heads {
                let mut scores = vec![0.0f32; seq * seq];
                for i in 0..seq {
                    for j in 0..seq {
                        let mut acc = 0.0f32;
                        for d in 0..hd {
                            let q = qkv[(b * seq + i) * 3 * c + h * hd + d];
                            let k = qkv[(b * seq + j) * 3 * c + c + h * hd + d];
                            acc += q * k;
                        }
                        scores[i * seq + j] = if j > i { -1e30 } else { acc * scale };
                    }
                    softmax_row(&mut scores[i * seq..(i + 1) * seq]);
                }
                for i in 0..seq {
                    for d in 0..hd {
                        let mut acc = 0.0f32;
                        for j in 0..seq {
                            let p = scores[i * seq + j];
                            let v = qkv[(b * seq + j) * 3 * c + 2 * c + h * hd + d];
                            acc += p * v;
                        }
                        attn[(b * seq + i) * c + h * hd + d] = acc;
                    }
                }
            }
        }

        let proj = linear(&attn, &lw.wproj, m, c, c);
        let x_attn: Vec<f32> = x.iter().zip(proj.iter()).map(|(a, b)| a + b).collect();
        let h2 = rms_norm(&x_attn, &lw.ln2, m, c, crate::model::EPS);
        let g = linear(&h2, &lw.w1, m, cfg.mlp(), c);
        let f = gelu(&g);
        let down = linear(&f, &lw.w2, m, c, cfg.mlp());
        x = x_attn.iter().zip(down.iter()).map(|(a, b)| a + b).collect();
    }

    let hf = rms_norm(&x, &w.ln_f, m, c, crate::model::EPS);
    linear(&hf, &w.lm_head, m, cfg.vocab, c)
}
