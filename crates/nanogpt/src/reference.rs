//! Independent CPU forward pass -- the composition the GPU path is checked
//! against.
//!
//! The op math comes from the shared CPU oracles (`dgemm::cpu` /
//! `dgops::ops::*::cpu` through `tensor::*_cpu`), so every kernel is pinned to
//! one CPU implementation; what is independent here is the wiring -- embedding,
//! attention, residuals and layer order -- which is the part no per-op test can
//! see.

use crate::model::{EPS, GptConfig, Weights};
use crate::tensor as t;

pub fn forward(w: &Weights, cfg: &GptConfig, tokens: &[u32], seq: usize) -> Vec<f32> {
    let c = cfg.n_embd;
    let heads = cfg.n_head;
    let hd = cfg.head_dim();
    let batch = tokens.len() / seq;
    let m = batch * seq;
    let scale = 1.0 / (hd as f32).sqrt();

    let pos_ids: Vec<u32> = (0..batch)
        .flat_map(|_| (0..seq).map(|x| x as u32))
        .collect();
    let tok = t::gather_rows_cpu(&w.tok_emb, tokens, c);
    let pos = t::gather_rows_cpu(&w.pos_emb, &pos_ids, c);
    let mut x = t::vec_add_cpu(&tok, &pos);

    for lw in &w.layers {
        let h1 = t::rms_norm_cpu(&x, &lw.ln1, m, c, EPS);
        let qkv = t::linear_cpu(&h1, &lw.wqkv, m, 3 * c, c);

        let mut attn = vec![0.0f32; m * c];
        for b in 0..batch {
            for h in 0..heads {
                let base = b * seq * 3 * c + h * hd;
                let q = &qkv[base..];
                let k = &qkv[base + c..];
                let mut scores = t::gemm_cpu(
                    q,
                    k,
                    vec![0.0; seq * seq],
                    seq,
                    seq,
                    hd,
                    3 * c,
                    3 * c,
                    seq,
                    false,
                    true,
                    0.0,
                );
                for i in 0..seq {
                    for j in 0..seq {
                        let idx = i * seq + j;
                        scores[idx] = if j > i { -1e30 } else { scores[idx] * scale };
                    }
                }
                let p = t::softmax_cpu(&scores, seq, seq);
                let v = &qkv[base + 2 * c..];
                let head_out = t::gemm_cpu(
                    &p,
                    v,
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
                );
                for row in 0..seq {
                    let dst = (b * seq + row) * c + h * hd;
                    attn[dst..dst + hd].copy_from_slice(&head_out[row * hd..(row + 1) * hd]);
                }
            }
        }

        let proj = t::linear_cpu(&attn, &lw.wproj, m, c, c);
        let x_attn = t::vec_add_cpu(&x, &proj);
        let h2 = t::rms_norm_cpu(&x_attn, &lw.ln2, m, c, EPS);
        let g = t::linear_cpu(&h2, &lw.w1, m, cfg.mlp(), c);
        let f = t::gelu_cpu(&g);
        let down = t::linear_cpu(&f, &lw.w2, m, c, cfg.mlp());
        x = t::vec_add_cpu(&x_attn, &down);
    }

    let hf = t::rms_norm_cpu(&x, &w.ln_f, m, c, EPS);
    t::linear_cpu(&hf, &w.lm_head, m, cfg.vocab, c)
}
