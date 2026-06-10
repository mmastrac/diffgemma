use crate::kernels::cpu::bf16_to_f32;
use crate::kernels::matmul;
use crate::safetensors::Error;
use crate::tensor::{Bf16Slice, TensorView};

const LM_HEAD_CHUNK: usize = 4096;

pub fn embed_tokens(
    out: &mut [f32],
    token_ids: &[u32],
    embed: Bf16Slice<'_>,
    hidden: usize,
    scale: f32,
) -> Result<(), Error> {
    assert_eq!(out.len(), token_ids.len() * hidden);
    for (t, &id) in token_ids.iter().enumerate() {
        let row_off = id as usize * hidden;
        let out_off = t * hidden;
        for h in 0..hidden {
            out[out_off + h] = bf16_to_f32(embed.get(row_off + h)) * scale;
        }
    }
    Ok(())
}

pub fn soft_embeddings_from_logits(
    out: &mut [f32],
    logits: &[f32],
    embed: Bf16Slice<'_>,
    seq_len: usize,
    vocab: usize,
    hidden: usize,
    scale: f32,
) {
    assert_eq!(logits.len(), seq_len * vocab);
    assert_eq!(out.len(), seq_len * hidden);
    for s in 0..seq_len {
        let logit_row = &logits[s * vocab..(s + 1) * vocab];
        let out_row = &mut out[s * hidden..(s + 1) * hidden];
        out_row.fill(0.0);
        let max_logit = logit_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        let mut probs = vec![0.0f32; vocab];
        for (i, l) in logit_row.iter().enumerate() {
            let p = (l - max_logit).exp();
            probs[i] = p;
            sum += p;
        }
        let inv = 1.0 / sum;
        for p in probs.iter_mut() {
            *p *= inv;
        }
        for (token, &prob) in probs.iter().enumerate() {
            if prob == 0.0 {
                continue;
            }
            let row_off = token * hidden;
            for h in 0..hidden {
                out_row[h] += prob * bf16_to_f32(embed.get(row_off + h)) * scale;
            }
        }
    }
}

/// Tied LM head: `logits = hidden @ embed^T`, computed in vocab chunks over bf16 weights.
pub fn lm_head_tied_bf16(
    logits: &mut [f32],
    hidden: &[f32],
    embed: TensorView<'_>,
    seq_len: usize,
    hidden_dim: usize,
    vocab_size: usize,
) -> Result<(), Error> {
    assert_eq!(hidden.len(), seq_len * hidden_dim);
    assert_eq!(logits.len(), seq_len * vocab_size);
    let embed_bf16 = embed.bf16()?;
    logits.fill(0.0);

    for v0 in (0..vocab_size).step_by(LM_HEAD_CHUNK) {
        let v1 = (v0 + LM_HEAD_CHUNK).min(vocab_size);
        let chunk = v1 - v0;
        let mut w_chunk = vec![0.0f32; chunk * hidden_dim];
        for v in v0..v1 {
            let dst = (v - v0) * hidden_dim;
            let src = v * hidden_dim;
            for h in 0..hidden_dim {
                w_chunk[dst + h] = bf16_to_f32(embed_bf16.get(src + h));
            }
        }
        for s in 0..seq_len {
            matmul::matmul_b_transpose(
                &mut logits[s * vocab_size + v0..s * vocab_size + v1],
                &hidden[s * hidden_dim..(s + 1) * hidden_dim],
                &w_chunk,
                1,
                hidden_dim,
                chunk,
                1.0,
                1.0,
            );
        }
    }
    Ok(())
}

pub fn logit_softcapping(logits: &mut [f32], cap: f32) {
    if cap <= 0.0 {
        return;
    }
    for v in logits.iter_mut() {
        *v = (*v / cap).tanh() * cap;
    }
}
