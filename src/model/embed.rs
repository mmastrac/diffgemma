use crate::buffer::Buffer;
use crate::fast_slice::FastBf16Slice;
use crate::kernels::matmul;
use crate::safetensors::Error;
use crate::tensor::{Bf16Slice, TensorView};

pub const LM_HEAD_CHUNK: usize = 4096;

pub fn embed_tokens(
    out: &mut [f32],
    token_ids: &[u32],
    embed: Bf16Slice<'_>,
    hidden: usize,
    scale: f32,
) -> Result<(), Error> {
    assert_eq!(out.len(), token_ids.len() * hidden);
    let table = FastBf16Slice::from_bf16(embed);
    for (t, &id) in token_ids.iter().enumerate() {
        let row_off = id as usize * hidden;
        let out_off = t * hidden;
        for h in 0..hidden {
            unsafe {
                out[out_off + h] = table.to_f32_unchecked(row_off + h) * scale;
            }
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
    prob_scratch: &mut Buffer<f32>,
) {
    assert_eq!(logits.len(), seq_len * vocab);
    assert_eq!(out.len(), seq_len * hidden);
    prob_scratch.ensure_len(vocab);
    let table = FastBf16Slice::from_bf16(embed);
    for s in 0..seq_len {
        let logit_row = &logits[s * vocab..(s + 1) * vocab];
        let out_row = &mut out[s * hidden..(s + 1) * hidden];
        out_row.fill(0.0);
        let max_logit = logit_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        {
            let mut probs = prob_scratch.as_fast_slice_mut();
            for i in 0..vocab {
                let p = (logit_row[i] - max_logit).exp();
                unsafe {
                    probs.write_unchecked(i, p);
                }
                sum += p;
            }
            let inv = 1.0 / sum;
            for i in 0..vocab {
                unsafe {
                    let p = probs.read_unchecked(i) * inv;
                    probs.write_unchecked(i, p);
                }
            }
        }
        let probs = prob_scratch.as_fast_slice();
        for token in 0..vocab {
            let prob = unsafe { probs.read_unchecked(token) };
            if prob == 0.0 {
                continue;
            }
            let row_off = token * hidden;
            for h in 0..hidden {
                unsafe {
                    out_row[h] += prob * table.to_f32_unchecked(row_off + h) * scale;
                }
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
    w_chunk: &mut Buffer<f32>,
) -> Result<(), Error> {
    assert_eq!(hidden.len(), seq_len * hidden_dim);
    assert_eq!(logits.len(), seq_len * vocab_size);
    let embed_bf16 = embed.bf16()?;
    let table = FastBf16Slice::from_bf16(embed_bf16);
    logits.fill(0.0);

    for v0 in (0..vocab_size).step_by(LM_HEAD_CHUNK) {
        let v1 = (v0 + LM_HEAD_CHUNK).min(vocab_size);
        let chunk = v1 - v0;
        w_chunk.ensure_len(chunk * hidden_dim);
        {
            let mut w = w_chunk.as_fast_slice_mut();
            for v in v0..v1 {
                let dst = (v - v0) * hidden_dim;
                let src = v * hidden_dim;
                for h in 0..hidden_dim {
                    unsafe {
                        w.write_unchecked(dst + h, table.to_f32_unchecked(src + h));
                    }
                }
            }
        }
        let w_slice = w_chunk.as_slice();
        for s in 0..seq_len {
            matmul::matmul_b_transpose(
                &mut logits[s * vocab_size + v0..s * vocab_size + v1],
                &hidden[s * hidden_dim..(s + 1) * hidden_dim],
                w_slice,
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
