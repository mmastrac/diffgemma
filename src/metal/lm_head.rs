//! Tied lm_head on GPU (q8 embed weights from `.dgq` blob).

use crate::metal::batch::GpuBatch;
use crate::metal::device::ComputePipeline;
use crate::metal::dgq_gpu::Q8LinearGpu;
use crate::metal::linear::f32_q8_linear_gpu_bufs;
use crate::model::embed::LM_HEAD_CHUNK;
use crate::safetensors::Error;

/// `logits = hidden @ embed^T` in vocab chunks; q8 weights read directly from the blob.
pub fn lm_head_tied_q8_gpu(
    batch: &mut GpuBatch<'_>,
    q8_pipeline: &ComputePipeline,
    hidden: &[f32],
    embed: &Q8LinearGpu,
    seq_len: usize,
    hidden_dim: usize,
    vocab_size: usize,
    logits: &mut [f32],
) -> Result<(), Error> {
    assert_eq!(hidden.len(), seq_len * hidden_dim);
    assert_eq!(logits.len(), seq_len * vocab_size);
    assert_eq!(embed.in_dim, hidden_dim);
    assert_eq!(embed.out_dim, vocab_size);

    let buf_a = batch.alloc_f32(hidden)?;
    logits.fill(0.0);

    for v0 in (0..vocab_size).step_by(LM_HEAD_CHUNK) {
        let v1 = (v0 + LM_HEAD_CHUNK).min(vocab_size);
        let chunk = v1 - v0;
        let w_chunk = embed.row_slice(v0, chunk);
        let buf_c = f32_q8_linear_gpu_bufs(
            batch,
            q8_pipeline,
            &buf_a,
            &w_chunk,
            seq_len,
            hidden_dim,
            chunk,
        )?;
        for s in 0..seq_len {
            let dst = &mut logits[s * vocab_size + v0..s * vocab_size + v1];
            batch.register_read_offset(buf_c.clone(), s * chunk, dst);
        }
    }
    Ok(())
}
