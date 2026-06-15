//! Tied lm_head on GPU (q8 embed weights from `.dgq` blob).

use crate::metal::batch::{set_bytes, GpuBatch};
use crate::metal::device::ComputePipeline;
use crate::metal::dgq_gpu::Q8LinearGpu;
use crate::metal::linear::f32_q8_linear_gpu_bufs;
use crate::metal::sampler::GpuLogitsBuf;
use crate::metal::sampler_kernels::GpuSamplerKernels;
use crate::model::embed::LM_HEAD_CHUNK;
use crate::safetensors::Error;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};

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

/// Same as [`lm_head_tied_q8_gpu`] but accumulates into a GPU logits buffer (no CPU readback).
pub fn lm_head_tied_q8_gpu_buf(
    batch: &mut GpuBatch<'_>,
    q8_pipeline: &ComputePipeline,
    sampler_kernels: &GpuSamplerKernels,
    kernels: &crate::metal::kernels::GpuKernels,
    hidden: &[f32],
    embed: &Q8LinearGpu,
    logits: &mut GpuLogitsBuf,
    seq_len: usize,
    hidden_dim: usize,
    vocab_size: usize,
) -> Result<(), Error> {
    assert_eq!(hidden.len(), seq_len * hidden_dim);
    assert_eq!(embed.in_dim, hidden_dim);
    assert_eq!(embed.out_dim, vocab_size);

    let total = seq_len * vocab_size;
    let buf_logits = logits
        .as_buf()
        .ok_or(Error::Format("gpu logits buffer not allocated"))?;
    zero_logits_buf(batch, kernels, buf_logits, total)?;
    let buf_a = batch.alloc_f32(hidden)?;

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
        let params = [
            seq_len as u32,
            chunk as u32,
            v0 as u32,
            vocab_size as u32,
        ];
        let tg_w = 16usize;
        let tg_h = 16usize;
        let grid = objc2_metal::MTLSize {
            width: crate::metal::batch::div_up(chunk, tg_w),
            height: crate::metal::batch::div_up(seq_len, tg_h),
            depth: 1,
        };
        let tg = objc2_metal::MTLSize {
            width: tg_w,
            height: tg_h,
            depth: 1,
        };
        batch.dispatch_with_grid(&sampler_kernels.scatter_vocab_chunk.pipeline, grid, tg, |enc| {
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_c), 0, 0);
                enc.setBuffer_offset_atIndex(Some(buf_logits), 0, 1);
            }
            set_bytes(enc, &params, 2);
        });
    }
    Ok(())
}

/// Zero-fill logits buffer before first chunk write (call once per forward).
pub fn zero_logits_buf(
    batch: &mut GpuBatch<'_>,
    kernels: &crate::metal::kernels::GpuKernels,
    buf: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
) -> Result<(), Error> {
    crate::metal::embed::vec_fill_zero_gpu_buf(batch, kernels, buf, len);
    Ok(())
}

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod tests {
    use super::*;
    use crate::dgq::store::DgqStore;
    use crate::metal::batch::GpuBatch;
    use crate::metal::buffer::{BufferPool, BufferPool as BP};
    use crate::metal::device::MetalContext;
    use crate::metal::dgq_gpu::{load_q8_linear, DgqGpuBlob};
    use crate::metal::sampler::GpuLogitsBuf;
    use crate::metal::sampler_kernels::GpuSamplerKernels;
    use crate::metal::kernels::GpuKernels;
    use std::sync::Arc;

    #[test]
    fn gpu_scatter_lm_head_matches_cpu_readback() {
        let dgq_dir = std::path::Path::new("/tmp/quantized-weights");
        if !dgq_dir.join("model.dgq.json").exists() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let store = DgqStore::open(dgq_dir).expect("open dgq");
        let ctx = MetalContext::new().expect("metal");
        let blob = DgqGpuBlob::from_store(&store, &ctx.device).expect("blob");
        let embed = load_q8_linear(&store, Arc::clone(&blob), "model.decoder.embed_tokens.weight")
            .expect("embed");
        let seq_len = 2usize;
        let hidden = embed.in_dim;
        let vocab = embed.out_dim;
        let hidden_data: Vec<f32> = (0..seq_len * hidden)
            .map(|i| ((i as f32 * 0.0003).sin() * 0.02))
            .collect();

        let mut pool = BufferPool::new();
        let prod = crate::kernels::sub::variant::KernelVariant::PRODUCTION;
        let q8_pipeline =
            crate::kernels::sub::gemm_q8_linear_f32::pipeline_for(&ctx, prod).expect("q8");
        let sampler_kernels = GpuSamplerKernels::new(&ctx).expect("sampler kernels");
        let kernels = GpuKernels::new(&ctx).expect("kernels");

        let mut cpu_logits = vec![0.0f32; seq_len * vocab];
        let mut batch = GpuBatch::begin_with_telemetry(&ctx.queue, &mut pool, &ctx.device, None).expect("batch");
        lm_head_tied_q8_gpu(
            &mut batch,
            &q8_pipeline,
            &hidden_data,
            &embed,
            seq_len,
            hidden,
            vocab,
            &mut cpu_logits,
        )
        .expect("cpu readback lm_head");
        batch.end().expect("end");

        let mut logits_buf = GpuLogitsBuf::new();
        logits_buf
            .ensure(&ctx.device, &mut pool, seq_len * vocab)
            .expect("ensure");
        let mut batch = GpuBatch::begin_with_telemetry(&ctx.queue, &mut pool, &ctx.device, None).expect("batch");
        lm_head_tied_q8_gpu_buf(
            &mut batch,
            &q8_pipeline,
            &sampler_kernels,
            &kernels,
            &hidden_data,
            &embed,
            &mut logits_buf,
            seq_len,
            hidden,
            vocab,
        )
        .expect("gpu buf lm_head");
        batch.end().expect("end");

        let mut gpu_logits = vec![0.0f32; seq_len * vocab];
        BP::read_f32(logits_buf.as_buf().expect("buf"), &mut gpu_logits);

        let mut max_err = 0.0f32;
        let mut nan_cpu = 0usize;
        let mut nan_gpu = 0usize;
        for (a, b) in cpu_logits.iter().zip(gpu_logits.iter()) {
            if a.is_nan() {
                nan_cpu += 1;
            }
            if b.is_nan() {
                nan_gpu += 1;
            }
            if !a.is_nan() && !b.is_nan() {
                max_err = max_err.max((a - b).abs());
            }
        }
        eprintln!(
            "lm_head scatter vs readback: max_err={max_err:.6} nan_cpu={nan_cpu} nan_gpu={nan_gpu} len={}",
            cpu_logits.len()
        );
        assert_eq!(nan_cpu, nan_gpu, "NaN mismatch");
        assert!(max_err < 1e-3, "max_err={max_err}");
    }
}
