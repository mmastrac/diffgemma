use crate::metal::batch::GpuBatch;
use crate::metal::device::ComputePipeline;
use crate::safetensors::Error;
use crate::tensor::Bf16Slice;

/// Pack PyTorch linear weights `[out, in]` into `[in, out]` for `bf16_gemm`.
pub fn transpose_weight_bf16(w: &[u16], out_dim: usize, in_dim: usize) -> Vec<u16> {
    let mut t = vec![0u16; in_dim * out_dim];
    for o in 0..out_dim {
        for i in 0..in_dim {
            t[i * out_dim + o] = w[o * in_dim + i];
        }
    }
    t
}

pub fn bf16_slice_to_vec(slice: Bf16Slice<'_>) -> Vec<u16> {
    (0..slice.len()).map(|i| slice.get(i)).collect()
}

#[derive(Clone)]
pub struct CachedLinear {
    pub w_t: Vec<u16>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl CachedLinear {
    pub fn from_bf16(w: Bf16Slice<'_>, out_dim: usize, in_dim: usize) -> Self {
        let w_raw = bf16_slice_to_vec(w);
        Self {
            w_t: transpose_weight_bf16(&w_raw, out_dim, in_dim),
            in_dim,
            out_dim,
        }
    }
}

/// Batched `y = x @ W^T` with pre-transposed weights; readback on `batch.end()`.
pub fn linear_cached_batched(
    batch: &mut GpuBatch<'_>,
    pipeline: &ComputePipeline,
    y: &mut [f32],
    x: &[f32],
    w: &CachedLinear,
    seq_len: usize,
) -> Result<(), Error> {
    if x.len() != seq_len * w.in_dim || y.len() != seq_len * w.out_dim {
        return Err(Error::Format("linear_cached_batched shape mismatch"));
    }
    let buf_a = batch.alloc_f32(x)?;
    let buf_b = batch.alloc_bf16(&w.w_t)?;
    let buf_c = batch.alloc_f32_out(y.len())?;
    batch.dispatch_gemm(
        &pipeline.pipeline,
        &buf_a,
        &buf_b,
        &buf_c,
        seq_len,
        w.out_dim,
        w.in_dim,
    );
    batch.register_read(buf_c, y);
    Ok(())
}
