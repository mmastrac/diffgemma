use crate::metal::gemm::{f32_to_bf16, Bf16Gemm};
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

pub fn pack_input_bf16(x: &[f32]) -> Vec<u16> {
    x.iter().map(|&v| f32_to_bf16(v)).collect()
}

/// `y[seq, out] = x[seq, in] @ W[out, in]^T` using bf16 GEMM.
pub fn linear_bf16(
    gemm: &mut Bf16Gemm,
    y: &mut [f32],
    x: &[f32],
    w: Bf16Slice<'_>,
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
) -> Result<(), Error> {
    if x.len() != seq_len * in_dim || y.len() != seq_len * out_dim || w.len() != out_dim * in_dim {
        return Err(Error::Format("linear_bf16 shape mismatch"));
    }
    let w_raw = bf16_slice_to_vec(w);
    let w_t = transpose_weight_bf16(&w_raw, out_dim, in_dim);
    gemm.matmul_f32_bf16(x, &w_t, y, seq_len, in_dim, out_dim)
}
