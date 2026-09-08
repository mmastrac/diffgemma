//! Affine int4 / int6 / int8 decode and the CPU GEMM oracles that read a
//! quantized row by index.

use super::bf16::bf16_to_f32;
use super::layout::{
    GROUP_SIZE, q4_matrix_bytes, q4_row_bytes, q6_row_bytes, q8_matrix_bytes, q8_row_bytes,
};

/// Dequant one Q4 row to f32.
pub fn dequant_row_q4(src: &[u8], in_dim: usize, dst: &mut [f32]) {
    assert_eq!(dst.len(), in_dim);
    let mut si = 0usize;
    let mut gi = 0usize;
    while gi < in_dim {
        let g_end = (gi + GROUP_SIZE).min(in_dim);
        let g_len = g_end - gi;
        let delta = bf16_to_f32(u16::from_le_bytes([src[si], src[si + 1]]));
        let min = bf16_to_f32(u16::from_le_bytes([src[si + 2], src[si + 3]]));
        si += 4;
        for j in 0..g_len {
            let byte = src[si + j / 2];
            let q = if j % 2 == 0 { byte & 0x0f } else { byte >> 4 } as f32;
            dst[gi + j] = delta * q + min;
        }
        si += GROUP_SIZE / 2;
        gi += GROUP_SIZE;
    }
}

pub fn dequant_matrix_q4(src: &[u8], out_dim: usize, in_dim: usize, dst: &mut [f32]) {
    assert_eq!(src.len(), q4_matrix_bytes(out_dim, in_dim));
    assert_eq!(dst.len(), out_dim * in_dim);
    let row_bytes = q4_row_bytes(in_dim);
    for row in 0..out_dim {
        dequant_row_q4(
            &src[row * row_bytes..(row + 1) * row_bytes],
            in_dim,
            &mut dst[row * in_dim..(row + 1) * in_dim],
        );
    }
}

#[inline]
pub fn q4_weight_at(src: &[u8], row: usize, col: usize, in_dim: usize) -> f32 {
    let row_bytes = q4_row_bytes(in_dim);
    let row_off = row * row_bytes;
    let g = col / GROUP_SIZE;
    let j = col % GROUP_SIZE;
    let si = row_off + g * (4 + GROUP_SIZE / 2);
    let delta = bf16_to_f32(u16::from_le_bytes([src[si], src[si + 1]]));
    let min = bf16_to_f32(u16::from_le_bytes([src[si + 2], src[si + 3]]));
    let byte = src[si + 4 + j / 2];
    let q = if j.is_multiple_of(2) {
        (byte & 0x0f) as f32
    } else {
        (byte >> 4) as f32
    };
    delta * q + min
}

/// CPU Q4 GEMM matching `gemm_linear_f32` (deterministic parity path).
pub fn q4_gemm_cpu(a: &[f32], m: usize, k: usize, w_q4: &[u8], n: usize, out: &mut [f32]) {
    assert_eq!(a.len(), m * k);
    assert_eq!(out.len(), m * n);
    assert_eq!(w_q4.len(), q4_matrix_bytes(n, k));
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[row * k + p] * q4_weight_at(w_q4, col, p, k);
            }
            out[row * n + col] = sum;
        }
    }
}

pub fn dequant_row_q8(src: &[u8], in_dim: usize, dst: &mut [f32]) {
    assert_eq!(dst.len(), in_dim);
    let scale = bf16_to_f32(u16::from_le_bytes([src[0], src[1]]));
    for i in 0..in_dim {
        let q = src[2 + i] as i8 as f32;
        dst[i] = q * scale;
    }
}

pub fn dequant_matrix_q8(src: &[u8], out_dim: usize, in_dim: usize, dst: &mut [f32]) {
    assert_eq!(src.len(), q8_matrix_bytes(out_dim, in_dim));
    assert_eq!(dst.len(), out_dim * in_dim);
    let row_bytes = q8_row_bytes(in_dim);
    for row in 0..out_dim {
        dequant_row_q8(
            &src[row * row_bytes..(row + 1) * row_bytes],
            in_dim,
            &mut dst[row * in_dim..(row + 1) * in_dim],
        );
    }
}

#[inline]
pub fn q8_weight_at(src: &[u8], row: usize, col: usize, in_dim: usize) -> f32 {
    let row_bytes = q8_row_bytes(in_dim);
    let row_off = row * row_bytes;
    let scale = bf16_to_f32(u16::from_le_bytes([src[row_off], src[row_off + 1]]));
    let q = src[row_off + 2 + col] as i8 as f32;
    q * scale
}

/// CPU Q8 GEMM: `y[M,N] = x[M,K] @ W[N,K]^T` (matches `gemm_q8`).
pub fn q8_gemm_cpu(a: &[f32], m: usize, k: usize, w_q8: &[u8], n: usize, out: &mut [f32]) {
    assert_eq!(a.len(), m * k);
    assert_eq!(out.len(), m * n);
    assert_eq!(w_q8.len(), q8_matrix_bytes(n, k));
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[row * k + p] * q8_weight_at(w_q8, col, p, k);
            }
            out[row * n + col] = sum;
        }
    }
}

/// CPU Q8 GEMM: `y[M,N] = x[M,K] @ W[K,N]` with K-indexed rows (matches `gemm_q8_rowk`).
pub fn q8_gemm_rowk_cpu(a: &[f32], m: usize, k: usize, w_q8: &[u8], n: usize, out: &mut [f32]) {
    assert_eq!(a.len(), m * k);
    assert_eq!(out.len(), m * n);
    assert_eq!(w_q8.len(), k * q8_row_bytes(n));
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[row * k + p] * q8_weight_at(w_q8, p, col, n);
            }
            out[row * n + col] = sum;
        }
    }
}

/// Dequantize one Q6 value (CPU mirror of the kernel decode; kernel computes
/// half-precision s*q+mn, this returns the f32 pre-rounding value for tests).
pub fn q6_weight_at(row_base: &[u8], col: usize) -> f32 {
    const BLOCK: usize = 4 + GROUP_SIZE * 6 / 8;
    let g = col / GROUP_SIZE;
    let j = col % GROUP_SIZE;
    let blk = &row_base[g * BLOCK..];
    let scale = bf16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
    let mn = bf16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
    let w = j / 4;
    let r = j % 4;
    let b = &blk[4 + w * 3..];
    let v: u32 = (b[0] as u32) | ((b[1] as u32) << 8) | ((b[2] as u32) << 16);
    let q = (v >> (6 * r)) & 0x3F;
    scale * q as f32 + mn
}

/// CPU q6 GEMM (engine fallback): y[m,n] = a[m,k] @ Wq6[n,k]^T.
pub fn q6_gemm_cpu(a: &[f32], m: usize, k: usize, w_q6: &[u8], n: usize, out: &mut [f32]) {
    let row_bytes = q6_row_bytes(k);
    let mut wrow = vec![0.0f32; k];
    for col in 0..n {
        let row = &w_q6[col * row_bytes..(col + 1) * row_bytes];
        for (c, w) in wrow.iter_mut().enumerate() {
            *w = q6_weight_at(row, c);
        }
        for r in 0..m {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[r * k + p] * wrow[p];
            }
            out[r * n + col] = acc;
        }
    }
}
