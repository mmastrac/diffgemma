//! Affine int4 / int8 quantization blocks (groups along K).

use crate::dgq::layout::{q4_matrix_bytes, q4_row_bytes, q8_matrix_bytes, q8_row_bytes, GROUP_SIZE};
use crate::kernels::cpu::bf16_to_f32;
use crate::safetensors::Error;

fn f32_to_bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn bf16_bytes_to_f32(src: &[u8], out: &mut [f32]) {
    let n = src.len() / 2;
    assert_eq!(out.len(), n);
    for i in 0..n {
        let bits = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
        out[i] = bf16_to_f32(bits);
    }
}

/// Quantize `[out, in]` bf16 row-major to Q4 blocks (K groups of 32).
pub fn quantize_bf16_matrix_q4(src: &[u8], out_dim: usize, in_dim: usize, dst: &mut [u8]) {
    let need = q4_matrix_bytes(out_dim, in_dim);
    assert_eq!(dst.len(), need);
    let mut row_f32 = vec![0.0f32; in_dim];
    let mut off = 0usize;
    for row in 0..out_dim {
        let row_src = &src[(row * in_dim * 2)..(row + 1) * in_dim * 2];
        bf16_bytes_to_f32(row_src, &mut row_f32);
        off += quantize_row_q4(&row_f32, in_dim, &mut dst[off..]);
    }
}

pub fn quantize_row_q4(row: &[f32], in_dim: usize, dst: &mut [u8]) -> usize {
    let need = q4_row_bytes(in_dim);
    assert!(dst.len() >= need);
    let mut off = 0usize;
    let mut gi = 0;
    while gi < in_dim {
        let g_end = (gi + GROUP_SIZE).min(in_dim);
        let g_len = g_end - gi;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &v in &row[gi..g_end] {
            min = min.min(v);
            max = max.max(v);
        }
        if min == f32::INFINITY {
            min = 0.0;
            max = 0.0;
        }
        let delta = if max - min < 1e-8 {
            1.0f32
        } else {
            (max - min) / 15.0
        };
        let scale_bits = f32_to_bf16_bits(delta).to_le_bytes();
        let min_bits = f32_to_bf16_bits(min).to_le_bytes();
        dst[off] = scale_bits[0];
        dst[off + 1] = scale_bits[1];
        dst[off + 2] = min_bits[0];
        dst[off + 3] = min_bits[1];
        off += 4;
        let mut nibbles = [0u8; GROUP_SIZE / 2];
        for j in 0..g_len {
            let q = if delta <= 0.0 {
                0u8
            } else {
                ((row[gi + j] - min) / delta).round().clamp(0.0, 15.0) as u8
            };
            if j % 2 == 0 {
                nibbles[j / 2] = q;
            } else {
                nibbles[j / 2] |= q << 4;
            }
        }
        dst[off..off + GROUP_SIZE / 2].copy_from_slice(&nibbles);
        off += GROUP_SIZE / 2;
        gi += GROUP_SIZE;
    }
    need
}

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
fn q4_weight_at(src: &[u8], row: usize, col: usize, in_dim: usize) -> f32 {
    let row_bytes = q4_row_bytes(in_dim);
    let row_off = row * row_bytes;
    let g = col / GROUP_SIZE;
    let j = col % GROUP_SIZE;
    let si = row_off + g * (4 + GROUP_SIZE / 2);
    let delta = bf16_to_f32(u16::from_le_bytes([src[si], src[si + 1]]));
    let min = bf16_to_f32(u16::from_le_bytes([src[si + 2], src[si + 3]]));
    let byte = src[si + 4 + j / 2];
    let q = if j % 2 == 0 {
        (byte & 0x0f) as f32
    } else {
        (byte >> 4) as f32
    };
    delta * q + min
}

/// CPU Q4 GEMM matching `f32_q4_linear.metal` (deterministic parity path).
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

/// Quantize `[out, in]` bf16 to per-row int8 + fp16 scale.
pub fn quantize_bf16_matrix_q8(src: &[u8], out_dim: usize, in_dim: usize, dst: &mut [u8]) {
    let need = q8_matrix_bytes(out_dim, in_dim);
    assert_eq!(dst.len(), need);
    let mut row_f32 = vec![0.0f32; in_dim];
    let mut off = 0usize;
    for row in 0..out_dim {
        let row_src = &src[(row * in_dim * 2)..(row + 1) * in_dim * 2];
        bf16_bytes_to_f32(row_src, &mut row_f32);
        off += quantize_row_q8(&row_f32, in_dim, &mut dst[off..]);
    }
}

pub fn quantize_row_q8(row: &[f32], in_dim: usize, dst: &mut [u8]) -> usize {
    let need = q8_row_bytes(in_dim);
    assert!(dst.len() >= need);
    let mut max_abs = 0.0f32;
    for &v in row {
        max_abs = max_abs.max(v.abs());
    }
    let scale = if max_abs < 1e-8 {
        1.0f32
    } else {
        max_abs / 127.0
    };
    let scale_bits = f32_to_bf16_bits(scale).to_le_bytes();
    dst[0] = scale_bits[0];
    dst[1] = scale_bits[1];
    for (i, &v) in row.iter().enumerate() {
        let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
        dst[2 + i] = q as u8;
    }
    need
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

pub fn quantize_expert_stack_q4(
    src: &[u8],
    experts: usize,
    out_dim: usize,
    in_dim: usize,
    dst: &mut [u8],
) -> Result<(), Error> {
    let stride = out_dim * in_dim * 2;
    let expert_q = q4_matrix_bytes(out_dim, in_dim);
    if src.len() != experts * stride {
        return Err(Error::Format("expert bf16 size mismatch"));
    }
    if dst.len() != experts * expert_q {
        return Err(Error::Format("expert q4 dst size mismatch"));
    }
    for e in 0..experts {
        quantize_bf16_matrix_q4(
            &src[e * stride..(e + 1) * stride],
            out_dim,
            in_dim,
            &mut dst[e * expert_q..(e + 1) * expert_q],
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_roundtrip_small_matrix() {
        let in_dim = 64;
        let out_dim = 4;
        let mut src = vec![0u8; out_dim * in_dim * 2];
        for row in 0..out_dim {
            for col in 0..in_dim {
                let v = ((row * in_dim + col) as f32 * 0.01 - 0.5) as f32;
                let bits = f32_to_bf16_bits(v).to_le_bytes();
                let i = (row * in_dim + col) * 2;
                src[i] = bits[0];
                src[i + 1] = bits[1];
            }
        }
        let mut q = vec![0u8; q4_matrix_bytes(out_dim, in_dim)];
        quantize_bf16_matrix_q4(&src, out_dim, in_dim, &mut q);
        let mut out = vec![0.0f32; out_dim * in_dim];
        dequant_matrix_q4(&q, out_dim, in_dim, &mut out);
        let mut orig = vec![0.0f32; out_dim * in_dim];
        bf16_bytes_to_f32(&src, &mut orig);
        let mut max_err = 0.0f32;
        for (a, b) in orig.iter().zip(out.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 0.15, "max_err={max_err}");
    }
}
