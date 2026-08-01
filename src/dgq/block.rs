//! Affine int4 / int8 quantization blocks (groups along K).

use crate::Error;
use crate::dgq::layout::{
    GROUP_SIZE, q4_matrix_bytes, q4_row_bytes, q6_matrix_bytes, q6_row_bytes, q8_matrix_bytes,
    q8_row_bytes,
};
use crate::shaders::cpu::bf16_to_f32;

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
pub(crate) fn q4_weight_at(src: &[u8], row: usize, col: usize, in_dim: usize) -> f32 {
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

#[inline]
pub(crate) fn q8_weight_at(src: &[u8], row: usize, col: usize, in_dim: usize) -> f32 {
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
        return Err(Error::Runtime("expert bf16 size mismatch"));
    }
    if dst.len() != experts * expert_q {
        return Err(Error::Runtime("expert q4 dst size mismatch"));
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
                let v = (row * in_dim + col) as f32 * 0.01 - 0.5;
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

pub fn quantize_expert_stack_q6(
    src: &[u8],
    experts: usize,
    out_dim: usize,
    in_dim: usize,
    dst: &mut [u8],
) -> Result<(), Error> {
    let stride = out_dim * in_dim * 2;
    let expert_q = q6_matrix_bytes(out_dim, in_dim);
    if src.len() != experts * stride {
        return Err(Error::Runtime("expert bf16 size mismatch"));
    }
    if dst.len() != experts * expert_q {
        return Err(Error::Runtime("expert q6 dst size mismatch"));
    }
    for e in 0..experts {
        quantize_bf16_matrix_q6(
            &src[e * stride..(e + 1) * stride],
            out_dim,
            in_dim,
            &mut dst[e * expert_q..(e + 1) * expert_q],
        );
    }
    Ok(())
}

/// Quantize one row to Q6 blocks: [scale bf16:2][min bf16:2][24B codes] per
/// 32-wide group. Codes are 6-bit (0..63), packed 4-per-24-bit-LE-word
/// (v = q0 | q1<<6 | q2<<12 | q3<<18 -> 3 bytes). Same affine semantics and
/// bf16 scale/min storage as q4 (scale precision measured immaterial).
pub fn quantize_row_q6(row: &[f32], in_dim: usize, dst: &mut [u8]) -> usize {
    let need = q6_row_bytes(in_dim);
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
            (max - min) / 63.0
        };
        let scale_bits = f32_to_bf16_bits(delta).to_le_bytes();
        let min_bits = f32_to_bf16_bits(min).to_le_bytes();
        dst[off] = scale_bits[0];
        dst[off + 1] = scale_bits[1];
        dst[off + 2] = min_bits[0];
        dst[off + 3] = min_bits[1];
        off += 4;
        let mut codes = [0u8; GROUP_SIZE];
        for j in 0..g_len {
            codes[j] = if delta <= 0.0 {
                0u8
            } else {
                ((row[gi + j] - min) / delta).round().clamp(0.0, 63.0) as u8
            };
        }
        for w in 0..GROUP_SIZE / 4 {
            let v: u32 = (codes[w * 4] as u32)
                | ((codes[w * 4 + 1] as u32) << 6)
                | ((codes[w * 4 + 2] as u32) << 12)
                | ((codes[w * 4 + 3] as u32) << 18);
            dst[off] = (v & 0xFF) as u8;
            dst[off + 1] = ((v >> 8) & 0xFF) as u8;
            dst[off + 2] = ((v >> 16) & 0xFF) as u8;
            off += 3;
        }
        gi = g_end;
    }
    need
}

/// Quantize `[out, in]` bf16 row-major to Q6 blocks.
pub fn quantize_bf16_matrix_q6(src: &[u8], out_dim: usize, in_dim: usize, dst: &mut [u8]) {
    let need = q6_matrix_bytes(out_dim, in_dim);
    assert_eq!(dst.len(), need);
    let mut row_f32 = vec![0.0f32; in_dim];
    let mut off = 0usize;
    for row in 0..out_dim {
        let row_src = &src[(row * in_dim * 2)..(row + 1) * in_dim * 2];
        bf16_bytes_to_f32(row_src, &mut row_f32);
        off += quantize_row_q6(&row_f32, in_dim, &mut dst[off..]);
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

#[cfg(test)]
mod q6_tests {
    use super::*;

    #[test]
    fn q6_roundtrip_error_bound() {
        let in_dim = 128usize;
        let row: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f32) * 0.37).sin() * 0.11 - 0.02)
            .collect();
        let mut dst = vec![0u8; q6_row_bytes(in_dim)];
        quantize_row_q6(&row, in_dim, &mut dst);
        let mut max_err = 0f32;
        let mut sq = 0f64;
        let mut ref_sq = 0f64;
        for (c, &w) in row.iter().enumerate() {
            let d = q6_weight_at(&dst, c) - w;
            max_err = max_err.max(d.abs());
            sq += (d as f64) * (d as f64);
            ref_sq += (w as f64) * (w as f64);
        }
        let rel = (sq / ref_sq).sqrt();
        // 6-bit affine on a smooth signal: well under 2% rel-RMS; each error
        // bounded by ~delta/2 + bf16 scale rounding.
        assert!(rel < 0.02, "rel-RMS {rel}");
        assert!(max_err < 0.006, "max err {max_err}");
    }

    #[test]
    fn q6_packing_exact_codes() {
        // Values on the exact quant lattice roundtrip code-exactly: (i*7)%64
        // covers 0..=63 (min 0, max 63 -> delta exactly 1.0, bf16-exact), so
        // codes == values and every 6-bit pack position is exercised.
        let in_dim = 32usize;
        let row: Vec<f32> = (0..in_dim).map(|i| ((i * 7) % 64) as f32).collect();
        let mut dst = vec![0u8; q6_row_bytes(in_dim)];
        quantize_row_q6(&row, in_dim, &mut dst);
        for (c, &w) in row.iter().enumerate() {
            let d = (q6_weight_at(&dst, c) - w).abs();
            assert!(d < 1e-6, "col {c}: {d}");
        }
    }
}
