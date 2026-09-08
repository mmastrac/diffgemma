//! Affine int4 / int8 quantization blocks (groups along K).

use crate::Error;
use crate::dgq::layout::{
    GROUP_SIZE, q4_matrix_bytes, q4_row_bytes, q6_matrix_bytes, q6_row_bytes, q8_matrix_bytes,
    q8_row_bytes,
};
use crate::shaders::cpu::bf16_to_f32;

/// Decode and CPU-GEMM oracle bodies for the block formats live in the shared
/// crate; these names keep their engine paths by re-exporting it. (`q8_weight_at`
/// has no engine caller of its own — it is re-exported for path stability.)
#[allow(unused_imports)]
pub use dgemm::format::block::{
    dequant_matrix_q4, dequant_matrix_q8, dequant_row_q4, dequant_row_q8, q4_gemm_cpu,
    q4_weight_at, q6_gemm_cpu, q6_weight_at, q8_gemm_cpu, q8_gemm_rowk_cpu, q8_weight_at,
};

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
