//! NVFP4 block quant/dequant (MLX 2-tier: E2M1 + per-block FP8 E4M3 scale).

use crate::dgq::fp4::{e2m1_from_f32, e2m1_to_f32, fp8_e4m3_from_f32, fp8_e4m3_to_f32, FP4_E2M1_MAX};
use crate::dgq::layout::{
    nvfp4_data_row_bytes, nvfp4_matrix_bytes, nvfp4_row_bytes, nvfp4_scales_row_bytes,
    NVFP4_GROUP_SIZE, NVFP4_HEADER_BYTES,
};
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

fn write_f32_le(dst: &mut [u8], v: f32) {
    dst[..4].copy_from_slice(&v.to_le_bytes());
}

fn read_f32_le(src: &[u8]) -> f32 {
    f32::from_le_bytes([src[0], src[1], src[2], src[3]])
}

/// Quantize one row along K into packed nibbles + block scales (no tensor header).
pub fn quantize_row_nvfp4(row: &[f32], in_dim: usize, dst: &mut [u8], global_scale: f32) -> usize {
    let need = nvfp4_row_bytes(in_dim);
    assert!(dst.len() >= need);
    let data_len = nvfp4_data_row_bytes(in_dim);
    let scales_len = nvfp4_scales_row_bytes(in_dim);
    let (data, scale_bytes) = dst.split_at_mut(data_len);
    let scales = &mut scale_bytes[..scales_len];
    data.fill(0);
    scales.fill(0);

    let inv_global = if global_scale.abs() < 1e-12 {
        1.0
    } else {
        1.0 / global_scale
    };

    let mut gi = 0usize;
    while gi < in_dim {
        let g_end = (gi + NVFP4_GROUP_SIZE).min(in_dim);
        let g_len = g_end - gi;
        let mut amax = 0.0f32;
        for &v in &row[gi..g_end] {
            amax = amax.max((v * inv_global).abs());
        }
        let scale_f = if amax < 1e-12 {
            0.0
        } else {
            amax / FP4_E2M1_MAX
        };
        let scale_bits = fp8_e4m3_from_f32(scale_f);
        let scale = fp8_e4m3_to_f32(scale_bits);
        scales[gi / NVFP4_GROUP_SIZE] = scale_bits;

        for j in 0..g_len {
            let idx = gi + j;
            let q = if scale == 0.0 {
                0u8
            } else {
                e2m1_from_f32(row[idx] * inv_global / scale)
            };
            let byte_i = idx / 2;
            if idx % 2 == 0 {
                data[byte_i] = (data[byte_i] & 0xf0) | (q & 0x0f);
            } else {
                data[byte_i] = (data[byte_i] & 0x0f) | (q << 4);
            }
        }
        gi += NVFP4_GROUP_SIZE;
    }
    need
}

pub fn dequant_row_nvfp4(src: &[u8], in_dim: usize, dst: &mut [f32], global_scale: f32) {
    assert_eq!(dst.len(), in_dim);
    let data_len = nvfp4_data_row_bytes(in_dim);
    let scales_len = nvfp4_scales_row_bytes(in_dim);
    assert!(src.len() >= data_len + scales_len);
    let data = &src[..data_len];
    let scales = &src[data_len..data_len + scales_len];

    for idx in 0..in_dim {
        let byte = data[idx / 2];
        let q = if idx % 2 == 0 {
            byte & 0x0f
        } else {
            byte >> 4
        };
        let scale = fp8_e4m3_to_f32(scales[idx / NVFP4_GROUP_SIZE]);
        dst[idx] = e2m1_to_f32(q) * scale * global_scale;
    }
}

pub fn dequant_matrix_nvfp4(
    src: &[u8],
    out_dim: usize,
    in_dim: usize,
    dst: &mut [f32],
    global_scale: f32,
) {
    let row_bytes = nvfp4_row_bytes(in_dim);
    assert_eq!(src.len(), out_dim * row_bytes);
    assert_eq!(dst.len(), out_dim * in_dim);
    for row in 0..out_dim {
        dequant_row_nvfp4(
            &src[row * row_bytes..(row + 1) * row_bytes],
            in_dim,
            &mut dst[row * in_dim..(row + 1) * in_dim],
            global_scale,
        );
    }
}

/// Quantize `[out, in]` f32 matrix; `dst` includes 4-byte global scale prefix (written as 1.0).
pub fn quantize_f32_matrix_nvfp4(src: &[f32], out_dim: usize, in_dim: usize, dst: &mut [u8]) {
    let need = nvfp4_matrix_bytes(out_dim, in_dim);
    assert_eq!(dst.len(), need);
    assert_eq!(src.len(), out_dim * in_dim);
    write_f32_le(&mut dst[..NVFP4_HEADER_BYTES], 1.0);
    let body = &mut dst[NVFP4_HEADER_BYTES..];
    let row_bytes = nvfp4_row_bytes(in_dim);
    for row in 0..out_dim {
        quantize_row_nvfp4(
            &src[row * in_dim..(row + 1) * in_dim],
            in_dim,
            &mut body[row * row_bytes..],
            1.0,
        );
    }
}

/// Quantize `[out, in]` bf16 matrix; `dst` includes 4-byte global scale prefix (written as 1.0).
pub fn quantize_bf16_matrix_nvfp4(src: &[u8], out_dim: usize, in_dim: usize, dst: &mut [u8]) {
    let need = nvfp4_matrix_bytes(out_dim, in_dim);
    assert_eq!(dst.len(), need);
    write_f32_le(&mut dst[..NVFP4_HEADER_BYTES], 1.0);
    let body = &mut dst[NVFP4_HEADER_BYTES..];
    let row_bytes = nvfp4_row_bytes(in_dim);
    let mut row_f32 = vec![0.0f32; in_dim];
    for row in 0..out_dim {
        let row_src = &src[(row * in_dim * 2)..(row + 1) * in_dim * 2];
        bf16_bytes_to_f32(row_src, &mut row_f32);
        quantize_row_nvfp4(&row_f32, in_dim, &mut body[row * row_bytes..], 1.0);
    }
}

pub fn quantize_expert_stack_f32_nvfp4(
    src: &[f32],
    experts: usize,
    out_dim: usize,
    in_dim: usize,
    dst: &mut [u8],
) -> Result<(), Error> {
    let per_expert = nvfp4_matrix_bytes(out_dim, in_dim);
    let need = experts * per_expert;
    if dst.len() != need || src.len() != experts * out_dim * in_dim {
        return Err(Error::Format("expert nvfp4 dst size mismatch"));
    }
    for e in 0..experts {
        quantize_f32_matrix_nvfp4(
            &src[e * out_dim * in_dim..(e + 1) * out_dim * in_dim],
            out_dim,
            in_dim,
            &mut dst[e * per_expert..(e + 1) * per_expert],
        );
    }
    Ok(())
}

pub fn dequant_matrix_nvfp4_payload(
    src: &[u8],
    out_dim: usize,
    in_dim: usize,
    dst: &mut [f32],
) -> Result<f32, Error> {
    let need = nvfp4_matrix_bytes(out_dim, in_dim);
    if src.len() != need {
        return Err(Error::Format("nvfp4 matrix size mismatch"));
    }
    let global_scale = read_f32_le(&src[..NVFP4_HEADER_BYTES]);
    dequant_matrix_nvfp4(&src[NVFP4_HEADER_BYTES..], out_dim, in_dim, dst, global_scale);
    Ok(global_scale)
}

#[inline]
pub fn nvfp4_weight_at(
    src: &[u8],
    row: usize,
    col: usize,
    in_dim: usize,
    global_scale: f32,
) -> f32 {
    let row_bytes = nvfp4_row_bytes(in_dim);
    let row_off = row * row_bytes;
    let data_len = nvfp4_data_row_bytes(in_dim);
    let byte = src[row_off + col / 2];
    let q = if col % 2 == 0 {
        byte & 0x0f
    } else {
        byte >> 4
    };
    let scale = fp8_e4m3_to_f32(src[row_off + data_len + col / NVFP4_GROUP_SIZE]);
    e2m1_to_f32(q) * scale * global_scale
}

/// CPU NVFP4 GEMM (oracle for Metal parity).
pub fn nvfp4_gemm_cpu(
    a: &[f32],
    m: usize,
    k: usize,
    w_nvfp4: &[u8],
    n: usize,
    global_scale: f32,
    out: &mut [f32],
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(out.len(), m * n);
    let row_bytes = nvfp4_row_bytes(k);
    assert_eq!(w_nvfp4.len(), n * row_bytes);
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            let row_off = col * row_bytes;
            for p in 0..k {
                sum += a[row * k + p] * nvfp4_weight_at(w_nvfp4, col, p, k, global_scale);
            }
            out[row * n + col] = sum;
        }
    }
}

pub fn quantize_expert_stack_nvfp4(
    src: &[u8],
    experts: usize,
    out_dim: usize,
    in_dim: usize,
    dst: &mut [u8],
) -> Result<(), Error> {
    let per_expert = nvfp4_matrix_bytes(out_dim, in_dim);
    let stride = out_dim * in_dim * 2;
    if src.len() != experts * stride {
        return Err(Error::Format("expert bf16 size mismatch"));
    }
    if dst.len() != experts * per_expert {
        return Err(Error::Format("expert nvfp4 dst size mismatch"));
    }
    for e in 0..experts {
        quantize_bf16_matrix_nvfp4(
            &src[e * stride..(e + 1) * stride],
            out_dim,
            in_dim,
            &mut dst[e * per_expert..(e + 1) * per_expert],
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvfp4_roundtrip_small_matrix() {
        let in_dim = 32usize;
        let out_dim = 2usize;
        let mut src = vec![0u8; out_dim * in_dim * 2];
        for row in 0..out_dim {
            for col in 0..in_dim {
                let v = (col as f32 * 0.1 - 1.5) + row as f32 * 0.05;
                let bits = f32_to_bf16_bits(v).to_le_bytes();
                let i = (row * in_dim + col) * 2;
                src[i] = bits[0];
                src[i + 1] = bits[1];
            }
        }
        let mut q = vec![0u8; nvfp4_matrix_bytes(out_dim, in_dim)];
        quantize_bf16_matrix_nvfp4(&src, out_dim, in_dim, &mut q);
        let mut out = vec![0.0f32; out_dim * in_dim];
        dequant_matrix_nvfp4_payload(&q, out_dim, in_dim, &mut out).unwrap();
        let mut orig = vec![0.0f32; out_dim * in_dim];
        bf16_bytes_to_f32(&src, &mut orig);
        let mut max_err = 0.0f32;
        for (a, b) in orig.iter().zip(out.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 0.35, "max_err={max_err}");
    }

    #[test]
    fn nvfp4_matches_mlx_linspace_row() {
        // Reference: mlx.quantize(linspace(-3,3,32), group_size=16, mode=nvfp4)
        let row: Vec<f32> = (0..32)
            .map(|i| -3.0 + (i as f32) * (6.0 / 31.0))
            .collect();
        let in_dim = row.len();
        let mut packed = vec![0u8; nvfp4_row_bytes(in_dim)];
        quantize_row_nvfp4(&row, in_dim, &mut packed, 1.0);
        let data_len = nvfp4_data_row_bytes(in_dim);
        assert_eq!(packed[data_len], 48);
        assert_eq!(packed[data_len + 1], 48);
        let mut dec = vec![0.0f32; in_dim];
        dequant_row_nvfp4(&packed, in_dim, &mut dec, 1.0);
        let mlx_ref = [
            -3.0, -3.0, -3.0, -2.0, -2.0, -2.0, -2.0, -1.5, -1.5, -1.5, -1.0, -0.75, -0.75,
            -0.5, -0.25, -0.0, 0.0, 0.25, 0.5, 0.75, 0.75, 1.0, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0,
            2.0, 3.0, 3.0, 3.0,
        ];
        for (got, want) in dec.iter().zip(mlx_ref.iter()) {
            assert!(
                (got - want).abs() < 1e-4,
                "dequant mismatch: got {got} want {want}"
            );
        }
    }
}
