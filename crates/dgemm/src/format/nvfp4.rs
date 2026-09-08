//! NVFP4 decode and its CPU GEMM oracle (MLX 2-tier: E2M1 codes + per-block
//! FP8 E4M3 scale).

use super::fp4::{e2m1_to_f32, fp8_e4m3_to_f32};
use super::layout::{
    NVFP4_GROUP_SIZE, NVFP4_HEADER_BYTES, nvfp4_data_row_bytes, nvfp4_matrix_bytes,
    nvfp4_row_bytes, nvfp4_scales_row_bytes,
};
use crate::Error;

fn read_f32_le(src: &[u8]) -> f32 {
    f32::from_le_bytes([src[0], src[1], src[2], src[3]])
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
        let q = if idx % 2 == 0 { byte & 0x0f } else { byte >> 4 };
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
    dequant_matrix_nvfp4(
        &src[NVFP4_HEADER_BYTES..],
        out_dim,
        in_dim,
        dst,
        global_scale,
    );
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
    let q = if col.is_multiple_of(2) {
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
            let _row_off = col * row_bytes;
            for p in 0..k {
                sum += a[row * k + p] * nvfp4_weight_at(w_nvfp4, col, p, k, global_scale);
            }
            out[row * n + col] = sum;
        }
    }
}
