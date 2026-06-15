//! CPU twins for `shaders/include/dequant.metal` — single decode reference.

use crate::dgq::layout::{nvfp4_row_bytes, NVFP4_HEADER_BYTES};
use crate::dgq::nvfp4::nvfp4_weight_at;
use crate::kernels::cpu::bf16_to_f32;

/// Mirror of `dequant_q4_group` (32 floats from one 20-byte Q4 block).
pub fn dequant_q4_group(g: &[u8; 20]) -> [f32; 32] {
    let s = bf16_to_f32(u16::from_le_bytes([g[0], g[1]]));
    let mn = bf16_to_f32(u16::from_le_bytes([g[2], g[3]]));
    let mut out = [0.0f32; 32];
    for i in 0..16 {
        let b = g[4 + i];
        out[2 * i] = s * (b & 0x0f) as f32 + mn;
        out[2 * i + 1] = s * (b >> 4) as f32 + mn;
    }
    out
}

/// Mirror of `q4_at_col` in `include/dequant.metal`.
pub fn q4_at_col(row_base: &[u8], col: usize) -> f32 {
    let g = col / 32;
    let j = col % 32;
    let blk = &row_base[g * 20..g * 20 + 20];
    let delta = bf16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
    let mn = bf16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
    let byte = blk[4 + j / 2];
    let q = if j & 1 != 0 {
        (byte >> 4) as f32
    } else {
        (byte & 0x0f) as f32
    };
    delta * q + mn
}

/// CPU oracle for `q4_group_k_order` dump layout (64 floats).
pub fn q4_group_k_order_row(row_base: &[u8], k0: usize, _in_dim: usize) -> Vec<f32> {
    let g_off = (k0 / 32) * 20;
    let grp: &[u8; 20] = row_base[g_off..g_off + 20].try_into().expect("group");
    let via_dequant = dequant_q4_group(grp);
    let mut out = Vec::with_capacity(64);
    out.extend_from_slice(&via_dequant);
    for m in 0..32 {
        out.push(q4_at_col(row_base, k0 + m));
    }
    out
}

/// CPU oracle for `nvfp4_tile_k_order` dump layout (64 floats).
pub fn nvfp4_tile_k_order_matrix(
    matrix: &[u8],
    row: usize,
    k0: usize,
    k_dim: usize,
) -> Vec<f32> {
    let gscale = f32::from_le_bytes(
        matrix[..NVFP4_HEADER_BYTES]
            .try_into()
            .expect("nvfp4 header"),
    );
    let body = &matrix[NVFP4_HEADER_BYTES..];
    let row_stride = nvfp4_row_bytes(k_dim);
    let row_base = &body[row * row_stride..(row + 1) * row_stride];
    let data_len = (k_dim + 1) / 2;
    let mut via_tile = [0.0f32; 32];
    for g in 0..2 {
        let g_idx = k0 / 16 + g;
        let scale = {
            let bits = row_base[data_len + g_idx];
            crate::dgq::fp4::fp8_e4m3_to_f32(bits) * gscale
        };
        let packed = &row_base[g_idx * 8..g_idx * 8 + 8];
        for i in 0..16 {
            let byte = packed[i / 2];
            let q = if i & 1 != 0 {
                byte >> 4
            } else {
                byte & 0x0f
            };
            via_tile[g * 16 + i] = crate::dgq::fp4::e2m1_to_f32(q) * scale;
        }
    }
    let mut out = Vec::with_capacity(64);
    out.extend_from_slice(&via_tile);
    for m in 0..32 {
        out.push(nvfp4_weight_at(body, row, k0 + m, k_dim, gscale));
    }
    out
}

/// Mirror of `q8_at` with scale already loaded.
pub fn q8_at(row_base: &[u8], col: usize, scale: f32) -> f32 {
    row_base[2 + col] as i8 as f32 * scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dgq::block::{dequant_row_q4, quantize_row_q4};
    use crate::dgq::layout::GROUP_SIZE;

    #[test]
    fn q4_group_matches_row_slice() {
        let row: Vec<f32> = (0..GROUP_SIZE).map(|i| (i as f32 * 0.07).sin()).collect();
        let mut q4 = vec![0u8; 20];
        quantize_row_q4(&row, GROUP_SIZE, &mut q4);
        let via_group = dequant_q4_group(q4.as_slice().try_into().unwrap());
        let mut via_row = vec![0.0f32; GROUP_SIZE];
        dequant_row_q4(&q4, GROUP_SIZE, &mut via_row);
        for i in 0..GROUP_SIZE {
            assert!(
                (via_group[i] - via_row[i]).abs() < 1e-5,
                "i={i} group={} row={}",
                via_group[i],
                via_row[i]
            );
        }
    }
}
