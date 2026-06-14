//! CPU twins for `shaders/include/dequant.metal` — single decode reference.

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
