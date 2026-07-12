//! bf16 blob helpers for monolith subkernel oracles.

use crate::shaders::cpu;

pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    cpu::bf16_to_f32(bits)
}

pub fn f32_to_bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

pub fn pack_bf16_slice(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        out.extend_from_slice(&f32_to_bf16_bits(v).to_le_bytes());
    }
    out
}

pub fn pack_bf16_scalar(v: f32) -> Vec<u8> {
    pack_bf16_slice(&[v])
}

/// bf16 arena store: round f32 to bf16 precision (matches `arena_store` in Metal).
pub fn store_bf16_round_half(v: f32) -> f32 {
    round_bf16_f32(v)
}

pub fn f32_slice_to_bf16_bits(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&v| f32_to_bf16_bits(v)).collect()
}

pub fn bf16_slice_to_f32(data: &[u16]) -> Vec<f32> {
    data.iter().map(|&b| bf16_bits_to_f32(b)).collect()
}

/// bf16-round f32 accumulator (f32 arena, e.g. MoE scatter / grouped GEMM).
pub fn round_bf16_f32(v: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(v))
}

#[cfg(test)]
mod tests {
    use super::{bf16_bits_to_f32, f32_to_bf16_bits, round_bf16_f32, store_bf16_round_half};

    #[test]
    fn store_bf16_round_half_matches_arena_store() {
        let v = 1.234567f32;
        assert_eq!(store_bf16_round_half(v), round_bf16_f32(v));
        let w = 0.001234567f32;
        assert_eq!(
            store_bf16_round_half(w),
            bf16_bits_to_f32(f32_to_bf16_bits(w))
        );
    }
}
