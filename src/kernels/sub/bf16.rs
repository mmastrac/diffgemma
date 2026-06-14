//! bf16 blob helpers for monolith subkernel oracles.

use crate::kernels::cpu;

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
