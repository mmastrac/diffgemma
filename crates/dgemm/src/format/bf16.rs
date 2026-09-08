//! bf16 bit-level conversion, the shared decode primitive.

/// Decode bf16 bits to f32 (matches `src/shaders/cpu/mod.rs::bf16_to_f32`).
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

pub fn f32_to_bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

pub fn bf16_bytes_to_f32(src: &[u8], out: &mut [f32]) {
    let n = src.len() / 2;
    assert_eq!(out.len(), n);
    for i in 0..n {
        let bits = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
        out[i] = bf16_to_f32(bits);
    }
}
