//! bf16 blob helpers for monolith subkernel oracles.
//!
//! Two roundings live here and they are not interchangeable. Buffer packing
//! truncates, because that is what the dgq quantizers do and what
//! `f32_slice_to_bf16_bits` writes into every fixture upload; an oracle that
//! models *loading* a bf16 buffer has to truncate with it. The activation
//! arena rounds to nearest even, so an oracle that models a *store* has to
//! round. Mixing the two costs exactly one bf16 ulp, which is enough to trip
//! the tight elementwise tolerances.

use crate::shaders::cpu;

pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    cpu::bf16_to_f32(bits)
}

/// Truncating f32 -> bf16 bits: the pack-production rounding.
pub fn f32_to_bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

/// Round-to-nearest-even f32 -> bf16 bits, mirroring `f32_round_bf16` in
/// `include/common.metal`. NaN/Inf pass through untouched so the rounding bump
/// cannot carry into the exponent.
pub fn f32_to_bf16_bits_rne(v: f32) -> u16 {
    let mut u = v.to_bits();
    if u & 0x7F80_0000 != 0x7F80_0000 {
        u += 0x7FFF + ((u >> 16) & 1);
    }
    (u >> 16) as u16
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

pub fn f32_slice_to_bf16_bits(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&v| f32_to_bf16_bits(v)).collect()
}

pub fn bf16_slice_to_f32(data: &[u16]) -> Vec<f32> {
    data.iter().map(|&b| bf16_bits_to_f32(b)).collect()
}

/// Value-round an f32 through a bf16 activation-arena store — the CPU twin of
/// `arena_store` / `arena_round_f32` in `include/arena.metal`.
pub fn arena_round_f32(v: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits_rne(v))
}

/// Value-truncate an f32 onto the bf16 grid — models *reading* a buffer packed
/// by [`f32_slice_to_bf16_bits`], not an arena store.
pub fn load_bf16_f32(v: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(v))
}

#[cfg(test)]
mod tests {
    use super::{arena_round_f32, f32_to_bf16_bits, f32_to_bf16_bits_rne, load_bf16_f32};

    #[test]
    fn rne_and_truncation_straddle_the_value() {
        // Mantissa 0x00C000 is three quarters of the way up its bf16 interval,
        // so truncation sheds the whole fraction while RNE takes the near side.
        let v = f32::from_bits(0x3F80_C000);
        let lo = load_bf16_f32(v);
        let hi = arena_round_f32(v);
        assert_ne!(lo, hi);
        assert!((hi - v).abs() < (lo - v).abs());
    }

    #[test]
    fn rne_ties_go_to_even() {
        // Exact midpoint between two bf16 values: mantissa 0x008000.
        let tie_to_even_down = f32::from_bits(0x3F80_8000);
        assert_eq!(f32_to_bf16_bits_rne(tie_to_even_down), 0x3F80);
        let tie_to_even_up = f32::from_bits(0x3F81_8000);
        assert_eq!(f32_to_bf16_bits_rne(tie_to_even_up), 0x3F82);
    }

    #[test]
    fn rne_leaves_nan_and_inf_alone() {
        for v in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            assert_eq!(f32_to_bf16_bits_rne(v), f32_to_bf16_bits(v));
        }
    }

    #[test]
    fn exact_bf16_values_are_unchanged_by_either_rounding() {
        for v in [0.0f32, -0.0, 1.0, -2.0, 0.5, 100.0] {
            assert_eq!(arena_round_f32(v), v);
            assert_eq!(load_bf16_f32(v), v);
        }
    }
}
