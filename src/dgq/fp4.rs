//! FP4 E2M1 and FP8 E4M3 helpers (MLX-compatible nvfp4 path).

pub const FP4_E2M1_MAX: f32 = 6.0;

/// Decode one E2M1 nibble (sign in bit 3).
pub fn e2m1_to_f32(bits: u8) -> f32 {
    let mag = match bits & 0x7 {
        0 => 0.0,
        1 => 0.5,
        2 => 1.0,
        3 => 1.5,
        4 => 2.0,
        5 => 3.0,
        6 => 4.0,
        _ => 6.0,
    };
    if bits & 0x8 != 0 {
        -mag
    } else {
        mag
    }
}

/// Quantize `x` to E2M1 (matches `mlx/backend/metal/kernels/fp4.h`).
pub fn e2m1_from_f32(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7;
    }
    let sign_bit = if x.is_sign_negative() { 0x8u8 } else { 0 };
    let x = x.abs();
    let mag = if x > 5.0 {
        0x7
    } else if x >= 3.5 {
        0x6
    } else if x > 2.5 {
        0x5
    } else if x >= 1.75 {
        0x4
    } else if x > 1.25 {
        0x3
    } else if x >= 0.75 {
        0x2
    } else if x > 0.25 {
        0x1
    } else {
        0x0
    };
    sign_bit | mag
}

/// FP8 E4M3 (PyTorch e4m3fn) → f32 (matches MLX `fp8_e4m3` operator float).
pub fn fp8_e4m3_to_f32(bits: u8) -> f32 {
    let v = ((bits & 127) as u16) << 7;
    let half = f16_bits_to_f32(v);
    let converted = half * 256.0;
    if bits & 128 != 0 {
        -converted
    } else {
        converted
    }
}

/// f32 → FP8 E4M3 (PyTorch e4m3fn, matches MLX `fp8_e4m3` constructor).
pub fn fp8_e4m3_from_f32(f: f32) -> u8 {
    const FP8_MAX: u32 = 543 << 21;
    const DENORM_MASK: u32 = 141 << 23;
    let mut f_bits = f.to_bits();
    let sign = f_bits & 0x8000_0000;
    f_bits ^= sign;
    let bits = if f_bits >= FP8_MAX {
        0x7Eu8
    } else if f_bits < 121 << 23 {
        let adjusted = f32::from_bits(f_bits) + f32::from_bits(DENORM_MASK);
        let adjusted_bits = adjusted.to_bits();
        (adjusted_bits - DENORM_MASK) as u8
    } else {
        let mant_odd = ((f_bits >> 20) & 1) as u32;
        f_bits = f_bits.wrapping_add((7u32.wrapping_sub(127) << 23) + 0x7_FFFF);
        f_bits = f_bits.wrapping_add(mant_odd);
        (f_bits >> 20) as u8
    };
    bits | ((sign >> 24) as u8)
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    if exp == 0 {
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let val = (mant as f32) * 2f32.powi(-24);
        return if sign == 1 { -val } else { val };
    }
    if exp == 0x1f {
        return if mant == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        };
    }
    f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_roundtrip_magnitudes() {
        for (bits, val) in [
            (0x0, 0.0),
            (0x1, 0.5),
            (0x2, 1.0),
            (0x3, 1.5),
            (0x4, 2.0),
            (0x5, 3.0),
            (0x6, 4.0),
            (0x7, 6.0),
        ] {
            assert!((e2m1_to_f32(bits) - val).abs() < 1e-6, "bits={bits}");
            assert_eq!(e2m1_from_f32(val), bits);
            assert_eq!(e2m1_from_f32(-val), bits | 0x8);
        }
    }

    #[test]
    fn fp8_scale_half() {
        let bits = fp8_e4m3_from_f32(0.5);
        assert_eq!(bits, 48);
        assert!((fp8_e4m3_to_f32(bits) - 0.5).abs() < 1e-3);
    }
}
