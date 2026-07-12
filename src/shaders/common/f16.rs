//! IEEE fp16 (`half`) bit conversion for monolith subkernel oracles.

pub fn f16_bits_to_f32(bits: u16) -> f32 {
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

pub fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.clamp(-65504.0, 65504.0).to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = (bits & 0x7fffff) as u32;
    if exp == 0xff {
        return sign | if mant == 0 { 0 } else { 0x7e00 };
    }
    if exp == 0 {
        return sign;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if new_exp <= 0 {
        return sign;
    }
    sign | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
}

pub fn f32_slice_to_f16(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&v| f32_to_f16_bits(v)).collect()
}

pub fn f16_slice_to_f32(data: &[u16]) -> Vec<f32> {
    data.iter().map(|&b| f16_bits_to_f32(b)).collect()
}

/// Round through fp16 like Metal `half(x)`.
pub fn round_half(v: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(v))
}

pub const SOFTCAP: f32 = 30.0;

pub fn softcap_f32(v: f32) -> f32 {
    let x = (v / SOFTCAP).clamp(-20.0, 20.0);
    round_half(x.tanh() * SOFTCAP)
}
