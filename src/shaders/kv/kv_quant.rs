//! CPU KV-cache quantizers (numerics reference + the pre-plumbing viability
//! proof for the TurboQuant rotated-KV memory lever, E8/M1).
//!
//! Group-32 symmetric, mirroring the shipped q8 path (`step_kv::kv_q8_*`):
//! - q8 row = `[hd × i8][hd/32 × f16 scale]`, `hd + hd/16` bytes.
//! - q4 row = `[hd/2 × packed i4][hd/32 × f16 scale]`, `hd/2 + hd/16` bytes
//!   (~0.28× f16 at hd=512). i4 clamped to [-7,7], scale = max_abs/7.
//!
//! The load-bearing question this module answers CHEAPLY, before any kernel
//! plumbing: does a per-head Walsh–Hadamard rotation make plain group-q4 KV
//! accurate enough to matter? Real KV has per-channel outlier spikes that
//! wreck a shared group scale; the rotation spreads each spike across the
//! head_dim (see `hadamard`), so q4 sees a near-uniform distribution.

use crate::shaders::f16::{f16_bits_to_f32, f32_to_f16_bits};
use crate::shaders::hadamard::fwht_normalized;

pub const KV_GROUP: usize = 32;

/// KV-cache storage format — the code threaded through the Rust API and set
/// as the `KV_FMT` uint function constant (index 4) the KV kernels specialize
/// on. Always a code, never a bool: the format is multi-state (f16 / q8 / q4)
/// and will grow. Group-32 for the quantized variants (mirrors the shipped
/// q8 lattice). `code()` MUST match the Metal `KV_FMT` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvFormat {
    /// f16 halfs — the direct-simdgroup-load default.
    F16,
    /// group-32 symmetric i8 + f16 scales (`[hd i8][hd/32 f16]`).
    Q8,
    /// group-32 q4 (`[hd/2 i4][hd/32 f16]`) — reserved for the rotated-KV
    /// memory lever (E8); not yet implemented in the kernels.
    Q4,
}

impl KvFormat {
    pub fn code(self) -> u32 {
        match self {
            KvFormat::F16 => 0,
            KvFormat::Q8 => 1,
            KvFormat::Q4 => 2,
        }
    }

    pub fn from_code(c: u32) -> Self {
        match c {
            1 => KvFormat::Q8,
            2 => KvFormat::Q4,
            _ => KvFormat::F16,
        }
    }

    /// Pipeline-cache label suffix.
    pub fn label(self) -> &'static str {
        match self {
            KvFormat::F16 => "kvf16",
            KvFormat::Q8 => "kvq8",
            KvFormat::Q4 => "kvq4",
        }
    }

    /// Bytes for one (slot, head, K-or-V) row of `hd` elements. MUST match the
    /// Metal `kv_row_bytes(hd, fmt)`.
    pub fn row_bytes(self, hd: u32) -> u64 {
        let hd = hd as u64;
        match self {
            KvFormat::F16 => hd * 2,
            KvFormat::Q8 => hd + (hd / KV_GROUP as u64) * 2,
            KvFormat::Q4 => hd / 2 + (hd / KV_GROUP as u64) * 2,
        }
    }
}

/// q8 group-32 round-trip of one head vector (matches `step_kv::kv_q8_pack_row`
/// + `kv_q8_read` numerics; reproduced here so this module is self-contained
///   and testable without the Metal path).
pub fn q8_roundtrip(src: &[f32], out: &mut [f32]) {
    let hd = src.len();
    for g in 0..hd / KV_GROUP {
        let grp = &src[g * KV_GROUP..(g + 1) * KV_GROUP];
        let mx = grp.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let sf = f16_bits_to_f32(f32_to_f16_bits((mx / 127.0).max(1e-8)));
        for j in 0..KV_GROUP {
            let q = (grp[j] / sf).round_ties_even().clamp(-127.0, 127.0);
            out[g * KV_GROUP + j] = q * sf;
        }
    }
}

/// Symmetric group-32 q4 round-trip of one head vector. i4 ∈ [-7,7],
/// scale = max_abs/7, f16-rounded scale (so the GPU mirror can be bit-exact
/// later). This is the numerics the M2 kernel will implement.
pub fn q4_roundtrip(src: &[f32], out: &mut [f32]) {
    let hd = src.len();
    for g in 0..hd / KV_GROUP {
        let grp = &src[g * KV_GROUP..(g + 1) * KV_GROUP];
        let mx = grp.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let sf = f16_bits_to_f32(f32_to_f16_bits((mx / 7.0).max(1e-8)));
        for j in 0..KV_GROUP {
            let q = (grp[j] / sf).round_ties_even().clamp(-7.0, 7.0);
            out[g * KV_GROUP + j] = q * sf;
        }
    }
}

/// Affine (asymmetric) group-32 q4 round-trip: 16 levels over the group's
/// actual [lo,hi] (scale=(hi-lo)/15, zero=lo), f16-rounded scale+zero. Mirrors
/// the shipped expert-q4 format; more accurate than symmetric when the group
/// isn't zero-centered (e.g. post-rotation V). Row = `[hd/2 i4][hd/32 ×
/// (f16 scale, f16 zero)]` = `hd/2 + hd/8` bytes.
pub fn q4_affine_roundtrip(src: &[f32], out: &mut [f32]) {
    let hd = src.len();
    for g in 0..hd / KV_GROUP {
        let grp = &src[g * KV_GROUP..(g + 1) * KV_GROUP];
        let lo = grp.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = grp.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sf = f16_bits_to_f32(f32_to_f16_bits(((hi - lo) / 15.0).max(1e-8)));
        let zf = f16_bits_to_f32(f32_to_f16_bits(lo));
        for j in 0..KV_GROUP {
            let q = ((grp[j] - zf) / sf).round_ties_even().clamp(0.0, 15.0);
            out[g * KV_GROUP + j] = zf + q * sf;
        }
    }
}

/// Generic affine (asymmetric) group-32 round-trip at `levels` steps
/// (q4 = 15, q6 = 63, q8 = 255) — for the weight-quant viability probe.
pub fn affine_roundtrip_levels(src: &[f32], out: &mut [f32], levels: f32) {
    let hd = src.len();
    for g in 0..hd / KV_GROUP {
        let grp = &src[g * KV_GROUP..(g + 1) * KV_GROUP];
        let lo = grp.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = grp.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sf = f16_bits_to_f32(f32_to_f16_bits(((hi - lo) / levels).max(1e-8)));
        let zf = f16_bits_to_f32(f32_to_f16_bits(lo));
        for j in 0..KV_GROUP {
            let q = ((grp[j] - zf) / sf).round_ties_even().clamp(0.0, levels);
            out[g * KV_GROUP + j] = zf + q * sf;
        }
    }
}

/// Single-scale symmetric q8 over the WHOLE row (one f16 scale, no groups) —
/// models dropping group scales entirely once rotation Gaussianizes the row
/// (post-rotation coordinates share one distribution, so per-group adaptivity
/// buys nothing; scale storage drops from hd/32 f16s to one).
pub fn q8_row_roundtrip(src: &[f32], out: &mut [f32]) {
    let mx = src.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let sf = f16_bits_to_f32(f32_to_f16_bits((mx / 127.0).max(1e-8)));
    for (o, &s) in out.iter_mut().zip(src) {
        *o = (s / sf).round_ties_even().clamp(-127.0, 127.0) * sf;
    }
}

/// Single-scale affine q4 over the whole row (one f16 scale + zero).
pub fn q4_affine_row_roundtrip(src: &[f32], out: &mut [f32]) {
    let lo = src.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = src.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sf = f16_bits_to_f32(f32_to_f16_bits(((hi - lo) / 15.0).max(1e-8)));
    let zf = f16_bits_to_f32(f32_to_f16_bits(lo));
    for (o, &s) in out.iter_mut().zip(src) {
        *o = zf + ((s - zf) / sf).round_ties_even().clamp(0.0, 15.0) * sf;
    }
}

/// Rotated row-scale q8: H, one-scale q8, H (self-inverse).
pub fn q8_row_rotated_roundtrip(src: &[f32], out: &mut [f32]) {
    rotated(src, out, q8_row_roundtrip);
}

/// Rotated row-scale affine q4.
pub fn q4_affine_row_rotated_roundtrip(src: &[f32], out: &mut [f32]) {
    rotated(src, out, q4_affine_row_roundtrip);
}

fn rotated<F: Fn(&[f32], &mut [f32])>(src: &[f32], out: &mut [f32], q: F) {
    let mut r = src.to_vec();
    fwht_normalized(&mut r);
    let mut rq = vec![0.0f32; r.len()];
    q(&r, &mut rq);
    fwht_normalized(&mut rq); // H⁻¹ = H
    out.copy_from_slice(&rq);
}

/// Rotated affine q4 round-trip.
pub fn q4_affine_rotated_roundtrip(src: &[f32], out: &mut [f32]) {
    rotated(src, out, q4_affine_roundtrip);
}

/// Rotated q4 round-trip: apply H, quantize to q4, dequantize, apply H again
/// (H is self-inverse). This is exactly what the M2 runtime path does — store
/// `H·kv` in q4, and the attention math undoes the rotation (K via the query's
/// matching H, V via the offline `W_o·Hᵀ` fold). `src.len()` must be pow2.
pub fn q4_rotated_roundtrip(src: &[f32], out: &mut [f32]) {
    let mut r = src.to_vec();
    fwht_normalized(&mut r);
    let mut rq = vec![0.0f32; r.len()];
    q4_roundtrip(&r, &mut rq);
    fwht_normalized(&mut rq); // H⁻¹ = H
    out.copy_from_slice(&rq);
}

/// Relative RMS error `‖a−b‖ / ‖a‖` (0 = exact).
pub fn rel_rms(reference: &[f32], approx: &[f32]) -> f32 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&a, &b) in reference.iter().zip(approx) {
        num += ((a - b) as f64).powi(2);
        den += (a as f64).powi(2);
    }
    if den == 0.0 {
        0.0
    } else {
        (num / den).sqrt() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A head vector with realistic KV structure: near-Gaussian bulk + a few
    /// per-channel outlier spikes (the thing that wrecks a shared group scale).
    fn kv_like(hd: usize, seed: u32) -> Vec<f32> {
        let mut v: Vec<f32> = (0..hd)
            .map(|i| {
                let t = (i as f32) * 0.017 + seed as f32 * 0.3;
                (t.sin() * 1.7 + (t * 2.3).cos() * 0.9) * 0.6
            })
            .collect();
        // Outlier channels: a handful of large spikes, ~15× the bulk.
        for &c in &[5usize, 37, 88, 190] {
            if c < hd {
                v[c] += if c % 2 == 0 { 14.0 } else { -12.0 };
            }
        }
        v
    }

    #[test]
    fn q4_roundtrip_bounded() {
        // Sanity: on smooth data (no outliers) plain q4 is already decent.
        let hd = 256;
        let src: Vec<f32> = (0..hd).map(|i| ((i as f32) * 0.02).sin()).collect();
        let mut out = vec![0.0f32; hd];
        q4_roundtrip(&src, &mut out);
        let e = rel_rms(&src, &out);
        assert!(e < 0.05, "smooth q4 rel-RMS {e} too high");
    }

    #[test]
    fn rotation_rescues_q4_on_outliers() {
        // THE THESIS (measured, synthetic outlier KV): the rotation spreads
        // per-channel spikes so a shared group scale stops being dominated by
        // one outlier. q4 is still 4 bits (won't match q8's 8), so the win is
        // CONDITIONING — rotation should cut the q4 error by a large factor.
        // We also print affine vs symmetric to steer the format choice; the
        // load-bearing real numbers come from the KV-dump probe, not here.
        for &hd in &[256usize, 512] {
            let (mut plain, mut sym_rot, mut aff, mut aff_rot, mut q8e) =
                (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
            let n = 8;
            for s in 0..n {
                let src = kv_like(hd, s);
                let mut o = vec![0.0f32; hd];
                q4_roundtrip(&src, &mut o);
                plain += rel_rms(&src, &o);
                q4_rotated_roundtrip(&src, &mut o);
                sym_rot += rel_rms(&src, &o);
                q4_affine_roundtrip(&src, &mut o);
                aff += rel_rms(&src, &o);
                q4_affine_rotated_roundtrip(&src, &mut o);
                aff_rot += rel_rms(&src, &o);
                q8_roundtrip(&src, &mut o);
                q8e += rel_rms(&src, &o);
            }
            for e in [&mut plain, &mut sym_rot, &mut aff, &mut aff_rot, &mut q8e] {
                *e /= n as f32;
            }
            println!(
                "hd={hd}: q4-plain {plain:.4}  q4-sym-rot {sym_rot:.4}  \
                 q4-affine {aff:.4}  q4-affine-rot {aff_rot:.4}  q8 {q8e:.4}"
            );
            // Robust, measured claims (regression guards; the informative
            // numbers are the printed line, and the load-bearing decision
            // comes from real KV + the mem-watch @262k check):
            // (1) rotation improves SYMMETRIC q4 (outliers stop dominating the
            //     shared symmetric scale).
            assert!(sym_rot < plain, "hd={hd}: sym rotation didn't help at all");
            // (2) affine q4 ALONE already beats symmetric plain q4 — its
            //     per-group [min,max] absorbs the outlier into an endpoint,
            //     which is why rotation barely moves affine (and can even hurt
            //     it: spreading outliers un-concentrates the bounded range).
            assert!(aff < plain, "hd={hd}: affine no better than symmetric");
            // (3) q4 (any variant) stays well above q8 — 4 bits vs 8; rotation
            //     fixes conditioning, not resolution.
            assert!(q8e < aff_rot && q8e < sym_rot, "hd={hd}: q8 not best");
        }
    }
}
