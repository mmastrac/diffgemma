//! Normalized Walsh–Hadamard transform for KV-cache rotation (TurboQuant /
//! QuaRot-style outlier suppression before low-bit quantization).
//!
//! `H` is the symmetric orthogonal Hadamard matrix scaled by `1/sqrt(n)`, so
//! `H = Hᵀ = H⁻¹` and `‖Hx‖ = ‖x‖`, `(Ha)·(Hb) = a·b`. That invariance is
//! what lets the rotation be attention-preserving:
//! - V cache: fold `H` into `W_v` (store `H·v`) and `Hᵀ = H` into `W_o`
//!   (`W_o·H`) offline — attention output is unchanged, but the stored V has
//!   its per-channel outliers spread across all channels so group-wise q4/q8
//!   quantizes far more accurately.
//! - K cache: apply `H` to both the query and the stored key AFTER RoPE at
//!   runtime (RoPE sits between `W_k` and storage, so K can't fold offline);
//!   `(Hq)·(Hk) = q·k` keeps the score exact.
//!
//! `n` must be a power of two (head_dim 256 sliding / 512 full — both are),
//! giving the O(n log n) in-place butterfly.

/// In-place normalized fast Walsh–Hadamard transform. `v.len()` must be a
/// power of two. Self-inverse: applying it twice returns the input (up to
/// f32 rounding).
pub fn fwht_normalized(v: &mut [f32]) {
    let n = v.len();
    debug_assert!(n.is_power_of_two() && n > 0, "FWHT length must be a power of two");
    // Unnormalized butterfly.
    let mut h = 1usize;
    while h < n {
        let mut i = 0usize;
        while i < n {
            for j in i..i + h {
                let a = v[j];
                let b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
            i += 2 * h;
        }
        h *= 2;
    }
    // Normalize by 1/sqrt(n) so the transform is orthogonal.
    let inv = 1.0f32 / (n as f32).sqrt();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Left-multiply a row-major `rows × n` matrix by `H` on the `n` axis:
/// each length-`n` row is replaced by `H·row`. Used to fold `H` into `W_v`
/// offline (rotate every output row of the v_proj weight; here a "row" is one
/// head_dim-length group being rotated). `n` must be a power of two.
pub fn fwht_rows(data: &mut [f32], rows: usize, n: usize) {
    debug_assert_eq!(data.len(), rows * n, "fwht_rows: data must be rows*n");
    for r in 0..rows {
        fwht_normalized(&mut data[r * n..(r + 1) * n]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn self_inverse() {
        for &n in &[1usize, 2, 4, 8, 256, 512] {
            let orig: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.37).sin()).collect();
            let mut v = orig.clone();
            fwht_normalized(&mut v);
            fwht_normalized(&mut v);
            for (a, b) in orig.iter().zip(&v) {
                assert!((a - b).abs() < 1e-4, "n={n}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn norm_preserving() {
        let n = 512;
        let orig: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).cos() * 3.0).collect();
        let n0: f32 = orig.iter().map(|x| x * x).sum();
        let mut v = orig.clone();
        fwht_normalized(&mut v);
        let n1: f32 = v.iter().map(|x| x * x).sum();
        assert!((n0 - n1).abs() / n0 < 1e-5, "‖Hx‖²={n1} vs ‖x‖²={n0}");
    }

    #[test]
    fn dot_product_preserving() {
        // The load-bearing property: (Ha)·(Hb) = a·b.
        let n = 256;
        let a: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.11).sin()).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.07 + 1.0).cos()).collect();
        let d0 = dot(&a, &b);
        let mut ha = a.clone();
        let mut hb = b.clone();
        fwht_normalized(&mut ha);
        fwht_normalized(&mut hb);
        let d1 = dot(&ha, &hb);
        assert!((d0 - d1).abs() < 1e-3, "(Ha)·(Hb)={d1} vs a·b={d0}");
    }

    #[test]
    fn spreads_a_spike() {
        // An outlier channel (single spike) becomes uniform-magnitude after H
        // — the reason group quantization gets accurate.
        let n = 256;
        let mut v = vec![0.0f32; n];
        v[3] = 10.0;
        fwht_normalized(&mut v);
        let expect = 10.0 / (n as f32).sqrt();
        for x in &v {
            assert!((x.abs() - expect).abs() < 1e-4, "spike not spread: {x} vs ±{expect}");
        }
    }
}
