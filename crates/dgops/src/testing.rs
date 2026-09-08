//! Shared test helpers and the per-op oracle matrix.

use crate::backend;

/// True when a GPU dispatch should actually run. On macOS a hosted CI runner
/// has no usable Metal device; a CUDA host is expected to have one.
pub fn gpu_available() -> bool {
    backend::available().is_some()
        && !(cfg!(target_os = "macos") && std::env::var_os("CI").is_some())
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 && nb == 0.0 {
        return 1.0;
    }
    // f32 rounding in sqrt(na)*sqrt(na) can put an exact match a hair off 1.0.
    (dot / (na.sqrt() * nb.sqrt())).clamp(-1.0, 1.0)
}

pub fn assert_oracle(a: &[f32], b: &[f32], max_tol: f32, min_cos: f32) {
    let max = max_abs_diff(a, b);
    // Bit-identical is the strongest possible pass; skip the cosine round-trip.
    let cos = if max == 0.0 { 1.0 } else { cosine_f32(a, b) };
    assert!(
        max <= max_tol && cos >= min_cos,
        "oracle mismatch max_abs={max:.6} cos={cos:.6} (tol={max_tol}, min_cos={min_cos})"
    );
    assert!(
        a.iter().all(|v| v.is_finite()),
        "output has non-finite values"
    );
}

/// One fixture: assert the CPU reference is finite and that the GPU result
/// matches it. Mirrors the engine's kernel_oracle_matrix, minus the format
/// axis (these ops are f32-only).
#[macro_export]
macro_rules! op_oracle_matrix {
    (
        mod $mod_name:ident,
        cpu = $cpu_fn:path,
        gpu = $gpu_fn:path,
        fixture = $fixture_fn:path,
        out_len = $out_len:expr,
        max_tol = $max_tol:expr,
        min_cos = $min_cos:expr $(,)?
    ) => {
        mod $mod_name {
            #[test]
            fn cpu_is_finite() {
                let fix = $fixture_fn();
                let out = $cpu_fn(&fix);
                assert_eq!(out.len(), $out_len(&fix));
                assert!(
                    out.iter().all(|v| v.is_finite()),
                    "cpu reference produced non-finite values"
                );
            }

            #[test]
            fn gpu_matches_cpu() {
                if !$crate::testing::gpu_available() {
                    return;
                }
                let fix = $fixture_fn();
                let cpu = $cpu_fn(&fix);
                let gpu = $gpu_fn(&fix).expect("gpu dispatch");
                $crate::testing::assert_oracle(&gpu, &cpu, $max_tol, $min_cos);
            }
        }
    };
}
