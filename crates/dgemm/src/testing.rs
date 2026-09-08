//! Test support: backend availability and the f32 oracle comparison.

/// True when a GPU dispatch should actually run. A hosted macOS CI runner has
/// no usable Metal device; a CUDA host is expected to have one.
pub fn gpu_available() -> bool {
    let backend = cfg!(target_os = "macos") || cfg!(feature = "cuda");
    backend && !(cfg!(target_os = "macos") && std::env::var_os("CI").is_some())
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

/// Assert the CPU oracle is finite and the GPU result matches it for one shape.
#[macro_export]
macro_rules! gemm_case {
    (
        mod $mod_name:ident,
        fixture = $fixture_fn:path,
        max_tol = $max_tol:expr,
        min_cos = $min_cos:expr $(,)?
    ) => {
        mod $mod_name {
            #[test]
            fn cpu_is_finite() {
                let call = $fixture_fn();
                let out = $crate::cpu(&call);
                assert_eq!(out.len(), $crate::out_len(&call));
                assert!(
                    out.iter().all(|v| v.is_finite()),
                    "cpu oracle produced non-finite values"
                );
            }

            #[test]
            fn gpu_matches_cpu() {
                if !$crate::testing::gpu_available() {
                    return;
                }
                let call = $fixture_fn();
                let cpu = $crate::cpu(&call);
                let gpu = $crate::gpu(&call).expect("gpu dispatch");
                $crate::testing::assert_oracle(&gpu, &cpu, $max_tol, $min_cos);
            }
        }
    };
}
