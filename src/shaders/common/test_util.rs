//! Tier-1 oracle helpers and the variant-matrix test macro.

use crate::Error;

/// Element storage format for the variant matrix (f32 today; bf16/fp4 for GEMM kernels).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemFormat {
    F32,
}

/// Quantized-model directory for real-weight tests: the in-repo model first,
/// then the legacy /tmp path. Tests gated on this skip gracefully when absent.
pub fn dgq_model_dir() -> Option<std::path::PathBuf> {
    for p in ["model/diffusiongemma-q4emb", "/tmp/quantized-weights"] {
        let d = std::path::PathBuf::from(p);
        if d.join("model.dgq.json").exists() {
            return Some(d);
        }
    }
    None
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
    dot / (na.sqrt() * nb.sqrt())
}

pub fn assert_oracle(a: &[f32], b: &[f32], max_tol: f32, min_cos: f32) {
    let max = max_abs_diff(a, b);
    let cos = cosine_f32(a, b);
    assert!(
        max <= max_tol && cos >= min_cos,
        "oracle mismatch max_abs={max:.6} cos={cos:.6} (tol={max_tol}, min_cos={min_cos})"
    );
    assert!(
        a.iter().all(|v| v.is_finite()),
        "output has non-finite values"
    );
}

/// Expand fixture × format cells into CPU-only and GPU-vs-CPU oracle tests.
#[macro_export]
macro_rules! kernel_oracle_matrix {
    (
        mod $mod_name:ident,
        cpu = $cpu_fn:path,
        cpu_oracle = $oracle_fn:path,
        gpu = $gpu_fn:path,
        fixture = $fixture_fn:path,
        out_len = $out_len:expr,
        formats: [ $( $fmt:ident ),* $(,)? ],
        max_tol = $max_tol:expr,
        min_cos = $min_cos:expr $(,)?
    ) => {
        mod $mod_name {
            use $crate::shaders::test_util::{assert_oracle, ElemFormat};

            $(
                mod $fmt {
                    use $crate::shaders::test_util::{assert_oracle, ElemFormat};

                    // (cpu_smoke was deleted: strictly dominated by
                    // cpu_matches_oracle, which runs the same CPU fn and asserts
                    // finiteness inside assert_oracle.)
                    #[test]
                    fn cpu_matches_oracle() {
                        let fix = $fixture_fn(ElemFormat::$fmt);
                        let out = $cpu_fn(&fix);
                        let oracle = $oracle_fn(&fix);
                        assert_eq!(out.len(), $out_len(&fix));
                        assert_oracle(&out, &oracle, $max_tol, $min_cos);
                    }

                    #[cfg(target_os = "macos")]
                    #[test]
                    fn gpu_matches_cpu() {
                        let fix = $fixture_fn(ElemFormat::$fmt);
                        let cpu = $cpu_fn(&fix);
                        let gpu = $gpu_fn(&fix, $crate::shaders::KernelVariant::PRODUCTION)
                            .expect("gpu dispatch");
                        assert_oracle(&gpu, &cpu, $max_tol, $min_cos);
                    }

                    #[cfg(target_os = "macos")]
                    #[test]
                    fn gpu_assert_variant_matches_cpu() {
                        let fix = $fixture_fn(ElemFormat::$fmt);
                        let cpu = $cpu_fn(&fix);
                        let gpu = $gpu_fn(&fix, $crate::shaders::KernelVariant::TEST_ASSERT)
                            .expect("gpu dispatch");
                        assert_oracle(&gpu, &cpu, $max_tol, $min_cos);
                    }
                }
            )*
        }
    };
}

pub type Result<T> = std::result::Result<T, Error>;
