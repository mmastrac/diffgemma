//! Test support: backend availability and the f32 oracle comparison.

/// True when a GPU dispatch should actually run. A hosted macOS CI runner has
/// no usable Metal device; a CUDA host is expected to have one.
pub fn gpu_available() -> bool {
    let backend = cfg!(target_os = "macos") || cfg!(feature = "cuda");
    backend && !(cfg!(target_os = "macos") && std::env::var_os("CI").is_some())
}

pub use gpukit::testing::{assert_oracle, cosine_f32, max_abs_diff};

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
