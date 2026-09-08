//! Kernel registration: one block declares what a kernel is (its name, its
//! backend bodies, its function-constant pipeline, its manifest row) and the
//! expansion emits the consts, the backend dispatch wrapper, and the tier-1
//! test module.
//!
//! Two arms, one per kernel home:
//!
//! - `op` — a backend-agnostic op whose Metal and CUDA bodies sit beside a CPU
//!   oracle (`crates/dgops`, `crates/dgemm`): emits `ENTRY`, `METAL`, the
//!   `CUBIN` cfg pair, `pub mod metal/cuda`, the backend-selecting `gpu()`
//!   wrapper, and the test module.
//! - `shader` — one Metal entry point with an FC-specialized pipeline and an
//!   optional manifest `SPEC` (the engine's `src/shaders`): emits `ENTRY`,
//!   `SHADER`, an optional `pipeline_for`, an optional `NAME`+`SPEC`, and the
//!   test module.
//!
//! Everything crate-specific (the error type, the Metal context type, the
//! manifest type, the CUDA-kernels cfg name) is passed in by the caller's shim,
//! so this crate stays backend- and crate-agnostic. The macro never generates
//! what actually differs between kernels — fixtures, CPU oracles, buffer
//! bindings, or dispatch bodies.
//!
//! Constraints worth knowing before editing:
//!
//! - `include_str!($metal)` resolves relative to the *invoking* file, so
//!   `metal = "gelu.metal"` means "beside this `mod.rs`".
//! - `include_bytes!(concat!(env!("OUT_DIR"), …))` uses the *consuming* crate's
//!   `OUT_DIR`, so `cuda = "ops/gelu/gelu"` maps to
//!   `$OUT_DIR/cuda/ops/gelu/gelu.cubin` (the build script's layout, unchanged).
//! - `pub mod metal;` emitted here resolves to the caller's directory.
//! - The `pipeline` body must be a caller-written closure: names introduced by
//!   this macro's transcriber are hygiene-invisible to caller tokens, so a
//!   generated `fn pipeline_for(ctx, variant)` cannot expose `ctx`/`variant` to
//!   a body written at the call site. The closure is coerced to a fn pointer so
//!   its parameters get their types from the generated signature.

/// Register one kernel. See the module docs for the two arms.
#[macro_export]
macro_rules! kernel {
    // ---- op: Metal + CUDA bodies beside a CPU oracle ------------------------
    (
        op,
        cuda_cfg = $cuda_cfg:ident,
        error = $err:path,
        assert_oracle = $assert_oracle:path,
        gpu_available = $gpu_available:path,
        name = $name:literal,
        metal = $metal:literal,
        cuda = $cuda:literal,
        fixture = $fixture:ty,
        tests = $tests:tt $(,)?
    ) => {
        pub const ENTRY: &str = $name;
        pub const METAL: &str = include_str!($metal);

        // The cubin is only embedded when nvcc produced one; otherwise the
        // backend reports that CUDA kernels were not built.
        #[cfg(all(feature = "cuda", $cuda_cfg))]
        const CUBIN: &[u8] =
            include_bytes!(concat!(env!("OUT_DIR"), "/cuda/", $cuda, ".cubin"));
        #[cfg(all(feature = "cuda", not($cuda_cfg)))]
        const CUBIN: &[u8] = &[];

        #[cfg(feature = "cuda")]
        pub mod cuda;
        #[cfg(target_os = "macos")]
        pub mod metal;

        /// Run this op on the backend this build has: Metal on macOS, CUDA
        /// elsewhere when the cuda feature is on.
        pub fn gpu(fix: &$fixture) -> Result<Vec<f32>, $err> {
            #[cfg(target_os = "macos")]
            {
                self::metal::gpu(fix)
            }
            #[cfg(all(feature = "cuda", not(target_os = "macos")))]
            {
                self::cuda::gpu(fix)
            }
            #[cfg(not(any(target_os = "macos", all(feature = "cuda", not(target_os = "macos")))))]
            {
                let _ = fix;
                Err(<$err>::Gpu(
                    "no GPU backend enabled (build with --features cuda on a CUDA host)",
                ))
            }
        }

        $crate::__op_tests!($assert_oracle, $gpu_available, $tests);
    };

    // ---- shader: one Metal entry, optional pipeline + manifest SPEC ---------
    (
        shader,
        error = $err:path,
        ctx = $ctx:ty,
        pipeline_ty = $pipeline_ty:ty,
        variant = $variant:ty,
        spec_macro = $spec_macro:path,
        assert_oracle = $assert_oracle:path,
        skip_gpu = $skip_gpu:path,
        elem_f32 = $elem_f32:path,
        name = $name:literal,
        entry = $entry:literal,
        metal = $metal:literal,
        $(pipeline = $pipeline:expr,)?
        $(spec = { $($spec:tt)* },)?
        tests = $tests:tt $(,)?
    ) => {
        pub const ENTRY: &str = $entry;
        pub const SHADER: &str = include_str!($metal);

        $(
            #[cfg(target_os = "macos")]
            pub fn pipeline_for(ctx: &$ctx, variant: $variant) -> Result<$pipeline_ty, $err> {
                let pipeline: fn(&$ctx, $variant) -> Result<$pipeline_ty, $err> = $pipeline;
                pipeline(ctx, variant)
            }
        )?

        $(
            /// Manifest name; equals `ENTRY` unless the kernel's entry point
            /// differs from its manifest name (see `attention_gemm`).
            pub const NAME: &str = $name;
            // The caller spec macro declares the const AND scatters it into
            // the gathered manifest, so a spec cannot exist unregistered.
            $spec_macro! {
                pub const SPEC {
                    name: NAME,
                    entry: ENTRY,
                    source: SHADER,
                    $($spec)*
                }
            }
        )?

        $crate::__shader_tests!($assert_oracle, $skip_gpu, $elem_f32, $variant, $tests);
    };
}
// ---------------------------------------------------------------------------
// Test expansion
//
// A kernel block declares its tests either as a verbatim block
// (`tests = { … }`) or as a fixture table (`tests = [ … ]`). A table entry is
//
//   case => fixture => (max_tol, min_cos)                  // default entry points
//   case => fixture => gpu_fn => (max_tol, min_cos)        // different gpu entry
//   case => fixture => no_variant => (max_tol, min_cos)    // gpu() takes no variant
//   case => { fixture = …, gpu = …, cpu = …, oracle = …, tol = …, cos = … }
//
// Generated tests live in `mod tests::<case>` inside the kernel module and reach
// the kernel's items through `use super::super::*;`, so an entry names a fixture
// once instead of repeating a fully qualified path per assertion. The expected
// output length is the fixture's own `out_len()` (op fixtures use `len()`), so a
// case never names a length function. The oracle defaults to the CPU reference:
// a case that declares a real oracle gets `cpu_matches_oracle`, the rest get
// `cpu_is_finite`.
// ---------------------------------------------------------------------------

/// Expand the `tests =` field of a shader kernel.
#[macro_export]
macro_rules! __shader_tests {
    ($assert_oracle:path, $skip_gpu:path, $elem_f32:path, $variant:ty, { $($tests:tt)* }) => {
        #[cfg(test)]
        mod tests {
            $($tests)*
        }
    };
    ($assert_oracle:path, $skip_gpu:path, $elem_f32:path, $variant:ty, [ $($table:tt)* ]) => {
        #[cfg(test)]
        mod tests {
            $crate::__shader_cases!($assert_oracle, $skip_gpu, $elem_f32, $variant, $($table)*);
        }
    };
}

/// One shader fixture-table case.
#[macro_export]
macro_rules! __shader_case {
    ($assert_oracle:path, $skip_gpu:path, $elem_f32:path, $variant:ty, $case:ident,
     $fixture:path, $gpu:path, $cpu:path, $oracle:path, $cpu_test:ident, $tol:expr,
     $cos:expr, variant) => {
        mod $case {
            use super::super::*;

            #[test]
            fn $cpu_test() {
                let fix = $fixture($elem_f32);
                let out = $cpu(&fix);
                let oracle = $oracle(&fix);
                assert_eq!(out.len(), fix.out_len());
                $assert_oracle(&out, &oracle, $tol, $cos);
            }

            #[cfg(target_os = "macos")]
            #[test]
            fn gpu_matches_cpu() {
                if $skip_gpu() {
                    return;
                }
                let fix = $fixture($elem_f32);
                let cpu = $cpu(&fix);
                let gpu = $gpu(&fix, <$variant>::PRODUCTION).expect("gpu dispatch");
                $assert_oracle(&gpu, &cpu, $tol, $cos);
            }

            #[cfg(target_os = "macos")]
            #[test]
            fn gpu_assert_variant_matches_cpu() {
                if $skip_gpu() {
                    return;
                }
                let fix = $fixture($elem_f32);
                let cpu = $cpu(&fix);
                let gpu = $gpu(&fix, <$variant>::TEST_ASSERT).expect("gpu dispatch");
                $assert_oracle(&gpu, &cpu, $tol, $cos);
            }
        }
    };
    ($assert_oracle:path, $skip_gpu:path, $elem_f32:path, $variant:ty, $case:ident,
     $fixture:path, $gpu:path, $cpu:path, $oracle:path, $cpu_test:ident, $tol:expr,
     $cos:expr, no_variant) => {
        mod $case {
            use super::super::*;

            #[test]
            fn $cpu_test() {
                let fix = $fixture($elem_f32);
                let out = $cpu(&fix);
                let oracle = $oracle(&fix);
                assert_eq!(out.len(), fix.out_len());
                $assert_oracle(&out, &oracle, $tol, $cos);
            }

            #[cfg(target_os = "macos")]
            #[test]
            fn gpu_matches_cpu() {
                if $skip_gpu() {
                    return;
                }
                let fix = $fixture($elem_f32);
                let cpu = $cpu(&fix);
                let gpu = $gpu(&fix).expect("gpu dispatch");
                $assert_oracle(&gpu, &cpu, $tol, $cos);
            }
        }
    };
}

/// Expand a shader fixture table into one `__shader_case!` per entry.
#[macro_export]
macro_rules! __shader_cases {
    ($assert_oracle:path, $skip_gpu:path, $elem_f32:path, $variant:ty $(,)?) => {};

    ($assert_oracle:path, $skip_gpu:path, $elem_f32:path, $variant:ty,
        $case:ident => { fixture = $fixture:path, gpu = $gpu:path, cpu = $cpu:path,
        oracle = $oracle:path, tol = $tol:expr, cos = $cos:expr } $(, $($rest:tt)*)? ) => {
        $crate::__shader_case!($assert_oracle, $skip_gpu, $elem_f32, $variant, $case, $fixture,
            $gpu, $cpu, $oracle, cpu_matches_oracle, $tol, $cos, variant);
        $crate::__shader_cases!($assert_oracle, $skip_gpu, $elem_f32, $variant $(, $($rest)*)?);
    };

    ($assert_oracle:path, $skip_gpu:path, $elem_f32:path, $variant:ty,
        $case:ident => { fixture = $fixture:path, gpu = $gpu:path, cpu = $cpu:path,
        tol = $tol:expr, cos = $cos:expr } $(, $($rest:tt)*)? ) => {
        $crate::__shader_case!($assert_oracle, $skip_gpu, $elem_f32, $variant, $case, $fixture,
            $gpu, $cpu, $cpu, cpu_is_finite, $tol, $cos, variant);
        $crate::__shader_cases!($assert_oracle, $skip_gpu, $elem_f32, $variant $(, $($rest)*)?);
    };

    ($assert_oracle:path, $skip_gpu:path, $elem_f32:path, $variant:ty,
        $case:ident => { fixture = $fixture:path, oracle = $oracle:path, tol = $tol:expr,
        cos = $cos:expr } $(, $($rest:tt)*)? ) => {
        $crate::__shader_case!($assert_oracle, $skip_gpu, $elem_f32, $variant, $case, $fixture,
            gpu, cpu, $oracle, cpu_matches_oracle, $tol, $cos, variant);
        $crate::__shader_cases!($assert_oracle, $skip_gpu, $elem_f32, $variant $(, $($rest)*)?);
    };

    ($assert_oracle:path, $skip_gpu:path, $elem_f32:path, $variant:ty,
        $case:ident => $fixture:ident => no_variant => ($tol:expr, $cos:expr)
        $(, $($rest:tt)*)? ) => {
        $crate::__shader_case!($assert_oracle, $skip_gpu, $elem_f32, $variant, $case, $fixture,
            gpu, cpu, cpu, cpu_is_finite, $tol, $cos, no_variant);
        $crate::__shader_cases!($assert_oracle, $skip_gpu, $elem_f32, $variant $(, $($rest)*)?);
    };

    ($assert_oracle:path, $skip_gpu:path, $elem_f32:path, $variant:ty,
        $case:ident => $fixture:ident => $gpu:ident => ($tol:expr, $cos:expr)
        $(, $($rest:tt)*)? ) => {
        $crate::__shader_case!($assert_oracle, $skip_gpu, $elem_f32, $variant, $case, $fixture,
            $gpu, cpu, cpu, cpu_is_finite, $tol, $cos, variant);
        $crate::__shader_cases!($assert_oracle, $skip_gpu, $elem_f32, $variant $(, $($rest)*)?);
    };

    ($assert_oracle:path, $skip_gpu:path, $elem_f32:path, $variant:ty,
        $case:ident => $fixture:ident => ($tol:expr, $cos:expr) $(, $($rest:tt)*)? ) => {
        $crate::__shader_case!($assert_oracle, $skip_gpu, $elem_f32, $variant, $case, $fixture,
            gpu, cpu, cpu, cpu_is_finite, $tol, $cos, variant);
        $crate::__shader_cases!($assert_oracle, $skip_gpu, $elem_f32, $variant $(, $($rest)*)?);
    };
}

/// Expand the `tests =` field of a portable op.
#[macro_export]
macro_rules! __op_tests {
    ($assert_oracle:path, $gpu_available:path, { $($tests:tt)* }) => {
        #[cfg(test)]
        mod tests {
            $($tests)*
        }
    };
    ($assert_oracle:path, $gpu_available:path, [ $($table:tt)* ]) => {
        #[cfg(test)]
        mod tests {
            $crate::__op_cases!($assert_oracle, $gpu_available, $($table)*);
        }
    };
}

/// One portable-op fixture-table case. The expected length is the fixture's own
/// `len()` unless the case names an explicit `out_len` function.
#[macro_export]
macro_rules! __op_case {
    ($assert_oracle:path, $gpu_available:path, $case:ident, $fixture:path, method, $gpu:path,
     $tol:expr, $cos:expr) => {
        mod $case {
            use super::super::*;

            #[test]
            fn cpu_is_finite() {
                let fix = $fixture();
                let out = cpu(&fix);
                assert_eq!(out.len(), fix.len());
                assert!(
                    out.iter().all(|v| v.is_finite()),
                    "cpu reference produced non-finite values"
                );
            }

            #[test]
            fn gpu_matches_cpu() {
                if !$gpu_available() {
                    return;
                }
                let fix = $fixture();
                let cpu = cpu(&fix);
                let gpu = $gpu(&fix).expect("gpu dispatch");
                $assert_oracle(&gpu, &cpu, $tol, $cos);
            }
        }
    };
    ($assert_oracle:path, $gpu_available:path, $case:ident, $fixture:path, $out_len:path,
     $gpu:path, $tol:expr, $cos:expr) => {
        mod $case {
            use super::super::*;

            #[test]
            fn cpu_is_finite() {
                let fix = $fixture();
                let out = cpu(&fix);
                assert_eq!(out.len(), $out_len(&fix));
                assert!(
                    out.iter().all(|v| v.is_finite()),
                    "cpu reference produced non-finite values"
                );
            }

            #[test]
            fn gpu_matches_cpu() {
                if !$gpu_available() {
                    return;
                }
                let fix = $fixture();
                let cpu = cpu(&fix);
                let gpu = $gpu(&fix).expect("gpu dispatch");
                $assert_oracle(&gpu, &cpu, $tol, $cos);
            }
        }
    };
}

/// Expand a portable-op fixture table into one `__op_case!` per entry.
#[macro_export]
macro_rules! __op_cases {
    ($assert_oracle:path, $gpu_available:path $(,)?) => {};

    ($assert_oracle:path, $gpu_available:path, $case:ident => { fixture = $fixture:path,
        gpu = $gpu:path, out_len = $out_len:path, tol = $tol:expr, cos = $cos:expr }
        $(, $($rest:tt)*)? ) => {
        $crate::__op_case!($assert_oracle, $gpu_available, $case, $fixture, $out_len, $gpu,
            $tol, $cos);
        $crate::__op_cases!($assert_oracle, $gpu_available $(, $($rest)*)?);
    };

    ($assert_oracle:path, $gpu_available:path, $case:ident => $fixture:ident
        => ($tol:expr, $cos:expr) $(, $($rest:tt)*)? ) => {
        $crate::__op_case!($assert_oracle, $gpu_available, $case, $fixture, method, gpu,
            $tol, $cos);
        $crate::__op_cases!($assert_oracle, $gpu_available $(, $($rest)*)?);
    };
}
