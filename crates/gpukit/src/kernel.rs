//! Kernel registration: one block declares what a kernel is (its name, its
//! backend bodies, its function-constant pipeline, its manifest row) and the
//! expansion emits the consts, the backend dispatch wrapper, and the tier-1
//! test module.
//!
//! Two arms, one per kernel home:
//!
//! - `op` — a backend-agnostic op whose Metal and CUDA bodies sit beside a CPU
//!   oracle (`crates/dgops`): emits `ENTRY`, `METAL`, the `CUDA` source, the
//!   two generated backend modules, the backend-selecting `gpu()` wrapper, and
//!   the test module. Its `abi` list is the single declaration of the kernel
//!   argument list: both backends' `gpu()` bodies are generated from it, so the
//!   two argument orders cannot drift (see `Generated dispatch` below).
//! - `shader` — one Metal entry point with an FC-specialized pipeline and an
//!   optional manifest `SPEC` (the engine's `src/shaders`): emits `ENTRY`,
//!   `SHADER`, an optional `pipeline_for`, an optional `NAME`+`SPEC`, and the
//!   test module.
//!
//! Everything crate-specific (the error type, the Metal context type, the
//! manifest type, the oracle assertion) is passed in by the caller's shim, so
//! this crate stays backend- and crate-agnostic. What differs between kernels
//! — fixtures, CPU oracles, kernel bodies — stays hand-written at the call
//! site; the only generated code is an `op` arm's host-side dispatch, and it is
//! generated for both backends from the one `abi` list.
//!
//! Constraints worth knowing before editing:
//!
//! - `include_str!($metal)` resolves relative to the *invoking* file, so
//!   `metal = "gelu.metal"` means "beside this `mod.rs`".
//! - `cuda = "gelu.cu"` embeds the CUDA C++ source beside the op, exactly like
//!   `metal`: NVRTC compiles it on first dispatch, so no build-time toolchain is
//!   needed and the artifact matches the running device's architecture.
//! - The `metal`/`cuda` modules emitted here resolve relative to the caller's
//!   directory (they are inline modules, so their `use super::*;` sees the op).
//! - The `pipeline` body must be a caller-written closure: names introduced by
//!   this macro's transcriber are hygiene-invisible to caller tokens, so a
//!   generated `fn pipeline_for(ctx, variant)` cannot expose `ctx`/`variant` to
//!   a body written at the call site. The closure is coerced to a fn pointer so
//!   its parameters get their types from the generated signature.
//! - The same rule is why an `op` arm's `fixture = Fixture => fix` names the
//!   binding: the `abi`/`launch`/`result` expressions are caller tokens, so the
//!   binding they read has to come from the call site. The generated `gpu()`
//!   takes an internal parameter and immediately rebinds it with
//!   `let $fixvar = ...` (a metavariable can name a local, not a parameter).
//! - The generated backend modules glob-import the op module (`use super::*;`)
//!   so the fixture type and helpers resolve inside them.

/// Register one kernel. See the module docs for the two arms.
#[macro_export]
macro_rules! kernel {
    // ---- op with a generated dispatch ABI ----------------------------------
    (
        op,
        error = $err:path,
        assert_oracle = $assert_oracle:path,
        gpu_available = $gpu_available:path,
        name = $name:literal,
        metal = $metal:literal,
        cuda = $cuda:literal,
        fixture = $fixture:ty => $fixvar:ident,
        abi = [ $($abi:tt)* ],
        launch = $launch:tt($largs:expr),
        result = ($res:ident, $rlen:expr),
        tests = $tests:tt $(,)?
    ) => {
        pub const ENTRY: &str = $name;
        pub const METAL: &str = include_str!($metal);
        pub const CUDA: &str = include_str!($cuda);

        #[cfg(target_os = "macos")]
        pub mod metal {
            use gpukit::metal::{BufferPool, CacheConfig};
            use super::*;

            pub fn gpu(__fixture: &$fixture) -> Result<Vec<f32>, $err> {
                let $fixvar = __fixture;
                let ctx = gpukit::metal::cached_context(CacheConfig::MEMORY)?;
                let pipeline = gpukit::metal::cached_pipeline(&ctx, super::METAL, super::ENTRY)?;
                let mut pool = BufferPool::new();
                $crate::__op_metal_prep!($err, ctx, pool, $($abi)*);
                $crate::__op_metal_dispatch!($launch, &ctx.queue, &pipeline.pipeline, $largs, enc,
                    $crate::__op_metal_bind!(enc, 0, $($abi)*))?;
                let mut out = vec![0.0f32; $rlen];
                BufferPool::read_f32(&$res, &mut out);
                Ok(out)
            }
        }

        #[cfg(feature = "cuda")]
        pub mod cuda {
            use gpukit::cuda::{BufferPool, KernelArgs};
            use super::*;

            pub fn gpu(__fixture: &$fixture) -> Result<Vec<f32>, $err> {
                let $fixvar = __fixture;
                let ctx = gpukit::cuda::cached_context()?;
                let kernel = gpukit::cuda::cached_source_kernel(super::CUDA, super::ENTRY)?;
                let mut pool = BufferPool::new();
                $crate::__op_cuda_prep!(ctx, pool, $($abi)*);
                let mut args = KernelArgs::new();
                $crate::__op_cuda_args!(args, $($abi)*);
                $crate::__op_cuda_dispatch!($launch, ctx, &kernel, $largs, &mut args)?;
                ctx.synchronize()?;
                let mut out = vec![0.0f32; $rlen];
                $res.read_f32(&mut out)?;
                Ok(out)
            }
        }

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
        oracle = $oracle:path, tol = $tol:expr, cos = $cos:expr } $($tail:tt)* ) => {
        $crate::__shader_case!($assert_oracle, $skip_gpu, $elem_f32, $variant, $case, $fixture,
            $gpu, $cpu, $oracle, cpu_matches_oracle, $tol, $cos, variant);
        $crate::__shader_cases!($assert_oracle, $skip_gpu, $elem_f32, $variant $($tail)*);
    };

    ($assert_oracle:path, $skip_gpu:path, $elem_f32:path, $variant:ty,
        $case:ident => { fixture = $fixture:path, gpu = $gpu:path, cpu = $cpu:path,
        tol = $tol:expr, cos = $cos:expr } $($tail:tt)* ) => {
        $crate::__shader_case!($assert_oracle, $skip_gpu, $elem_f32, $variant, $case, $fixture,
            $gpu, $cpu, $cpu, cpu_is_finite, $tol, $cos, variant);
        $crate::__shader_cases!($assert_oracle, $skip_gpu, $elem_f32, $variant $($tail)*);
    };

    ($assert_oracle:path, $skip_gpu:path, $elem_f32:path, $variant:ty,
        $case:ident => { fixture = $fixture:path, oracle = $oracle:path, tol = $tol:expr,
        cos = $cos:expr } $($tail:tt)* ) => {
        $crate::__shader_case!($assert_oracle, $skip_gpu, $elem_f32, $variant, $case, $fixture,
            gpu, cpu, $oracle, cpu_matches_oracle, $tol, $cos, variant);
        $crate::__shader_cases!($assert_oracle, $skip_gpu, $elem_f32, $variant $($tail)*);
    };

    ($assert_oracle:path, $skip_gpu:path, $elem_f32:path, $variant:ty,
        $case:ident => $fixture:ident => no_variant => ($tol:expr, $cos:expr)
        $($tail:tt)* ) => {
        $crate::__shader_case!($assert_oracle, $skip_gpu, $elem_f32, $variant, $case, $fixture,
            gpu, cpu, cpu, cpu_is_finite, $tol, $cos, no_variant);
        $crate::__shader_cases!($assert_oracle, $skip_gpu, $elem_f32, $variant $($tail)*);
    };

    ($assert_oracle:path, $skip_gpu:path, $elem_f32:path, $variant:ty,
        $case:ident => $fixture:ident => $gpu:ident => ($tol:expr, $cos:expr)
        $($tail:tt)* ) => {
        $crate::__shader_case!($assert_oracle, $skip_gpu, $elem_f32, $variant, $case, $fixture,
            $gpu, cpu, cpu, cpu_is_finite, $tol, $cos, variant);
        $crate::__shader_cases!($assert_oracle, $skip_gpu, $elem_f32, $variant $($tail)*);
    };

    ($assert_oracle:path, $skip_gpu:path, $elem_f32:path, $variant:ty,
        $case:ident => $fixture:ident => ($tol:expr, $cos:expr) $($tail:tt)* ) => {
        $crate::__shader_case!($assert_oracle, $skip_gpu, $elem_f32, $variant, $case, $fixture,
            gpu, cpu, cpu, cpu_is_finite, $tol, $cos, variant);
        $crate::__shader_cases!($assert_oracle, $skip_gpu, $elem_f32, $variant $($tail)*);
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
        $($tail:tt)* ) => {
        $crate::__op_case!($assert_oracle, $gpu_available, $case, $fixture, $out_len, $gpu,
            $tol, $cos);
        $crate::__op_cases!($assert_oracle, $gpu_available $($tail)*);
    };

    ($assert_oracle:path, $gpu_available:path, $case:ident => $fixture:ident
        => ($tol:expr, $cos:expr) $($tail:tt)* ) => {
        $crate::__op_case!($assert_oracle, $gpu_available, $case, $fixture, method, gpu,
            $tol, $cos);
        $crate::__op_cases!($assert_oracle, $gpu_available $($tail)*);
    };
}

// ---------------------------------------------------------------------------
// Generated dispatch (op arm with `abi`)
//
// An op whose bindings are a flat list of buffers and scalars declares them
// once; both backends' gpu() bodies are generated from that list, so the two
// argument orders cannot drift. Entry grammar, in kernel-argument order:
//
//   in(buf = expr)        read-only f32 buffer, size expr.len()
//   inout(buf = expr)     read-write f32 buffer (the result when named by
//                         `result`), size expr.len()
//   out(buf = size)       write-only f32 buffer of `size` elements
//   in_u32(buf = expr)    read-only u32 buffer, size expr.len()
//   u32(expr) f32(expr)   scalar arguments
//   u32x2(a, b)           two u32 dims: one packed uint2 on Metal, two args on CUDA
//   pod(expr)             repr(C) POD scalar argument
//
// `launch = 1d(expr) | rows(expr)` picks the dispatch shape and
// `result = (buf, len_expr)` the buffer copied back to the host.
// ---------------------------------------------------------------------------

/// Metal: allocate and upload every buffer of an op ABI.
#[macro_export]
macro_rules! __op_metal_prep {
    ($err:path, $ctx:ident, $pool:ident $(,)?) => {};
    ($err:path, $ctx:ident, $pool:ident, in($n:ident = $e:expr) $($tail:tt)*) => {
        let $n = $pool
            .allocate(&$ctx.device, $e.len() * 4)
            .ok_or(<$err>::Gpu("buffer alloc"))?;
        gpukit::metal::BufferPool::write_f32(&$n, &$e);
        $crate::__op_metal_prep!($err, $ctx, $pool $($tail)*);
    };
    ($err:path, $ctx:ident, $pool:ident, inout($n:ident = $e:expr) $($tail:tt)*) => {
        let $n = $pool
            .allocate(&$ctx.device, $e.len() * 4)
            .ok_or(<$err>::Gpu("buffer alloc"))?;
        gpukit::metal::BufferPool::write_f32(&$n, &$e);
        $crate::__op_metal_prep!($err, $ctx, $pool $($tail)*);
    };
    ($err:path, $ctx:ident, $pool:ident, out($n:ident = $size:expr) $($tail:tt)*) => {
        let $n = $pool
            .allocate(&$ctx.device, ($size) * 4)
            .ok_or(<$err>::Gpu("buffer alloc"))?;
        $crate::__op_metal_prep!($err, $ctx, $pool $($tail)*);
    };
    ($err:path, $ctx:ident, $pool:ident, in_u32($n:ident = $e:expr) $($tail:tt)*) => {
        let $n = $pool
            .allocate(&$ctx.device, $e.len() * 4)
            .ok_or(<$err>::Gpu("buffer alloc"))?;
        gpukit::metal::BufferPool::write_bytes(&$n, unsafe {
            std::slice::from_raw_parts($e.as_ptr().cast::<u8>(), $e.len() * 4)
        });
        $crate::__op_metal_prep!($err, $ctx, $pool $($tail)*);
    };
    ($err:path, $ctx:ident, $pool:ident, u32x2($a:expr, $b:expr) $($tail:tt)*) => {
        $crate::__op_metal_prep!($err, $ctx, $pool $($tail)*);
    };
    ($err:path, $ctx:ident, $pool:ident, $head:ident($e:expr) $($tail:tt)*) => {
        $crate::__op_metal_prep!($err, $ctx, $pool $($tail)*);
    };
}

/// Metal: bind every buffer and scalar, in order.
#[macro_export]
macro_rules! __op_metal_bind {
    ($enc:ident, $idx:expr $(,)?) => {};
    ($enc:ident, $idx:expr, in($n:ident = $e:expr) $($tail:tt)*) => {
        gpukit::metal::bind_buffer($enc, &$n, $idx);
        $crate::__op_metal_bind!($enc, $idx + 1 $($tail)*);
    };
    ($enc:ident, $idx:expr, inout($n:ident = $e:expr) $($tail:tt)*) => {
        gpukit::metal::bind_buffer($enc, &$n, $idx);
        $crate::__op_metal_bind!($enc, $idx + 1 $($tail)*);
    };
    ($enc:ident, $idx:expr, out($n:ident = $size:expr) $($tail:tt)*) => {
        gpukit::metal::bind_buffer($enc, &$n, $idx);
        $crate::__op_metal_bind!($enc, $idx + 1 $($tail)*);
    };
    ($enc:ident, $idx:expr, in_u32($n:ident = $e:expr) $($tail:tt)*) => {
        gpukit::metal::bind_buffer($enc, &$n, $idx);
        $crate::__op_metal_bind!($enc, $idx + 1 $($tail)*);
    };
    ($enc:ident, $idx:expr, u32x2($a:expr, $b:expr) $($tail:tt)*) => {
        gpukit::metal::set_bytes($enc, &[$a as u32, $b as u32], $idx);
        $crate::__op_metal_bind!($enc, $idx + 1 $($tail)*);
    };
    ($enc:ident, $idx:expr, u32($e:expr) $($tail:tt)*) => {
        gpukit::metal::set_bytes($enc, &($e as u32), $idx);
        $crate::__op_metal_bind!($enc, $idx + 1 $($tail)*);
    };
    ($enc:ident, $idx:expr, f32($e:expr) $($tail:tt)*) => {
        gpukit::metal::set_bytes($enc, &($e as f32), $idx);
        $crate::__op_metal_bind!($enc, $idx + 1 $($tail)*);
    };
    ($enc:ident, $idx:expr, pod($e:expr) $($tail:tt)*) => {
        gpukit::metal::set_bytes($enc, &$e, $idx);
        $crate::__op_metal_bind!($enc, $idx + 1 $($tail)*);
    };
}

/// CUDA: allocate and upload every buffer of an op ABI.
#[macro_export]
macro_rules! __op_cuda_prep {
    ($ctx:ident, $pool:ident $(,)?) => {};
    ($ctx:ident, $pool:ident, in($n:ident = $e:expr) $($tail:tt)*) => {
        let $n = $pool.allocate($ctx, $e.len() * 4)?;
        $n.write_f32(&$e)?;
        $crate::__op_cuda_prep!($ctx, $pool $($tail)*);
    };
    ($ctx:ident, $pool:ident, inout($n:ident = $e:expr) $($tail:tt)*) => {
        let $n = $pool.allocate($ctx, $e.len() * 4)?;
        $n.write_f32(&$e)?;
        $crate::__op_cuda_prep!($ctx, $pool $($tail)*);
    };
    ($ctx:ident, $pool:ident, out($n:ident = $size:expr) $($tail:tt)*) => {
        let $n = $pool.allocate($ctx, ($size) * 4)?;
        $crate::__op_cuda_prep!($ctx, $pool $($tail)*);
    };
    ($ctx:ident, $pool:ident, in_u32($n:ident = $e:expr) $($tail:tt)*) => {
        let $n = $pool.allocate($ctx, $e.len() * 4)?;
        $n.write_bytes(unsafe {
            std::slice::from_raw_parts($e.as_ptr().cast::<u8>(), $e.len() * 4)
        })?;
        $crate::__op_cuda_prep!($ctx, $pool $($tail)*);
    };
    ($ctx:ident, $pool:ident, u32x2($a:expr, $b:expr) $($tail:tt)*) => {
        $crate::__op_cuda_prep!($ctx, $pool $($tail)*);
    };
    ($ctx:ident, $pool:ident, $head:ident($e:expr) $($tail:tt)*) => {
        $crate::__op_cuda_prep!($ctx, $pool $($tail)*);
    };
}

/// CUDA: append every buffer and scalar to the launch arguments, in order.
#[macro_export]
macro_rules! __op_cuda_args {
    ($args:ident $(,)?) => {};
    ($args:ident, in($n:ident = $e:expr) $($tail:tt)*) => {
        $args.device_ptr($n.device_ptr());
        $crate::__op_cuda_args!($args $($tail)*);
    };
    ($args:ident, inout($n:ident = $e:expr) $($tail:tt)*) => {
        $args.device_ptr($n.device_ptr());
        $crate::__op_cuda_args!($args $($tail)*);
    };
    ($args:ident, out($n:ident = $size:expr) $($tail:tt)*) => {
        $args.device_ptr($n.device_ptr());
        $crate::__op_cuda_args!($args $($tail)*);
    };
    ($args:ident, in_u32($n:ident = $e:expr) $($tail:tt)*) => {
        $args.device_ptr($n.device_ptr());
        $crate::__op_cuda_args!($args $($tail)*);
    };
    ($args:ident, u32x2($a:expr, $b:expr) $($tail:tt)*) => {
        $args.u32($a as u32);
        $args.u32($b as u32);
        $crate::__op_cuda_args!($args $($tail)*);
    };
    ($args:ident, u32($e:expr) $($tail:tt)*) => {
        $args.u32($e as u32);
        $crate::__op_cuda_args!($args $($tail)*);
    };
    ($args:ident, f32($e:expr) $($tail:tt)*) => {
        $args.f32($e as f32);
        $crate::__op_cuda_args!($args $($tail)*);
    };
    ($args:ident, pod($e:expr) $($tail:tt)*) => {
        $args.bytes(gpukit::cuda::pod_bytes(&$e));
        $crate::__op_cuda_args!($args $($tail)*);
    };
}

/// Metal: pick the dispatch shape declared by the op ABI.
#[macro_export]
macro_rules! __op_metal_dispatch {
    (1d, $queue:expr, $pipe:expr, $n:expr, $enc:ident, $($binds:tt)*) => {
        gpukit::metal::dispatch_1d($queue, $pipe, $n, |$enc| { $($binds)*; })
    };
    (rows, $queue:expr, $pipe:expr, $n:expr, $enc:ident, $($binds:tt)*) => {
        gpukit::metal::dispatch_rows($queue, $pipe, $n, |$enc| { $($binds)*; })
    };
}

/// CUDA: pick the dispatch shape declared by the op ABI.
#[macro_export]
macro_rules! __op_cuda_dispatch {
    (1d, $ctx:expr, $kernel:expr, $n:expr, $args:expr) => {
        gpukit::cuda::launch_1d($ctx, $kernel, $n, $args)
    };
    (rows, $ctx:expr, $kernel:expr, $n:expr, $args:expr) => {
        gpukit::cuda::launch_rows($ctx, $kernel, $n, 256, $args)
    };
}
