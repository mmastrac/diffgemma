//! `op_kernel!` — the dgops binding of `gpukit::kernel!`.
//!
//! Binds this crate's error type and the build script's CUDA-kernels cfg, so an
//! op's registration block names only what is specific to that op.

#[macro_export]
macro_rules! op_kernel {
    ($($rest:tt)*) => {
        gpukit::kernel! {
            op,
            error = crate::Error,
            assert_oracle = crate::testing::assert_oracle,
            gpu_available = crate::testing::gpu_available,
            $($rest)*
        }
    };
}
