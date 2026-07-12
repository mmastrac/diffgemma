//! Shared helpers for tiled quant GEMM subkernels (function constants 1–3).

pub const THREADS_PER_TG: usize = 128;
pub const M_TILE: usize = 32;

pub use crate::shaders::gpu_common::div_up;

/// N-axis tile width for the block GEMM family.
pub fn n_tile() -> usize {
    128
}

#[cfg(target_os = "macos")]
pub fn dispatch_shape(m: usize, n: usize) -> (objc2_metal::MTLSize, objc2_metal::MTLSize) {
    use objc2_metal::MTLSize;
    (
        MTLSize {
            width: div_up(n, n_tile()),
            height: div_up(m, M_TILE),
            depth: 1,
        },
        MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        },
    )
}
