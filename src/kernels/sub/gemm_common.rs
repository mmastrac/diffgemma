//! Shared helpers for tiled quant GEMM subkernels (function constants 1–3).

pub const THREADS_PER_TG: usize = 128;

pub fn div_up(value: usize, group: usize) -> usize {
    (value + group - 1) / group
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn dispatch_shape(m: usize, n: usize) -> (objc2_metal::MTLSize, objc2_metal::MTLSize) {
    use objc2_metal::MTLSize;
    (
        MTLSize {
            width: div_up(n, 32),
            height: div_up(m, 32),
            depth: 1,
        },
        MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        },
    )
}
