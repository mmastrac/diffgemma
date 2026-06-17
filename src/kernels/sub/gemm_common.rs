//! Shared helpers for tiled quant GEMM subkernels (function constants 1–3).

pub const THREADS_PER_TG: usize = 128;
pub const M_TILE: usize = 32;

pub fn div_up(value: usize, group: usize) -> usize {
    (value + group - 1) / group
}

/// N-axis tile width for block GEMM (`DGQ_GEMM_N_TILE`: 128 default, set `32` to opt out).
pub fn n_tile() -> usize {
    static TILE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *TILE.get_or_init(|| match std::env::var("DGQ_GEMM_N_TILE").as_deref() {
        Ok("32") => 32,
        Ok(v) => v.parse::<usize>().unwrap_or(128),
        Err(_) => 128,
    })
}

#[cfg(all(feature = "metal", target_os = "macos"))]
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
