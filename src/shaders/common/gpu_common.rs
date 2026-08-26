//! Shared Metal helpers for tier-1 subkernel GPU tests. The dispatch
//! mechanics live in `gpukit`; only diffgemma-shaped grids stay here.

#[cfg(target_os = "macos")]
pub use gpukit::metal::{dispatch_1d, dispatch_1d_ranged, dispatch_grid, dispatch_rows, set_bytes};

pub fn div_up(value: usize, group: usize) -> usize {
    value.div_ceil(group)
}

/// `scatter_vocab_chunk` grid (matches lm_head).
#[cfg(target_os = "macos")]
pub fn scatter_vocab_grid(
    seq_len: usize,
    chunk_cols: usize,
) -> (objc2_metal::MTLSize, objc2_metal::MTLSize) {
    use objc2_metal::MTLSize;
    const TG_W: usize = 16;
    const TG_H: usize = 16;
    (
        MTLSize {
            width: div_up(chunk_cols, TG_W),
            height: div_up(seq_len, TG_H),
            depth: 1,
        },
        MTLSize {
            width: TG_W,
            height: TG_H,
            depth: 1,
        },
    )
}
