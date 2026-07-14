//! CPU oracle for `gemm_int_sparse`: pure-Rust int8 dot-product MoE expert GEMM
//! on the bench side-channel fixture. The oracle forever (AGENTS.md §5 Tier 1).
//!
//! Mirrors the kernel's math exactly:
//!   - per-row absmax int8 quantization on A and W (`q = round(w / scale)`,
//!     `scale = max(|w_row|) / 127`)
//!   - `int32 += int8 * int8` per output cell
//!   - `f32 out = int32_acc * scale_a[row] * scale_w[col]`
//!
//! Layout:
//!   - `a_int8[slot, K]` row-major int8
//!   - `w_int8` packed per expert, each expert's rows `[N, K]` row-major
//!     starting at `jobs[e].w_byte_off` (byte offset into w_int8)
//!   - `scale_a[slot]` per-A-row f32 absmax scale
//!   - `scale_w` per-output-COLUMN f32 scale, shared across experts (the bench
//!     re-quants all expert rows with the same per-column scale envelope — see
//!     the kernel header comment)
//!
//! Block-sparse dispatch: for each block in `route`, walk its `BM` rows
//! (bounded by `row_starts[e+1] - row0`) and compute each output cell.

use crate::metal::{BlockGroupedJob, RouteScratch};

/// Per-row absmax int8 quantization: returns (int8 packed row-major, f32 scale).
/// `scale = max(|x|) / 127`; `q = round(x / scale)` clamped to [-128, 127].
pub fn quantize_row_int8(x: &[f32]) -> (Vec<i8>, f32) {
    let mut max_abs = 0.0f32;
    for &v in x {
        max_abs = max_abs.max(v.abs());
    }
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
    let q: Vec<i8> = x
        .iter()
        .map(|&v| {
            let r = (v / scale).round();
            let r = r.clamp(-128.0, 127.0);
            r as i8
        })
        .collect();
    (q, scale)
}

/// Quantize a [rows, K] row-major f32 matrix into a row-major int8 blob plus
/// per-row f32 scales.
pub fn quantize_matrix_int8_row_major(x: &[f32], rows: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; rows * k];
    let mut s = vec![0.0f32; rows];
    for r in 0..rows {
        let row = &x[r * k..(r + 1) * k];
        let (rq, rs) = quantize_row_int8(row);
        q[r * k..(r + 1) * k].copy_from_slice(&rq);
        s[r] = rs;
    }
    (q, s)
}

/// Quantize a [N, K] row-major f32 expert weight matrix into a row-major int8
/// blob plus per-N-row (per-output-column) f32 scales. This is the W layout
/// the kernel reads: `w_int8[expert_offset + col*K + k]`.
pub fn quantize_expert_int8_row_major(w: &[f32], n: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
    quantize_matrix_int8_row_major(w, n, k)
}

/// CPU oracle: block-sparse int8 GEMM mirroring `gemm_int_sparse`.
///
/// - `a_int8`: `[slots, K]` row-major int8 activations
/// - `w_int8`: packed int8 expert weights blob (each expert `[N, K]` row-major
///   at byte offset `jobs[e].w_byte_off`)
/// - `scale_a`: `[slots]` per-row f32 scales
/// - `scale_w`: `[N]` per-output-column f32 scales (shared across experts)
/// - `route` / `row_starts` / `jobs`: block-sparse dispatch metadata
/// - `bm`: block height (matches the kernel's INT_BM)
/// - returns: `[slots, N]` row-major f32 output
pub fn gemm_int_sparse_cpu(
    a_int8: &[i8],
    w_int8: &[u8],
    scale_a: &[f32],
    scale_w: &[f32],
    jobs: &[BlockGroupedJob],
    row_starts: &[u32],
    route: &RouteScratch,
    n: usize,
    k: usize,
    bm: usize,
) -> Vec<f32> {
    let slots = row_starts.last().copied().unwrap_or(0) as usize;
    let mut out = vec![0.0f32; slots * n];
    for blk in 0..route.num_blocks as usize {
        let e = route.block_expert[blk] as usize;
        if e >= jobs.len() {
            continue;
        }
        let row0 = route.block_row0[blk] as usize;
        let m1 = row_starts[e + 1] as usize;
        let m_tile = bm.min(m1.saturating_sub(row0));
        if m_tile == 0 {
            continue;
        }
        let w_off = jobs[e].w_byte_off as usize;
        for di in 0..m_tile {
            let slot = row0 + di;
            let sa = scale_a[slot];
            let a_off = slot * k;
            for col in 0..n {
                let sw = scale_w[col];
                let w_off_col = w_off + col * k;
                let mut acc: i32 = 0;
                for kk in 0..k {
                    let a = a_int8[a_off + kk] as i32;
                    let w = w_int8[w_off_col + kk] as i8 as i32;
                    acc += a * w;
                }
                out[slot * n + col] = acc as f32 * sa * sw;
            }
        }
    }
    out
}
