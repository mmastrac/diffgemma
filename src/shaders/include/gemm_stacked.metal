#ifndef DGQ_INCLUDE_GEMM_STACKED_METAL
#define DGQ_INCLUDE_GEMM_STACKED_METAL

#include <metal_stdlib>
using namespace metal;

/// One output segment in a stacked multi-matrix GEMM (QKV, dense gate+up).
struct GemmStackedSeg {
    uint n_cols;
    uint y_col0;
    uint y_row_cols;
    uint _pad;
    ulong w_off;
    ulong y_byte_off;
};

constant uint GEMM_STACKED_MAX_SEGS = 3u;

inline uint stacked_seg_index(uint global_col, constant const GemmStackedSeg *segs, uint n_segs) {
    uint acc = 0u;
    for (uint s = 0u; s < n_segs; s++) {
        acc += segs[s].n_cols;
        if (global_col < acc) {
            return s;
        }
    }
    return n_segs > 0u ? n_segs - 1u : 0u;
}

inline uint stacked_local_col(uint global_col, constant const GemmStackedSeg *segs, uint seg) {
    uint acc = 0u;
    for (uint s = 0u; s < seg; s++) {
        acc += segs[s].n_cols;
    }
    return global_col - acc;
}

/// Fast segment lookup (n_segs <= 3) using precomputed exclusive column ends.
inline uint stacked_seg_index_fast(
    uint global_col,
    uint n_segs,
    uint end0,
    uint end1,
    uint end2
) {
    if (n_segs <= 1u) {
        return 0u;
    }
    if (global_col < end0) {
        return 0u;
    }
    if (n_segs == 2u) {
        return 1u;
    }
    if (global_col < end1) {
        return 1u;
    }
    return 2u;
}

inline uint stacked_local_col_fast(uint global_col, uint seg, uint end0, uint end1) {
    if (seg == 0u) {
        return global_col;
    }
    if (seg == 1u) {
        return global_col - end0;
    }
    return global_col - end1;
}

#endif
