#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#include "gemm_fc.metal"
#include "gemm_block_tile.metal"
#include "dequant.metal"
#include "arena.metal"
#include "gemm_stacked.metal"

inline void gemm_block_stacked_q4_impl(
    device const ushort *x,
    device ushort *y_arena,
    device const uchar *blob,
    constant GemmStackedSeg *segs,
    uint n_segs,
    uint M,
    uint3 tgid,
    uint ltid,
    uint sgid,
    threadgroup half tx[GEMM_M_TILE][GEMM_K_TILE],
    threadgroup half tw[GEMM_N_TILE_MAX][GEMM_K_TILE],
    threadgroup float ty[GEMM_M_TILE][GEMM_N_TILE_MAX]
) {
    const uint N = GEMM_N;
    const uint K = GEMM_K;
    const uint m0 = tgid.y * GEMM_M_TILE;
    const uint n0 = tgid.x * GEMM_N_TILE;
    simdgroup_float8x8 acc0(0.f), acc1(0.f), acc2(0.f), acc3(0.f);
    simdgroup_float8x8 acc4(0.f), acc5(0.f), acc6(0.f), acc7(0.f);
    simdgroup_float8x8 acc8(0.f), acc9(0.f), acc10(0.f), acc11(0.f);
    simdgroup_float8x8 acc12(0.f), acc13(0.f), acc14(0.f), acc15(0.f);

    const ulong rowB = q4_row_bytes(K);

    uint seg_end0 = 0u;
    uint seg_end1 = 0u;
    uint seg_end2 = 0u;
    ulong seg_body0 = 0ul;
    ulong seg_body1 = 0ul;
    ulong seg_body2 = 0ul;
    if (n_segs > 0u) {
        seg_end0 = segs[0].n_cols;
        seg_body0 = segs[0].w_off;
    }
    if (n_segs > 1u) {
        seg_end1 = seg_end0 + segs[1].n_cols;
        seg_body1 = segs[1].w_off;
    }
    if (n_segs > 2u) {
        seg_end2 = seg_end1 + segs[2].n_cols;
        seg_body2 = segs[2].w_off;
    }

    for (uint k0 = 0u; k0 < K; k0 += GEMM_K_TILE) {
        gemm_load_a_tile(x, M, K, m0, k0, ltid, tx);
        if (ltid < GEMM_N_TILE) {
            const uint r = ltid;
            const uint global_n = n0 + r;
            if (global_n >= N) {
                gemm_block_zero_tw_row(tw, r);
            } else {
                const uint seg = stacked_seg_index_fast(global_n, n_segs, seg_end0, seg_end1, seg_end2);
                const uint local_n = stacked_local_col_fast(global_n, seg, seg_end0, seg_end1);
                const ulong body = seg == 0u ? seg_body0 : (seg == 1u ? seg_body1 : seg_body2);
                dequant_q4_group_half_tg(
                    blob + body + (ulong)local_n * rowB + (ulong)(k0 / 32u) * 20ul,
                    &tw[r][0]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_block_mma_k32(
            tx, tw, sgid, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10,
            acc11, acc12, acc13, acc14, acc15);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    gemm_block_mma_store(
        ty, sgid, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11,
        acc12, acc13, acc14, acc15);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ltid; i < GEMM_M_TILE * GEMM_N_TILE; i += 128u) {
        const uint mm = i / GEMM_N_TILE;
        const uint nn = i % GEMM_N_TILE;
        const uint global_n = n0 + nn;
        if (m0 + mm < M && global_n < N) {
            const uint seg = stacked_seg_index_fast(global_n, n_segs, seg_end0, seg_end1, seg_end2);
            const GemmStackedSeg sg = segs[seg];
            const uint local_n = stacked_local_col_fast(global_n, seg, seg_end0, seg_end1);
            const ulong out_idx = (ulong)(m0 + mm) * (ulong)sg.y_row_cols + (ulong)sg.y_col0 + (ulong)local_n;
            device ushort *dst = (device ushort *)((device uchar *)y_arena + sg.y_byte_off);
            arena_store(dst, out_idx, ty[mm][nn]);
        }
    }
}

inline void gemm_block_stacked_nvfp4_impl(
    device const ushort *x,
    device ushort *y_arena,
    device const uchar *blob,
    constant GemmStackedSeg *segs,
    uint n_segs,
    uint M,
    uint3 tgid,
    uint ltid,
    uint sgid,
    threadgroup half tx[GEMM_M_TILE][GEMM_K_TILE],
    threadgroup half tw[GEMM_N_TILE_MAX][GEMM_K_TILE],
    threadgroup float ty[GEMM_M_TILE][GEMM_N_TILE_MAX]
) {
    const uint N = GEMM_N;
    const uint K = GEMM_K;
    const uint m0 = tgid.y * GEMM_M_TILE;
    const uint n0 = tgid.x * GEMM_N_TILE;
    simdgroup_float8x8 acc0(0.f), acc1(0.f), acc2(0.f), acc3(0.f);
    simdgroup_float8x8 acc4(0.f), acc5(0.f), acc6(0.f), acc7(0.f);
    simdgroup_float8x8 acc8(0.f), acc9(0.f), acc10(0.f), acc11(0.f);
    simdgroup_float8x8 acc12(0.f), acc13(0.f), acc14(0.f), acc15(0.f);

    uint seg_end0 = 0u;
    uint seg_end1 = 0u;
    uint seg_end2 = 0u;
    ulong seg_w0 = 0ul;
    ulong seg_w1 = 0ul;
    ulong seg_w2 = 0ul;
    float seg_gscale0 = 0.f;
    float seg_gscale1 = 0.f;
    float seg_gscale2 = 0.f;
    if (n_segs > 0u) {
        seg_end0 = segs[0].n_cols;
        seg_w0 = segs[0].w_off;
        seg_gscale0 = as_type<float>(*(device const uint *)(blob + seg_w0));
    }
    if (n_segs > 1u) {
        seg_end1 = seg_end0 + segs[1].n_cols;
        seg_w1 = segs[1].w_off;
        seg_gscale1 = as_type<float>(*(device const uint *)(blob + seg_w1));
    }
    if (n_segs > 2u) {
        seg_end2 = seg_end1 + segs[2].n_cols;
        seg_w2 = segs[2].w_off;
        seg_gscale2 = as_type<float>(*(device const uint *)(blob + seg_w2));
    }
    const ulong rowB = nvfp4_row_bytes(K);
    const device uchar *body0 = blob + seg_w0 + 4ul;
    const device uchar *body1 = blob + seg_w1 + 4ul;
    const device uchar *body2 = blob + seg_w2 + 4ul;

    for (uint k0 = 0u; k0 < K; k0 += GEMM_K_TILE) {
        gemm_load_a_tile(x, M, K, m0, k0, ltid, tx);
        if (ltid < GEMM_N_TILE) {
            const uint r = ltid;
            const uint global_n = n0 + r;
            if (global_n >= N) {
                gemm_block_zero_tw_row(tw, r);
            } else {
                const uint seg = stacked_seg_index_fast(global_n, n_segs, seg_end0, seg_end1, seg_end2);
                const uint local_n = stacked_local_col_fast(global_n, seg, seg_end0, seg_end1);
                const device uchar *body = seg == 0u ? body0 : (seg == 1u ? body1 : body2);
                const float gscale = seg == 0u ? seg_gscale0 : (seg == 1u ? seg_gscale1 : seg_gscale2);
                device const uchar *row = body + (ulong)local_n * rowB;
                dequant_nvfp4_tile_half_fused_tg(row, K, k0, &tw[r][0], gscale);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_block_mma_k32(
            tx, tw, sgid, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10,
            acc11, acc12, acc13, acc14, acc15);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    gemm_block_mma_store(
        ty, sgid, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11,
        acc12, acc13, acc14, acc15);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ltid; i < GEMM_M_TILE * GEMM_N_TILE; i += 128u) {
        const uint mm = i / GEMM_N_TILE;
        const uint nn = i % GEMM_N_TILE;
        const uint global_n = n0 + nn;
        if (m0 + mm < M && global_n < N) {
            const uint seg = stacked_seg_index_fast(global_n, n_segs, seg_end0, seg_end1, seg_end2);
            const GemmStackedSeg sg = segs[seg];
            const uint local_n = stacked_local_col_fast(global_n, seg, seg_end0, seg_end1);
            const ulong out_idx = (ulong)(m0 + mm) * (ulong)sg.y_row_cols + (ulong)sg.y_col0 + (ulong)local_n;
            device ushort *dst = (device ushort *)((device uchar *)y_arena + sg.y_byte_off);
            arena_store(dst, out_idx, ty[mm][nn]);
        }
    }
}

/// Stacked GEMM entry: dispatches to format-specific implementations (Q4 / NVFP4).
/// `GEMM_N` = sum(segs[i].n_cols). K from `GEMM_K` (FC6).
kernel void gemm_block_stacked(
    device const ushort *x [[buffer(0)]],
    device ushort *y_arena [[buffer(1)]],
    device const uchar *blob [[buffer(2)]],
    constant GemmStackedSeg *segs [[buffer(3)]],
    constant uint &n_segs [[buffer(4)]],
    constant uint &M [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]
) {
    const uint ltid = lid.x;
    threadgroup half tx[GEMM_M_TILE][GEMM_K_TILE];
    threadgroup half tw[GEMM_N_TILE_MAX][GEMM_K_TILE];
    threadgroup float ty[GEMM_M_TILE][GEMM_N_TILE_MAX];
    if (K_QUANT_FORMAT == QUANT_NVFP4) {
        gemm_block_stacked_nvfp4_impl(x, y_arena, blob, segs, n_segs, M, tgid, ltid, sgid, tx, tw, ty);
    } else {
        gemm_block_stacked_q4_impl(x, y_arena, blob, segs, n_segs, M, tgid, ltid, sgid, tx, tw, ty);
    }
}
