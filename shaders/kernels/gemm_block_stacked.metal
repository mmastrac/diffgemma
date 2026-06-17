#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#include "gemm_fc.metal"
#include "gemm_block_tile.metal"
#include "dequant.metal"
#include "arena.metal"
#include "gemm_stacked.metal"
#include "gemm_stacked_fc.metal"

inline void gemm_block_stacked_q4_impl(
    device const ushort *x,
    device ushort *y_arena,
    device const uchar *blob,
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

    for (uint k0 = 0u; k0 < K; k0 += GEMM_K_TILE) {
        gemm_load_a_tile(x, M, K, m0, k0, ltid, tx);
        if (ltid < GEMM_N_TILE) {
            const uint r = ltid;
            const uint global_n = n0 + r;
            if (global_n >= N) {
                gemm_block_zero_tw_row(tw, r);
            } else if (STACKED_N_SEGS == 1u) {
                dequant_q4_group_half_tg(
                    blob + STACKED_W_OFF0 + (ulong)global_n * rowB + (ulong)(k0 / 32u) * 20ul,
                    &tw[r][0]);
            } else if (STACKED_N_SEGS == 2u) {
                if (global_n < STACKED_END0) {
                    dequant_q4_group_half_tg(
                        blob + STACKED_W_OFF0 + (ulong)global_n * rowB + (ulong)(k0 / 32u) * 20ul,
                        &tw[r][0]);
                } else {
                    const uint local_n = global_n - STACKED_END0;
                    dequant_q4_group_half_tg(
                        blob + STACKED_W_OFF1 + (ulong)local_n * rowB + (ulong)(k0 / 32u) * 20ul,
                        &tw[r][0]);
                }
            } else {
                if (global_n < STACKED_END0) {
                    dequant_q4_group_half_tg(
                        blob + STACKED_W_OFF0 + (ulong)global_n * rowB + (ulong)(k0 / 32u) * 20ul,
                        &tw[r][0]);
                } else if (global_n < STACKED_END1) {
                    const uint local_n = global_n - STACKED_END0;
                    dequant_q4_group_half_tg(
                        blob + STACKED_W_OFF1 + (ulong)local_n * rowB + (ulong)(k0 / 32u) * 20ul,
                        &tw[r][0]);
                } else {
                    const uint local_n = global_n - STACKED_END1;
                    dequant_q4_group_half_tg(
                        blob + STACKED_W_OFF2 + (ulong)local_n * rowB + (ulong)(k0 / 32u) * 20ul,
                        &tw[r][0]);
                }
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
            const uint seg = stacked_fc_seg_index(global_n);
            const uint local_n = stacked_fc_local_n(global_n, seg);
            const ulong out_idx = (ulong)(m0 + mm) * (ulong)stacked_fc_y_row_cols(seg)
                + (ulong)stacked_fc_y_col0(seg) + (ulong)local_n;
            device ushort *dst = (device ushort *)((device uchar *)y_arena + stacked_fc_y_byte_off(seg));
            arena_store(dst, out_idx, ty[mm][nn]);
        }
    }
}

inline void gemm_block_stacked_nvfp4_impl(
    device const ushort *x,
    device ushort *y_arena,
    device const uchar *blob,
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

    const float gscale0 = as_type<float>(*(device const uint *)(blob + STACKED_W_OFF0));
    const float gscale1 = as_type<float>(*(device const uint *)(blob + STACKED_W_OFF1));
    const float gscale2 = as_type<float>(*(device const uint *)(blob + STACKED_W_OFF2));
    const ulong rowB = nvfp4_row_bytes(K);
    const device uchar *body0 = blob + STACKED_W_OFF0 + 4ul;
    const device uchar *body1 = blob + STACKED_W_OFF1 + 4ul;
    const device uchar *body2 = blob + STACKED_W_OFF2 + 4ul;

    for (uint k0 = 0u; k0 < K; k0 += GEMM_K_TILE) {
        gemm_load_a_tile(x, M, K, m0, k0, ltid, tx);
        if (ltid < GEMM_N_TILE) {
            const uint r = ltid;
            const uint global_n = n0 + r;
            if (global_n >= N) {
                gemm_block_zero_tw_row(tw, r);
            } else if (STACKED_N_SEGS == 1u) {
                device const uchar *row = body0 + (ulong)global_n * rowB;
                dequant_nvfp4_tile_half_fused_tg(row, K, k0, &tw[r][0], gscale0);
            } else if (STACKED_N_SEGS == 2u) {
                if (global_n < STACKED_END0) {
                    device const uchar *row = body0 + (ulong)global_n * rowB;
                    dequant_nvfp4_tile_half_fused_tg(row, K, k0, &tw[r][0], gscale0);
                } else {
                    const uint local_n = global_n - STACKED_END0;
                    device const uchar *row = body1 + (ulong)local_n * rowB;
                    dequant_nvfp4_tile_half_fused_tg(row, K, k0, &tw[r][0], gscale1);
                }
            } else {
                if (global_n < STACKED_END0) {
                    device const uchar *row = body0 + (ulong)global_n * rowB;
                    dequant_nvfp4_tile_half_fused_tg(row, K, k0, &tw[r][0], gscale0);
                } else if (global_n < STACKED_END1) {
                    const uint local_n = global_n - STACKED_END0;
                    device const uchar *row = body1 + (ulong)local_n * rowB;
                    dequant_nvfp4_tile_half_fused_tg(row, K, k0, &tw[r][0], gscale1);
                } else {
                    const uint local_n = global_n - STACKED_END1;
                    device const uchar *row = body2 + (ulong)local_n * rowB;
                    dequant_nvfp4_tile_half_fused_tg(row, K, k0, &tw[r][0], gscale2);
                }
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
            const uint seg = stacked_fc_seg_index(global_n);
            const uint local_n = stacked_fc_local_n(global_n, seg);
            const ulong out_idx = (ulong)(m0 + mm) * (ulong)stacked_fc_y_row_cols(seg)
                + (ulong)stacked_fc_y_col0(seg) + (ulong)local_n;
            device ushort *dst = (device ushort *)((device uchar *)y_arena + stacked_fc_y_byte_off(seg));
            arena_store(dst, out_idx, ty[mm][nn]);
        }
    }
}

/// Stacked GEMM: segment table is compile-time FC12–27 (per fused layout / layer).
/// `GEMM_N` = sum(segs[i].n_cols). K from `GEMM_K` (FC6).
kernel void gemm_block_stacked(
    device const ushort *x [[buffer(0)]],
    device ushort *y_arena [[buffer(1)]],
    device const uchar *blob [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]
) {
    const uint ltid = lid.x;
    threadgroup half tx[GEMM_M_TILE][GEMM_K_TILE];
    threadgroup half tw[GEMM_N_TILE_MAX][GEMM_K_TILE];
    threadgroup float ty[GEMM_M_TILE][GEMM_N_TILE_MAX];
    if (K_QUANT_FORMAT == QUANT_NVFP4) {
        gemm_block_stacked_nvfp4_impl(x, y_arena, blob, M, tgid, ltid, sgid, tx, tw, ty);
    } else {
        gemm_block_stacked_q4_impl(x, y_arena, blob, M, tgid, ltid, sgid, tx, tw, ty);
    }
}
