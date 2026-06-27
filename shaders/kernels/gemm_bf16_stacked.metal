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

// bf16 stacked GEMM: same fused N-segment layout as gemm_block_stacked (FC12-27
// segment table, output scatter via stacked_fc_*), but weights are bf16 [N,K]
// rows read straight into the half tile (no dequant) -- the bf16 analog of how
// gemm_bf16 derives from gemm_block. Used to fuse QKV / gate+up on the bf16
// default path (the q4 stacked path can't read bf16 weights). Bit-identical to
// the per-segment gemm_bf16 split (same per-column dot products + tiling).

/// Load one N-tile row of bf16 weights into tw_buf[slot][r] at K-tile k0,
/// segment-aware (global_n -> (segment, local_n) via the FC table).
inline void stacked_load_tw_bf16(
    device const uchar *blob,
    uint N, uint n0, uint ltid, uint k0,
    ulong w0, ulong w1, ulong w2,
    threadgroup half tw_buf[2u][GEMM_N_TILE_MAX][GEMM_K_TILE],
    uint slot
) {
    if (ltid >= GEMM_N_TILE) {
        return;
    }
    const uint r = ltid;
    const uint global_n = n0 + r;
    if (global_n >= N) {
        gemm_block_zero_tw_row(tw_buf[slot], r);
        return;
    }
    const ulong rowB = (ulong)GEMM_K * 2ul;  // bf16 row stride in bytes
    ulong wbase;
    uint local_n;
    if (STACKED_N_SEGS == 1u) {
        wbase = w0; local_n = global_n;
    } else if (STACKED_N_SEGS == 2u) {
        if (global_n < STACKED_END0) { wbase = w0; local_n = global_n; }
        else { wbase = w1; local_n = global_n - STACKED_END0; }
    } else {
        if (global_n < STACKED_END0) { wbase = w0; local_n = global_n; }
        else if (global_n < STACKED_END1) { wbase = w1; local_n = global_n - STACKED_END0; }
        else { wbase = w2; local_n = global_n - STACKED_END1; }
    }
    device const ushort *row =
        (device const ushort *)(blob + wbase + (ulong)local_n * rowB) + k0;
    for (uint kk = 0u; kk < GEMM_K_TILE; ++kk) {
        tw_buf[slot][r][kk] = half(bf16_to_f32(row[kk]));
    }
}

kernel void gemm_bf16_stacked(
    device const ushort *x [[buffer(0)]],
    device ushort *y_arena [[buffer(1)]],
    device const uchar *blob [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]
) {
    const uint N = GEMM_N;
    const uint K = GEMM_K;
    const uint m0 = tgid.y * GEMM_M_TILE;
    const uint n0 = tgid.x * GEMM_N_TILE;
    const uint ltid = lid.x;

    threadgroup half tx_buf[2u][GEMM_M_TILE][GEMM_K_TILE];
    threadgroup half tw_buf[2u][GEMM_N_TILE_MAX][GEMM_K_TILE];

    simdgroup_float8x8 acc0(0.f), acc1(0.f), acc2(0.f), acc3(0.f);
    simdgroup_float8x8 acc4(0.f), acc5(0.f), acc6(0.f), acc7(0.f);
    simdgroup_float8x8 acc8(0.f), acc9(0.f), acc10(0.f), acc11(0.f);
    simdgroup_float8x8 acc12(0.f), acc13(0.f), acc14(0.f), acc15(0.f);

    const ulong w0 = STACKED_W_OFF0;
    const ulong w1 = STACKED_W_OFF1;
    const ulong w2 = STACKED_W_OFF2;
    const uint n_k_tiles = (K + GEMM_K_TILE - 1u) / GEMM_K_TILE;

    // --- Prime: load tile 0 into buffer 0 ---
    {
        const uint k0 = 0u;
        gemm_load_a_tile(x, M, K, m0, k0, ltid, tx_buf[0u]);
        stacked_load_tw_bf16(blob, N, n0, ltid, k0, w0, w1, w2, tw_buf, 0u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Steady state: MMA on cur buffer, prefetch into nxt buffer ---
    for (uint ti = 0u; ti < n_k_tiles; ++ti) {
        const uint cur = ti & 1u;
        const uint nxt = 1u - cur;

        gemm_block_mma_k32(
            tx_buf[cur], tw_buf[cur], sgid, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7,
            acc8, acc9, acc10, acc11, acc12, acc13, acc14, acc15);

        if (ti + 1u < n_k_tiles) {
            const uint k0 = (ti + 1u) * GEMM_K_TILE;
            gemm_load_a_tile(x, M, K, m0, k0, ltid, tx_buf[nxt]);
            stacked_load_tw_bf16(blob, N, n0, ltid, k0, w0, w1, w2, tw_buf, nxt);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store: ty aliases over tw_buf (dead after MMA); scatter per segment.
    threadgroup float (*ty)[GEMM_N_TILE_MAX] = (threadgroup float (*)[GEMM_N_TILE_MAX]) tw_buf;
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
            device ushort *dst =
                (device ushort *)((device uchar *)y_arena + stacked_fc_y_byte_off(seg));
            arena_store(dst, out_idx, ty[mm][nn]);
        }
    }
}
