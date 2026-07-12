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

/// Dequant one N-tile row into tw_buf slot `slot` at K-tile `k0`.
/// Shared by q4/nvfp4/raw(bf16) stacked variants.
inline void stacked_load_tw(
    device const uchar *blob,
    uint N, uint n0, uint ltid, uint k0,
    ulong body0, ulong body1, ulong body2,
    float gscale0, float gscale1, float gscale2,
    ulong rowB, bool is_nvfp4, bool is_raw,
    threadgroup half tw_buf[2u][GEMM_N_TILE_MAX][GEMM_K_TILE],
    uint slot
) {
    if (ltid >= GEMM_N_TILE) return;
    const uint r = ltid;
    const uint global_n = n0 + r;
    if (global_n >= N) {
        gemm_block_zero_tw_row(tw_buf[slot], r);
        return;
    }
    if (is_raw) {
        // bf16 [N,K] weights read straight into the half tile (no dequant).
        // Resolve segment -> (wbase, local_n); rowB = K*2, body{0,1,2} = seg w_off.
        ulong wbase;
        uint local_n;
        if (STACKED_N_SEGS == 1u) {
            wbase = body0; local_n = global_n;
        } else if (STACKED_N_SEGS == 2u) {
            if (global_n < STACKED_END0) { wbase = body0; local_n = global_n; }
            else { wbase = body1; local_n = global_n - STACKED_END0; }
        } else {
            if (global_n < STACKED_END0) { wbase = body0; local_n = global_n; }
            else if (global_n < STACKED_END1) { wbase = body1; local_n = global_n - STACKED_END0; }
            else { wbase = body2; local_n = global_n - STACKED_END1; }
        }
        device const ushort *row =
            (device const ushort *)(blob + wbase + (ulong)local_n * rowB) + k0;
        for (uint kk = 0u; kk < GEMM_K_TILE; ++kk) {
            tw_buf[slot][r][kk] = half(bf16_to_f32(row[kk]));
        }
    } else if (is_nvfp4) {
        if (STACKED_N_SEGS == 1u) {
            device const uchar *row = blob + body0 + (ulong)global_n * rowB;
            dequant_nvfp4_tile_half_fused_tg(row, GEMM_K, k0, &tw_buf[slot][r][0], gscale0);
        } else if (STACKED_N_SEGS == 2u) {
            if (global_n < STACKED_END0) {
                device const uchar *row = blob + body0 + (ulong)global_n * rowB;
                dequant_nvfp4_tile_half_fused_tg(row, GEMM_K, k0, &tw_buf[slot][r][0], gscale0);
            } else {
                const uint local_n = global_n - STACKED_END0;
                device const uchar *row = blob + body1 + (ulong)local_n * rowB;
                dequant_nvfp4_tile_half_fused_tg(row, GEMM_K, k0, &tw_buf[slot][r][0], gscale1);
            }
        } else {
            if (global_n < STACKED_END0) {
                device const uchar *row = blob + body0 + (ulong)global_n * rowB;
                dequant_nvfp4_tile_half_fused_tg(row, GEMM_K, k0, &tw_buf[slot][r][0], gscale0);
            } else if (global_n < STACKED_END1) {
                const uint local_n = global_n - STACKED_END0;
                device const uchar *row = blob + body1 + (ulong)local_n * rowB;
                dequant_nvfp4_tile_half_fused_tg(row, GEMM_K, k0, &tw_buf[slot][r][0], gscale1);
            } else {
                const uint local_n = global_n - STACKED_END1;
                device const uchar *row = blob + body2 + (ulong)local_n * rowB;
                dequant_nvfp4_tile_half_fused_tg(row, GEMM_K, k0, &tw_buf[slot][r][0], gscale2);
            }
        }
    } else {
        if (STACKED_N_SEGS == 1u) {
            dequant_q4_group_half_tg(
                blob + body0 + (ulong)global_n * rowB + (ulong)(k0 / 32u) * 20ul,
                &tw_buf[slot][r][0]);
        } else if (STACKED_N_SEGS == 2u) {
            if (global_n < STACKED_END0) {
                dequant_q4_group_half_tg(
                    blob + body0 + (ulong)global_n * rowB + (ulong)(k0 / 32u) * 20ul,
                    &tw_buf[slot][r][0]);
            } else {
                const uint local_n = global_n - STACKED_END0;
                dequant_q4_group_half_tg(
                    blob + body1 + (ulong)local_n * rowB + (ulong)(k0 / 32u) * 20ul,
                    &tw_buf[slot][r][0]);
            }
        } else {
            if (global_n < STACKED_END0) {
                dequant_q4_group_half_tg(
                    blob + body0 + (ulong)global_n * rowB + (ulong)(k0 / 32u) * 20ul,
                    &tw_buf[slot][r][0]);
            } else if (global_n < STACKED_END1) {
                const uint local_n = global_n - STACKED_END0;
                dequant_q4_group_half_tg(
                    blob + body1 + (ulong)local_n * rowB + (ulong)(k0 / 32u) * 20ul,
                    &tw_buf[slot][r][0]);
            } else {
                const uint local_n = global_n - STACKED_END1;
                dequant_q4_group_half_tg(
                    blob + body2 + (ulong)local_n * rowB + (ulong)(k0 / 32u) * 20ul,
                    &tw_buf[slot][r][0]);
            }
        }
    }
}

inline void gemm_block_stacked_impl(
    device const ushort *x,
    device ushort *y_arena,
    device const uchar *blob,
    uint M,
    uint3 tgid,
    uint ltid,
    uint sgid,
    threadgroup half tx_buf[2u][GEMM_M_TILE][GEMM_K_TILE],
    threadgroup half tw_buf[2u][GEMM_N_TILE_MAX][GEMM_K_TILE]
) {
    const uint N = GEMM_N;
    const uint K = GEMM_K;
    const uint m0 = tgid.y * GEMM_M_TILE;
    const uint n0 = tgid.x * GEMM_N_TILE;
    simdgroup_float8x8 acc0(0.f), acc1(0.f), acc2(0.f), acc3(0.f);
    simdgroup_float8x8 acc4(0.f), acc5(0.f), acc6(0.f), acc7(0.f);
    simdgroup_float8x8 acc8(0.f), acc9(0.f), acc10(0.f), acc11(0.f);
    simdgroup_float8x8 acc12(0.f), acc13(0.f), acc14(0.f), acc15(0.f);

    const bool is_nvfp4 = (K_QUANT_FORMAT == QUANT_NVFP4);
    const bool is_raw = (K_QUANT_FORMAT == QUANT_RAW);  // bf16 [N,K], no dequant
    ulong rowB = 0ul;
    if (is_nvfp4) {
        rowB = nvfp4_row_bytes(K);
    } else if (is_raw) {
        rowB = (ulong)K * 2ul;  // bf16 row stride in bytes
    } else {
        rowB = q4_row_bytes(K);
    }
    // Body offsets (skip 4-byte gscale header for nvfp4 segments; raw/q4 have none).
    const ulong body0 = STACKED_W_OFF0 + (is_nvfp4 ? 4ul : 0ul);
    const ulong body1 = STACKED_W_OFF1 + (is_nvfp4 ? 4ul : 0ul);
    const ulong body2 = STACKED_W_OFF2 + (is_nvfp4 ? 4ul : 0ul);
    const float gscale0 = as_type<float>(*(device const uint *)(blob + STACKED_W_OFF0));
    const float gscale1 = as_type<float>(*(device const uint *)(blob + STACKED_W_OFF1));
    const float gscale2 = as_type<float>(*(device const uint *)(blob + STACKED_W_OFF2));

    const uint n_k_tiles = (K + GEMM_K_TILE - 1u) / GEMM_K_TILE;

    // --- Prime: load tile 0 into buffer 0 ---
    {
        const uint k0 = 0u;
        gemm_load_a_tile(x, M, K, m0, k0, ltid, tx_buf[0u]);
        stacked_load_tw(
            blob, N, n0, ltid, k0,
            body0, body1, body2, gscale0, gscale1, gscale2,
            rowB, is_nvfp4, is_raw, tw_buf, 0u);
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
            stacked_load_tw(
                blob, N, n0, ltid, k0,
                body0, body1, body2, gscale0, gscale1, gscale2,
                rowB, is_nvfp4, is_raw, tw_buf, nxt);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store: ty aliases over tw_buf (dead after MMA).
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
            device ushort *dst = (device ushort *)((device uchar *)y_arena + stacked_fc_y_byte_off(seg));
            arena_store(dst, out_idx, ty[mm][nn]);
        }
    }
}

/// Stacked GEMM: segment table is compile-time FC12–27 (per fused layout / layer).
/// `GEMM_N` = sum(segs[i].n_cols). K from `GEMM_K` (FC6).
///
/// Double-buffered K-tile: MMA on tile i overlaps with dequant of tile i+1.
/// One barrier per K-tile (vs two in the single-buffered variant).
/// ty store tile aliases over tw_buf (dead after MMA) to fit 32 KB TG memory.
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
    threadgroup half tx_buf[2u][GEMM_M_TILE][GEMM_K_TILE];
    threadgroup half tw_buf[2u][GEMM_N_TILE_MAX][GEMM_K_TILE];
    gemm_block_stacked_impl(x, y_arena, blob, M, tgid, ltid, sgid, tx_buf, tw_buf);
}
