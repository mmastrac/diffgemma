#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#include "gemm_fc.metal"
#include "gemm_block_tile.metal"
#include "dequant.metal"
#include "arena.metal"

/// Load one GEMM_K_TILE-wide bf16 weight K-tile for row `r` (one thread per row).
/// Weights are bf16 [N,K]; reading straight into the half tile is lossless.
inline void gemm_load_w_tile_bf16(
    device const ushort *w,
    uint N,
    uint K,
    uint n0,
    uint k0,
    uint r,
    threadgroup half tw[GEMM_N_TILE_MAX][GEMM_K_TILE]
) {
    if (n0 + r < N) {
        device const ushort *row = w + (ulong)(n0 + r) * K + k0;
        for (uint kk = 0u; kk < GEMM_K_TILE; ++kk) {
            tw[r][kk] = half(bf16_to_f32(row[kk]));
        }
    } else {
        gemm_block_zero_tw_row(tw, r);
    }
}

/// y[M,N] = x[M,K] @ W[N,K]^T with bf16 weights (no quantization). Same tiling as
/// gemm_block (double-buffered K-tile, GEMM_N_TILE-wide output, 16 accumulators)
/// but reads the weight tile straight from bf16 rows -- lossless into the half
/// tiles since the source weights are bf16. threadgroup (128,1,1).
kernel void gemm_bf16(
    device const ushort *x [[buffer(0)]],
    device ushort *y [[buffer(1)]],
    device const uchar *blob [[buffer(2)]],
    constant ulong &w_off [[buffer(3)]],
    constant uint &M [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]
) {
    const uint N = GEMM_N, K = GEMM_K;
    // Double-buffered K-tile storage: ping-pong so MMA on tile i overlaps with load of tile i+1.
    threadgroup half tx_buf[2u][GEMM_M_TILE][GEMM_K_TILE];
    threadgroup half tw_buf[2u][GEMM_N_TILE_MAX][GEMM_K_TILE];
    const uint m0 = tgid.y * GEMM_M_TILE;
    const uint n0 = tgid.x * GEMM_N_TILE;
    const uint ltid = lid.x;
    simdgroup_float8x8 acc0(0.f), acc1(0.f), acc2(0.f), acc3(0.f);
    simdgroup_float8x8 acc4(0.f), acc5(0.f), acc6(0.f), acc7(0.f);
    simdgroup_float8x8 acc8(0.f), acc9(0.f), acc10(0.f), acc11(0.f);
    simdgroup_float8x8 acc12(0.f), acc13(0.f), acc14(0.f), acc15(0.f);

    device const ushort *w = (device const ushort *)(blob + w_off);
    const uint n_k_tiles = (K + GEMM_K_TILE - 1u) / GEMM_K_TILE;

    // --- Prime: load tile 0 into buffer 0 ---
    {
        const uint k0 = 0u;
        gemm_load_a_tile(x, M, K, m0, k0, ltid, tx_buf[0u]);
        if (ltid < GEMM_N_TILE) {
            gemm_load_w_tile_bf16(w, N, K, n0, k0, ltid, tw_buf[0u]);
        }
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
            if (ltid < GEMM_N_TILE) {
                gemm_load_w_tile_bf16(w, N, K, n0, k0, ltid, tw_buf[nxt]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store accumulators to threadgroup f32 tile (aliased over tw_buf, dead after MMA),
    // then to global memory.
    threadgroup float (*ty)[GEMM_N_TILE_MAX] = (threadgroup float (*)[GEMM_N_TILE_MAX]) tw_buf;
    gemm_block_mma_store(
        ty, sgid, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11,
        acc12, acc13, acc14, acc15);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ltid; i < GEMM_M_TILE * GEMM_N_TILE; i += 128u) {
        const uint mm = i / GEMM_N_TILE;
        const uint nn = i % GEMM_N_TILE;
        if (m0 + mm < M && n0 + nn < N) {
            arena_store(y, (ulong)(m0 + mm) * N + n0 + nn, ty[mm][nn]);
        }
    }
}
