#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#include "gemm_fc.metal"
#include "gemm_block_tile.metal"
#include "dequant.metal"
#include "qgemm_grouped.metal"
#include "moe_router_device.metal"

/// Block-sparse grouped MoE GEMM (megablocks-style ragged GEMM).
///
/// Identical math to `gemm_block_grouped`, but the ragged M dimension is
/// pre-tiled into fixed <=32-row blocks (`route->block_expert/block_row0`,
/// built in `moe_bucket_fill` phase 1). Grid is `(N_tiles, num_blocks)`:
/// each threadgroup computes exactly ONE 32-row tile of ONE expert, so there is
/// no per-expert `m_base` loop and every block is an independent, uniform unit
/// of work — better load balance / scheduling than one variable-length
/// threadgroup per expert. Result is bit-identical to the grouped kernel.
kernel void gemm_block_sparse(
    device const float *a [[buffer(0)]],
    device const uchar *blob [[buffer(1)]],
    device float *c [[buffer(2)]],
    device const GroupedJob *jobs [[buffer(3)]],
    device const uint *row_starts [[buffer(4)]],
    constant uint &num_jobs [[buffer(5)]],
    device const RouteScratch *route [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]
) {
    const uint N = GEMM_N;
    const uint K = GEMM_K;

    const uint b = tgid.y;
    if (b >= route->num_blocks) {
        return;
    }
    const uint e = route->block_expert[b];
    if (e >= num_jobs) {
        return;
    }

    const uint row0 = route->block_row0[b];
    const uint m1 = row_starts[e + 1u];
    const uint M_tile = min(GEMM_M_TILE, m1 - row0);
    if (M_tile == 0u) {
        return;
    }

    const uint n0 = tgid.x * GEMM_N_TILE;
    if (n0 >= N) {
        return;
    }

    const GroupedJob job = jobs[e];
    const ulong w_off = job.w_byte_off;

    threadgroup half tx_buf[2u][GEMM_M_TILE][GEMM_K_TILE];
    threadgroup half tw_buf[2u][GEMM_N_TILE_MAX][GEMM_K_TILE];
    const uint ltid = lid.x;

    const bool is_nvfp4 = (K_QUANT_FORMAT == QUANT_NVFP4);
    ulong body = w_off;
    ulong rowB = 0ul;
    if (is_nvfp4) {
        body = w_off + 4ul;
        rowB = nvfp4_row_bytes(K);
    } else {
        rowB = q4_row_bytes(K);
    }

    simdgroup_float8x8 acc0(0.f), acc1(0.f), acc2(0.f), acc3(0.f);
    simdgroup_float8x8 acc4(0.f), acc5(0.f), acc6(0.f), acc7(0.f);
    simdgroup_float8x8 acc8(0.f), acc9(0.f), acc10(0.f), acc11(0.f);
    simdgroup_float8x8 acc12(0.f), acc13(0.f), acc14(0.f), acc15(0.f);

    const uint n_k_tiles = (K + GEMM_K_TILE - 1u) / GEMM_K_TILE;

    // --- Prime: load K-tile 0 into buffer 0 ---
    {
        const uint k0 = 0u;
        gemm_load_a_tile_f32(a, M_tile, K, row0, k0, ltid, tx_buf[0u]);
        if (ltid < GEMM_N_TILE) {
            const uint r = ltid;
            const uint global_n = n0 + r;
            if (global_n >= N) {
                gemm_block_zero_tw_row(tw_buf[0u], r);
            } else if (is_nvfp4) {
                device const uchar *row = blob + body + (ulong)global_n * rowB;
                const float gscale = as_type<float>(*(device const uint *)(blob + w_off));
                dequant_nvfp4_tile_half_fused_tg(row, K, k0, &tw_buf[0u][r][0], gscale);
            } else {
                dequant_q4_group_half_tg(
                    blob + body + (ulong)global_n * rowB + (ulong)(k0 / 32u) * 20ul,
                    &tw_buf[0u][r][0]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Steady state: MMA on cur buffer, prefetch into nxt buffer ---
    for (uint ti = 0u; ti < n_k_tiles; ++ti) {
        const uint cur = ti & 1u;
        const uint nxt = 1u - cur;

        gemm_block_mma_k32(
            tx_buf[cur], tw_buf[cur], sgid,
            acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7,
            acc8, acc9, acc10, acc11, acc12, acc13, acc14, acc15);

        if (ti + 1u < n_k_tiles) {
            const uint k0 = (ti + 1u) * GEMM_K_TILE;
            gemm_load_a_tile_f32(a, M_tile, K, row0, k0, ltid, tx_buf[nxt]);
            if (ltid < GEMM_N_TILE) {
                const uint r = ltid;
                const uint global_n = n0 + r;
                if (global_n >= N) {
                    gemm_block_zero_tw_row(tw_buf[nxt], r);
                } else if (is_nvfp4) {
                    device const uchar *row = blob + body + (ulong)global_n * rowB;
                    const float gscale = as_type<float>(*(device const uint *)(blob + w_off));
                    dequant_nvfp4_tile_half_fused_tg(row, K, k0, &tw_buf[nxt][r][0], gscale);
                } else {
                    dequant_q4_group_half_tg(
                        blob + body + (ulong)global_n * rowB + (ulong)(k0 / 32u) * 20ul,
                        &tw_buf[nxt][r][0]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float (*ty)[GEMM_N_TILE_MAX] = (threadgroup float (*)[GEMM_N_TILE_MAX]) tw_buf;
    gemm_block_mma_store(
        ty, sgid, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11,
        acc12, acc13, acc14, acc15);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ltid; i < GEMM_M_TILE * GEMM_N_TILE; i += 128u) {
        const uint mm = i / GEMM_N_TILE;
        const uint nn = i % GEMM_N_TILE;
        if (mm < M_tile && n0 + nn < N) {
            c[(ulong)(row0 + mm) * N + n0 + nn] = f32_round_bf16(ty[mm][nn]);
        }
    }
}
