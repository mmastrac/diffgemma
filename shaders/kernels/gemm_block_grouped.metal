#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#include "gemm_fc.metal"
#include "gemm_block_tile.metal"
#include "dequant.metal"
#include "qgemm_grouped.metal"

/// Tiled grouped MoE GEMM: one expert segment per threadgroup row.
/// `C[m,N] = A[m,K] @ W[job]^T` for rows `m in [row_starts[e], row_starts[e+1])`.
/// Format from K_QUANT_FORMAT (FC3). N/K from GEMM_N/GEMM_K (FC5–6).
kernel void gemm_block_grouped(
    device const float *a [[buffer(0)]],
    device const uchar *blob [[buffer(1)]],
    device float *c [[buffer(2)]],
    device const GroupedJob *jobs [[buffer(3)]],
    device const uint *row_starts [[buffer(4)]],
    constant uint &num_jobs [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]
) {
    const uint N = GEMM_N;
    const uint K = GEMM_K;
    const uint e = tgid.y;
    if (e >= num_jobs) {
        return;
    }

    // Tier-2: distinguish "expert has 0 tokens" from "bucketing never ran".
    // row_starts[num_jobs] is total assignments; must be >0 when canvas has tokens.
    if (K_SHAPE_ASSERT && e == 0u && tgid.x == 0u && lid.x == 0u && row_starts[num_jobs] == 0u) {
        c[0] = as_type<float>(0x7fc00000u);
        return;
    }

    const uint m0 = row_starts[e];
    const uint m1 = row_starts[e + 1u];
    const uint M = m1 - m0;
    if (M == 0u) {
        return;
    }

    const uint n0 = tgid.x * GEMM_N_TILE;
    if (n0 >= N) {
        return;
    }

    const GroupedJob job = jobs[e];
    const ulong w_off = job.w_byte_off;

    threadgroup half tx[GEMM_M_TILE][GEMM_K_TILE];
    threadgroup half tw[GEMM_N_TILE_MAX][GEMM_K_TILE];
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

    for (uint m_base = 0u; m_base < M; m_base += GEMM_M_TILE) {
        const uint M_tile = min(GEMM_M_TILE, M - m_base);
        const uint row0 = m0 + m_base;
        simdgroup_float8x8 acc0(0.f), acc1(0.f), acc2(0.f), acc3(0.f);
        simdgroup_float8x8 acc4(0.f), acc5(0.f), acc6(0.f), acc7(0.f);
        simdgroup_float8x8 acc8(0.f), acc9(0.f), acc10(0.f), acc11(0.f);
        simdgroup_float8x8 acc12(0.f), acc13(0.f), acc14(0.f), acc15(0.f);

        for (uint k0 = 0u; k0 < K; k0 += GEMM_K_TILE) {
            gemm_load_a_tile_f32(a, M_tile, K, row0, k0, ltid, tx);
            if (ltid < GEMM_N_TILE) {
                const uint r = ltid;
                const uint global_n = n0 + r;
                if (global_n >= N) {
                    gemm_block_zero_tw_row(tw, r);
                } else if (is_nvfp4) {
                    device const uchar *row = blob + body + (ulong)global_n * rowB;
                    const float gscale = as_type<float>(*(device const uint *)(blob + w_off));
                    dequant_nvfp4_tile_half_fused_tg(row, K, k0, &tw[r][0], gscale);
                } else {
                    dequant_q4_group_half_tg(
                        blob + body + (ulong)global_n * rowB + (ulong)(k0 / 32u) * 20ul,
                        &tw[r][0]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            gemm_block_mma_k32(
                tx, tw, sgid, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10,
                acc11, acc12, acc13, acc14, acc15);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        threadgroup float ty[GEMM_M_TILE][GEMM_N_TILE_MAX];
        gemm_block_mma_store(
            ty, sgid, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11,
            acc12, acc13, acc14, acc15);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = ltid; i < GEMM_M_TILE * GEMM_N_TILE; i += 128u) {
            const uint mm = i / GEMM_N_TILE;
            const uint nn = i % GEMM_N_TILE;
            if (mm < M_tile && n0 + nn < N) {
                c[(ulong)(m0 + m_base + mm) * N + n0 + nn] = f32_round_bf16(ty[mm][nn]);
            }
        }
    }
}
