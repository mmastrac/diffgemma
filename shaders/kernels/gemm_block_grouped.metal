#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#include "gemm_fc.metal"
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

    const uint n0 = tgid.x * 32u;
    if (n0 >= N) {
        return;
    }

    const GroupedJob job = jobs[e];
    const ulong w_off = job.w_byte_off;

    threadgroup half tx[32][32];
    threadgroup half tw[32][32];
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

    for (uint m_base = 0u; m_base < M; m_base += 32u) {
        const uint M_tile = min(32u, M - m_base);
        const uint row0 = m0 + m_base;
        simdgroup_float8x8 acc0(0.f), acc1(0.f), acc2(0.f), acc3(0.f);

        for (uint k0 = 0u; k0 < K; k0 += 32u) {
            gemm_load_a_tile_f32(a, M_tile, K, row0, k0, ltid, tx);
            if (ltid < 32u) {
                const uint r = ltid;
                const uint global_n = n0 + r;
                if (global_n >= N) {
                    for (uint i = 0u; i < 32u; ++i) {
                        tw[r][i] = half(0);
                    }
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
            for (uint kk = 0u; kk < 32u; kk += 8u) {
                simdgroup_half8x8 a_tile, b0, b1, b2, b3;
                simdgroup_load(a_tile, &tx[8u * sgid][kk], 32);
                simdgroup_load(b0, &tw[0][kk], 32, ulong2(0, 0), true);
                simdgroup_load(b1, &tw[8][kk], 32, ulong2(0, 0), true);
                simdgroup_load(b2, &tw[16][kk], 32, ulong2(0, 0), true);
                simdgroup_load(b3, &tw[24][kk], 32, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc0, a_tile, b0, acc0);
                simdgroup_multiply_accumulate(acc1, a_tile, b1, acc1);
                simdgroup_multiply_accumulate(acc2, a_tile, b2, acc2);
                simdgroup_multiply_accumulate(acc3, a_tile, b3, acc3);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        threadgroup float ty[32][32];
        simdgroup_store(acc0, &ty[8u * sgid][0], 32);
        simdgroup_store(acc1, &ty[8u * sgid][8], 32);
        simdgroup_store(acc2, &ty[8u * sgid][16], 32);
        simdgroup_store(acc3, &ty[8u * sgid][24], 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = ltid; i < 32u * 32u; i += 128u) {
            const uint mm = i / 32u;
            const uint nn = i % 32u;
            if (mm < M_tile && n0 + nn < N) {
                c[(ulong)(m0 + m_base + mm) * N + n0 + nn] = f32_round_bf16(ty[mm][nn]);
            }
        }
    }
}
