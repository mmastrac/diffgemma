#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#include "gemm_fc.metal"
#include "dequant.metal"

/// y[M,N] = x[M,K] @ W[N,K]^T ; format from K_QUANT_FORMAT (FC3). Threadgroup (128,1,1).
kernel void gemm_block(
    device const half *x [[buffer(0)]],
    device half *y [[buffer(1)]],
    device const uchar *blob [[buffer(2)]],
    constant ulong &w_off [[buffer(3)]],
    constant uint &M [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]
) {
    const uint N = GEMM_N, K = GEMM_K;
    threadgroup half tx[32][32];
    threadgroup half tw[32][32];
    uint m0 = tgid.y * 32, n0 = tgid.x * 32;
    uint ltid = lid.x;
    simdgroup_float8x8 acc0(0.f), acc1(0.f), acc2(0.f), acc3(0.f);

    const bool is_nvfp4 = (K_QUANT_FORMAT == QUANT_NVFP4);
    float gscale = 0.f;
    ulong body = w_off;
    ulong rowB = 0ul;
    if (is_nvfp4) {
        gscale = as_type<float>(*(device const uint *)(blob + w_off));
        body = w_off + 4ul;
        rowB = nvfp4_row_bytes(K);
    } else {
        rowB = q4_row_bytes(K);
    }

    for (uint k0 = 0; k0 < K; k0 += 32) {
        for (uint i = ltid; i < 32 * 32; i += 128) {
            uint mm = i / 32, kk = i % 32;
            tx[mm][kk] = (m0 + mm < M) ? x[(ulong)(m0 + mm) * K + k0 + kk] : half(0);
        }
        for (uint r = ltid; r < 32; r += 128) {
            if (is_nvfp4) {
                device const uchar *row = blob + body + (ulong)(n0 + r) * rowB;
                dequant_nvfp4_tile_half_fused_tg(row, K, k0, &tw[r][0], gscale);
            } else {
                dequant_q4_group_half_tg(
                    blob + body + (ulong)(n0 + r) * rowB + (ulong)(k0 / 32) * 20ul,
                    &tw[r][0]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_half8x8 a, b0, b1, b2, b3;
            simdgroup_load(a, &tx[8 * sgid][kk], 32);
            simdgroup_load(b0, &tw[0][kk], 32, ulong2(0, 0), true);
            simdgroup_load(b1, &tw[8][kk], 32, ulong2(0, 0), true);
            simdgroup_load(b2, &tw[16][kk], 32, ulong2(0, 0), true);
            simdgroup_load(b3, &tw[24][kk], 32, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc0, a, b0, acc0);
            simdgroup_multiply_accumulate(acc1, a, b1, acc1);
            simdgroup_multiply_accumulate(acc2, a, b2, acc2);
            simdgroup_multiply_accumulate(acc3, a, b3, acc3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float ty[32][32];
    simdgroup_store(acc0, &ty[8 * sgid][0], 32);
    simdgroup_store(acc1, &ty[8 * sgid][8], 32);
    simdgroup_store(acc2, &ty[8 * sgid][16], 32);
    simdgroup_store(acc3, &ty[8 * sgid][24], 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ltid; i < 32 * 32; i += 128) {
        uint mm = i / 32, nn = i % 32;
        if (m0 + mm < M && n0 + nn < N) {
            y[(ulong)(m0 + mm) * N + n0 + nn] = half(ty[mm][nn]);
        }
    }
}
