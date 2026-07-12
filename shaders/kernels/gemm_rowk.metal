#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#include "gemm_fc.metal"
#include "dequant.metal"
#include "arena.metal"

/// Row-K GEMM: y[M,N] = x[M,K] @ W[K,N], weight rows indexed by K (vocab /
/// SC softembed). Generic over BOTH axes tunable-style:
///   K_QUANT_FORMAT (FC3)  = weight format: QUANT_Q8 (dequant) or QUANT_RAW (bf16).
///   K_X_FP16       (FC10) = activation input dtype: fp16 vs bf16 arena.
///   K_ROWK_OUT_ARENA (FC30) = output/fusion mode:
///       false → y is `float[M,N]`, ACCUMULATE (`y += ...`), tied-embed lm_head.
///       true  → y is arena `ushort[M,N]`, OVERWRITE (`arena_store`), SC softembed.
constant bool K_X_FP16 [[function_constant(10)]];
constant bool K_ROWK_OUT_ARENA_DEF [[function_constant(30)]];
constant bool K_ROWK_OUT_ARENA =
    is_function_constant_defined(K_ROWK_OUT_ARENA_DEF) ? K_ROWK_OUT_ARENA_DEF : false;

kernel void gemm_rowk(
    device const ushort *x [[buffer(0)]],
    device uchar *y [[buffer(1)]],
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
    device const ushort *w = (device const ushort *)(blob + w_off);
    for (uint k0 = 0; k0 < K; k0 += 32) {
        for (uint i = ltid; i < 32 * 32; i += 128) {
            uint mm = i / 32, kk = i % 32;
            float xv = (m0 + mm < M)
                ? (K_X_FP16
                    ? float(((device const half *)x)[(ulong)(m0 + mm) * K + k0 + kk])
                    : arena_load(x, (ulong)(m0 + mm) * K + k0 + kk))
                : 0.f;
            tx[mm][kk] = half(xv);
        }
        for (uint i = ltid; i < 32 * 32; i += 128) {
            uint nn = i / 32, kk = i % 32;
            if (K_QUANT_FORMAT == QUANT_Q8) {
                device const uchar *rb = blob + w_off + (ulong)(k0 + kk) * q8_row_bytes(N);
                tw[nn][kk] = half(q8_at(rb, n0 + nn, bf16_bytes(rb)));
            } else {  // QUANT_RAW: bf16 [K,N] weights, no dequant
                tw[nn][kk] = half(bf16_to_f32(w[(ulong)(k0 + kk) * N + n0 + nn]));
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
            const ulong idx = (ulong)(m0 + mm) * N + n0 + nn;
            if (K_ROWK_OUT_ARENA) {
                arena_store((device ushort *)y, idx, ty[mm][nn]);
            } else {
                ((device float *)y)[idx] += ty[mm][nn];
            }
        }
    }
}
