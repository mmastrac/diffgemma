#include <metal_stdlib>
using namespace metal;

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

struct GemmParams {
    uint m, n, k;
    uint lda, ldb, ldc;
    float alpha, beta;
    uint trans_a, trans_b;
};

/// C = alpha * op(A) @ op(B) + beta * C, one 64x64 block tile per threadgroup.
kernel void gemm_f32(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant GemmParams &p [[buffer(3)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    threadgroup float As[BM * BK];
    threadgroup float Bs[BK * BN];

    uint tx = lid & 15u;
    uint ty = lid >> 4u;
    uint row0 = tgp.y * BM;
    uint col0 = tgp.x * BN;

    float acc[TM][TN];
    for (uint i = 0; i < TM; i++) {
        for (uint j = 0; j < TN; j++) {
            acc[i][j] = 0.0f;
        }
    }

    for (uint k0 = 0; k0 < p.k; k0 += BK) {
        for (uint i = 0; i < TM; i++) {
            uint m = row0 + ty * TM + i;
            for (uint kk = tx; kk < BK; kk += 16u) {
                uint k = k0 + kk;
                float v = 0.0f;
                if (m < p.m && k < p.k) {
                    v = p.trans_a != 0u ? A[(ulong)k * p.lda + m] : A[(ulong)m * p.lda + k];
                }
                As[(ty * TM + i) * BK + kk] = v;
            }
        }
        for (uint kk = 0; kk < BK; kk++) {
            uint k = k0 + kk;
            for (uint j = 0; j < TN; j++) {
                uint n = col0 + tx * TN + j;
                float v = 0.0f;
                if (k < p.k && n < p.n) {
                    v = p.trans_b != 0u ? B[(ulong)n * p.ldb + k] : B[(ulong)k * p.ldb + n];
                }
                Bs[kk * BN + tx * TN + j] = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < BK; kk++) {
            for (uint i = 0; i < TM; i++) {
                float a = As[(ty * TM + i) * BK + kk];
                for (uint j = 0; j < TN; j++) {
                    acc[i][j] += a * Bs[kk * BN + tx * TN + j];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < TM; i++) {
        uint m = row0 + ty * TM + i;
        if (m >= p.m) {
            continue;
        }
        for (uint j = 0; j < TN; j++) {
            uint n = col0 + tx * TN + j;
            if (n >= p.n) {
                continue;
            }
            ulong idx = (ulong)m * p.ldc + n;
            float prior = (p.beta == 0.0f) ? 0.0f : p.beta * C[idx];
            C[idx] = p.alpha * acc[i][j] + prior;
        }
    }
}
