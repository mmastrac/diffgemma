// Tiled f32 GEMM. Mirrors gemm.metal.
// C = alpha * op(A) @ op(B) + beta * C

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

struct GemmParams {
    unsigned m, n, k;
    unsigned lda, ldb, ldc;
    float alpha, beta;
    unsigned trans_a, trans_b;
};

extern "C" __global__ void gemm_f32(
    const float *A,
    const float *B,
    float *C,
    GemmParams p
) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    unsigned tx = threadIdx.x & 15u;
    unsigned ty = threadIdx.x >> 4u;
    unsigned row0 = blockIdx.y * BM;
    unsigned col0 = blockIdx.x * BN;

    float acc[TM][TN];
    for (unsigned i = 0; i < TM; i++) {
        for (unsigned j = 0; j < TN; j++) {
            acc[i][j] = 0.0f;
        }
    }

    for (unsigned k0 = 0; k0 < p.k; k0 += BK) {
        for (unsigned i = 0; i < TM; i++) {
            unsigned m = row0 + ty * TM + i;
            for (unsigned kk = tx; kk < BK; kk += 16u) {
                unsigned k = k0 + kk;
                float v = 0.0f;
                if (m < p.m && k < p.k) {
                    v = p.trans_a != 0u ? A[(size_t)k * p.lda + m] : A[(size_t)m * p.lda + k];
                }
                As[(ty * TM + i) * BK + kk] = v;
            }
        }
        for (unsigned kk = 0; kk < BK; kk++) {
            unsigned k = k0 + kk;
            for (unsigned j = 0; j < TN; j++) {
                unsigned n = col0 + tx * TN + j;
                float v = 0.0f;
                if (k < p.k && n < p.n) {
                    v = p.trans_b != 0u ? B[(size_t)n * p.ldb + k] : B[(size_t)k * p.ldb + n];
                }
                Bs[kk * BN + tx * TN + j] = v;
            }
        }
        __syncthreads();

        for (unsigned kk = 0; kk < BK; kk++) {
            for (unsigned i = 0; i < TM; i++) {
                float a = As[(ty * TM + i) * BK + kk];
                for (unsigned j = 0; j < TN; j++) {
                    acc[i][j] += a * Bs[kk * BN + tx * TN + j];
                }
            }
        }
        __syncthreads();
    }

    for (unsigned i = 0; i < TM; i++) {
        unsigned m = row0 + ty * TM + i;
        if (m >= p.m) {
            continue;
        }
        for (unsigned j = 0; j < TN; j++) {
            unsigned n = col0 + tx * TN + j;
            if (n >= p.n) {
                continue;
            }
            size_t idx = (size_t)m * p.ldc + n;
            float prior = (p.beta == 0.0f) ? 0.0f : p.beta * C[idx];
            C[idx] = p.alpha * acc[i][j] + prior;
        }
    }
}
