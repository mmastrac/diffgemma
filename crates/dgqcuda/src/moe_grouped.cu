// Grouped MoE expert GEMM (q4 weights). Tokens are bucketed by expert on the
// host, so the flat row space is expert-major: row r of the grouped matrix
// belongs to the bucket containing r and to token tok_idx[r]. Each block
// computes a [BM x BN] tile of ONE expert, decoding the expert q4 weight tile
// once into shared memory and reusing it across all BM tokens — the reuse the
// per-token GEMV loop cannot get, and the reason the experts dominated the
// denoise step.
//
// Q4 layout (dgemm::format::layout): a row is groups of 32 weights, each group
// 4 bytes of (bf16 delta, bf16 min) followed by 16 bytes of packed nibbles,
// low nibble = even index. w[o][k] = delta * q + min.

#define MOE_BM 32
#define MOE_BN 64
#define MOE_BK 32
#define MOE_TM 4
#define MOE_TN 4
#define MOE_THREADS 128
#define Q4_GROUP 32

__device__ __forceinline__ float moe_bf16(unsigned short bits) {
    return __uint_as_float((unsigned)bits << 16u);
}

// Decode one q4 weight value: expert-local row row, input index k.
__device__ __forceinline__ float moe_q4_at(
    const unsigned char *w, unsigned row, unsigned k, unsigned row_bytes
) {
    const unsigned char *r = w + (size_t)row * row_bytes;
    const unsigned g = k / Q4_GROUP;
    const unsigned j = k - g * Q4_GROUP;
    const unsigned char *blk = r + g * (4u + Q4_GROUP / 2u);
    const float delta = moe_bf16((unsigned short)(blk[0] | ((unsigned)blk[1] << 8)));
    const float mn = moe_bf16((unsigned short)(blk[2] | ((unsigned)blk[3] << 8)));
    const unsigned char byte = blk[4u + j / 2u];
    const unsigned q = (j & 1u) ? (unsigned)(byte >> 4u) : (unsigned)(byte & 0x0fu);
    return delta * (float)q + mn;
}

// c[global_row, col] = sum_k a[tok_idx[global_row], k] * w_e[col, k]
extern "C" __global__ void dgq_moe_gate_up(
    const float *a, const unsigned char *w_blob, float *c,
    const unsigned *row_starts, const unsigned *tok_idx, const unsigned *experts,
    unsigned k_dim, unsigned n_dim, unsigned num_jobs, unsigned row_bytes
) {
    __shared__ float as[MOE_BM * MOE_BK];
    __shared__ float ws[MOE_BN * MOE_BK];
    const unsigned tx = threadIdx.x & 15u;
    const unsigned ty = threadIdx.x >> 4u;
    const unsigned col0 = blockIdx.x * MOE_BN;
    const unsigned row0 = blockIdx.y * MOE_BM;
    // A launch with no buckets must not read row_starts[0] as a row count.
    if (num_jobs == 0u) return;
    const unsigned row_end = row_starts[num_jobs];
    // One row block can straddle a bucket boundary, so walk the buckets it
    // overlaps instead of resolving a single job for the whole block.
    if (col0 >= n_dim || row0 >= row_end) return;

    float acc[MOE_TM][MOE_TN];
    for (unsigned i = 0u; i < MOE_TM; i++)
        for (unsigned j = 0u; j < MOE_TN; j++) acc[i][j] = 0.0f;

    // A is the bucketed row space (dgq_moe_gather wrote it expert-major), so a
    // block row IS a bucketed row. The A tile is therefore the same for every
    // job the block overlaps; only the weight tile changes. The accumulators
    // must be shared across those jobs and written once, or the last job would
    // overwrite the others.
    for (unsigned k0 = 0u; k0 < k_dim; k0 += MOE_BK) {
        for (unsigned i = 0u; i < MOE_TM; i++) {
            const unsigned r = row0 + ty * MOE_TM + i;
            for (unsigned kk = tx; kk < MOE_BK; kk += 16u) {
                const unsigned k = k0 + kk;
                as[(ty * MOE_TM + i) * MOE_BK + kk] =
                    (r < row_end && k < k_dim) ? a[(size_t)r * k_dim + k] : 0.0f;
            }
        }
        for (unsigned job = 0u; job < num_jobs; job++) {
            const unsigned job_start = row_starts[job];
            const unsigned job_end = row_starts[job + 1u];
            if (job_end <= row0) continue;
            if (job_start >= row0 + MOE_BM) break;
            const unsigned char *w =
                w_blob + (size_t)experts[job] * (size_t)n_dim * (size_t)row_bytes;
            for (unsigned kk = 0u; kk < MOE_BK; kk++) {
                const unsigned k = k0 + kk;
                for (unsigned j = 0u; j < MOE_TN; j++) {
                    const unsigned col = col0 + tx * MOE_TN + j;
                    ws[kk * MOE_BN + tx * MOE_TN + j] =
                        (col < n_dim && k < k_dim) ? moe_q4_at(w, col, k, row_bytes) : 0.0f;
                }
            }
            __syncthreads();
            for (unsigned kk = 0u; kk < MOE_BK; kk++) {
                for (unsigned i = 0u; i < MOE_TM; i++) {
                    const unsigned r = row0 + ty * MOE_TM + i;
                    if (r >= job_end || r < job_start) continue;
                    const float av = as[(ty * MOE_TM + i) * MOE_BK + kk];
                    for (unsigned j = 0u; j < MOE_TN; j++) {
                        acc[i][j] += av * ws[kk * MOE_BN + tx * MOE_TN + j];
                    }
                }
            }
            __syncthreads();
        }
    }

    for (unsigned i = 0u; i < MOE_TM; i++) {
        const unsigned r = row0 + ty * MOE_TM + i;
        if (r >= row_end) continue;
        for (unsigned j = 0u; j < MOE_TN; j++) {
            const unsigned col = col0 + tx * MOE_TN + j;
            if (col < n_dim) c[(size_t)r * n_dim + col] = acc[i][j];
        }
    }
}

// Down projection: same shape, expert weight rows are row_bytes apart and
// the k dimension is the expert intermediate width.
extern "C" __global__ void dgq_moe_down(
    const float *a, const unsigned char *w_blob, float *c,
    const unsigned *row_starts, const unsigned *tok_idx, const unsigned *experts,
    unsigned k_dim, unsigned n_dim, unsigned num_jobs, unsigned row_bytes
) {
    __shared__ float as[MOE_BM * MOE_BK];
    __shared__ float ws[MOE_BN * MOE_BK];
    const unsigned tx = threadIdx.x & 15u;
    const unsigned ty = threadIdx.x >> 4u;
    const unsigned col0 = blockIdx.x * MOE_BN;
    const unsigned row0 = blockIdx.y * MOE_BM;
    if (num_jobs == 0u) return;
    const unsigned row_end = row_starts[num_jobs];
    if (col0 >= n_dim || row0 >= row_end) return;

    float acc[MOE_TM][MOE_TN];
    for (unsigned i = 0u; i < MOE_TM; i++)
        for (unsigned j = 0u; j < MOE_TN; j++) acc[i][j] = 0.0f;

    // Down takes the expert-major activation rows directly, so the A tile is
    // the block rows themselves and only the weight tile depends on the job.
    for (unsigned k0 = 0u; k0 < k_dim; k0 += MOE_BK) {
        for (unsigned i = 0u; i < MOE_TM; i++) {
            const unsigned r = row0 + ty * MOE_TM + i;
            for (unsigned kk = tx; kk < MOE_BK; kk += 16u) {
                const unsigned k = k0 + kk;
                as[(ty * MOE_TM + i) * MOE_BK + kk] =
                    (r < row_end && k < k_dim) ? a[(size_t)r * k_dim + k] : 0.0f;
            }
        }
        for (unsigned job = 0u; job < num_jobs; job++) {
            const unsigned job_start = row_starts[job];
            const unsigned job_end = row_starts[job + 1u];
            if (job_end <= row0) continue;
            if (job_start >= row0 + MOE_BM) break;
            const unsigned char *w =
                w_blob + (size_t)experts[job] * (size_t)n_dim * (size_t)row_bytes;
            for (unsigned kk = 0u; kk < MOE_BK; kk++) {
                const unsigned k = k0 + kk;
                for (unsigned j = 0u; j < MOE_TN; j++) {
                    const unsigned col = col0 + tx * MOE_TN + j;
                    ws[kk * MOE_BN + tx * MOE_TN + j] =
                        (col < n_dim && k < k_dim) ? moe_q4_at(w, col, k, row_bytes) : 0.0f;
                }
            }
            __syncthreads();
            for (unsigned kk = 0u; kk < MOE_BK; kk++) {
                for (unsigned i = 0u; i < MOE_TM; i++) {
                    const unsigned r = row0 + ty * MOE_TM + i;
                    if (r >= job_end || r < job_start) continue;
                    const float av = as[(ty * MOE_TM + i) * MOE_BK + kk];
                    for (unsigned j = 0u; j < MOE_TN; j++) {
                        acc[i][j] += av * ws[kk * MOE_BN + tx * MOE_TN + j];
                    }
                }
            }
            __syncthreads();
        }
    }

    for (unsigned i = 0u; i < MOE_TM; i++) {
        const unsigned r = row0 + ty * MOE_TM + i;
        if (r >= row_end) continue;
        for (unsigned j = 0u; j < MOE_TN; j++) {
            const unsigned col = col0 + tx * MOE_TN + j;
            if (col < n_dim) c[(size_t)r * n_dim + col] = acc[i][j];
        }
    }

}

extern "C" __global__ void dgq_moe_swiglu_weighted(
    const float *gu, float *act, const float *row_w,
    unsigned row_start, unsigned rows, unsigned inter
) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned n = rows * inter;
    if (i >= n) return;
    const unsigned r = i / inter;
    const unsigned c = i - r * inter;
    const unsigned grow = row_start + r;
    const float g = gu[(size_t)grow * (size_t)(inter * 2u) + c];
    const float u = gu[(size_t)grow * (size_t)(inter * 2u) + inter + c];
    const float x3 = g * g * g;
    const float uu = 0.7978846f * (g + 0.044715f * x3);
    const float t = (uu > 8.0f) ? 1.0f : (uu < -8.0f) ? -1.0f : tanhf(uu);
    const float gelu = 0.5f * g * (1.0f + t);
    act[i] = gelu * u * row_w[grow];
    if (i == 0u) {
    }
}

// dst[token, :] += src[row, :] for every expert-major row (the routing weight
// is already folded in by dgq_moe_swiglu_weighted).
extern "C" __global__ void dgq_moe_scatter(
    float *dst, const float *src, const unsigned *tok_idx,
    unsigned row_start, unsigned rows, unsigned hidden
) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned n = rows * hidden;
    if (i >= n) return;
    const unsigned r = i / hidden;
    const unsigned d = i - r * hidden;
    const unsigned grow = row_start + r;
    atomicAdd(&dst[(size_t)tok_idx[grow] * hidden + d], src[i]);
}
// Gather the bucketed input rows: rows_a[r, :] = x[tok_idx[r], :].
extern "C" __global__ void dgq_moe_gather(
    const float *x, float *rows_a, const unsigned *tok_idx,
    unsigned rows, unsigned hidden
) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned n = rows * hidden;
    if (i >= n) return;
    const unsigned r = i / hidden;
    const unsigned d = i - r * hidden;
    rows_a[i] = x[(size_t)tok_idx[r] * hidden + d];
}
