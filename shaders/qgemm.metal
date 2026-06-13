#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

inline float bf16_bits_to_f32(uint bits) {
    return as_type<float>(bits << 16);
}

/// Q4_1 row layout: groups of 32 along K; each group = fp16 scale + fp16 min + 16 nibbles.
inline float q4_weight_at(
    device const uchar *base,
    uint row,
    uint col,
    uint in_dim,
    uint groups_per_row,
    uint row_stride
) {
    uint g = col / 32u;
    uint j = col % 32u;
    device const uchar *blk = base + row * row_stride + g * 20u;
    float delta = bf16_bits_to_f32(uint(blk[0]) | (uint(blk[1]) << 8));
    float mn = bf16_bits_to_f32(uint(blk[2]) | (uint(blk[3]) << 8));
    uchar byte = blk[4u + j / 2u];
    float q = (j & 1u) ? float(byte >> 4) : float(byte & 0x0fu);
    return delta * q + mn;
}

/// Dot one Q4 group (32 K elements) against activations.
inline float dot_q4_group(
    device const float *a_row,
    device const uchar *blk,
    uint base_k
) {
    float delta = bf16_bits_to_f32(uint(blk[0]) | (uint(blk[1]) << 8));
    float mn = bf16_bits_to_f32(uint(blk[2]) | (uint(blk[3]) << 8));
    float sum = 0.0f;
    for (uint j = 0; j < 16u; j++) {
        uchar byte = blk[4u + j];
        float q0 = float(byte & 0x0fu);
        float q1 = float(byte >> 4);
        sum = fma(a_row[base_k + j * 2u], delta * q0 + mn, sum);
        sum = fma(a_row[base_k + j * 2u + 1u], delta * q1 + mn, sum);
    }
    return sum;
}

struct Q4GroupedJob {
    uint w_byte_off;
    uint groups_per_row;
};

/// Grouped MoE Q4 GEMM: flattened `[total_m,K] @ expert weights^T` in one dispatch.
kernel void f32_q4_linear_grouped(
    device const float *a [[buffer(0)]],
    device const uchar *w_blob [[buffer(1)]],
    device float *c [[buffer(2)]],
    device const Q4GroupedJob *jobs [[buffer(3)]],
    device const uint *row_starts [[buffer(4)]],
    constant uint3 &dims [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint k = dims.x;
    uint n = dims.y;
    uint num_jobs = dims.z;
    uint global_row = gid.y;
    uint col = gid.x;
    if (col >= n) {
        return;
    }

    uint lo = 0u;
    uint hi = num_jobs;
    while (lo + 1u < hi) {
        uint mid = (lo + hi) >> 1;
        if (row_starts[mid] <= global_row) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    uint job_id = lo;
    if (global_row >= row_starts[job_id + 1u]) {
        return;
    }

    Q4GroupedJob job = jobs[job_id];
    device const uchar *w = w_blob + job.w_byte_off;
    uint row_stride = job.groups_per_row * 20u;

    float sum = 0.0f;
    for (uint g = simd_lane; g < job.groups_per_row; g += 32u) {
        device const uchar *blk = w + col * row_stride + g * 20u;
        sum += dot_q4_group(a + global_row * k, blk, g * 32u);
    }
    sum = simd_sum(sum);
    if (simd_lane == 0u) {
        c[global_row * n + col] = sum;
    }
}

/// C[M,N] = A[M,K] @ W[N,K]^T with Q4 PyTorch row-major W stored as in `.dgq`.
kernel void f32_q4_linear(
    device const float *a [[buffer(0)]],
    device const uchar *w [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant uint4 &dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint m = dims.x;
    uint n = dims.y;
    uint k_dim = dims.z;
    uint groups_per_row = dims.w;
    uint row_stride = groups_per_row * 20u;
    uint row = gid.y;
    uint col = gid.x;
    if (row >= m || col >= n) {
        return;
    }

    float sum = 0.0f;
    for (uint p = 0; p < k_dim; p++) {
        float av = a[row * k_dim + p];
        float wv = q4_weight_at(w, col, p, k_dim, groups_per_row, row_stride);
        sum += av * wv;
    }
    c[row * n + col] = sum;
}

/// Dequant full Q4 `[N,K]` matrix to f32 row-major (MPS scratch path).
kernel void dequant_q4_matrix(
    device const uchar *q4 [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint3 &dims [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint n = dims.x;
    uint k_dim = dims.y;
    uint groups_per_row = dims.z;
    uint row_stride = groups_per_row * 20u;
    uint row = gid.y;
    uint col = gid.x;
    if (row >= n || col >= k_dim) {
        return;
    }
    out[row * k_dim + col] = q4_weight_at(q4, row, col, k_dim, groups_per_row, row_stride);
}
inline float q8_weight_at(
    device const uchar *base,
    uint row,
    uint col,
    uint in_dim,
    uint row_stride
) {
    device const uchar *r = base + row * row_stride;
    float scale = bf16_bits_to_f32(uint(r[0]) | (uint(r[1]) << 8));
    int q = int(*((device const char *)(r + 2 + col)));
    return float(q) * scale;
}

/// C[M,N] = A[M,K] @ W[N,K]^T with Q8 row-major W as in `.dgq`.
kernel void f32_q8_linear(
    device const float *a [[buffer(0)]],
    device const uchar *w [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant uint3 &dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint m = dims.x;
    uint n = dims.y;
    uint k_dim = dims.z;
    uint row_stride = 2u + k_dim;
    uint row = gid.y;
    uint col = gid.x;
    if (row >= m || col >= n) {
        return;
    }

    float sum = 0.0f;
    for (uint p = 0; p < k_dim; p++) {
        float av = a[row * k_dim + p];
        float wv = q8_weight_at(w, col, p, k_dim, row_stride);
        sum += av * wv;
    }
    c[row * n + col] = sum;
}

/// C[M,N] = A[M,K] @ W[K,N] — W rows index K (vocab), row stride (2 + N).
/// Soft-embed: probs[seq,vocab] @ embed[vocab,hidden].
kernel void f32_q8_linear_kxn(
    device const float *a [[buffer(0)]],
    device const uchar *w [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant uint3 &dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint m = dims.x;
    uint n = dims.y;
    uint k_dim = dims.z;
    uint row_stride = 2u + n;
    uint row = gid.y;
    uint col = gid.x;
    if (row >= m || col >= n) {
        return;
    }

    float sum = 0.0f;
    for (uint p = 0; p < k_dim; p++) {
        float av = a[row * k_dim + p];
        float wv = q8_weight_at(w, p, col, n, row_stride);
        sum += av * wv;
    }
    c[row * n + col] = sum;
}

inline float fp16_bits_to_f32(ushort bits) {
    uint sign = (bits >> 15) & 1u;
    uint exp = (bits >> 10) & 0x1fu;
    uint mant = bits & 0x3ffu;
    if (exp == 0u) {
        if (mant == 0u) {
            return sign ? -0.0f : 0.0f;
        }
        return (sign ? -1.0f : 1.0f) * float(mant) * exp2(-24.0f);
    }
    if (exp == 0x1fu) {
        if (mant == 0u) {
            return sign ? -INFINITY : INFINITY;
        }
        return NAN;
    }
    uint f32_bits = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    return as_type<float>(f32_bits);
}

inline float fp8_e4m3_to_f32(uchar b) {
    ushort v = ushort(b & 127u) << 7;
    float converted = fp16_bits_to_f32(v) * 256.0f;
    return (b & 128u) ? -converted : converted;
}

inline float e2m1_to_f32(uint q) {
    float mag = 0.0f;
    switch (q & 7u) {
        case 1: mag = 0.5f; break;
        case 2: mag = 1.0f; break;
        case 3: mag = 1.5f; break;
        case 4: mag = 2.0f; break;
        case 5: mag = 3.0f; break;
        case 6: mag = 4.0f; break;
        case 7: mag = 6.0f; break;
        default: mag = 0.0f; break;
    }
    return (q & 8u) ? -mag : mag;
}

inline uint nvfp4_data_row_bytes(uint in_dim) { return (in_dim + 1u) / 2u; }
inline uint nvfp4_scales_row_bytes(uint in_dim) { return (in_dim + 15u) / 16u; }
inline uint nvfp4_row_bytes(uint in_dim) {
    return nvfp4_data_row_bytes(in_dim) + nvfp4_scales_row_bytes(in_dim);
}

inline float nvfp4_weight_at(
    device const uchar *base,
    uint row,
    uint col,
    uint in_dim
) {
    float gscale = as_type<float>(*(device const uint *)(base));
    uint data_len = nvfp4_data_row_bytes(in_dim);
    uint row_stride = nvfp4_row_bytes(in_dim);
    device const uchar *row_base = base + 4u + row * row_stride;
    uint g = col / 16u;
    float scale = fp8_e4m3_to_f32(row_base[data_len + g]) * gscale;
    uchar byte = row_base[col / 2u];
    uint q = (col & 1u) ? uint(byte >> 4) : uint(byte & 0x0fu);
    return e2m1_to_f32(q) * scale;
}

kernel void f32_nvfp4_linear(
    device const float *a [[buffer(0)]],
    device const uchar *w [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant uint3 &dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint m = dims.x;
    uint n = dims.y;
    uint k_dim = dims.z;
    uint row = gid.y;
    uint col = gid.x;
    if (row >= m || col >= n) {
        return;
    }
    float sum = 0.0f;
    for (uint p = 0; p < k_dim; p++) {
        float av = a[row * k_dim + p];
        float wv = nvfp4_weight_at(w, col, p, k_dim);
        sum += av * wv;
    }
    c[row * n + col] = sum;
}

kernel void dequant_nvfp4_matrix(
    device const uchar *nvfp4 [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint2 &dims [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint n = dims.x;
    uint k_dim = dims.y;
    uint row = gid.y;
    uint col = gid.x;
    if (row >= n || col >= k_dim) {
        return;
    }
    out[row * k_dim + col] = nvfp4_weight_at(nvfp4, row, col, k_dim);
}
