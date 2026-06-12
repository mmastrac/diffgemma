#include <metal_stdlib>
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

/// Q8 row layout per output row: fp16 scale + int8 weights along K.
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
