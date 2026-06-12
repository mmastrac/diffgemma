#include <metal_stdlib>
using namespace metal;

inline float bf16_bits_to_f32(uint16_t bits) {
    return as_type<float>((uint)bits << 16);
}

/// C[M,N] = A[M,K] @ B[K,N], row-major bf16 inputs, f32 accumulation/output.
kernel void bf16_gemm(
    device const ushort *a [[buffer(0)]],
    device const ushort *b [[buffer(1)]],
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
        float av = bf16_bits_to_f32(a[row * k_dim + p]);
        float bv = bf16_bits_to_f32(b[p * n + col]);
        sum += av * bv;
    }
    c[row * n + col] = sum;
}

/// C[M,N] = A[M,K] @ W[N,K]^T with PyTorch linear weight W stored row-major [N,K].
kernel void f32_bf16_linear(
    device const float *a [[buffer(0)]],
    device const ushort *w [[buffer(1)]],
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
        float bv = bf16_bits_to_f32(w[col * k_dim + p]);
        sum += av * bv;
    }
    c[row * n + col] = sum;
}

/// C[M,N] = A[M,K] @ B[K,N], f32 activations and bf16 weights.
kernel void f32_bf16_gemm(
    device const float *a [[buffer(0)]],
    device const ushort *b [[buffer(1)]],
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
        float bv = bf16_bits_to_f32(b[p * n + col]);
        sum += av * bv;
    }
    c[row * n + col] = sum;
}
