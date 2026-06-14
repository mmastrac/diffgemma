#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "dequant.metal"

/// `C[M,N] = A[M,K] @ W[N,K]^T` scalar f32 GEMM; format from K_QUANT_FORMAT (Q4 or NVFP4).
kernel void gemm_linear_f32(
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
    uint row = gid.y;
    uint col = gid.x;
    if (K_SHAPE_ASSERT && (m == 0u || n == 0u || k_dim == 0u)) {
        return;
    }
    if (row >= m || col >= n) {
        return;
    }

    float sum = 0.0f;
    if (K_QUANT_FORMAT == QUANT_NVFP4) {
        for (uint p = 0u; p < k_dim; p++) {
            float av = a[row * k_dim + p];
            sum += av * nvfp4_at_col(w, col, p, k_dim);
        }
    } else {
        uint row_stride = groups_per_row * 20u;
        device const uchar *row_base = w + ulong(col) * row_stride;
        for (uint p = 0u; p < k_dim; p++) {
            float av = a[row * k_dim + p];
            sum += av * q4_at_col(row_base, p, k_dim);
        }
    }
    c[row * n + col] = sum;
}
