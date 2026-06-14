#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "dequant.metal"

/// `C[M,N] = A[M,K] @ W[K,N]` — weight rows indexed by K (embed / SC softembed).
kernel void gemm_q8_linear_kxn_f32(
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
    if (K_SHAPE_ASSERT && (m == 0u || n == 0u || k_dim == 0u)) {
        return;
    }
    if (row >= m || col >= n) {
        return;
    }

    float sum = 0.0f;
    for (uint p = 0u; p < k_dim; p++) {
        float av = a[row * k_dim + p];
        device const uchar *w_row = w + ulong(p) * row_stride;
        float scale = bf16_bytes(w_row);
        sum += av * q8_at(w_row, col, scale);
    }
    c[row * n + col] = sum;
}
