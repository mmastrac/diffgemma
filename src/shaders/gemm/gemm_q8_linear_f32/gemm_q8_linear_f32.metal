#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "dequant.metal"

/// Q8 linear GEMV `C[M,N] = A[M,K] @ W` with Q8 row weights. Generic over the
/// weight K-order (K_W_KXN / FC4): false = W[N,K]^T (`[scale][i8:K]` per output
/// row, scale hoisted); true = W[K,N] (`[scale][i8:N]` per K row, embed / SC
/// softembed). FC3 stays inert (not the format axis).
constant bool K_W_KXN_DEF [[function_constant(4)]];
constant bool K_W_KXN = is_function_constant_defined(K_W_KXN_DEF) ? K_W_KXN_DEF : false;

kernel void gemm_q8_linear_f32(
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
    if (K_SHAPE_ASSERT && (m == 0u || n == 0u || k_dim == 0u)) {
        return;
    }
    if (row >= m || col >= n) {
        return;
    }

    float sum = 0.0f;
    if (K_W_KXN) {
        // W[K,N]: one [scale][i8:N] row per K; scale + col-index inside the loop.
        uint row_stride = 2u + n;
        for (uint p = 0u; p < k_dim; p++) {
            float av = a[row * k_dim + p];
            device const uchar *w_row = w + ulong(p) * row_stride;
            float scale = bf16_bytes(w_row);
            sum += av * q8_at(w_row, col, scale);
        }
    } else {
        // W[N,K]^T: one [scale][i8:K] row per output col; scale hoisted.
        uint row_stride = 2u + k_dim;
        device const uchar *w_row = w + ulong(col) * row_stride;
        float scale = bf16_bytes(w_row);
        for (uint p = 0u; p < k_dim; p++) {
            float av = a[row * k_dim + p];
            sum += av * q8_at(w_row, p, scale);
        }
    }
    c[row * n + col] = sum;
}
