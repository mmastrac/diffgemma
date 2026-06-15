#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "dequant.metal"

/// Dequant block weight matrix `[N,K]` to f32 row-major (scratch / validation).
/// Format from K_QUANT_FORMAT (FC3): Q4 affine or NVFP4.
/// Dispatch: `row = gid.y`, `col = gid.x`; grid width = K, height = N (see P1.10).
kernel void dequant_block_matrix(
    device const uchar *w [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint3 &dims [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint n = dims.x;
    uint k_dim = dims.y;
    uint groups_per_row = dims.z;
    uint row = gid.y;
    uint col = gid.x;
    if (K_SHAPE_ASSERT && (n == 0u || k_dim == 0u)) {
        return;
    }
    if (row >= n || col >= k_dim) {
        return;
    }
    if (K_QUANT_FORMAT == QUANT_NVFP4) {
        out[row * k_dim + col] = nvfp4_at_col(w, row, col, k_dim);
    } else {
        uint row_stride = groups_per_row * 20u;
        device const uchar *row_base = w + ulong(row) * row_stride;
        out[row * k_dim + col] = q4_at_col(row_base, col, k_dim);
    }
}
