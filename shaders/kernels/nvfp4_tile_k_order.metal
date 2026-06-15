#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "dequant.metal"

/// Compare 32-wide NVFP4 tile decode: path A = fused half tile, path B = `nvfp4_at_col`.
/// Writes 64 floats to dump [[buffer(4)]] when K_DUMP_STAGE >= 1.
kernel void nvfp4_tile_k_order(
    device const uchar *matrix [[buffer(0)]],
    constant uint &k0 [[buffer(1)]],
    constant uint &k_dim [[buffer(2)]],
    constant uint &row [[buffer(3)]],
    device float *dump [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    K_ELEMENTWISE_GUARD();
    if (K_DUMP_STAGE < 1u) {
        return;
    }
    if (K_SHAPE_ASSERT && (k_dim == 0u || k0 + 32u > k_dim)) {
        return;
    }
    if (i > 0u) {
        return;
    }
    float gscale = as_type<float>(*(device const uint *)(matrix));
    device const uchar *body = matrix + 4u;
    ulong row_stride = nvfp4_row_bytes(k_dim);
    device const uchar *row_base = body + ulong(row) * row_stride;

    thread half via_tile[32];
    dequant_nvfp4_tile_half_fused(row_base, k_dim, k0, via_tile, gscale);
    for (uint m = 0u; m < 32u; ++m) {
        dump[m] = float(via_tile[m]);
    }
    for (uint m = 0u; m < 32u; ++m) {
        dump[32u + m] = nvfp4_at_col(matrix, row, k0 + m, k_dim);
    }
}
