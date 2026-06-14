#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "dequant.metal"

/// Compare 32-wide Q4 group decode: path A = `dequant_q4_group`, path B = `q4_at_col`.
/// Writes 64 floats to dump [[buffer(3)]] when K_DUMP_STAGE >= 1 (compiled out otherwise).
kernel void q4_group_k_order(
    device const uchar* row_base [[buffer(0)]],
    constant uint& k0 [[buffer(1)]],
    constant uint& in_dim [[buffer(2)]],
    device float* dump [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    K_ELEMENTWISE_GUARD();
    if (K_DUMP_STAGE < 1u) {
        return;
    }
    if (K_SHAPE_ASSERT && (in_dim == 0u || k0 + 32u > in_dim)) {
        return;
    }
    if (i > 0u) {
        return;
    }
    device const uchar* grp = row_base + ulong(k0 / 32u) * 20ul;
    thread float via_dequant[32];
    dequant_q4_group(grp, via_dequant);
    for (uint m = 0u; m < 32u; ++m) {
        dump[m] = via_dequant[m];
    }
    for (uint m = 0u; m < 32u; ++m) {
        dump[32u + m] = q4_at_col(row_base, k0 + m, in_dim);
    }
}
