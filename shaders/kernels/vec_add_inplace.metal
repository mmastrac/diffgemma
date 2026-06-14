#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"

kernel void vec_add_inplace(
    device float *out [[buffer(0)]],
    device const float *addend [[buffer(1)]],
    constant uint &len [[buffer(2)]],
    device float *dump [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (K_SHAPE_ASSERT && len == 0u) return;
    if (gid >= len) return;
    K_ELEMENTWISE_GUARD();
    if (K_DUMP_STAGE >= 1u) dump[gid] = addend[gid];
    out[gid] += addend[gid];
}
