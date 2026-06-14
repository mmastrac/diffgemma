#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_KERNEL_COMMON_METAL
#include "common.metal"
#endif

kernel void vec_fill_zero(
    device float *x [[buffer(0)]],
    constant uint2 &range [[buffer(1)]],
    device float *dump [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint base = range.x;
    uint len = range.y;
    if (K_SHAPE_ASSERT && len == 0u) return;
    if (gid >= len) return;
    K_ELEMENTWISE_GUARD();
    uint i = base + gid;
    if (K_DUMP_STAGE >= 1u) dump[gid] = x[i];
    x[i] = 0.0f;
}
