#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_KERNEL_COMMON_METAL
#include "common.metal"
#endif

kernel void half_scale(
    device half *y [[buffer(0)]],
    constant uint &n [[buffer(1)]],
    constant float &scale [[buffer(2)]],
    device float *dump [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (K_SHAPE_ASSERT && n == 0u) return;
    if (gid >= n) return;
    float v = float(y[gid]) * scale;
    if (K_DUMP_STAGE >= 1u) dump[gid] = v;
    K_ELEMENTWISE_GUARD();
    y[gid] = half(v);
}
