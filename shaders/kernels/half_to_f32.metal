#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_KERNEL_COMMON_METAL
#include "common.metal"
#endif

kernel void half_to_f32(
    device const half *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant uint &base [[buffer(2)]],
    constant uint &len [[buffer(3)]],
    device float *dump [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (K_SHAPE_ASSERT && len == 0u) return;
    if (gid >= len) return;
    K_ELEMENTWISE_GUARD();
    uint i = base + gid;
    float v = float(x[i]);
    if (K_DUMP_STAGE >= 1u) dump[gid] = v;
    y[i] = v;
}
