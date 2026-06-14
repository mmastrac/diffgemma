#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_KERNEL_COMMON_METAL
#include "common.metal"
#endif

kernel void f32_to_half(
    device const float *x [[buffer(0)]],
    device half *y [[buffer(1)]],
    constant uint &base [[buffer(2)]],
    constant uint &len [[buffer(3)]],
    device float *dump [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (K_SHAPE_ASSERT && len == 0u) return;
    if (gid >= len) return;
    K_ELEMENTWISE_GUARD();
    uint i = base + gid;
    float v = x[i];
    if (K_DUMP_STAGE >= 1u) dump[gid] = v;
    y[i] = half(v);
}
