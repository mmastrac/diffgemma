#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_KERNEL_COMMON_METAL
#include "common.metal"
#endif

kernel void residual_f32b(
    device const half *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device half *y [[buffer(2)]],
    device float *dump [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    float v = float(a[i]) + b[i];
    if (K_DUMP_STAGE >= 1u) dump[i] = v;
    (void)K_USE_FP4;
    y[i] = half(v);
}
