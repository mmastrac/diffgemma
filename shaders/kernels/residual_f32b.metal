#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"

kernel void residual_f32b(
    device const half *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device half *y [[buffer(2)]],
    device float *dump [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    float v = float(a[i]) + b[i];
    if (K_DUMP_STAGE >= 1u) dump[i] = v;
    K_ELEMENTWISE_GUARD();
    y[i] = half(v);
}
