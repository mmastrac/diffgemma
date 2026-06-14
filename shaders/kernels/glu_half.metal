#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_KERNEL_COMMON_METAL
#include "common.metal"
#endif
#ifndef DGQ_KERNEL_ACTIVATIONS_METAL
#include "activations.metal"
#endif

kernel void glu_half(
    device const half *gate [[buffer(0)]],
    device const half *up [[buffer(1)]],
    device half *y [[buffer(2)]],
    device float *dump [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    float v = gelu_tanh(float(gate[i])) * float(up[i]);
    if (K_DUMP_STAGE >= 1u) dump[i] = v;
    (void)K_USE_FP4;
    y[i] = half(v);
}
