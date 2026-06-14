#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_KERNEL_COMMON_METAL
#include "common.metal"
#endif

#ifndef DGQ_KERNEL_ACTIVATIONS_METAL
#include "activations.metal"
#endif

kernel void gelu_pytorch_tanh(
    device float *x [[buffer(0)]],
    constant uint &len [[buffer(1)]],
    device float *dump [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (K_SHAPE_ASSERT && len == 0u) {
        return;
    }
    if (gid >= len) {
        return;
    }
    (void)K_USE_FP4;
    float v = x[gid];
    if (K_DUMP_STAGE >= 1u) {
        dump[gid] = v;
    }
    x[gid] = gelu_tanh(v);
}
