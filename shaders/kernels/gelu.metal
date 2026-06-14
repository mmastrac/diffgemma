#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_KERNEL_COMMON_METAL
#include "common.metal"
#endif
#ifndef DGQ_INCLUDE_ACTIVATIONS_METAL
#error "gelu: bundle must include shaders/include/activations.metal"
#endif

/// In-place GELU (PyTorch tanh approximation).
kernel void gelu(
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
    K_ELEMENTWISE_GUARD();
    float v = x[gid];
    if (K_DUMP_STAGE >= 1u) {
        dump[gid] = v;
    }
    x[gid] = gelu_tanh(v);
}
