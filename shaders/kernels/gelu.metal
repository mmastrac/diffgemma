#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "activations.metal"

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
