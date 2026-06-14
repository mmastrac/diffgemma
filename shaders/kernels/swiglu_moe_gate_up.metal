#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_FC_AXES_METAL
#include "fc_axes.metal"
#endif
#ifndef DGQ_INCLUDE_ACTIVATIONS_METAL
#error "swiglu_moe_gate_up: bundle must include shaders/include/activations.metal"
#endif

/// MoE gate_up `[batch, 2*moe_inter]` → gelu(gate)*up. Layout is fixed by .dgq manifest.
kernel void swiglu_moe_gate_up(
    device const float *gate_up [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint2 &dims [[buffer(2)]],
    device float *dump [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    K_ELEMENTWISE_GUARD();

    uint batch = dims.x;
    uint moe_inter = dims.y;
    uint row = gid / moe_inter;
    uint col = gid % moe_inter;
    if (K_SHAPE_ASSERT && (batch == 0u || moe_inter == 0u)) {
        return;
    }
    if (row >= batch) {
        return;
    }
    uint off = row * (2u * moe_inter) + col;
    float g = gelu_tanh(gate_up[off]);
    float u = gate_up[off + moe_inter];
    if (K_DUMP_STAGE >= 1u) {
        dump[gid] = u;
    }
    out[gid] = g * u;
}
