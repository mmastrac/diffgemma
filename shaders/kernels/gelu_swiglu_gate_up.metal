#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_KERNEL_COMMON_METAL
#include "common.metal"
#endif

#ifndef DGQ_KERNEL_ACTIVATIONS_METAL
#include "activations.metal"
#endif

/// MoE gate_up layout `[batch, 2 * moe_inter]` -> activated `[batch, moe_inter]`.
kernel void gelu_swiglu_gate_up(
    device const float *gate_up [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint2 &dims [[buffer(2)]],
    device float *dump [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint batch_size = dims.x;
    uint moe_inter = dims.y;
    uint row = gid / moe_inter;
    uint col = gid % moe_inter;
    if (K_SHAPE_ASSERT && (batch_size == 0u || moe_inter == 0u)) {
        return;
    }
    if (row >= batch_size) {
        return;
    }
    (void)K_USE_FP4;
    uint off = row * (2u * moe_inter) + col;
    float g = gelu_tanh(gate_up[off]);
    float u = gate_up[off + moe_inter];
    if (K_DUMP_STAGE >= 1u) {
        dump[gid] = u;
    }
    out[gid] = g * u;
}
