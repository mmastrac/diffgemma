#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "moe_router_device.metal"

/// Bucketing phases 0/1/2 (monolith k_bucket_fill).
kernel void moe_bucket_fill(
    device RouteScratch *R [[buffer(0)]],
    constant uint &phase [[buffer(1)]],
    constant RouterDims &dims [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (K_SHAPE_ASSERT && (dims.canvas == 0u || dims.top_k == 0u || dims.n_experts == 0u)) {
        return;
    }
    K_ELEMENTWISE_GUARD();

    if (phase == 0u) {
        uint slots = dims.canvas * dims.top_k;
        if (i < slots) {
            uint tok = i / dims.top_k;
            uint kk = i % dims.top_k;
            atomic_fetch_add_explicit(
                (device atomic_uint *)&R->count[R->expert[tok][kk]],
                1u,
                memory_order_relaxed);
        }
    } else if (phase == 1u) {
        if (i == 0u) {
            uint s = 0u;
            for (uint e = 0u; e < dims.n_experts; ++e) {
                R->row_start[e] = s;
                s += R->count[e];
                R->count[e] = 0u;
            }
            R->row_start[dims.n_experts] = s;
            R->num_slots = s;
        }
    } else {
        uint slots = dims.canvas * dims.top_k;
        if (i < slots) {
            uint tok = i / dims.top_k;
            uint kk = i % dims.top_k;
            uint e = R->expert[tok][kk];
            uint slot = R->row_start[e]
                + atomic_fetch_add_explicit(
                      (device atomic_uint *)&R->count[e], 1u, memory_order_relaxed);
            R->token_list[slot] = tok;
            R->slot_list[slot] = kk;
        }
    }
}
