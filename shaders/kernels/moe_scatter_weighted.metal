#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "moe_router_device.metal"
#include "moe_grouped_device.metal"

/// Weighted scatter: `[num_slots, hidden]` expert rows → canvas `moe_out`.
kernel void moe_scatter_weighted(
    device const float *expert_out [[buffer(0)]],
    device float *moe_out [[buffer(1)]],
    device const RouteScratch *R [[buffer(2)]],
    constant uint &hidden [[buffer(3)]],
    constant uint &elem_base [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (K_SHAPE_ASSERT && hidden == 0u) {
        return;
    }
    K_ELEMENTWISE_GUARD();
    uint elem = elem_base + gid;
    uint slot = elem / hidden;
    uint d = elem % hidden;
    if (slot >= R->num_slots) {
        return;
    }
    uint tok = R->token_list[slot];
    uint kk = R->slot_list[slot];
    float w = float(R->weight[tok][kk]);
    atomic_add_f32(
        (device atomic_uint *)&moe_out[(ulong)tok * hidden + d],
        w * expert_out[(ulong)slot * hidden + d]);
}
