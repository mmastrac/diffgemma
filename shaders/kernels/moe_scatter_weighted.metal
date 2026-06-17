#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "common.metal"
#include "moe_router_device.metal"
#include "arena.metal"

/// Weighted scatter: `[num_slots, hidden]` expert rows → canvas `moe_out`.
/// One threadgroup per `(tok, d)`; 8 threads reduce top-k expert slots (no float atomics).
kernel void moe_scatter_weighted(
    device const float *expert_out [[buffer(0)]],
    device float *moe_out [[buffer(1)]],
    device const RouteScratch *R [[buffer(2)]],
    constant uint &hidden [[buffer(3)]],
    constant uint &canvas [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint kk [[thread_index_in_threadgroup]]
) {
    if (K_SHAPE_ASSERT && (hidden == 0u || canvas == 0u)) {
        return;
    }
    K_ELEMENTWISE_GUARD();
    uint d = tgid.x;
    uint tok = tgid.y;
    if (tok >= canvas || d >= hidden || kk >= MOE_MAX_TOP_K) {
        return;
    }

    threadgroup float terms[MOE_MAX_TOP_K];
    uint slot = R->token_slot[tok][kk];
    float w = bf16_to_f32(as_type<ushort>(R->weight[tok][kk]));
    float v = expert_out[(ulong)slot * hidden + d];
    terms[kk] = f32_round_bf16(w * f32_round_bf16(v));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kk == 0u) {
        float sum = 0.f;
        for (uint i = 0u; i < MOE_MAX_TOP_K; ++i) {
            sum += terms[i];
        }
        moe_out[(ulong)tok * hidden + d] = sum;
    }
}
