#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "common.metal"
#include "moe_router_device.metal"
#include "arena.metal"

/// Weighted scatter: `[num_slots, hidden]` expert rows → canvas `moe_out`.
///
/// One threadgroup per `(tok, d-tile)`, 256 threads each owning one `d`. The
/// token's TOP_K slots+weights are loaded once into threadgroup memory, then
/// every thread reduces them for its column. Reads `expert_out[slot*hidden + d]`
/// are coalesced across the d-tile (consecutive threads → consecutive d). This
/// replaces the old (tok,d)-per-threadgroup / 8-thread layout (75% idle lanes,
/// strided reads). Bit-identical: same slot order, same per-term round, same f32
/// accumulate.
kernel void moe_scatter_weighted(
    device const float *expert_out [[buffer(0)]],
    device float *moe_out [[buffer(1)]],
    device const RouteScratch *R [[buffer(2)]],
    constant uint &hidden [[buffer(3)]],
    constant uint &canvas [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (K_SHAPE_ASSERT && (hidden == 0u || canvas == 0u)) {
        return;
    }
    K_ELEMENTWISE_GUARD();
    const uint tok = tgid.y;
    if (tok >= canvas) {
        return;
    }

    // Load this token's TOP_K slot indices + weights once.
    threadgroup uint t_slot[MOE_MAX_TOP_K];
    threadgroup float t_w[MOE_MAX_TOP_K];
    if (tid < MOE_MAX_TOP_K) {
        t_slot[tid] = R->token_slot[tok][tid];
        t_w[tid] = bf16_to_f32(as_type<ushort>(R->weight[tok][tid]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint d = tgid.x * 256u + tid;
    if (d >= hidden) {
        return;
    }

    float sum = 0.f;
    for (uint i = 0u; i < MOE_MAX_TOP_K; ++i) {
        const float v = expert_out[(ulong)t_slot[i] * hidden + d];
        sum += f32_round_bf16(t_w[i] * f32_round_bf16(v));
    }
    moe_out[(ulong)tok * hidden + d] = sum;
}
