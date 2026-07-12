#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "common.metal"
#include "attention_device.metal"
#include "moe_router_device.metal"
#include "arena.metal"

/// Top-k tail of the MoE router, reading PRECOMPUTED logits (DGQ_ROUTER_GEMM):
/// the [canvas, n_experts] logits come from the fast bf16 GEMM
/// (scaled-norm(stream) @ router_proj^T) instead of the per-thread serial dot
/// products in `moe_router` (which ran at ~77 GFLOP/s). `router_hscale` is a
/// uniform logit scale (linear in the GEMM input), applied here before the
/// softmax; selection/softmax/per-expert-scale/RouteScratch writes are
/// verbatim from `moe_router`'s tail. One thread per token.
kernel void moe_router_topk(
    device const ushort *logits_in [[buffer(0)]],
    device const uchar *blob [[buffer(1)]],
    device const LayerOffsets *L [[buffer(2)]],
    device RouteScratch *R [[buffer(3)]],
    constant RouterDims &dims [[buffer(4)]],
    device DebugStatus *dbg [[buffer(5)]],
    uint tok [[thread_position_in_grid]]
) {
    if (K_SHAPE_ASSERT
        && (dims.canvas == 0u || dims.hidden == 0u || dims.n_experts == 0u || dims.top_k == 0u)) {
        return;
    }
    K_ELEMENTWISE_GUARD();
    if (tok >= dims.canvas || dims.n_experts > MOE_MAX_EXPERTS || dims.top_k > MOE_MAX_TOP_K) {
        return;
    }

    float logits[128];
    device const ushort *lr = logits_in + (ulong)tok * dims.n_experts;
    for (uint e = 0u; e < dims.n_experts; ++e) {
        logits[e] = arena_load(lr, e) * dims.router_hscale;
    }

    uint pick[8];
    for (uint kk = 0u; kk < dims.top_k; ++kk) {
        float best = -1e30f;
        uint bi = 0u;
        for (uint i = 0u; i < dims.n_experts; ++i) {
            bool taken = false;
            for (uint p = 0u; p < kk; ++p) {
                taken = taken || (pick[p] == i);
            }
            if (!taken && logits[i] > best) {
                best = logits[i];
                bi = i;
            }
        }
        pick[kk] = bi;
    }
    float mx = logits[pick[0]];
    for (uint kk = 1u; kk < dims.top_k; ++kk) {
        mx = max(mx, logits[pick[kk]]);
    }
    float sum = 0.f;
    for (uint kk = 0u; kk < dims.top_k; ++kk) {
        sum += exp(logits[pick[kk]] - mx);
    }
    device const uchar *pes = blob + L->per_expert_scale;
    float inv = (sum > 0.f) ? (1.f / sum) : 0.f;
    for (uint kk = 0u; kk < dims.top_k; ++kk) {
        float w = exp(logits[pick[kk]] - mx) * inv * bf16_bytes(pes + 2ul * pick[kk]);
        dgq_assert_index(dbg, DbgKernelMoeRouter, pick[kk], dims.n_experts);
        R->expert[tok][kk] = pick[kk];
        // ALWAYS-bf16 weight bits (scatter reads bf16).
        R->weight[tok][kk] = as_type<half>(arena_bf16_bits(w));
    }
}
