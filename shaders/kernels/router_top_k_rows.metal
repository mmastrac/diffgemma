#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_KERNEL_COMMON_METAL
#include "common.metal"
#endif

/// Top-k expert routing per row from softmax probs `[rows, experts]`.
/// Tie-break: higher prob wins; equal prob → lower expert index wins (bit-exact vs CPU oracle).
kernel void router_top_k_rows(
    device const float *probs [[buffer(0)]],
    device const float *per_expert_scale [[buffer(1)]],
    device uint *out_indices [[buffer(2)]],
    device float *out_weights [[buffer(3)]],
    constant uint3 &params [[buffer(4)]],
    device float *dump [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    uint rows = params.x;
    uint experts = params.y;
    uint k = params.z;
    if (K_SHAPE_ASSERT && (rows == 0u || experts == 0u || k == 0u)) {
        return;
    }
    if (row >= rows || k == 0u || k > 32u) {
        return;
    }
    K_ELEMENTWISE_GUARD();

    float top_p[32];
    uint top_i[32];
    for (uint i = 0u; i < k; i++) {
        top_p[i] = -1.0f;
        top_i[i] = 0u;
    }

    for (uint e = 0u; e < experts; e++) {
        float p = probs[row * experts + e];
        uint insert_at = k;
        for (uint i = 0u; i < k; i++) {
            if (p > top_p[i] || (p == top_p[i] && e < top_i[i])) {
                insert_at = i;
                break;
            }
        }
        if (insert_at < k) {
            for (uint j = k - 1u; j > insert_at; j--) {
                top_p[j] = top_p[j - 1u];
                top_i[j] = top_i[j - 1u];
            }
            top_p[insert_at] = p;
            top_i[insert_at] = e;
        }
    }

    float sum = 0.0f;
    for (uint i = 0u; i < k; i++) {
        sum += top_p[i];
    }
    float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    if (K_DUMP_STAGE >= 1u) {
        dump[row * 2u] = sum;
        dump[row * 2u + 1u] = inv;
    }

    uint off = row * k;
    for (uint i = 0u; i < k; i++) {
        out_indices[off + i] = top_i[i];
        out_weights[off + i] = top_p[i] * inv * per_expert_scale[top_i[i]];
    }
}
