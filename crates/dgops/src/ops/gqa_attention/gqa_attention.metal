#include <metal_stdlib>
using namespace metal;

/// Must stay layout-identical to GqaAttentionParams in mod.rs and .cu.
struct GqaAttentionParams {
    uint seq_len;
    uint total_kv;
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    uint n_groups;
    uint mask_kind;
    uint sliding_window;
    uint kv_cache_len;
    float mask_neg;
    uint rotary_dim;
    uint num_heads_rope;
    uint elem_offset;
};

constant uint MASK_NEG_BITS = 0u;

/// Causal (optionally windowed) GQA attention, one threadgroup per query head.
kernel void gqa_attention(
    device const float *q [[buffer(0)]],
    device const float *kv [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant GqaAttentionParams &p [[buffer(3)]],
    constant uint2 &dims [[buffer(4)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint3 tpg3 [[threads_per_threadgroup]]
) {
    const uint row = tgp.y;
    const uint tpg = tpg3.x;
    const uint seq_len = dims.x;
    const uint n_heads = dims.y;
    const uint hd = p.head_dim;
    const uint nkv = p.n_kv_heads;
    const uint tok = row / n_heads;
    const uint qh = row % n_heads;
    if (tok >= seq_len || qh >= n_heads) {
        return;
    }
    const uint kvh = qh / p.n_groups;
    const uint q_off = (tok * n_heads + qh) * hd;

    threadgroup float red_sum[256];

    float m = -1.0e30f;
    float l = 0.0f;
    float acc[512];
    for (uint d = 0; d < hd; d++) {
        acc[d] = 0.0f;
    }

    for (uint t = 0; t < p.total_kv; t++) {
        // Causal: a query at position kv_cache_len + tok sees keys up to that.
        if (t > p.kv_cache_len + tok) {
            break;
        }
        if (p.sliding_window > 0 && t + p.sliding_window <= p.kv_cache_len + tok) {
            continue;
        }
        const uint k_off = t * nkv * hd * 2 + kvh * hd;
        float partial = 0.0f;
        for (uint d = lid; d < hd; d += tpg) {
            partial += q[q_off + d] * kv[k_off + d];
        }
        red_sum[lid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tpg / 2; s > 0; s >>= 1) {
            if (lid < s) {
                red_sum[lid] += red_sum[lid + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float dot = red_sum[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float mn = max(m, dot);
        float corr = exp(m - mn);
        float p_t = exp(dot - mn);
        m = mn;
        l = l * corr + p_t;
        const uint v_off = k_off + nkv * hd;
        for (uint d = lid; d < hd; d += tpg) {
            acc[d] = acc[d] * corr + p_t * kv[v_off + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (uint d = lid; d < hd; d += tpg) {
        out[q_off + d] = acc[d] * inv;
    }
}
