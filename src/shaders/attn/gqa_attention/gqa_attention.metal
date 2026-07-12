#include <metal_stdlib>
using namespace metal;

#include "gqa_device.metal"

kernel void gqa_attention(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device float *out [[buffer(3)]],
    device const uchar *decoder_mask [[buffer(4)]],
    device const long *positions [[buffer(5)]],
    constant GqaParams &p [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint h = gid.x;
    uint qi = gid.y;
    if (h >= p.n_heads || qi >= p.seq_len) {
        return;
    }

    uint hd = p.head_dim;
    uint kv_dim = p.n_kv_heads * hd;
    uint kv_h = h / p.n_groups;
    uint o_off = (qi * p.n_heads + h) * hd;

    // Skip keys that are masked for this query on every pass — bit-identical
    // (see gqa_ki_bounds) and the difference between O(kv_len) and O(window)
    // work per query on sliding layers.
    const uint2 kb = gqa_ki_bounds(p, qi, positions);

    float max_val = -1e30f;
    for (uint ki = kb.x; ki < kb.y; ki++) {
        float dot = gqa_score(p, q, k, qi, ki, h, decoder_mask, positions);
        max_val = max(max_val, dot);
    }

    float sum = 0.0f;
    for (uint ki = kb.x; ki < kb.y; ki++) {
        float dot = gqa_score(p, q, k, qi, ki, h, decoder_mask, positions);
        sum += exp(dot - max_val);
    }
    float inv_sum = 1.0f / sum;

    for (uint d = 0; d < hd; d++) {
        out[o_off + d] = 0.0f;
    }
    for (uint ki = kb.x; ki < kb.y; ki++) {
        float dot = gqa_score(p, q, k, qi, ki, h, decoder_mask, positions);
        float w = exp(dot - max_val) * inv_sum;
        uint v_off = ki * kv_dim + kv_h * hd;
        for (uint d = 0; d < hd; d++) {
            out[o_off + d] += w * v[v_off + d];
        }
    }
}
