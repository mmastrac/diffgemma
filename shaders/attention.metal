#include <metal_stdlib>
using namespace metal;

struct GqaParams {
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
};

constant uint MASK_CAUSAL_SLIDING = 0;
constant uint MASK_ENCODER_EXTEND = 1;
constant uint MASK_DECODER_BITMAP = 2;

kernel void apply_rope_heads(
    device float *x [[buffer(0)]],
    device const float *freqs [[buffer(1)]],
    constant GqaParams &p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint h = gid.x;
    uint s = gid.y;
    if (h >= p.num_heads_rope || s >= p.seq_len) {
        return;
    }

    uint off = (s * p.num_heads_rope + h) * p.head_dim;
    uint foff = s * p.rotary_dim;
    uint rot_half = p.rotary_dim / 2;

    for (uint d = 0; d < rot_half; d++) {
        float cos_val = freqs[foff + 2 * d];
        float sin_val = freqs[foff + 2 * d + 1];
        float x0 = x[off + d];
        float x1 = x[off + d + rot_half];
        x[off + d] = x0 * cos_val - x1 * sin_val;
        x[off + d + rot_half] = x0 * sin_val + x1 * cos_val;
    }
}

inline float gqa_dot(
    device const float *q,
    device const float *k,
    uint q_off,
    uint k_off,
    uint hd
) {
    float dot = 0.0f;
    for (uint d = 0; d < hd; d++) {
        dot += q[q_off + d] * k[k_off + d];
    }
    return dot;
}

inline bool gqa_masked(
    constant GqaParams &p,
    uint qi,
    uint ki,
    device const uchar *decoder_mask,
    device const int *positions
) {
    if (p.mask_kind == MASK_DECODER_BITMAP) {
        return decoder_mask[qi * p.total_kv + ki] == 0;
    }
    if (p.mask_kind == MASK_ENCODER_EXTEND) {
        int pos_q = int(positions[qi]);
        int pos_k;
        if (ki < p.kv_cache_len) {
            pos_k = int(ki);
        } else {
            pos_k = positions[ki - p.kv_cache_len];
        }
        bool masked = pos_k > pos_q;
        if (p.sliding_window > 0 && pos_k + int(p.sliding_window) <= pos_q) {
            masked = true;
        }
        return masked;
    }
    bool masked = ki > qi;
    if (p.sliding_window > 0 && ki + p.sliding_window <= qi) {
        masked = true;
    }
    return masked;
}

inline float gqa_score(
    constant GqaParams &p,
    device const float *q,
    device const float *k,
    uint qi,
    uint ki,
    uint h,
    device const uchar *decoder_mask,
    device const int *positions
) {
    uint hd = p.head_dim;
    uint kv_dim = p.n_kv_heads * hd;
    uint kv_h = h / p.n_groups;
    uint q_off = (qi * p.n_heads + h) * hd;
    uint k_off = ki * kv_dim + kv_h * hd;
    float dot = gqa_dot(q, k, q_off, k_off, hd);
    if (gqa_masked(p, qi, ki, decoder_mask, positions)) {
        dot = p.mask_neg;
    }
    return dot;
}

kernel void gqa_attention(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device float *out [[buffer(3)]],
    device const uchar *decoder_mask [[buffer(4)]],
    device const int *positions [[buffer(5)]],
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

    float max_val = -1e30f;
    for (uint ki = 0; ki < p.total_kv; ki++) {
        float dot = gqa_score(p, q, k, qi, ki, h, decoder_mask, positions);
        max_val = max(max_val, dot);
    }

    float sum = 0.0f;
    for (uint ki = 0; ki < p.total_kv; ki++) {
        float dot = gqa_score(p, q, k, qi, ki, h, decoder_mask, positions);
        sum += exp(dot - max_val);
    }
    float inv_sum = 1.0f / sum;

    for (uint d = 0; d < hd; d++) {
        out[o_off + d] = 0.0f;
    }
    for (uint ki = 0; ki < p.total_kv; ki++) {
        float dot = gqa_score(p, q, k, qi, ki, h, decoder_mask, positions);
        float w = exp(dot - max_val) * inv_sum;
        uint v_off = ki * kv_dim + kv_h * hd;
        for (uint d = 0; d < hd; d++) {
            out[o_off + d] += w * v[v_off + d];
        }
    }
}
