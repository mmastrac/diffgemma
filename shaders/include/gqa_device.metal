#ifndef GQA_DEVICE_METAL
#define GQA_DEVICE_METAL

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
    uint elem_offset;
};

constant uint MASK_CAUSAL_SLIDING = 0;
constant uint MASK_ENCODER_EXTEND = 1;
constant uint MASK_DECODER_BITMAP = 2;

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
    device const long *positions
) {
    if (p.mask_kind == MASK_DECODER_BITMAP) {
        return decoder_mask[qi * p.total_kv + ki] == 0;
    }
    if (p.mask_kind == MASK_ENCODER_EXTEND) {
        long pos_q = positions[qi];
        long pos_k;
        if (ki < p.kv_cache_len) {
            pos_k = long(ki);
        } else {
            pos_k = positions[ki - p.kv_cache_len];
        }
        bool masked = pos_k > pos_q;
        if (p.sliding_window > 0 && pos_k + long(p.sliding_window) <= pos_q) {
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
    device const long *positions
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

#endif
