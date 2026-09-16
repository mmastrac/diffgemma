// Causal (optionally windowed) GQA attention, one thread block per query head.
// Mirrors gqa_attention.metal.

struct GqaAttentionParams {
    unsigned seq_len;
    unsigned total_kv;
    unsigned n_heads;
    unsigned n_kv_heads;
    unsigned head_dim;
    unsigned n_groups;
    unsigned mask_kind;
    unsigned sliding_window;
    unsigned kv_cache_len;
    float mask_neg;
    unsigned rotary_dim;
    unsigned num_heads_rope;
    unsigned elem_offset;
};

extern "C" __global__ void gqa_attention(
    const float *q,
    const float *kv,
    float *out,
    GqaAttentionParams p,
    unsigned seq_len,
    unsigned n_heads
) {
    const unsigned row = blockIdx.x;
    const unsigned hd = p.head_dim;
    const unsigned nkv = p.n_kv_heads;
    const unsigned tok = row / n_heads;
    const unsigned qh = row % n_heads;
    if (tok >= seq_len || qh >= n_heads) {
        return;
    }
    const unsigned kvh = qh / p.n_groups;
    const unsigned q_off = (tok * n_heads + qh) * hd;

    __shared__ float red[256];

    float m = -1.0e30f;
    float l = 0.0f;
    float acc[512];
    for (unsigned d = 0; d < hd; d++) {
        acc[d] = 0.0f;
    }

    for (unsigned t = 0; t < p.total_kv; t++) {
        if (t > p.kv_cache_len + tok) {
            break;
        }
        if (p.sliding_window > 0 && t + p.sliding_window <= p.kv_cache_len + tok) {
            continue;
        }
        const unsigned k_off = t * nkv * hd * 2 + kvh * hd;
        float partial = 0.0f;
        for (unsigned d = threadIdx.x; d < hd; d += blockDim.x) {
            partial += q[q_off + d] * kv[k_off + d];
        }
        red[threadIdx.x] = partial;
        __syncthreads();
        for (unsigned s = blockDim.x / 2u; s > 0u; s >>= 1u) {
            if (threadIdx.x < s) {
                red[threadIdx.x] += red[threadIdx.x + s];
            }
            __syncthreads();
        }
        const float dot = red[0];
        __syncthreads();

        const float mn = fmaxf(m, dot);
        const float corr = expf(m - mn);
        const float p_t = expf(dot - mn);
        m = mn;
        l = l * corr + p_t;
        const unsigned v_off = k_off + nkv * hd;
        for (unsigned d = threadIdx.x; d < hd; d += blockDim.x) {
            acc[d] = acc[d] * corr + p_t * kv[v_off + d];
        }
        __syncthreads();
    }

    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (unsigned d = threadIdx.x; d < hd; d += blockDim.x) {
        out[q_off + d] = acc[d] * inv;
    }
}
