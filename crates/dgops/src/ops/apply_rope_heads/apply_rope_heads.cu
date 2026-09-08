// Split-half / proportional RoPE, in place. Mirrors apply_rope_heads.metal.
// sincosf/tanhf/expf/rsqrtf are device builtins: NVRTC has no <cmath>.

struct GqaRopeParams {
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

extern "C" __global__ void apply_rope_heads(
    float *x,
    const float *freqs,
    GqaRopeParams p,
    unsigned num_heads,
    unsigned seq_len
) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_heads * seq_len) {
        return;
    }
    const unsigned h = gid % num_heads;
    const unsigned s = gid / num_heads;

    const unsigned off = p.elem_offset + (s * num_heads + h) * p.head_dim;
    const unsigned foff = s * p.rotary_dim;
    const unsigned rot_half = p.rotary_dim / 2u;
    const unsigned half_head = p.head_dim / 2u;
    const bool proportional = p.rotary_dim < p.head_dim;

    for (unsigned d = 0; d < rot_half; d++) {
        float cos_val = freqs[foff + 2u * d];
        float sin_val = freqs[foff + 2u * d + 1u];
        unsigned i1 = proportional ? (half_head + d) : (d + rot_half);
        float x0 = x[off + d];
        float x1 = x[off + i1];
        x[off + d] = x0 * cos_val - x1 * sin_val;
        x[off + i1] = x0 * sin_val + x1 * cos_val;
    }
}
