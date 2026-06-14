#ifndef DGQ_INCLUDE_ATTENTION_METAL
#define DGQ_INCLUDE_ATTENTION_METAL

// bf16_bytes: shaders/include/common.metal (prepended by Rust concat)

constant float ATTN_RMS_EPS = 1e-6f;

struct LayerOffsets {
    ulong input_ln, q_proj, q_norm, k_proj, k_norm, v_proj, o_proj, post_attn_ln;
    ulong pre_ff_ln, mlp_gate, mlp_up, mlp_down, post_ff_ln_1;
    ulong router_scale, router_proj, per_expert_scale, pre_ff_ln_2;
    ulong experts_gate_up, experts_down, post_ff_ln_2, post_ff_ln, layer_scalar;
    ulong kv_region;
    uint head_dim;
    uint n_kv_heads;
    uint is_full;
    uint _pad;
};

struct AttnDims {
    uint canvas;
    uint n_q_heads;
};

/// Split-half RoPE on the first `rot` dims; inv_freq denominator is full `head_dim`.
inline void apply_split_half_rope(
    device half *src,
    uint rot,
    uint head_dim,
    float theta,
    uint pos
) {
    const uint half_rot = rot / 2u;
    for (uint d = 0u; d < half_rot; ++d) {
        float inv_freq = pow(theta, -2.0f * float(d) / float(head_dim));
        float a = float(pos) * inv_freq;
        float c = cos(a);
        float s = sin(a);
        float x0 = float(src[d]);
        float x1 = float(src[d + half_rot]);
        src[d] = half(x0 * c - x1 * s);
        src[d + half_rot] = half(x0 * s + x1 * c);
    }
}

#endif
