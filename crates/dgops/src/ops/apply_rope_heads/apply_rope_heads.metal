#include <metal_stdlib>
using namespace metal;

/// Must stay layout-identical to GqaRopeParams in mod.rs and .cu.
struct GqaRopeParams {
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

/// Split-half / proportional RoPE: one thread per (head, position), in place.
kernel void apply_rope_heads(
    device float *x [[buffer(0)]],
    device const float *freqs [[buffer(1)]],
    constant GqaRopeParams &p [[buffer(2)]],
    constant uint2 &dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint num_heads = dims.x;
    const uint seq_len = dims.y;
    const uint h = gid % num_heads;
    const uint s = gid / num_heads;
    if (h >= num_heads || s >= seq_len) {
        return;
    }

    const uint off = p.elem_offset + (s * num_heads + h) * p.head_dim;
    const uint foff = s * p.rotary_dim;
    const uint rot_half = p.rotary_dim / 2;
    const uint half_head = p.head_dim / 2;
    const bool proportional = p.rotary_dim < p.head_dim;

    for (uint d = 0; d < rot_half; d++) {
        float cos_val = freqs[foff + 2 * d];
        float sin_val = freqs[foff + 2 * d + 1];
        uint i1 = proportional ? (half_head + d) : (d + rot_half);
        float x0 = x[off + d];
        float x1 = x[off + i1];
        x[off + d] = x0 * cos_val - x1 * sin_val;
        x[off + i1] = x0 * sin_val + x1 * cos_val;
    }
}
