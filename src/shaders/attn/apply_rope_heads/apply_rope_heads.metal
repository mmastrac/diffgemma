#include <metal_stdlib>
using namespace metal;

#include "gqa_device.metal"

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

    uint off = p.elem_offset + (s * p.num_heads_rope + h) * p.head_dim;
    uint foff = s * p.rotary_dim;
    uint rot_half = p.rotary_dim / 2;
    uint half_head = p.head_dim / 2;
    bool proportional = p.rotary_dim < p.head_dim;

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
