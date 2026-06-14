#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "common.metal"
#include "dequant.metal"

/// Gather Q8 embed rows by token id: out[tok,d] = dequant(embed[id], d) * embed_scale.
kernel void embed_gather(
    device const uchar *blob [[buffer(0)]],
    device const uint *ids [[buffer(1)]],
    device half *out [[buffer(2)]],
    constant ulong &w_off [[buffer(3)]],
    constant uint2 &dims [[buffer(4)]],
    constant float &embed_scale [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint hidden = dims.x;
    const uint num_tokens = dims.y;
    const uint tok = gid.y;
    const uint d = gid.x;
    if (K_SHAPE_ASSERT && (hidden == 0u || num_tokens == 0u)) {
        return;
    }
    if (tok >= num_tokens || d >= hidden) {
        return;
    }
    K_ELEMENTWISE_GUARD();
    device const uchar *row = blob + w_off + (ulong)ids[tok] * q8_row_bytes(hidden);
    out[(ulong)tok * hidden + d] = half(q8_at(row, d, bf16_bytes(row)) * embed_scale);
}
