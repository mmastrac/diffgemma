#include <metal_stdlib>
using namespace metal;

/// One threadgroup per gathered row.
kernel void gather_rows(
    device float *out [[buffer(0)]],
    device const float *src [[buffer(1)]],
    device const uint *indices [[buffer(2)]],
    constant uint2 &dims [[buffer(3)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    uint num_indices = dims.x;
    uint hidden = dims.y;
    uint t = tgp.y;
    if (t >= num_indices) {
        return;
    }
    uint row = indices[t];
    const device float *s = src + (ulong)row * hidden;
    device float *d = out + (ulong)t * hidden;
    for (uint i = lid; i < hidden; i += 256u) {
        d[i] = s[i];
    }
}
