#include <metal_stdlib>
using namespace metal;

/// One threadgroup per source row; every element is accumulated with a device
/// atomic add, so repeated indices accumulate rather than overwrite.
kernel void scatter_add_rows(
    device float *dst [[buffer(0)]],
    device const uint *indices [[buffer(1)]],
    device const float *src [[buffer(2)]],
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
    device atomic_float *d = (device atomic_float *)(dst + (ulong)row * hidden);
    const device float *s = src + (ulong)t * hidden;
    for (uint i = lid; i < hidden; i += 256u) {
        atomic_fetch_add_explicit(&d[i], s[i], memory_order_relaxed);
    }
}
