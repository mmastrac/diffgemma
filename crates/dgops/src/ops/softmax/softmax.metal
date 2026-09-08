#include <metal_stdlib>
using namespace metal;

/// One threadgroup per row: max reduction, exp, sum reduction, normalize.
kernel void softmax_rows(
    device float *x [[buffer(0)]],
    constant uint2 &dims [[buffer(1)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    const uint TG = 256u;
    uint rows = dims.x;
    uint cols = dims.y;
    uint row = tgp.y;
    if (row >= rows) {
        return;
    }
    device float *r = x + (ulong)row * cols;
    threadgroup float scratch[256];

    float local_max = -1e30f;
    for (uint c = lid; c < cols; c += TG) {
        local_max = max(local_max, r[c]);
    }
    scratch[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TG / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            scratch[lid] = max(scratch[lid], scratch[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint c = lid; c < cols; c += TG) {
        float e = exp(r[c] - row_max);
        r[c] = e;
        local_sum += e;
    }
    scratch[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TG / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            scratch[lid] += scratch[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = lid; c < cols; c += TG) {
        r[c] *= inv;
    }
}
