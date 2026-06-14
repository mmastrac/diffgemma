#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"

/// Stable row softmax in-place over row-major `[rows, cols]`.
kernel void softmax_rows(
    device float *x [[buffer(0)]],
    constant uint2 &dims [[buffer(1)]],
    device float *dump [[buffer(2)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    const uint TG = 256u;
    uint row = tgp.y;
    uint rows = dims.x;
    uint cols = dims.y;
    if (K_SHAPE_ASSERT && (cols == 0u || rows == 0u)) {
        return;
    }
    if (row >= rows) {
        return;
    }
    K_ELEMENTWISE_GUARD();

    device float *r = x + row * cols;
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
    float row_sum = scratch[0];
    if (K_DUMP_STAGE >= 1u && lid == 0u) {
        dump[row * 2u] = row_max;
        dump[row * 2u + 1u] = row_sum;
    }
    float inv = 1.0f / row_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = lid; c < cols; c += TG) {
        r[c] *= inv;
    }
}
