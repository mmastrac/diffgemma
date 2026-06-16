#include <metal_stdlib>
using namespace metal;

/// Shannon entropy per row from logits (softmax then `-sum p log p`).
kernel void row_entropy(
    device const float *logits [[buffer(0)]],
    device float *entropies [[buffer(1)]],
    constant uint2 &dims [[buffer(2)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    const uint TG = 256u;
    uint row = tgp.y;
    uint rows = dims.x;
    uint cols = dims.y;
    if (row >= rows) {
        return;
    }

    device const float *r = logits + row * cols;
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
        local_sum += exp(r[c] - row_max);
    }
    scratch[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TG / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            scratch[lid] += scratch[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_h = 0.0f;
    for (uint c = lid; c < cols; c += TG) {
        float p = exp(r[c] - row_max) * inv_sum;
        if (p > 0.0f) {
            local_h -= p * log(p);
        }
    }
    scratch[lid] = local_h;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TG / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            scratch[lid] += scratch[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        entropies[row] = scratch[0];
    }
}
