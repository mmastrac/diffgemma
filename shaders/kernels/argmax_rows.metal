#include <metal_stdlib>
using namespace metal;

/// Per-row argmax over `[rows, cols]` logits (temperature-scaled).
kernel void argmax_rows(
    device const float *logits [[buffer(0)]],
    device uint *out [[buffer(1)]],
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
    threadgroup float scratch_val[256];
    threadgroup uint scratch_idx[256];

    float local_best = -1e30f;
    uint local_idx = 0u;
    for (uint c = lid; c < cols; c += TG) {
        float v = r[c];
        if (v > local_best || (v == local_best && c < local_idx)) {
            local_best = v;
            local_idx = c;
        }
    }
    scratch_val[lid] = local_best;
    scratch_idx[lid] = local_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TG / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            float va = scratch_val[lid];
            float vb = scratch_val[lid + s];
            uint ia = scratch_idx[lid];
            uint ib = scratch_idx[lid + s];
            if (vb > va || (vb == va && ib < ia)) {
                scratch_val[lid] = vb;
                scratch_idx[lid] = ib;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        out[row] = scratch_idx[0];
    }
}
