#include <metal_stdlib>
using namespace metal;

kernel void softmax_backward(
    device const float *probs [[buffer(0)]],
    device const float *dp [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint2 &dims [[buffer(3)]],
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
    threadgroup float scratch[256];

    const device float *pr = probs + (ulong)row * cols;
    const device float *dr = dp + (ulong)row * cols;
    device float *orow = out + (ulong)row * cols;

    float s = 0.0f;
    for (uint i = lid; i < cols; i += TG) {
        s += pr[i] * dr[i];
    }
    scratch[lid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = TG / 2u; step > 0u; step >>= 1u) {
        if (lid < step) {
            scratch[lid] += scratch[lid + step];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = scratch[0];
    for (uint i = lid; i < cols; i += TG) {
        orow[i] = pr[i] * (dr[i] - row_sum);
    }
}
