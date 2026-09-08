#include <metal_stdlib>
using namespace metal;

/// One threadgroup per row; 256 threads reduce and then write the row.
kernel void rms_norm_rows(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint2 &dims [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    const uint TG = 256u;
    uint rows = dims.x;
    uint hidden = dims.y;
    uint row = tgp.y;
    if (row >= rows) {
        return;
    }
    threadgroup float scratch[256];

    const device float *xr = x + (ulong)row * hidden;
    device float *orow = out + (ulong)row * hidden;

    float sum = 0.0f;
    for (uint i = lid; i < hidden; i += TG) {
        float v = xr[i];
        sum += v * v;
    }
    scratch[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TG / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            scratch[lid] += scratch[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(scratch[0] / float(hidden) + eps);
    for (uint i = lid; i < hidden; i += TG) {
        orow[i] = xr[i] * inv * weight[i];
    }
}
