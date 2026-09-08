#include <metal_stdlib>
using namespace metal;

/// One threadgroup per row: reduce sum(x^2) and sum(dy*w*x) together, then
/// write dx = inv*dy*w - inv^3 * x * dot / hidden.
kernel void rms_norm_backward(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *dy [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint2 &dims [[buffer(4)]],
    constant float &eps [[buffer(5)]],
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
    threadgroup float scratch_sq[256];
    threadgroup float scratch_dot[256];

    const device float *xr = x + (ulong)row * hidden;
    const device float *dr = dy + (ulong)row * hidden;
    device float *orow = out + (ulong)row * hidden;

    float sum_sq = 0.0f;
    float dot = 0.0f;
    for (uint i = lid; i < hidden; i += TG) {
        float xv = xr[i];
        sum_sq += xv * xv;
        dot += dr[i] * weight[i] * xv;
    }
    scratch_sq[lid] = sum_sq;
    scratch_dot[lid] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TG / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            scratch_sq[lid] += scratch_sq[lid + s];
            scratch_dot[lid] += scratch_dot[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(scratch_sq[0] / float(hidden) + eps);
    float inv3 = inv * inv * inv;
    float coef = inv3 * scratch_dot[0] / float(hidden);
    for (uint i = lid; i < hidden; i += TG) {
        orow[i] = inv * dr[i] * weight[i] - coef * xr[i];
    }
}
