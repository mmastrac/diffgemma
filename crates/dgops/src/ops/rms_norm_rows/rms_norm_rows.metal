#include <metal_stdlib>
using namespace metal;

/// Per-row Gemma RMSNorm: one thread per row, f32 I/O, optional affine scale.
kernel void rms_norm_rows(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint2 &dims [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint seq_len = dims.x;
    const uint hidden = dims.y;
    const uint s = gid;
    if (s >= seq_len) {
        return;
    }

    const uint off = s * hidden;
    float sum_sq = 0.0f;
    for (uint i = 0; i < hidden; i++) {
        float v = x[off + i];
        sum_sq += v * v;
    }
    float rms_inv = rsqrt(sum_sq / float(hidden) + eps);
    for (uint i = 0; i < hidden; i++) {
        out[off + i] = x[off + i] * rms_inv * weight[i];
    }
}
