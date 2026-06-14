#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_KERNEL_COMMON_METAL
#include "common.metal"
#endif

/// Per-row Gemma RMSNorm: `out[s,:] = x[s,:] / rms(x[s,:]) * weight`.
kernel void rms_norm_rows(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint2 &dims [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    device float *dump [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint seq_len = dims.x;
    uint hidden = dims.y;
    uint s = gid;
    if (K_SHAPE_ASSERT && (hidden == 0u || seq_len == 0u)) {
        return;
    }
    if (s >= seq_len) {
        return;
    }

    uint off = s * hidden;
    float sum_sq = 0.0f;
    for (uint i = 0; i < hidden; i++) {
        float v = x[off + i];
        sum_sq += v * v;
    }
    float rms_inv = rsqrt(sum_sq / float(hidden) + eps);
    if (K_DUMP_STAGE >= 1u) {
        dump[s] = rms_inv;
    }
    (void)K_USE_FP4;
    for (uint i = 0; i < hidden; i++) {
        out[off + i] = x[off + i] * rms_inv * weight[i];
    }
}
