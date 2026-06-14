#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_KERNEL_COMMON_METAL
#include "common.metal"
#endif

/// Per-row RMSNorm without affine weight: `out[s,:] = x[s,:] / rms(x[s,:])`.
kernel void rms_norm_rows_no_scale(
    device const float *x [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint2 &dims [[buffer(2)]],
    constant float &eps [[buffer(3)]],
    device float *dump [[buffer(4)]],
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
        out[off + i] = x[off + i] * rms_inv;
    }
}
