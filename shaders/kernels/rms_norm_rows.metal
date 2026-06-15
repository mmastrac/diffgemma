#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"

constant bool K_AFFINE [[function_constant(4)]];

/// Per-row Gemma RMSNorm (engine path, 1 thread/row, f32 I/O).
/// `K_AFFINE`: multiply by weight; else normalize only. Weight buffer unused when false.
kernel void rms_norm_rows(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint2 &dims [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    device float *dump [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    K_ELEMENTWISE_GUARD();
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
    K_ELEMENTWISE_GUARD();
    for (uint i = 0; i < hidden; i++) {
        float v = x[off + i] * rms_inv;
        if (K_AFFINE) {
            v *= weight[i];
        }
        out[off + i] = v;
    }
}
