#include <metal_stdlib>
using namespace metal;

constant float GELU_TANH_COEF = 0.7978845608028654;

kernel void rms_norm_rows(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint2 &dims [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint seq_len = dims.x;
    uint hidden = dims.y;
    uint s = gid;
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
    for (uint i = 0; i < hidden; i++) {
        out[off + i] = x[off + i] * rms_inv * weight[i];
    }
}

kernel void rms_norm_rows_no_scale(
    device const float *x [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint2 &dims [[buffer(2)]],
    constant float &eps [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint seq_len = dims.x;
    uint hidden = dims.y;
    uint s = gid;
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
    for (uint i = 0; i < hidden; i++) {
        out[off + i] = x[off + i] * rms_inv;
    }
}

kernel void vec_add_inplace(
    device float *out [[buffer(0)]],
    device const float *addend [[buffer(1)]],
    constant uint &len [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    out[gid] += addend[gid];
}

kernel void vec_mul_inplace(
    device float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    constant uint &len [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    a[gid] *= b[gid];
}

kernel void vec_scale_inplace(
    device float *x [[buffer(0)]],
    constant float &scale [[buffer(1)]],
    constant uint &len [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    x[gid] *= scale;
}

kernel void router_scale_rows(
    device float *x [[buffer(0)]],
    device const float *scale [[buffer(1)]],
    constant uint2 &dims [[buffer(2)]],
    constant float &root [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint seq_len = dims.x;
    uint hidden = dims.y;
    uint s = gid;
    if (s >= seq_len) {
        return;
    }
    uint off = s * hidden;
    for (uint i = 0; i < hidden; i++) {
        x[off + i] *= scale[i] * root;
    }
}

inline float gelu_tanh(float x) {
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanh(GELU_TANH_COEF * (x + 0.044715f * x3)));
}

kernel void gelu_pytorch_tanh(
    device float *x [[buffer(0)]],
    constant uint &len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    x[gid] = gelu_tanh(x[gid]);
}

kernel void swiglu_mul(
    device float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    constant uint &len [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    gate[gid] = gate[gid] * up[gid];
}
