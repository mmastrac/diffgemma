#include <metal_stdlib>
using namespace metal;

constant float GELU_TANH_COEF = 0.7978845608028654;

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

kernel void gather_rows(
    device const float *src [[buffer(0)]],
    device const uint *indices [[buffer(1)]],
    device float *dst [[buffer(2)]],
    constant uint2 &dims [[buffer(3)]],
    constant uint &batch_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint hidden = dims.y;
    uint bi = gid / hidden;
    uint h = gid % hidden;
    if (bi >= batch_size) {
        return;
    }
    uint tok = indices[bi];
    dst[bi * hidden + h] = src[tok * hidden + h];
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

/// Copy `[rows, chunk]` columns starting at `v0` from row-major `[rows, vocab]` probs.
kernel void gather_prob_cols(
    device const float *probs [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint4 &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint rows = params.x;
    uint vocab = params.y;
    uint v0 = params.z;
    uint chunk = params.w;
    uint col = gid.x;
    uint row = gid.y;
    if (row >= rows || col >= chunk) {
        return;
    }
    out[row * chunk + col] = probs[row * vocab + v0 + col];
}

kernel void vec_fill_zero(
    device float *x [[buffer(0)]],
    constant uint2 &range [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    uint base = range.x;
    uint len = range.y;
    uint i = base + gid;
    if (gid >= len) {
        return;
    }
    x[i] = 0.0f;
}
