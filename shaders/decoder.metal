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

inline float gelu_tanh(float x) {
    float x3 = x * x * x;
    float u = GELU_TANH_COEF * (x + 0.044715f * x3);
    // Metal fast-math tanh can return NaN for large |u|; GELU only needs ±1 limits.
    float t = (u > 8.0f) ? 1.0f : (u < -8.0f) ? -1.0f : tanh(u);
    return 0.5f * x * (1.0f + t);
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

/// MoE gate_up layout `[batch, 2 * moe_inter]` -> activated `[batch, moe_inter]`.
kernel void gelu_swiglu_gate_up(
    device const float *gate_up [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint2 &dims [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint batch_size = dims.x;
    uint moe_inter = dims.y;
    uint row = gid / moe_inter;
    uint col = gid % moe_inter;
    if (row >= batch_size) {
        return;
    }
    uint off = row * (2u * moe_inter) + col;
    float g = gelu_tanh(gate_up[off]);
    float u = gate_up[off + moe_inter];
    out[gid] = g * u;
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

/// Top-k expert routing per row from softmax probs `[rows, experts]`.
/// Tie-break: higher prob wins; equal prob → lower expert index wins (bit-exact vs CPU oracle).
/// Outputs renormed top-k weights scaled by `per_expert_scale`.
kernel void router_top_k_rows(
    device const float *probs [[buffer(0)]],
    device const float *per_expert_scale [[buffer(1)]],
    device uint *out_indices [[buffer(2)]],
    device float *out_weights [[buffer(3)]],
    constant uint3 &params [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    uint rows = params.x;
    uint experts = params.y;
    uint k = params.z;
    if (row >= rows || k == 0u || k > 32u) {
        return;
    }

    float top_p[32];
    uint top_i[32];
    for (uint i = 0u; i < k; i++) {
        top_p[i] = -1.0f;
        top_i[i] = 0u;
    }

    for (uint e = 0u; e < experts; e++) {
        float p = probs[row * experts + e];
        uint insert_at = k;
        for (uint i = 0u; i < k; i++) {
            if (p > top_p[i] || (p == top_p[i] && e < top_i[i])) {
                insert_at = i;
                break;
            }
        }
        if (insert_at < k) {
            for (uint j = k - 1u; j > insert_at; j--) {
                top_p[j] = top_p[j - 1u];
                top_i[j] = top_i[j - 1u];
            }
            top_p[insert_at] = p;
            top_i[insert_at] = e;
        }
    }

    float sum = 0.0f;
    for (uint i = 0u; i < k; i++) {
        sum += top_p[i];
    }
    float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;

    uint off = row * k;
    for (uint i = 0u; i < k; i++) {
        out_indices[off + i] = top_i[i];
        out_weights[off + i] = top_p[i] * inv * per_expert_scale[top_i[i]];
    }
}
