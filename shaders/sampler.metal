#include <metal_stdlib>
using namespace metal;

/// Copy `len` f32 elements device-to-device.
kernel void copy_f32(
    device const float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant uint &len [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    dst[gid] = src[gid];
}

kernel void logit_softcapping(
    device float *x [[buffer(0)]],
    constant float &cap [[buffer(1)]],
    constant uint &len [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    if (cap <= 0.0f) {
        return;
    }
    float v = x[gid];
    x[gid] = tanh(v / cap) * cap;
}

/// Divide every element by scalar temperature.
kernel void scale_logits(
    device float *x [[buffer(0)]],
    constant float &inv_t [[buffer(1)]],
    constant uint &len [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    x[gid] *= inv_t;
}

/// Scatter `[seq, chunk]` GEMM output into `[seq, vocab]` logits at column offset `v0`.
kernel void scatter_vocab_chunk(
    device const float *chunk [[buffer(0)]],
    device float *logits [[buffer(1)]],
    constant uint4 &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint seq_len = params.x;
    uint chunk_cols = params.y;
    uint v0 = params.z;
    uint vocab = params.w;
    uint row = gid.y;
    uint col = gid.x;
    if (row >= seq_len || col >= chunk_cols) {
        return;
    }
    logits[row * vocab + v0 + col] = chunk[row * chunk_cols + col];
}

/// Per-row argmax over `[rows, cols]` logits (temperature-scaled).
kernel void argmax_rows(
    device const float *logits [[buffer(0)]],
    device uint *out [[buffer(1)]],
    constant uint2 &dims [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    const uint TG = 256u;
    uint rows = dims.x;
    uint cols = dims.y;
    if (row >= rows) {
        return;
    }

    device const float *r = logits + row * cols;
    threadgroup float scratch_val[256];
    threadgroup uint scratch_idx[256];

    float local_best = -1e30f;
    uint local_idx = 0u;
    for (uint c = lid; c < cols; c += TG) {
        float v = r[c];
        if (v > local_best || (v == local_best && c < local_idx)) {
            local_best = v;
            local_idx = c;
        }
    }
    scratch_val[lid] = local_best;
    scratch_idx[lid] = local_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TG / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            float va = scratch_val[lid];
            float vb = scratch_val[lid + s];
            uint ia = scratch_idx[lid];
            uint ib = scratch_idx[lid + s];
            if (vb > va || (vb == va && ib < ia)) {
                scratch_val[lid] = vb;
                scratch_idx[lid] = ib;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        out[row] = scratch_idx[0];
    }
}

/// Shannon entropy per row from logits (softmax then `-sum p log p`).
kernel void row_entropy(
    device const float *logits [[buffer(0)]],
    device float *entropies [[buffer(1)]],
    constant uint2 &dims [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    const uint TG = 256u;
    uint rows = dims.x;
    uint cols = dims.y;
    if (row >= rows) {
        return;
    }

    device const float *r = logits + row * cols;
    threadgroup float scratch[256];

    float local_max = -1e30f;
    for (uint c = lid; c < cols; c += TG) {
        local_max = max(local_max, r[c]);
    }
    scratch[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TG / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            scratch[lid] = max(scratch[lid], scratch[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint c = lid; c < cols; c += TG) {
        local_sum += exp(r[c] - row_max);
    }
    scratch[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TG / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            scratch[lid] += scratch[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_h = 0.0f;
    for (uint c = lid; c < cols; c += TG) {
        float p = exp(r[c] - row_max) * inv_sum;
        if (p > 0.0f) {
            local_h -= p * log(p);
        }
    }
    scratch[lid] = local_h;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TG / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            scratch[lid] += scratch[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        entropies[row] = scratch[0];
    }
}

/// Categorical sample per row from row-major probabilities `[rows, cols]`.
kernel void sample_from_probs_rows(
    device const float *probs [[buffer(0)]],
    device const float *rand [[buffer(1)]],
    device uint *out [[buffer(2)]],
    constant uint2 &dims [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    uint rows = dims.x;
    uint cols = dims.y;
    if (row >= rows) {
        return;
    }

    device const float *r = probs + row * cols;
    float target = rand[row];
    float cum = 0.0f;
    uint chosen = 0u;
    for (uint c = 0u; c < cols; c++) {
        cum += r[c];
        if (target < cum) {
            chosen = c;
            break;
        }
    }
    out[row] = chosen;
}
