#include <metal_stdlib>
using namespace metal;

// MTP draft head, M=1 kernels. Activations f32, weights bf16 as shipped.
// Simplicity over throughput: one draft token is ~0.6 GFLOP and every op
// here is memory-bound at these shapes.

struct MtpDims {
    uint out_dim;
    uint in_dim;
};

// One simdgroup per output row: lanes stride the row in bfloat4 steps, so
// each simdgroup reads contiguous 256-byte bursts (the thread-per-row form
// made adjacent lanes read a full row apart and left the lm_head sweep
// hopelessly uncoalesced). in_dim must be a multiple of 4; every head dim is.
kernel void mtp_matvec_bf16(
    device const bfloat *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant MtpDims &dims [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tg_id * 8 + simd_id;
    if (row >= dims.out_dim) {
        return;
    }
    ulong base = ulong(row) * dims.in_dim;
    device const bfloat4 *w4 = (device const bfloat4 *)(w + base);
    device const float4 *x4 = (device const float4 *)x;
    uint n4 = dims.in_dim / 4;
    float acc = 0.0f;
    for (uint i = lane; i < n4; i += 32) {
        acc += dot(float4(w4[i]), x4[i]);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        out[row] = acc;
    }
}

// Rows of length `hidden`, one shared weight vector (row = a canvas row or
// one attention head).
kernel void mtp_rmsnorm(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint2 &dims [[buffer(3)]],  // rows, hidden
    constant float &eps [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= dims.x) {
        return;
    }
    uint hidden = dims.y;
    ulong off = ulong(gid) * hidden;
    float sum_sq = 0.0f;
    for (uint i = 0; i < hidden; i++) {
        float v = x[off + i];
        sum_sq += v * v;
    }
    float inv = 1.0f / sqrt(sum_sq / float(hidden) + eps);
    for (uint i = 0; i < hidden; i++) {
        out[off + i] = x[off + i] * inv * weight[i];
    }
}

// Both Gemma 4 flavors pair d with head_dim/2 + d and use head_dim in the
// frequency exponent; they differ only in rot_dim (sliding: full head,
// full/global: head_dim/4). Thread per (head, pair).
kernel void mtp_rope(
    device float *q [[buffer(0)]],
    constant uint4 &p [[buffer(1)]],  // n_heads, head_dim, rot_dim, pos
    constant float &theta [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint half_rot = p.z / 2;
    uint h = gid / half_rot;
    uint d = gid % half_rot;
    if (h >= p.x) {
        return;
    }
    ulong off = ulong(h) * p.y;
    uint pair = p.y / 2 + d;
    float inv_freq = pow(theta, -2.0f * float(d) / float(p.y));
    float a = float(p.w) * inv_freq;
    float c = cos(a);
    float s = sin(a);
    float x0 = q[off + d];
    float x1 = q[off + pair];
    q[off + d] = x0 * c - x1 * s;
    q[off + pair] = x0 * s + x1 * c;
}

struct MtpAttnDims {
    uint n_q;
    uint n_kv;
    uint hd;
    uint seq;
    uint kv_len;
};

// One simdgroup per query head, online softmax, scale 1.0 (Gemma 4). K/V are
// f32 head-major [n_kv, seq, hd] planes. Lanes stride the head dim so K/V
// reads coalesce; scores reduce with simd_sum; V accumulates lane-locally
// with running rescale. hd must be a multiple of 32 (256 and 512 are).
kernel void mtp_attend(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant MtpAttnDims &d [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint head = tg_id * 8 + simd_id;
    if (head >= d.n_q) {
        return;
    }
    uint group = d.n_q / d.n_kv;
    ulong qoff = ulong(head) * d.hd;
    ulong plane = ulong(head / group) * d.seq * d.hd;
    uint per_lane = d.hd / 32;
    float qr[16];
    float acc[16];
    for (uint j = 0; j < per_lane; j++) {
        qr[j] = q[qoff + lane + 32 * j];
        acc[j] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;
    for (uint t = 0; t < d.kv_len; t++) {
        ulong koff = plane + ulong(t) * d.hd;
        float part = 0.0f;
        for (uint j = 0; j < per_lane; j++) {
            part += qr[j] * k[koff + lane + 32 * j];
        }
        float sc = simd_sum(part);
        float mn = max(m, sc);
        float corr = exp(m - mn);
        float w = exp(sc - mn);
        l = l * corr + w;
        for (uint j = 0; j < per_lane; j++) {
            acc[j] = acc[j] * corr + w * v[koff + lane + 32 * j];
        }
        m = mn;
    }
    for (uint j = 0; j < per_lane; j++) {
        out[qoff + lane + 32 * j] = acc[j] / l;
    }
}

kernel void mtp_gelu_mul(
    device float *g [[buffer(0)]],
    device const float *u [[buffer(1)]],
    constant uint &len [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    float x = g[gid];
    float x3 = x * x * x;
    // Fast-math tanh is the naive exp form, NaN past |x| ~ 44; tanh saturates
    // to +-1 well before the clamp, so this is exact.
    float t = clamp(0.7978846f * (x + 0.044715f * x3), -15.0f, 15.0f);
    float gelu = 0.5f * x * (1.0f + tanh(t));
    g[gid] = gelu * u[gid];
}

// h = (h + x) * scale; scale 1.0 for plain residual adds, layer_scalar when
// fused with the layer's closing multiply.
kernel void mtp_add_scale(
    device float *h [[buffer(0)]],
    device const float *x [[buffer(1)]],
    constant uint &len [[buffer(2)]],
    constant float &scale [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    h[gid] = (h[gid] + x[gid]) * scale;
}

// Fused-round support: argmax and embed feedback stay on-GPU so a K-token
// draft round is one command buffer, no CPU round-trips.

struct ArgmaxPair {
    float v;
    uint i;
};

// Stage 1: 256 threadgroups scan strided; one (max, index) partial per group.
// Ties resolve to the lower index, matching the CPU scan.
kernel void mtp_argmax_stage1(
    device const float *logits [[buffer(0)]],
    device ArgmaxPair *partials [[buffer(1)]],
    constant uint &len [[buffer(2)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    float best = -INFINITY;
    uint arg = 0;
    for (uint i = tg_id * 256 + tid; i < len; i += 256 * 256) {
        float v = logits[i];
        if (v > best || (v == best && i < arg)) {
            best = v;
            arg = i;
        }
    }
    threadgroup ArgmaxPair sg[8];
    float mv = simd_max(best);
    // Lowest index among lanes holding the simd max.
    uint cand = (best == mv) ? arg : 0xFFFFFFFFu;
    uint mi = simd_min(cand);
    if (lane == 0) {
        sg[simd_id] = ArgmaxPair{mv, mi};
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        ArgmaxPair out = sg[0];
        for (uint s = 1; s < 8; s++) {
            ArgmaxPair p = sg[s];
            if (p.v > out.v || (p.v == out.v && p.i < out.i)) {
                out = p;
            }
        }
        partials[tg_id] = out;
    }
}

// Stage 2: one threadgroup folds the 256 partials and appends the winning
// token id to the round's token list.
kernel void mtp_argmax_stage2(
    device const ArgmaxPair *partials [[buffer(0)]],
    device uint *toks [[buffer(1)]],
    constant uint &slot [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) {
        return;
    }
    ArgmaxPair out = partials[0];
    for (uint s = 1; s < 256; s++) {
        ArgmaxPair p = partials[s];
        if (p.v > out.v || (p.v == out.v && p.i < out.i)) {
            out = p;
        }
    }
    toks[slot] = out.i;
}

// x[0..2816] = target_embed[toks[slot]] * scale (the bf16-rounded sqrt(2816)).
kernel void mtp_gather_embed(
    device const bfloat *embed [[buffer(0)]],
    device const uint *toks [[buffer(1)]],
    device float *x [[buffer(2)]],
    constant uint &slot [[buffer(3)]],
    constant float &scale [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= 2816) {
        return;
    }
    ulong base = ulong(toks[slot]) * 2816;
    x[gid] = float(embed[base + gid]) * scale;
}
