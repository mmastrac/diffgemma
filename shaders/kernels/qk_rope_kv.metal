#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "common.metal"
#include "attention_device.metal"
#include "sampler_device.metal"

/// Per-head Q/K RMSNorm + split-half RoPE + KV cache write (monolith step path).
kernel void qk_rope_kv(
    device half *q [[buffer(0)]],
    device half *k [[buffer(1)]],
    device half *v [[buffer(2)]],
    device half *kvcache [[buffer(3)]],
    device const uchar *blob [[buffer(4)]],
    device const LayerOffsets *L [[buffer(5)]],
    constant StepParams &P [[buffer(6)]],
    constant AttnDims &dims [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (K_SHAPE_ASSERT && (dims.canvas == 0u || dims.n_q_heads == 0u)) {
        return;
    }
    K_ELEMENTWISE_GUARD();

    const uint hd = L->head_dim;
    const uint nkv = L->n_kv_heads;
    const uint tok = gid.x;
    const uint h = gid.y;
    if (tok >= dims.canvas) {
        return;
    }
    const uint pos = P.kv_len + tok;
    const bool full = L->is_full != 0u;
    const uint rot = full ? (hd / 4u) : hd;
    const float theta = full ? 1.0e6f : 1.0e4f;

    const bool isQ = h < dims.n_q_heads;
    const bool isK = !isQ && h < (dims.n_q_heads + nkv);
    const uint hh = isQ ? h : (h - dims.n_q_heads) % nkv;
    device half *src = isQ ? (q + (ulong)tok * dims.n_q_heads * hd + hh * hd)
                     : isK ? (k + (ulong)tok * nkv * hd + hh * hd)
                     : ((L->v_proj != 0ul ? v : k) + (ulong)tok * nkv * hd + hh * hd);

    float ss = 0.f;
    for (uint i = 0u; i < hd; ++i) {
        float t = float(src[i]);
        ss += t * t;
    }
    float inv = rsqrt(ss / float(hd) + ATTN_RMS_EPS);
    ulong noff = isQ ? L->q_norm : isK ? L->k_norm : 0ul;
    if (isQ || isK) {
        // Full-attn layers alias V from raw k_proj: do not mutate `k` in place on the K path.
        float head[512];
        for (uint i = 0u; i < hd; ++i) {
            head[i] = float(src[i]) * inv * bf16_bytes(blob + noff + 2ul * i);
        }
        if (full) {
            apply_proportional_rope_f32(head, rot, hd, theta, pos);
        } else {
            apply_split_half_rope_f32(head, rot, hd, theta, pos);
        }
        if (isK) {
            device half *dst = kvcache + L->kv_region / 2 + (ulong)pos * nkv * hd * 2u + hh * hd;
            for (uint i = 0u; i < hd; ++i) {
                dst[i] = half(head[i]);
            }
            if (L->v_proj != 0ul) {
                for (uint i = 0u; i < hd; ++i) {
                    src[i] = half(head[i]);
                }
            }
        } else {
            for (uint i = 0u; i < hd; ++i) {
                src[i] = half(head[i]);
            }
        }
    } else {
        device half *dst = kvcache + L->kv_region / 2 + (ulong)pos * nkv * hd * 2u
            + (ulong)nkv * hd + hh * hd;
        for (uint i = 0u; i < hd; ++i) {
            dst[i] = half(float(src[i]) * inv);
        }
    }
}
