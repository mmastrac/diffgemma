#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "common.metal"
#include "attention_device.metal"
#include "arena.metal"
#include "sampler_device.metal"

/// Canvas queries attend all KV positions 0..kv_len+canvas-1 (no causal mask).
kernel void attention(
    device const ushort *q [[buffer(0)]],
    device const half *kvcache [[buffer(1)]],
    device ushort *out [[buffer(2)]],
    device const LayerOffsets *L [[buffer(3)]],
    constant StepParams &P [[buffer(4)]],
    constant AttnDims &dims [[buffer(5)]],
    device DebugStatus *dbg [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tpg [[threads_per_threadgroup]]
) {
    if (K_SHAPE_ASSERT && (dims.canvas == 0u || dims.n_q_heads == 0u)) {
        return;
    }
    K_ELEMENTWISE_GUARD();

    const uint hd = L->head_dim;
    const uint nkv = L->n_kv_heads;
    const uint tok = tgid.x;
    const uint qh = tgid.y;
    if (tok >= dims.canvas || qh >= dims.n_q_heads) {
        return;
    }
    const uint ltid = lid.x;
    const uint tpg_w = tpg.x;
    const uint kvh = qh / (dims.n_q_heads / nkv);
    const uint T = P.kv_len + dims.canvas;
    device const ushort *qv = q + (ulong)tok * dims.n_q_heads * hd + qh * hd;
    device const half *base = kvcache + L->kv_region / 2;
    threadgroup float red[8];
    float m = -INFINITY;
    float l = 0.f;
    float acc[8];
    const uint per = (hd + tpg_w - 1u) / tpg_w;
    const uint nsg = (tpg_w + 31u) / 32u;
    // KV axis: T is runtime (kv_len + canvas); loop 0..T streams softmax — no fixed score buffer.
    // acc[8]/red[8] are sized for max head_dim=512 @ tpg_w=64 (per=8, nsg=2); assert under tier-2.
    if (K_SHAPE_ASSERT && (per > 8u || nsg > 8u)) {
        if (tok == 0u && qh == 0u && ltid == 0u) {
            arena_store(out, 0, as_type<float>(0x7fc00000u));
        }
        return;
    }
    for (uint i = 0u; i < per; ++i) {
        acc[i] = 0.f;
    }
    for (uint t = 0u; t < T; ++t) {
        device const half *kk = base + (ulong)t * nkv * hd * 2u + kvh * hd;
        float d = 0.f;
        for (uint i = ltid; i < hd; i += tpg_w) {
            d += arena_load(qv, i) * float(kk[i]);
        }
        d = simd_sum(d);
        const uint sg = ltid / 32u;
        if ((ltid & 31u) == 0u) {
            red[sg] = d;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        d = 0.f;
        for (uint i = 0u; i < nsg; ++i) {
            d += red[i];
        }
        float mn = max(m, d);
        float corr = exp(m - mn);
        float p = exp(d - mn);
        l = l * corr + p;
        m = mn;
        device const half *vv = base + (ulong)t * nkv * hd * 2u + (ulong)nkv * hd + kvh * hd;
        for (uint i = 0u; i < per; ++i) {
            uint idx = ltid + i * tpg_w;
            if (idx < hd) {
                acc[i] = acc[i] * corr + p * float(vv[idx]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device ushort *ov = out + (ulong)tok * dims.n_q_heads * hd + qh * hd;
    if (ltid == 0u) {
        dgq_assert_positive_f32(dbg, DbgKernelAttention, l, (tok << 16u) | qh);
    }
    for (uint i = 0u; i < per; ++i) {
        uint idx = ltid + i * tpg_w;
        if (idx < hd) {
            float y = acc[i] / l;
            dgq_assert_finite_f32(dbg, DbgKernelAttention, y, idx);
            arena_store(ov, idx, y);
        }
    }
}
