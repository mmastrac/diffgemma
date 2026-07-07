#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "common.metal"
#include "attention_device.metal"
#include "arena.metal"
#include "sampler_device.metal"

// Session-wide KV storage format: q8 (group-32) for long-context sessions,
// f16 otherwise. Unset (oracle/test compiles) = f16.
constant uint KV_FMT_FC [[function_constant(4)]];
constant uint KV_FMT = is_function_constant_defined(KV_FMT_FC) ? KV_FMT_FC : 0u;
constant bool KV_Q8 = (KV_FMT == 1u);
constant bool KV_Q4 = (KV_FMT == 2u);

/// Canvas queries attend all KV positions 0..kv_len+canvas-1 (no causal mask)
/// when dims.causal==0 (denoise). When dims.causal!=0 (prefill), query at absolute
/// position kv_len+tok attends only [0..kv_len+tok] (causal); window<=1024 so for
/// prompts <=1024 plain causal == the engine's causal-sliding.
kernel void attention(
    device const ushort *q [[buffer(0)]],
    device const ushort *kvcache [[buffer(1)]],
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
    // Causal (prefill): query at absolute pos kv_len+tok attends only [0..pos].
    const uint T_all = P.kv_len + dims.canvas;
    const uint T = (dims.causal != 0u) ? min(T_all, P.kv_len + tok + 1u) : T_all;
    // Sliding window (dims.window != 0, sliding layers): keys below t_lo are
    // masked out. Denoise (bidirectional): canvas attends the last window-1
    // encoder positions + all canvas -> t_lo = kv_len - (window-1) (MLX
    // _make_decoder_masks). Causal prefill: query q attends [q-(window-1), q]
    // (engine CausalSliding). Keys are contiguous, so the mask is just a loop
    // start; t_lo = 0 (bit-identical) until the context outgrows the window.
    uint t_lo = 0u;
    if (dims.window != 0u) {
        const uint wm1 = dims.window - 1u;
        const uint qpos = (dims.causal != 0u) ? (P.kv_len + tok) : P.kv_len;
        t_lo = (qpos > wm1) ? (qpos - wm1) : 0u;
    }
    device const ushort *qv = q + (ulong)tok * dims.n_q_heads * hd + qh * hd;
    device const ushort *base = kvcache + L->kv_region / 2;
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
    device const uchar *base_b = (device const uchar *)kvcache + L->kv_region;
    const ulong q8_stride = kv_slot_stride_bytes(nkv, hd, KV_FMT);
    const ulong q8_row = kv_row_bytes(hd, KV_FMT);
    for (uint t = t_lo; t < T; ++t) {
        const uint ts = kv_slot_of(L, t);
        device const ushort *kk = base + (ulong)ts * nkv * hd * 2u + kvh * hd;
        device const uchar *krow = base_b + (ulong)ts * q8_stride + kvh * q8_row;
        float d = 0.f;
        for (uint i = ltid; i < hd; i += tpg_w) {
            const float kvv = KV_Q8 ? kv_q8_load(krow, i, hd) : kv_load(kk, i);
            d += arena_load(qv, i) * kvv;
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
        device const ushort *vv = base + (ulong)ts * nkv * hd * 2u + (ulong)nkv * hd + kvh * hd;
        device const uchar *vrow = base_b + (ulong)ts * q8_stride + ((ulong)nkv + kvh) * q8_row;
        for (uint i = 0u; i < per; ++i) {
            uint idx = ltid + i * tpg_w;
            if (idx < hd) {
                const float vval = KV_Q8 ? kv_q8_load(vrow, idx, hd) : kv_load(vv, idx);
                acc[i] = acc[i] * corr + p * vval;
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
