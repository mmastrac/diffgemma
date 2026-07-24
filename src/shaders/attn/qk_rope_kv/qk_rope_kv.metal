#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "arena.metal"
#include "attention_device.metal"
#include "sampler_device.metal"

// Session-wide KV storage format: q8 (group-32) for long-context sessions,
// f16 otherwise. Unset (oracle/test compiles) = f16.
constant uint KV_FMT_FC [[function_constant(4)]];
constant uint KV_FMT = is_function_constant_defined(KV_FMT_FC) ? KV_FMT_FC : 0u;
constant bool KV_Q8 = (KV_FMT == 1u);
constant bool KV_Q4 = (KV_FMT == 2u);

// During fast prefill, every layer ALSO writes post-RoPE K / normed V
// into an f32 side cache (buffer 9, bound at the layer's side offset; same
// kv_slot_of addressing — ring for sliding layers, linear for full). The
// prefill attention reads the side instead of the f16 cache, so
// chunk-boundary rounding never compounds. Denoise is untouched (the variant
// is dispatched only for prefill; the f16/q8 monolithic write above stays
// authoritative for denoise).
constant bool KV_F32_SIDE_FC [[function_constant(30)]];
constant bool KV_F32_SIDE =
    is_function_constant_defined(KV_F32_SIDE_FC) && KV_F32_SIDE_FC;

/// Per-head Q/K RMSNorm + split-half RoPE + KV cache write (monolith step path).
kernel void qk_rope_kv(
    device ushort *q [[buffer(0)]],
    device ushort *k [[buffer(1)]],
    device ushort *v [[buffer(2)]],
    device ushort *kvcache [[buffer(3)]],
    device const uchar *blob [[buffer(4)]],
    device const LayerOffsets *L [[buffer(5)]],
    constant StepParams &P [[buffer(6)]],
    constant AttnDims &dims [[buffer(7)]],
    device DebugStatus *dbg [[buffer(8)]],
    device float *kv_f32_side [[buffer(9), function_constant(KV_F32_SIDE_FC)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (K_SHAPE_ASSERT && (dims.canvas == 0u || dims.n_q_heads == 0u)) {
        return;
    }
    K_ELEMENTWISE_GUARD();

    const uint hd = L->head_dim;
    dgq_assert_fast(hd > 0u && hd <= 512u, dbg, DbgErrBadDim, DbgKernelQkRopeKv, hd);
    const uint nkv = L->n_kv_heads;
    const uint tok = gid.x;
    const uint h = gid.y;
    if (tok >= dims.canvas) {
        return;
    }
    const uint pos = P.kv_len + tok;
    // Prefill pads the tail chunk with zero rows; their positions are >=
    // kv_write_end and must never reach the cache (a pad position wraps onto
    // a LIVE sliding-ring slot and clobbers the oldest window content).
    const bool kv_write = pos < P.kv_write_end;
    const bool full = L->is_full != 0u;
    const uint rot = full ? (hd / 4u) : hd;
    const float theta = full ? 1.0e6f : 1.0e4f;

    const bool isQ = h < dims.n_q_heads;
    const bool isK = !isQ && h < (dims.n_q_heads + nkv);
    const uint hh = isQ ? h : (h - dims.n_q_heads) % nkv;
    device ushort *src = isQ ? (q + (ulong)tok * dims.n_q_heads * hd + hh * hd)
                     : isK ? (k + (ulong)tok * nkv * hd + hh * hd)
                     : ((L->v_proj != 0ul ? v : k) + (ulong)tok * nkv * hd + hh * hd);

    float ss = 0.f;
    for (uint i = 0u; i < hd; ++i) {
        float t = arena_load(src, i);
        ss += t * t;
    }
    float inv = rsqrt(ss / float(hd) + ATTN_RMS_EPS);
    dgq_assert_finite_f32(dbg, DbgKernelQkRopeKv, inv, tok);
    ulong noff = isQ ? L->q_norm : isK ? L->k_norm : 0ul;
    if (isQ || isK) {
        // Full-attn layers alias V from raw k_proj: do not mutate `k` in place on the K path.
        float head[512];
        for (uint i = 0u; i < hd; ++i) {
            head[i] = arena_load(src, i) * inv * bf16_bytes(blob + noff + 2ul * i);
        }
        if (full) {
            apply_proportional_rope_f32(head, rot, hd, theta, pos);
        } else {
            apply_split_half_rope_f32(head, rot, hd, theta, pos);
        }
        if (isK) {
            // RoPE uses the ABSOLUTE pos; only the storage slot is ring-mapped.
            if (!kv_write) {
            } else if (KV_Q8) {
                device uchar *row = (device uchar *)kvcache + L->kv_region
                    + (ulong)kv_slot_of(L, pos) * kv_slot_stride_bytes(nkv, hd, KV_FMT)
                    + hh * kv_row_bytes(hd, KV_FMT);
                kv_q8_store_row(row, head, hd);
            } else {
                device ushort *dst = kvcache + L->kv_region / 2
                    + (ulong)kv_slot_of(L, pos) * nkv * hd * 2u + hh * hd;
                for (uint i = 0u; i < hd; ++i) {
                    kv_store(dst, i, head[i]);  // KV cache is f16 (see kv_store)
                }
            }
            if (KV_F32_SIDE && kv_write) {
                device float *sd = kv_f32_side
                    + (ulong)kv_slot_of(L, pos) * nkv * hd * 2u + hh * hd;
                for (uint i = 0u; i < hd; ++i) {
                    sd[i] = head[i];
                }
            }
            if (L->v_proj != 0ul) {
                for (uint i = 0u; i < hd; ++i) {
                    arena_store(src, i, head[i]);
                }
            }
        } else {
            for (uint i = 0u; i < hd; ++i) {
                arena_store(src, i, head[i]);
            }
        }
    } else {
        if (!kv_write) {
        } else if (KV_Q8) {
            float head[512];
            for (uint i = 0u; i < hd; ++i) {
                head[i] = arena_load(src, i) * inv;
            }
            device uchar *row = (device uchar *)kvcache + L->kv_region
                + (ulong)kv_slot_of(L, pos) * kv_slot_stride_bytes(nkv, hd, KV_FMT)
                + ((ulong)nkv + hh) * kv_row_bytes(hd, KV_FMT);
            kv_q8_store_row(row, head, hd);
        } else {
            device ushort *dst = kvcache + L->kv_region / 2
                + (ulong)kv_slot_of(L, pos) * nkv * hd * 2u + (ulong)nkv * hd + hh * hd;
            for (uint i = 0u; i < hd; ++i) {
                kv_store(dst, i, arena_load(src, i) * inv);  // KV cache is f16
            }
        }
        if (KV_F32_SIDE && kv_write) {
            device float *sd = kv_f32_side
                + (ulong)kv_slot_of(L, pos) * nkv * hd * 2u
                + (ulong)nkv * hd + hh * hd;
            for (uint i = 0u; i < hd; ++i) {
                sd[i] = arena_load(src, i) * inv;
            }
        }
    }
}
