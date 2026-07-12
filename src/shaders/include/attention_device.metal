#ifndef DGQ_INCLUDE_ATTENTION_DEVICE_METAL
#define DGQ_INCLUDE_ATTENTION_DEVICE_METAL

#include <metal_stdlib>
using namespace metal;

#include "common.metal"
#include "arena.metal"

constant float ATTN_RMS_EPS = 1e-6f;

struct LayerOffsets {
    ulong input_ln, q_proj, q_norm, k_proj, k_norm, v_proj, o_proj, post_attn_ln;
    ulong pre_ff_ln, mlp_gate, mlp_up, mlp_down, post_ff_ln_1;
    ulong router_scale, router_proj, per_expert_scale, pre_ff_ln_2;
    ulong experts_gate_up, experts_down, post_ff_ln_2, post_ff_ln, layer_scalar;
    ulong kv_region;
    uint head_dim;
    uint n_kv_heads;
    uint is_full;
    // KV slot mapping: 0 = linear (full layers); else pow2-1 ring mask
    // (sliding layers, slot = pos & mask — only the last window-1 + canvas
    // positions are live, so the ring never aliases a readable key).
    uint kv_ring_mask;
};

/// Absolute KV position -> slot index within the layer's region.
inline uint kv_slot_of(constant LayerOffsets *L, uint pos) {
    return (L->kv_ring_mask != 0u) ? (pos & L->kv_ring_mask) : pos;
}
inline uint kv_slot_of(device const LayerOffsets *L, uint pos) {
    return (L->kv_ring_mask != 0u) ? (pos & L->kv_ring_mask) : pos;
}

/// KV cache element load/store. The cache stores **f16** (was bf16): the
/// values are RMS-normed K (post-RoPE) and V with |x| <~ 22 (measured), well
/// inside f16 range, and f16's 10 mantissa bits beat bf16's 7 everywhere in
/// the live range. f16 storage also lets the MMA attention kernels
/// simdgroup_load K/V tiles STRAIGHT from device memory (no bf16->half
/// staging pass) — the enabler for long-context attention throughput.
inline float kv_load(device const ushort *p, uint i) {
    return float(as_type<half>(p[i]));
}
inline void kv_store(device ushort *p, uint i, float v) {
    p[i] = as_type<ushort>(half(v));
}

// ---- q8 KV (long-context sessions; session-wide KV_Q8 function constant) ----
//
// Row = one (slot, head, K-or-V) vector: [hd x i8][hd/32 x f16 group scales].
// Group-32 symmetric: measured 0.92% rel-RMS on real KV (vs 2.25% per-vector
// — KV has outlier channels), 1.88x fewer bytes than f16. Slot layout stays
// [K rows x nkv][V rows x nkv].
constant uint KV_Q8_GROUP = 32u;

// KV storage format code (mirrors Rust KvFormat::code(); the KV_FMT function
// constant): 0 = f16, 1 = q8 (group-32 i8+f16 scales), 2 = q4 (group-32
// i4+f16 scales, reserved for the rotated-KV lever). Row bytes per format:
inline ulong kv_row_bytes(uint hd, uint fmt) {
    switch (fmt) {
        case 1u: return (ulong)hd + (ulong)(hd / KV_Q8_GROUP) * 2ul;        // q8
        case 2u: return (ulong)(hd / 2u) + (ulong)(hd / KV_Q8_GROUP) * 2ul; // q4
        default: return (ulong)hd * 2ul;                                    // f16
    }
}

/// Byte stride between consecutive KV slots (one position's K+V, all heads).
inline ulong kv_slot_stride_bytes(uint nkv, uint hd, uint fmt) {
    return 2ul * (ulong)nkv * kv_row_bytes(hd, fmt);
}

/// Dequantize one element of a q8 row (scalar fallback — the MMA kernels use
/// the vectorized group dequant below).
inline float kv_q8_load(device const uchar *row, uint d, uint hd) {
    const half s = *(device const half *)(row + hd + (d / KV_Q8_GROUP) * 2u);
    return float(*(device const char *)(row + d)) * float(s);
}

/// Dequantize one 32-group of a q8 row into a BLOCKED [chunk][key][d%8] tile
/// (`tile` = &kq8[0][0][0], element (c,key,d) at c*64 + key*8 + d). ONE scale
/// load + 8 x (char4 load -> half4 multiply -> aligned half4 store): the
/// per-element scale-gather/divide version of this was 4x slower end-to-end.
inline void kv_q8_dequant_group_blocked(
    device const uchar *row,
    uint g,
    uint hd,
    uint key,
    threadgroup half *tile
) {
    const half s = *(device const half *)(row + hd + g * 2u);
    device const char4 *src = (device const char4 *)(row + g * KV_Q8_GROUP);
    for (uint j4 = 0u; j4 < KV_Q8_GROUP / 4u; ++j4) {
        const char4 c = src[j4];
        const uint d0 = g * KV_Q8_GROUP + j4 * 4u;
        // 4 consecutive d stay inside one 8-wide chunk (d0 is a multiple of 4)
        // and the target is 8-byte aligned -> one half4 store.
        threadgroup half4 *dst = (threadgroup half4 *)(tile + (d0 >> 3u) * 64u + key * 8u + (d0 & 7u));
        *dst = half4(float4(c.x, c.y, c.z, c.w)) * s;
    }
}

/// Quantize one 32-group into a q8 row (group-32 symmetric, rint =
/// round-half-even to match the Rust mirror's round_ties_even).
inline void kv_q8_store_group(device uchar *row, uint g, thread const float *src, uint hd) {
    float mx = 0.f;
    for (uint j = 0u; j < KV_Q8_GROUP; ++j) {
        mx = max(mx, fabs(src[j]));
    }
    const float scale = max(mx / 127.0f, 1e-8f);
    // Quantize with the f16-ROUNDED scale so dequant is exact w.r.t. the
    // stored scale (and the CPU mirror agrees bit-for-bit).
    const half sh = half(scale);
    const float sf = float(sh);
    *(device half *)(row + hd + g * 2u) = sh;
    for (uint j = 0u; j < KV_Q8_GROUP; ++j) {
        const float q = rint(src[j] / sf);
        *(device char *)(row + g * KV_Q8_GROUP + j) = char(clamp(q, -127.0f, 127.0f));
    }
}

/// Quantize `hd` floats into a q8 row.
inline void kv_q8_store_row(device uchar *row, thread const float *src, uint hd) {
    for (uint g = 0u; g < hd / KV_Q8_GROUP; ++g) {
        float grp[KV_Q8_GROUP];
        for (uint j = 0u; j < KV_Q8_GROUP; ++j) {
            grp[j] = src[g * KV_Q8_GROUP + j];
        }
        kv_q8_store_group(row, g, grp, hd);
    }
}

struct AttnDims {
    uint canvas;
    uint n_q_heads;
    uint causal;   // 0 = bidirectional all-valid (denoise); 1 = causal (prefill)
    uint window;   // sliding-window size (0 = unwindowed/full-attention layer)
};

/// KV block range for flash-decode style sequential-block attention
/// (full-attention layers at long kv). Blocks are dispatched IN ORDER with
/// the online-softmax state (m, l, unnormalized O — all f32) persisted in a
/// scratch buffer between dispatches, so the result is bit-identical to one
/// monolithic pass; the win is SLC locality (every threadgroup streams the
/// same <=block-sized key window per dispatch instead of the whole cache).
struct AttnBlockRange {
    uint t_begin;    // first key position of this block (8-aligned)
    uint t_end;      // one past the last key position of this block
    uint is_first;   // init state instead of loading it
    uint is_last;    // normalize + write `out` instead of storing state
};

/// Split-half RoPE on the first `rot` dims; inv_freq denominator is full `head_dim`.
inline void apply_split_half_rope_f32(
    thread float *src,
    uint rot,
    uint head_dim,
    float theta,
    uint pos
) {
    const uint half_rot = rot / 2u;
    for (uint d = 0u; d < half_rot; ++d) {
        float inv_freq = pow(theta, -2.0f * float(d) / float(head_dim));
        float a = float(pos) * inv_freq;
        float c = cos(a);
        float s = sin(a);
        float x0 = src[d];
        float x1 = src[d + half_rot];
        src[d] = x0 * c - x1 * s;
        src[d + half_rot] = x0 * s + x1 * c;
    }
}

/// Gemma proportional RoPE: rotate left[i] with right[i] for i < rot/2.
inline void apply_proportional_rope_f32(
    thread float *src,
    uint rotary_dim,
    uint head_dim,
    float theta,
    uint pos
) {
    const uint half_head = head_dim / 2u;
    const uint half_rot = rotary_dim / 2u;
    for (uint d = 0u; d < half_rot; ++d) {
        float inv_freq = pow(theta, -2.0f * float(d) / float(head_dim));
        float a = float(pos) * inv_freq;
        float c = cos(a);
        float s = sin(a);
        float x0 = src[d];
        float x1 = src[half_head + d];
        src[d] = x0 * c - x1 * s;
        src[half_head + d] = x0 * s + x1 * c;
    }
}

inline void apply_split_half_rope(
    device ushort *src,
    uint rot,
    uint head_dim,
    float theta,
    uint pos
) {
    const uint half_rot = rot / 2u;
    for (uint d = 0u; d < half_rot; ++d) {
        float inv_freq = pow(theta, -2.0f * float(d) / float(head_dim));
        float a = float(pos) * inv_freq;
        float c = cos(a);
        float s = sin(a);
        float x0 = arena_load(src, d);
        float x1 = arena_load(src, d + half_rot);
        arena_store(src, d, x0 * c - x1 * s);
        arena_store(src, d + half_rot, x0 * s + x1 * c);
    }
}

#endif
