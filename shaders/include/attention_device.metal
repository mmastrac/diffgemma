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
