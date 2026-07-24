#ifndef DGQ_INCLUDE_ARENA_METAL
#define DGQ_INCLUDE_ARENA_METAL

#include "common.metal"
#include "arena_fc.metal"

// Activation arena: 2-byte slots — bf16 by default (settled precision policy),
// fp16 when compiled with K_ARENA_F16 (fp16 prefill stream: 2^-11 relative
// step instead of 2^-8; M0 range trace shows all planes peak <130 vs fp16 max
// 65504). Same byte layout either way; the FC folds at compile time.

inline ushort arena_bf16_bits(float x) {
    return ushort(as_type<uint>(f32_round_bf16(x)) >> 16);
}

// Explicit-bf16 aliases (logits / large-range buffers) — NOT switched by
// K_ARENA_F16.
inline float arena_load_bf16(device const ushort *buf, ulong i) {
    return bf16_to_f32(buf[i]);
}
inline void arena_store_bf16(device ushort *buf, ulong i, float x) {
    buf[i] = arena_bf16_bits(x);
}

inline float arena_load(device const ushort *buf, ulong i) {
    return K_ARENA_F16 ? float(as_type<half>(buf[i])) : bf16_to_f32(buf[i]);
}
inline void arena_store(device ushort *buf, ulong i, float x) {
    buf[i] = K_ARENA_F16 ? as_type<ushort>(half(x)) : arena_bf16_bits(x);
}

inline ushort arena_f32_to_bits(float x) {
    return K_ARENA_F16 ? as_type<ushort>(half(x)) : arena_bf16_bits(x);
}

/// Value-round for f32-SLOT arenas (MoE expert planes): the slots stay f32 but
/// values are rounded to the arena dtype for parity with the ushort planes.
inline float arena_round_f32(float x) {
    return K_ARENA_F16 ? float(half(x)) : f32_round_bf16(x);
}

#endif
