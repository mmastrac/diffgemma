#ifndef DGQ_INCLUDE_ARENA_METAL
#define DGQ_INCLUDE_ARENA_METAL

#include "common.metal"

// Activation arena: 2-byte bf16 slots (settled precision policy: bf16 acts;
// f16 only for [0,1]-range values via explicit half buffers).

inline ushort arena_bf16_bits(float x) {
    return ushort(as_type<uint>(f32_round_bf16(x)) >> 16);
}

// Explicit-bf16 aliases (logits / large-range buffers).
inline float arena_load_bf16(device const ushort *buf, ulong i) {
    return bf16_to_f32(buf[i]);
}
inline void arena_store_bf16(device ushort *buf, ulong i, float x) {
    buf[i] = arena_bf16_bits(x);
}

inline float arena_load(device const ushort *buf, ulong i) {
    return bf16_to_f32(buf[i]);
}
inline void arena_store(device ushort *buf, ulong i, float x) {
    buf[i] = arena_bf16_bits(x);
}

inline ushort arena_f32_to_bits(float x) {
    return arena_bf16_bits(x);
}

#endif
