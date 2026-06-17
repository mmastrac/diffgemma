#ifndef DGQ_INCLUDE_ARENA_METAL
#define DGQ_INCLUDE_ARENA_METAL

#include "common.metal"

/// bf16 activation arena: 2-byte slots hold raw bf16 bits (MLX-like), not fp16.
inline ushort arena_f32_to_bits(float x) {
    return ushort(as_type<uint>(f32_round_bf16(x)) >> 16);
}

inline float arena_load(device const ushort *buf, ulong i) {
    return bf16_to_f32(buf[i]);
}

inline void arena_store(device ushort *buf, ulong i, float x) {
    buf[i] = arena_f32_to_bits(x);
}

#endif
