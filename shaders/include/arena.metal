#ifndef DGQ_INCLUDE_ARENA_METAL
#define DGQ_INCLUDE_ARENA_METAL

#include "common.metal"
#include "act_fc.metal"   // K_ACT_F16 (FC9 only)

// Activation arena: 2-byte slots. bf16 (7-mantissa, wide exponent) by default;
// f16 (10-mantissa) when K_ACT_F16 (DGQ_ACT_F16). Measured activations are <~170
// (f16 max 65504), so f16 has ample range. Logits use the explicit *_bf16 path.

inline ushort arena_bf16_bits(float x) {
    return ushort(as_type<uint>(f32_round_bf16(x)) >> 16);
}

// Always-bf16 (logits / large-range buffers; ignores K_ACT_F16).
inline float arena_load_bf16(device const ushort *buf, ulong i) {
    return bf16_to_f32(buf[i]);
}
inline void arena_store_bf16(device ushort *buf, ulong i, float x) {
    buf[i] = arena_bf16_bits(x);
}

// Toggleable activation precision.
inline float arena_load(device const ushort *buf, ulong i) {
    return K_ACT_F16 ? float(as_type<half>(buf[i])) : bf16_to_f32(buf[i]);
}
inline void arena_store(device ushort *buf, ulong i, float x) {
    buf[i] = K_ACT_F16 ? as_type<ushort>(half(x)) : arena_bf16_bits(x);
}

inline ushort arena_f32_to_bits(float x) {
    return K_ACT_F16 ? as_type<ushort>(half(x)) : arena_bf16_bits(x);
}

#endif
