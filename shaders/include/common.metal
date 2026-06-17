#ifndef DGQ_INCLUDE_COMMON_METAL
#define DGQ_INCLUDE_COMMON_METAL

#include <metal_stdlib>
using namespace metal;

inline float bf16_bytes(device const uchar *p) {
    return as_type<float>((uint(p[0]) | (uint(p[1]) << 8)) << 16);
}

inline float bf16_to_f32(ushort b) {
    return as_type<float>(uint(b) << 16);
}

/// Truncate f32 to bf16 precision (matches Rust `f32_to_bf16_bits` / MLX matmul store).
inline float f32_round_bf16(float x) {
    return as_type<float>(as_type<uint>(x) & 0xFFFF0000u);
}

/// fp16 buffer store: bf16-round then IEEE fp16 (logits / KV cache slots).
inline half half_store_bf16_round(float x) {
    return half(f32_round_bf16(x));
}

#endif
