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

/// f32 -> bf16 precision, round-to-nearest-even.
///
/// This used to mask the low 16 bits, which is truncation toward zero: every
/// activation store shed a uniform fraction of an ulp instead of a signed half
/// ulp, so the arena carried a systematic downward bias rather than zero-mean
/// rounding error. Measured on the step-1 preamble, that cost 0.28% of the
/// row's l2 per store -- a scale-free RMS norm came out at 0.9972 instead of 1,
/// and truncation reproduced the engine's own output to cos 1.000000000 where
/// round-to-nearest did not.
///
/// The old comment justified it as matching "Rust `f32_to_bf16_bits` / MLX
/// matmul store". The first half held (the pack quantizers truncate too, which
/// is why CPU/GPU parity never caught this); the second did not. MLX rounds to
/// nearest even on both paths -- `mlx/types/bf16.h` does
/// `in.u += (in.u >> 16 & 1) + 0x7FFF` and its Metal side uses the native
/// `bfloat`, whose conversion is RNE.
inline float f32_round_bf16(float x) {
    uint u = as_type<uint>(x);
    // Leave NaN/Inf alone: the rounding bump would carry into the exponent.
    if ((u & 0x7F800000u) != 0x7F800000u) {
        u += 0x7FFFu + ((u >> 16) & 1u);
    }
    return as_type<float>(u & 0xFFFF0000u);
}

#endif
