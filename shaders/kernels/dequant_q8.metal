#ifndef DGQ_KERNEL_DEQUANT_Q8_METAL
#define DGQ_KERNEL_DEQUANT_Q8_METAL

#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_KERNEL_BF16_METAL
#include "bf16.metal"
#endif

// q8_row: [scale bf16:2][i8 weights:K], w = scale * q
inline ulong q8_row_bytes(uint K) {
    return ulong(K) + 2ul;
}

inline float q8_at(device const uchar *row_base, uint col, float s) {
    return float(*((device const char *)(row_base + 2 + col))) * s;
}

#endif
