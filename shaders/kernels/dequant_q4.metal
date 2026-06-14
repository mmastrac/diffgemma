#ifndef DGQ_KERNEL_DEQUANT_Q4_METAL
#define DGQ_KERNEL_DEQUANT_Q4_METAL

#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_KERNEL_BF16_METAL
#include "bf16.metal"
#endif

// q4_block: [scale:2][min:2][nibbles:16], w = scale*q + min
inline void dequant_q4_group(device const uchar *g, thread float *out32) {
    float s = bf16_bytes(g);
    float mn = bf16_bytes(g + 2);
    for (uint i = 0; i < 16; ++i) {
        uchar b = g[4 + i];
        out32[2 * i] = s * float(b & 0x0F) + mn;
        out32[2 * i + 1] = s * float(b >> 4) + mn;
    }
}

inline ulong q4_row_bytes(uint K) { return ulong(K / 32) * 20ul; }

#endif
