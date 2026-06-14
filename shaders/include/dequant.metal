#ifndef DGQ_INCLUDE_DEQUANT_METAL
#define DGQ_INCLUDE_DEQUANT_METAL

#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_INCLUDE_COMMON_METAL
#include "common.metal"
#endif

// ---- Q4 affine (32-wide groups) ----
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

inline ulong q4_row_bytes(uint K) {
    return ulong(K / 32) * 20ul;
}

// ---- Q8 per-row ----
// q8_row: [scale bf16:2][i8 weights:K], w = scale * q
inline ulong q8_row_bytes(uint K) {
    return ulong(K) + 2ul;
}

inline float q8_at(device const uchar *row_base, uint col, float s) {
    return float(*((device const char *)(row_base + 2 + col))) * s;
}

// ---- NVFP4 ----
// nvfp4_block: [f32 global_scale:4] + per row [data:ceil(K/2)][scales:ceil(K/16)]

inline float fp16_bits_to_f32(ushort bits) {
    uint sign = (bits >> 15) & 1u;
    uint exp = (bits >> 10) & 0x1fu;
    uint mant = bits & 0x3ffu;
    if (exp == 0u) {
        if (mant == 0u) {
            return sign ? -0.0f : 0.0f;
        }
        return (sign ? -1.0f : 1.0f) * float(mant) * exp2(-24.0f);
    }
    if (exp == 0x1fu) {
        if (mant == 0u) {
            return sign ? -INFINITY : INFINITY;
        }
        return NAN;
    }
    uint f32_bits = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    return as_type<float>(f32_bits);
}

inline float fp8_e4m3_to_f32(uchar b) {
    ushort v = ushort(b & 127u) << 7;
    float converted = fp16_bits_to_f32(v) * 256.0f;
    return (b & 128u) ? -converted : converted;
}

inline float e2m1_to_f32(uint q) {
    float mag = 0.f;
    switch (q & 7u) {
        case 1: mag = 0.5f; break;
        case 2: mag = 1.0f; break;
        case 3: mag = 1.5f; break;
        case 4: mag = 2.0f; break;
        case 5: mag = 3.0f; break;
        case 6: mag = 4.0f; break;
        case 7: mag = 6.0f; break;
        default: mag = 0.f; break;
    }
    return (q & 8u) ? -mag : mag;
}

inline ulong nvfp4_row_bytes(uint K) {
    return ulong((K + 1u) / 2u + (K + 15u) / 16u);
}

inline ulong nvfp4_matrix_bytes(uint out_dim, uint K) {
    return 4ul + ulong(out_dim) * nvfp4_row_bytes(K);
}

inline void dequant_nvfp4_group(device const uchar *row, uint K, uint g,
                                thread float *out16, float gscale) {
    uint data_len = (K + 1u) / 2u;
    float scale = fp8_e4m3_to_f32(row[data_len + g]) * gscale;
    device const uchar *packed = row + g * 8u;
    for (uint i = 0; i < 16u; ++i) {
        uchar byte = packed[i / 2u];
        uint q = (i & 1u) ? uint(byte >> 4) : uint(byte & 0x0Fu);
        out16[i] = e2m1_to_f32(q) * scale;
    }
}

inline void dequant_nvfp4_tile(device const uchar *row, uint K, uint k0,
                               thread float *out32, float gscale) {
    dequant_nvfp4_group(row, K, k0 / 16u, out32, gscale);
    dequant_nvfp4_group(row, K, k0 / 16u + 1u, out32 + 16, gscale);
}

#endif
