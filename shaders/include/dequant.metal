#ifndef DGQ_INCLUDE_DEQUANT_METAL
#define DGQ_INCLUDE_DEQUANT_METAL

#include <metal_stdlib>
using namespace metal;

#include "common.metal"

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

/// Fused GEMM path: dequant one Q4 group directly to half (skips f32 tile + half() round-trip).
inline void dequant_q4_group_half(device const uchar *g, thread half *out32) {
    half s = half(bf16_bytes(g));
    half mn = half(bf16_bytes(g + 2));
    for (uint i = 0; i < 16; ++i) {
        uchar b = g[4 + i];
        out32[2 * i] = s * half(b & 0x0F) + mn;
        out32[2 * i + 1] = s * half(b >> 4) + mn;
    }
}

inline void dequant_q4_group_half_tg(device const uchar *g, threadgroup half *out32) {
    half s = half(bf16_bytes(g));
    half mn = half(bf16_bytes(g + 2));
    for (uint i = 0; i < 16; ++i) {
        uchar b = g[4 + i];
        out32[2 * i] = s * half(b & 0x0F) + mn;
        out32[2 * i + 1] = s * half(b >> 4) + mn;
    }
}

/// Column-indexed Q4 decode (parity vs `dequant_q4_group` / CPU `q4_weight_at`).
inline float q4_at_col(device const uchar *row_base, uint col, uint K) {
    uint g = col / 32u;
    uint j = col % 32u;
    device const uchar *blk = row_base + ulong(g) * 20ul;
    float delta = bf16_bytes(blk);
    float mn = bf16_bytes(blk + 2);
    uchar byte = blk[4u + j / 2u];
    float q = (j & 1u) ? float(byte >> 4) : float(byte & 0x0fu);
    return delta * q + mn;
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

inline half e2m1_to_half(uint q) {
    const half mags[8] = {
        half(0), half(0.5), half(1), half(1.5),
        half(2), half(3), half(4), half(6),
    };
    half mag = mags[q & 7u];
    return (q & 8u) ? -mag : mag;
}

inline half fp8_e4m3_to_half(uchar b) {
    ushort v = ushort(b & 127u) << 7;
    half converted = half(fp16_bits_to_f32(v)) * half(256.0);
    return (b & 128u) ? -converted : converted;
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

inline void dequant_nvfp4_group_half(device const uchar *row, uint K, uint g,
                                     thread half *out16, half gscale_h) {
    uint data_len = (K + 1u) / 2u;
    half scale = fp8_e4m3_to_half(row[data_len + g]) * gscale_h;
    device const uchar *packed = row + g * 8u;
    for (uint i = 0; i < 16u; ++i) {
        uchar byte = packed[i / 2u];
        uint q = (i & 1u) ? uint(byte >> 4) : uint(byte & 0x0Fu);
        out16[i] = e2m1_to_half(q) * scale;
    }
}

inline void dequant_nvfp4_tile(device const uchar *row, uint K, uint k0,
                               thread float *out32, float gscale) {
    dequant_nvfp4_group(row, K, k0 / 16u, out32, gscale);
    dequant_nvfp4_group(row, K, k0 / 16u + 1u, out32 + 16, gscale);
}

inline void dequant_nvfp4_tile_half(device const uchar *row, uint K, uint k0,
                                    thread half *out32, half gscale_h) {
    dequant_nvfp4_group_half(row, K, k0 / 16u, out32, gscale_h);
    dequant_nvfp4_group_half(row, K, k0 / 16u + 1u, out32 + 16, gscale_h);
}

/// Fused 32-wide NVFP4 tile: two group scales decoded once, direct half output.
inline void dequant_nvfp4_tile_half_fused(device const uchar *row, uint K, uint k0,
                                          thread half *out32, half gscale_h) {
    uint data_len = (K + 1u) / 2u;
    uint g0 = k0 / 16u;
    uint g1 = g0 + 1u;
    half scale0 = fp8_e4m3_to_half(row[data_len + g0]) * gscale_h;
    half scale1 = fp8_e4m3_to_half(row[data_len + g1]) * gscale_h;
    device const uchar *packed0 = row + g0 * 8u;
    device const uchar *packed1 = row + g1 * 8u;
    for (uint i = 0; i < 16u; ++i) {
        uchar byte = packed0[i / 2u];
        uint q = (i & 1u) ? uint(byte >> 4) : uint(byte & 0x0Fu);
        out32[i] = e2m1_to_half(q) * scale0;
        byte = packed1[i / 2u];
        q = (i & 1u) ? uint(byte >> 4) : uint(byte & 0x0Fu);
        out32[16u + i] = e2m1_to_half(q) * scale1;
    }
}

inline void dequant_nvfp4_tile_half_fused_tg(device const uchar *row, uint K, uint k0,
                                             threadgroup half *out32, half gscale_h) {
    uint data_len = (K + 1u) / 2u;
    uint g0 = k0 / 16u;
    uint g1 = g0 + 1u;
    half scale0 = fp8_e4m3_to_half(row[data_len + g0]) * gscale_h;
    half scale1 = fp8_e4m3_to_half(row[data_len + g1]) * gscale_h;
    device const uchar *packed0 = row + g0 * 8u;
    device const uchar *packed1 = row + g1 * 8u;
    for (uint i = 0; i < 16u; ++i) {
        uchar byte = packed0[i / 2u];
        uint q = (i & 1u) ? uint(byte >> 4) : uint(byte & 0x0Fu);
        out32[i] = e2m1_to_half(q) * scale0;
        byte = packed1[i / 2u];
        q = (i & 1u) ? uint(byte >> 4) : uint(byte & 0x0Fu);
        out32[16u + i] = e2m1_to_half(q) * scale1;
    }
}

/// Column decode for one NVFP4 matrix row (`matrix` includes 4-byte global scale header).
inline float nvfp4_at_col(device const uchar *matrix, uint row, uint col, uint K) {
    float gscale = as_type<float>(*(device const uint *)(matrix));
    device const uchar *body = matrix + 4u;
    ulong row_stride = nvfp4_row_bytes(K);
    device const uchar *row_base = body + ulong(row) * row_stride;
    uint data_len = (K + 1u) / 2u;
    uint g = col / 16u;
    float scale = fp8_e4m3_to_f32(row_base[data_len + g]) * gscale;
    uchar byte = row_base[col / 2u];
    uint q = (col & 1u) ? uint(byte >> 4) : uint(byte & 0x0fu);
    return e2m1_to_f32(q) * scale;
}

#endif
