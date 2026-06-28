#ifndef DGQ_INCLUDE_DEQUANT_METAL
#define DGQ_INCLUDE_DEQUANT_METAL

#include <metal_stdlib>
using namespace metal;

#include "common.metal"
#include "act_fc.metal"   // K_ACT_F16 (FC9 only)

/// Read one activation arena slot into half: f16 reinterpret when K_ACT_F16, else
/// bf16->f32->half. Matches arena_store's precision (arena.metal).
inline half arena_act_half(device const ushort *x, ulong i) {
    return K_ACT_F16 ? as_type<half>(x[i]) : half(bf16_to_f32(x[i]));
}

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

/// Partial Q4 group decode into threadgroup tile (8 consecutive columns).
inline void dequant_q4_group_half_tg_range(
    device const uchar *g,
    threadgroup half *out,
    uint col_start,
    uint col_count
) {
    half s = half(bf16_bytes(g));
    half mn = half(bf16_bytes(g + 2));
    for (uint i = 0u; i < col_count; ++i) {
        const uint j = col_start + i;
        uchar b = g[4u + j / 2u];
        const float q = (j & 1u) ? float(b >> 4) : float(b & 0x0fu);
        out[i] = s * half(q) + mn;
    }
}

/// Load one 32×32 activation K-tile (f32 activations) with all 128 threads.
inline void gemm_load_a_tile_f32(
    device const float *a,
    uint M_tile,
    uint K,
    uint row0,
    uint k0,
    uint ltid,
    threadgroup half tx[32][32]
) {
    for (uint i = ltid; i < 32u * 32u; i += 128u) {
        const uint mm = i / 32u;
        const uint kk = i % 32u;
        tx[mm][kk] = (mm < M_tile)
            ? half(a[(ulong)(row0 + mm) * K + k0 + kk])
            : half(0);
    }
}

/// Gathering A-tile load for fused MoE gate_up: instead of a pre-materialized
/// [slots, K] f32 buffer, read the source token row straight from the bf16
/// `moein` [canvas, K] via `token_list[slot]`. Saves the separate gather pass +
/// the 23MB/layer f32 round-trip; `moein` is only `canvas` rows (cache-hot,
/// TOP_K-reused). bf16->f32->half == bf16->half, so bit-identical to gather+f32.
inline void gemm_load_a_tile_gather_bf16(
    device const ushort *moein,
    device const uint *token_list,
    uint M_tile,
    uint K,
    uint row0,
    uint k0,
    uint ltid,
    threadgroup half tx[32][32]
) {
    for (uint i = ltid; i < 32u * 32u; i += 128u) {
        const uint mm = i / 32u;
        const uint kk = i % 32u;
        if (mm < M_tile) {
            const uint tok = token_list[row0 + mm];
            tx[mm][kk] = arena_act_half(moein, (ulong)tok * K + k0 + kk);
        } else {
            tx[mm][kk] = half(0);
        }
    }
}

inline void gemm_prefetch_a_tile_f32(
    device const float *a,
    uint M_tile,
    uint K,
    uint row0,
    uint k_next,
    uint ltid,
    threadgroup half tx_next[32][32]
) {
    for (uint i = ltid - 32u; i < 32u * 32u; i += 96u) {
        const uint mm = i / 32u;
        const uint kk = i % 32u;
        tx_next[mm][kk] = (mm < M_tile)
            ? half(a[(ulong)(row0 + mm) * K + k_next + kk])
            : half(0);
    }
}

/// Load one 32×32 bf16 activation K-tile with all 128 threads.
inline void gemm_load_a_tile(
    device const ushort *x,
    uint M,
    uint K,
    uint m0,
    uint k0,
    uint ltid,
    threadgroup half tx[32][32]
) {
    for (uint i = ltid; i < 32u * 32u; i += 128u) {
        const uint mm = i / 32u;
        const uint kk = i % 32u;
        tx[mm][kk] = (m0 + mm < M)
            ? arena_act_half(x, (ulong)(m0 + mm) * K + k0 + kk)
            : half(0);
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

/// Half-precision column decode for parallel GEMM tile fill.
inline half q4_at_col_half(device const uchar *row_base, uint col) {
    uint g = col / 32u;
    uint j = col % 32u;
    device const uchar *blk = row_base + ulong(g) * 20ul;
    half s = half(bf16_bytes(blk));
    half mn = half(bf16_bytes(blk + 2));
    uchar byte = blk[4u + j / 2u];
    float q = (j & 1u) ? float(byte >> 4) : float(byte & 0x0fu);
    return s * half(q) + mn;
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

inline float fp8_e4m3_to_f32(uchar b) {
    uint s = (b >> 7) & 1u;
    uint e = (b >> 3) & 0xFu;
    uint m = b & 0x7u;
    float val;
    if (e == 0u) {
        val = float(m) * exp2(-9.0f);
    } else if (e == 15u && m == 7u) {
        val = NAN;
    } else {
        val = (1.0f + float(m) * 0.125f) * exp2(float(int(e) - 7));
    }
    return s ? -val : val;
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
    return half(fp8_e4m3_to_f32(b));
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
                                     thread half *out16, float gscale) {
    uint data_len = (K + 1u) / 2u;
    float scale = fp8_e4m3_to_f32(row[data_len + g]) * gscale;
    device const uchar *packed = row + g * 8u;
    for (uint i = 0; i < 16u; ++i) {
        uchar byte = packed[i / 2u];
        uint q = (i & 1u) ? uint(byte >> 4) : uint(byte & 0x0Fu);
        out16[i] = half(e2m1_to_f32(q) * scale);
    }
}

inline void dequant_nvfp4_tile(device const uchar *row, uint K, uint k0,
                               thread float *out32, float gscale) {
    dequant_nvfp4_group(row, K, k0 / 16u, out32, gscale);
    dequant_nvfp4_group(row, K, k0 / 16u + 1u, out32 + 16, gscale);
}

inline void dequant_nvfp4_tile_half(device const uchar *row, uint K, uint k0,
                                    thread half *out32, float gscale) {
    dequant_nvfp4_group_half(row, K, k0 / 16u, out32, gscale);
    dequant_nvfp4_group_half(row, K, k0 / 16u + 1u, out32 + 16, gscale);
}

/// Fused 32-wide NVFP4 tile: two group scales decoded once, direct half output.
inline void dequant_nvfp4_tile_half_fused(device const uchar *row, uint K, uint k0,
                                          thread half *out32, float gscale) {
    uint data_len = (K + 1u) / 2u;
    uint g0 = k0 / 16u;
    uint g1 = g0 + 1u;
    float scale0 = fp8_e4m3_to_f32(row[data_len + g0]) * gscale;
    float scale1 = fp8_e4m3_to_f32(row[data_len + g1]) * gscale;
    device const uchar *packed0 = row + g0 * 8u;
    device const uchar *packed1 = row + g1 * 8u;
    for (uint i = 0; i < 16u; ++i) {
        uchar byte = packed0[i / 2u];
        uint q = (i & 1u) ? uint(byte >> 4) : uint(byte & 0x0Fu);
        out32[i] = half(e2m1_to_f32(q) * scale0);
        byte = packed1[i / 2u];
        q = (i & 1u) ? uint(byte >> 4) : uint(byte & 0x0Fu);
        out32[16u + i] = half(e2m1_to_f32(q) * scale1);
    }
}

inline void dequant_nvfp4_tile_half_fused_tg(device const uchar *row, uint K, uint k0,
                                             threadgroup half *out32, float gscale) {
    uint data_len = (K + 1u) / 2u;
    uint g0 = k0 / 16u;
    uint g1 = g0 + 1u;
    float scale0 = fp8_e4m3_to_f32(row[data_len + g0]) * gscale;
    float scale1 = fp8_e4m3_to_f32(row[data_len + g1]) * gscale;
    device const uchar *packed0 = row + g0 * 8u;
    device const uchar *packed1 = row + g1 * 8u;
    for (uint i = 0; i < 16u; ++i) {
        uchar byte = packed0[i / 2u];
        uint q = (i & 1u) ? uint(byte >> 4) : uint(byte & 0x0Fu);
        out32[i] = half(e2m1_to_f32(q) * scale0);
        byte = packed1[i / 2u];
        q = (i & 1u) ? uint(byte >> 4) : uint(byte & 0x0Fu);
        out32[16u + i] = half(e2m1_to_f32(q) * scale1);
    }
}

/// Partial NVFP4 32-wide K-tile decode (8 consecutive columns).
inline void dequant_nvfp4_tile_half_fused_tg_range(
    device const uchar *row,
    uint K,
    uint k0,
    threadgroup half *out,
    float gscale,
    uint col_start,
    uint col_count
) {
    uint data_len = (K + 1u) / 2u;
    for (uint i = 0u; i < col_count; ++i) {
        const uint col = col_start + i;
        const uint g = col / 16u;
        const float scale = fp8_e4m3_to_f32(row[data_len + g]) * gscale;
        device const uchar *packed = row + g * 8u;
        const uint j = col % 16u;
        uchar byte = packed[j / 2u];
        uint q = (j & 1u) ? uint(byte >> 4) : uint(byte & 0x0Fu);
        out[i] = half(e2m1_to_f32(q) * scale);
    }
}

/// 128-thread cooperative weight tile: threads 0-31 run fast row decode; 32-127 prefetch
/// the next activation K-slice into `tx_next` while weights decode (software pipeline).
inline void dequant_q4_tw_tile_cooperative(
    device const uchar *blob,
    ulong body,
    ulong rowB,
    uint k0,
    uint n0,
    uint N,
    uint ltid,
    threadgroup half tw[32][32],
    device const ushort *x,
    uint M,
    uint K,
    uint m0,
    uint k_next,
    threadgroup half tx_next[32][32]
) {
    if (ltid < 32u) {
        const uint r = ltid;
        const uint global_n = n0 + r;
        if (global_n >= N) {
            for (uint i = 0u; i < 32u; ++i) {
                tw[r][i] = half(0);
            }
        } else {
            dequant_q4_group_half_tg(
                blob + body + (ulong)global_n * rowB + (ulong)(k0 / 32u) * 20ul,
                &tw[r][0]);
        }
    } else if (k_next < K) {
        for (uint i = ltid - 32u; i < 32u * 32u; i += 96u) {
            const uint mm = i / 32u;
            const uint kk = i % 32u;
            tx_next[mm][kk] = (m0 + mm < M)
                ? arena_act_half(x, (ulong)(m0 + mm) * K + k_next + kk)
                : half(0);
        }
    }
}

inline void dequant_nvfp4_tw_tile_cooperative(
    device const uchar *matrix,
    uint K,
    uint k0,
    uint n0,
    uint N,
    uint ltid,
    threadgroup half tw[32][32],
    device const ushort *x,
    uint M,
    uint m0,
    uint k_next,
    threadgroup half tx_next[32][32]
) {
    const float gscale = as_type<float>(*(device const uint *)(matrix));
    const device uchar *body = matrix + 4u;
    const ulong rowB = nvfp4_row_bytes(K);
    if (ltid < 32u) {
        const uint r = ltid;
        const uint global_n = n0 + r;
        if (global_n >= N) {
            for (uint i = 0u; i < 32u; ++i) {
                tw[r][i] = half(0);
            }
        } else {
            device const uchar *row = body + (ulong)global_n * rowB;
            dequant_nvfp4_tile_half_fused_tg(row, K, k0, &tw[r][0], gscale);
        }
    } else if (k_next < K) {
        for (uint i = ltid - 32u; i < 32u * 32u; i += 96u) {
            const uint mm = i / 32u;
            const uint kk = i % 32u;
            tx_next[mm][kk] = (m0 + mm < M)
                ? arena_act_half(x, (ulong)(m0 + mm) * K + k_next + kk)
                : half(0);
        }
    }
}

/// 128-thread parallel fill: 4 threads × 32 rows, 8 columns each.
inline void dequant_nvfp4_tw_tile_parallel(
    device const uchar *matrix,
    uint K,
    uint k0,
    uint n0,
    uint N,
    uint ltid,
    threadgroup half tw[32][32]
) {
    const float gscale = as_type<float>(*(device const uint *)(matrix));
    const device uchar *body = matrix + 4u;
    const ulong rowB = nvfp4_row_bytes(K);
    const uint r = ltid / 4u;
    const uint part = ltid % 4u;
    const uint col_start = part * 8u;
    const uint global_n = n0 + r;
    if (global_n >= N) {
        for (uint i = 0u; i < 8u; ++i) {
            tw[r][col_start + i] = half(0);
        }
        return;
    }
    device const uchar *row = body + (ulong)global_n * rowB;
    dequant_nvfp4_tile_half_fused_tg_range(row, K, k0, &tw[r][col_start], gscale, col_start, 8u);
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

inline half nvfp4_at_col_half(device const uchar *matrix, uint row, uint col, uint K) {
    return half(nvfp4_at_col(matrix, row, col, K));
}

#endif
