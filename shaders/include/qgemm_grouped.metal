#ifndef DGQ_INCLUDE_QGEMM_GROUPED_METAL
#define DGQ_INCLUDE_QGEMM_GROUPED_METAL

#include "dequant.metal"

struct GroupedJob {
    ulong w_byte_off;
    uint groups_per_row;
    uint _pad_job;
};

inline float dot_q4_group(
    device const float *a_row,
    device const uchar *blk,
    uint base_k
) {
    thread float w32[32];
    dequant_q4_group(blk, w32);
    float sum = 0.0f;
    for (uint i = 0; i < 32u; ++i) {
        sum = fma(a_row[base_k + i], w32[i], sum);
    }
    return sum;
}

inline float dot_nvfp4_k32(
    device const float *a_row,
    device const uchar *row,
    uint k_dim,
    uint k0,
    float gscale
) {
    uint data_len = (k_dim + 1u) / 2u;
    half gscale_h = half(gscale);
    uint g0 = k0 / 16u;
    uint g1 = g0 + 1u;
    half scale0 = fp8_e4m3_to_half(row[data_len + g0]) * gscale_h;
    half scale1 = fp8_e4m3_to_half(row[data_len + g1]) * gscale_h;
    device const uchar *packed0 = row + g0 * 8u;
    device const uchar *packed1 = row + g1 * 8u;
    float sum = 0.0f;
    uint n = min(32u, k_dim - k0);
    for (uint i = 0; i < n; ++i) {
        uint local = i & 15u;
        bool upper = (i >= 16u);
        half scale = upper ? scale1 : scale0;
        device const uchar *packed = upper ? packed1 : packed0;
        uchar byte = packed[local / 2u];
        uint q = (local & 1u) ? uint(byte >> 4) : uint(byte & 0x0Fu);
        sum = fma(a_row[k0 + i], float(e2m1_to_half(q) * scale), sum);
    }
    return sum;
}

inline float dot_nvfp4_k32_half(
    device const half *a_row,
    device const uchar *row,
    uint k_dim,
    uint k0,
    float gscale
) {
    uint data_len = (k_dim + 1u) / 2u;
    half gscale_h = half(gscale);
    uint g0 = k0 / 16u;
    uint g1 = g0 + 1u;
    half scale0 = fp8_e4m3_to_half(row[data_len + g0]) * gscale_h;
    half scale1 = fp8_e4m3_to_half(row[data_len + g1]) * gscale_h;
    device const uchar *packed0 = row + g0 * 8u;
    device const uchar *packed1 = row + g1 * 8u;
    float sum = 0.0f;
    uint n = min(32u, k_dim - k0);
    for (uint i = 0; i < n; ++i) {
        uint local = i & 15u;
        bool upper = (i >= 16u);
        half scale = upper ? scale1 : scale0;
        device const uchar *packed = upper ? packed1 : packed0;
        uchar byte = packed[local / 2u];
        uint q = (local & 1u) ? uint(byte >> 4) : uint(byte & 0x0Fu);
        sum = fma(float(a_row[k0 + i]), float(e2m1_to_half(q) * scale), sum);
    }
    return sum;
}

inline float dot_nvfp4_k32_act(
    threadgroup const float *a_row,
    device const uchar *row,
    uint k_dim,
    uint k0,
    float gscale
) {
    uint data_len = (k_dim + 1u) / 2u;
    half gscale_h = half(gscale);
    uint g0 = k0 / 16u;
    uint g1 = g0 + 1u;
    half scale0 = fp8_e4m3_to_half(row[data_len + g0]) * gscale_h;
    half scale1 = fp8_e4m3_to_half(row[data_len + g1]) * gscale_h;
    device const uchar *packed0 = row + g0 * 8u;
    device const uchar *packed1 = row + g1 * 8u;
    float sum = 0.0f;
    uint n = min(32u, k_dim - k0);
    for (uint i = 0; i < n; ++i) {
        uint local = i & 15u;
        bool upper = (i >= 16u);
        half scale = upper ? scale1 : scale0;
        device const uchar *packed = upper ? packed1 : packed0;
        uchar byte = packed[local / 2u];
        uint q = (local & 1u) ? uint(byte >> 4) : uint(byte & 0x0Fu);
        sum = fma(a_row[k0 + i], float(e2m1_to_half(q) * scale), sum);
    }
    return sum;
}

inline uint resolve_grouped_job(
    device const uint *row_starts,
    uint num_jobs,
    uint global_row
) {
    uint lo = 0u;
    uint hi = num_jobs;
    while (lo + 1u < hi) {
        uint mid = (lo + hi) >> 1;
        if (row_starts[mid] <= global_row) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

#endif
