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
    thread float w32[32];
    dequant_nvfp4_tile(row, k_dim, k0, w32, gscale);
    float sum = 0.0f;
    uint n = min(32u, k_dim - k0);
    for (uint i = 0; i < n; ++i) {
        sum = fma(a_row[k0 + i], w32[i], sum);
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
