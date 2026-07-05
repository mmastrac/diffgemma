#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "qgemm_grouped.metal"

/// Grouped MoE block GEMM: flattened `[total_m,K] @ expert weights^T` in one dispatch.
/// Format from K_QUANT_FORMAT (FC3): Q4 affine or NVFP4.
kernel void gemm_linear_grouped(
    device const float *a [[buffer(0)]],
    device const uchar *w_blob [[buffer(1)]],
    device float *c [[buffer(2)]],
    device const GroupedJob *jobs [[buffer(3)]],
    device const uint *row_starts [[buffer(4)]],
    constant uint3 &dims [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint k = dims.x;
    uint n = dims.y;
    uint num_jobs = dims.z;
    uint global_row = tgid.y;
    uint col = tgid.x;
    if (K_SHAPE_ASSERT && (k == 0u || n == 0u || num_jobs == 0u)) {
        return;
    }
    if (col >= n) {
        return;
    }

    uint job_id = resolve_grouped_job(row_starts, num_jobs, global_row);
    if (global_row >= row_starts[job_id + 1u]) {
        return;
    }

    GroupedJob job = jobs[job_id];
    device const uchar *w = w_blob + job.w_byte_off;

    float sum = 0.0f;
    if (K_QUANT_FORMAT == QUANT_NVFP4) {
        float gscale = as_type<float>(*(device const uint *)(w));
        device const uchar *body = w + 4u;
        ulong row_stride = nvfp4_row_bytes(k);
        uint num_tiles = (k + 31u) / 32u;
        for (uint t = simd_lane; t < num_tiles; t += 32u) {
            uint k0 = t * 32u;
            device const uchar *row = body + (ulong)col * row_stride;
            sum += dot_nvfp4_k32(a + global_row * k, row, k, k0, gscale);
        }
    } else if (K_QUANT_FORMAT == QUANT_Q6) {
        ulong row_stride = ulong(job.groups_per_row) * 28ul;
        for (uint g = simd_lane; g < job.groups_per_row; g += 32u) {
            device const uchar *blk = w + (ulong)col * row_stride + (ulong)g * 28ul;
            sum += dot_q6_group(a + global_row * k, blk, g * 32u);
        }
    } else {
        ulong row_stride = ulong(job.groups_per_row) * 20ul;
        for (uint g = simd_lane; g < job.groups_per_row; g += 32u) {
            device const uchar *blk = w + (ulong)col * row_stride + (ulong)g * 20ul;
            sum += dot_q4_group(a + global_row * k, blk, g * 32u);
        }
    }
    sum = simd_sum(sum);
    if (simd_lane == 0u) {
        c[global_row * n + col] = sum;
    }
}
