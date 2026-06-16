#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"

/// Per-row max + sumexp over half logits (post-softcap; tempered when stored for SC) -> {mx, sum}.
kernel void logit_rowstats(
    device const half *logits [[buffer(0)]],
    device float *rowstat [[buffer(1)]],
    constant uint2 &dims [[buffer(2)]],
    device DebugStatus *dbg [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    const uint rows = dims.x, cols = dims.y;
    if (K_SHAPE_ASSERT && (cols == 0u || rows == 0u)) {
        return;
    }
    if (row >= rows) {
        return;
    }
    K_ELEMENTWISE_GUARD();

    threadgroup float r_mx[8];
    threadgroup float r_sum[8];
    device const half *lr = logits + (ulong)row * cols;
    float mx = -INFINITY;
    for (uint v = lid; v < cols; v += tpg) {
        mx = max(mx, float(lr[v]));
    }
    mx = simd_max(mx);
    uint sg = lid / 32u, nsg = (tpg + 31u) / 32u;
    if ((lid & 31u) == 0u) {
        r_mx[sg] = mx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        for (uint i = 1u; i < nsg; ++i) {
            r_mx[0] = max(r_mx[0], r_mx[i]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = r_mx[0];
    float sum = 0.f;
    for (uint v = lid; v < cols; v += tpg) {
        sum += exp(float(lr[v]) - mx);
    }
    sum = simd_sum(sum);
    if ((lid & 31u) == 0u) {
        r_sum[sg] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float s = 0.f;
        for (uint i = 0u; i < nsg; ++i) {
            s += r_sum[i];
        }
        dgq_assert_positive_f32(dbg, DbgKernelLogitRowstats, s, row);
        rowstat[row * 2u] = mx;
        rowstat[row * 2u + 1u] = s;
    }
}
