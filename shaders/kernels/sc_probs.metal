#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "arena.metal"
#include "sc_prob_scale.metal"

/// Materialize softmax rows from precomputed row stats (SC GEMM fast path).
kernel void sc_probs(
    device const ushort *logits [[buffer(0)]],
    device const float *rowstat [[buffer(1)]],
    device ushort *probs [[buffer(2)]],
    constant uint2 &dims [[buffer(3)]],
    device DebugStatus *dbg [[buffer(4)]],
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

    float mx = rowstat[row * 2u];
    float sum = rowstat[row * 2u + 1u];
    if (lid == 0u) {
        dgq_assert_positive_f32(dbg, DbgKernelScProbs, sum, row);
    }
    device const ushort *lr = logits + (ulong)row * cols;
    // Store probs as fp16 (10 mantissa bits), not bf16 (7) — see sc_prob_cols.metal.
    // SC_PROB_GEMM_SCALE keeps a near-uniform prob out of fp16's denormal range
    // (sc_prob_scale.metal); the GEMM caller (x_fp16=true) divides it back out.
    for (uint v = lid; v < cols; v += tpg) {
        ((device half *)probs)[(ulong)row * cols + v] = half((exp(arena_load_bf16(lr, v) - mx) / sum) * SC_PROB_GEMM_SCALE);
    }
}
