#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"

/// Materialize softmax rows from precomputed row stats (SC GEMM fast path).
kernel void sc_probs(
    device const half *logits [[buffer(0)]],
    device const float *rowstat [[buffer(1)]],
    device half *probs [[buffer(2)]],
    constant uint2 &dims [[buffer(3)]],
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
    device const half *lr = logits + (ulong)row * cols;
    for (uint v = lid; v < cols; v += tpg) {
        probs[(ulong)row * cols + v] = half(exp(float(lr[v]) - mx) / sum);
    }
}
