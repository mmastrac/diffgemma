#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "arena.metal"
#include "sc_prob_scale.metal"

/// Softmax a vocab column slice for SC: probs[row, v0+col] from logits + rowstat.
kernel void sc_prob_cols(
    device const ushort *logits [[buffer(0)]],
    device const float *rowstat [[buffer(1)]],
    device ushort *probs [[buffer(2)]],
    constant uint4 &params [[buffer(3)]],
    device DebugStatus *dbg [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint rows = params.x;
    const uint vocab = params.y;
    const uint v0 = params.z;
    const uint chunk = params.w;
    const uint col = gid.x;
    const uint row = gid.y;
    if (K_SHAPE_ASSERT && (rows == 0u || chunk == 0u || vocab == 0u)) {
        return;
    }
    if (row >= rows || col >= chunk) {
        return;
    }
    K_ELEMENTWISE_GUARD();

    float mx = rowstat[row * 2u];
    float sum = rowstat[row * 2u + 1u];
    if (col == 0u) {
        dgq_assert_positive_f32(dbg, DbgKernelScProbs, sum, row);
    }
    uint v = v0 + col;
    if (v >= vocab) {
        ((device half *)probs)[(ulong)row * chunk + col] = half(0);
        return;
    }
    float x = arena_load_bf16(logits, (ulong)row * vocab + v);
    // Store probs as fp16 (10 mantissa bits), not bf16 (7): the soft-embed sums
    // over the full vocab, and bf16-rounding each prob adds noise worst for
    // spread distributions. SC_PROB_GEMM_SCALE pushes a near-uniform prob
    // (~2^-18, an fp16 denormal below the 2^-14 normal min) into fp16's normal
    // range; the GEMM caller (x_fp16=true) divides it back out.
    ((device half *)probs)[(ulong)row * chunk + col] = half((exp(x - mx) / sum) * SC_PROB_GEMM_SCALE);
}
