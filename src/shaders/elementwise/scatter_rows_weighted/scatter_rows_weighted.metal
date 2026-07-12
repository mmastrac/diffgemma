#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"

/// Weighted scatter of grouped MoE arena rows back to token rows (engine f32
/// prefill path). For each (tok, d):
///   out[tok*hidden + d] = sum_k weights[tok*top_k + k] * arena[rows[tok*top_k + k]*hidden + d]
/// with k ascending. The (rows, weights) lists are built on the CPU in expert-JOB
/// order — the same order `scatter_weighted_expert_outputs` accumulates in — so
/// the f32 sum sequence (and thus rounding) is bit-identical to the CPU scatter.
kernel void scatter_rows_weighted(
    device const float *arena [[buffer(0)]],
    device const uint *rows [[buffer(1)]],
    device const float *weights [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint3 &dims [[buffer(4)]],   // [seq_len, hidden, top_k]
    device float *dump [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint seq_len = dims.x;
    const uint hidden = dims.y;
    const uint top_k = dims.z;
    if (K_SHAPE_ASSERT && (seq_len == 0u || hidden == 0u || top_k == 0u)) return;
    K_ELEMENTWISE_GUARD();
    const uint tok = gid / hidden;
    const uint d = gid % hidden;
    if (tok >= seq_len) return;
    float acc = 0.f;
    for (uint k = 0u; k < top_k; ++k) {
        const uint slot = tok * top_k + k;
        // Separate mul/add roundings (no FMA contraction): the CPU scatter this
        // must bit-match does `out += w * x` as two rounded f32 ops, but Metal
        // fast-math is free to fuse them into one. The volatile intermediate
        // forces the product to materialize as a rounded f32 first.
        volatile float term = weights[slot] * arena[(ulong)rows[slot] * hidden + d];
        acc += term;
    }
    if (K_DUMP_STAGE >= 1u) dump[gid] = acc;
    out[gid] = acc;
}
