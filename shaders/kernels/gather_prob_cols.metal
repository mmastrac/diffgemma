#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_FC_AXES_METAL
#include "fc_axes.metal"
#endif

kernel void gather_prob_cols(
    device const float *probs [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint4 &params [[buffer(2)]],
    device float *dump [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint rows = params.x;
    uint vocab = params.y;
    uint v0 = params.z;
    uint chunk = params.w;
    uint col = gid.x;
    uint row = gid.y;
    if (K_SHAPE_ASSERT && (rows == 0u || chunk == 0u || vocab == 0u)) return;
    if (row >= rows || col >= chunk) return;
    K_ELEMENTWISE_GUARD();
    float v = probs[row * vocab + v0 + col];
    if (K_DUMP_STAGE >= 1u) dump[row * chunk + col] = v;
    out[row * chunk + col] = v;
}
