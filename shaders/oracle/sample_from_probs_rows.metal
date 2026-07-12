#include <metal_stdlib>
using namespace metal;

/// Categorical sample per row (one TG per row; lane 0 inverse-CDF scan).
kernel void sample_from_probs_rows(
    device const float *probs [[buffer(0)]],
    device const float *rand [[buffer(1)]],
    device uint *out [[buffer(2)]],
    constant uint2 &dims [[buffer(3)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    uint row = tgp.y;
    uint rows = dims.x;
    uint cols = dims.y;
    if (row >= rows || lid != 0u) {
        return;
    }

    device const float *r = probs + row * cols;
    float target = rand[row];
    float cum = 0.0f;
    uint chosen = 0u;
    for (uint c = 0u; c < cols; c++) {
        cum += r[c];
        if (target < cum) {
            chosen = c;
            break;
        }
    }
    out[row] = chosen;
}
