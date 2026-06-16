#include <metal_stdlib>
using namespace metal;

/// Divide every element by scalar temperature.
kernel void scale_logits(
    device float *x [[buffer(0)]],
    constant float &inv_t [[buffer(1)]],
    constant uint2 &range [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint base = range.x;
    uint len = range.y;
    uint i = base + gid;
    if (gid >= len) {
        return;
    }
    x[i] *= inv_t;
}
