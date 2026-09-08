#include <metal_stdlib>
using namespace metal;

kernel void vec_fill_zero(
    device float *x [[buffer(0)]],
    constant uint2 &range [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    uint base = range.x;
    uint count = range.y;
    if (gid >= count) {
        return;
    }
    x[base + gid] = 0.0f;
}
