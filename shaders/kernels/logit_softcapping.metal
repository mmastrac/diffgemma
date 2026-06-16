#include <metal_stdlib>
using namespace metal;

kernel void logit_softcapping(
    device float *x [[buffer(0)]],
    constant float &cap [[buffer(1)]],
    constant uint2 &range [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint base = range.x;
    uint len = range.y;
    uint i = base + gid;
    if (gid >= len) {
        return;
    }
    if (cap <= 0.0f) {
        return;
    }
    float v = x[i];
    x[i] = tanh(v / cap) * cap;
}
