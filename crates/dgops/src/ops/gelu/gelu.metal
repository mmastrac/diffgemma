#include <metal_stdlib>
using namespace metal;

constant float GELU_TANH_COEF = 0.7978845608028654f;

inline float gelu_tanh(float x) {
    float x3 = x * x * x;
    float u = GELU_TANH_COEF * (x + 0.044715f * x3);
    float t = (u > 8.0f) ? 1.0f : (u < -8.0f) ? -1.0f : tanh(u);
    return 0.5f * x * (1.0f + t);
}

kernel void gelu(
    device float *x [[buffer(0)]],
    constant uint &len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    x[gid] = gelu_tanh(x[gid]);
}
