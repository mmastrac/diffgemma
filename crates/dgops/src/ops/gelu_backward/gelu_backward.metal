#include <metal_stdlib>
using namespace metal;

constant float GELU_TANH_COEF = 0.7978846f;

inline float gelu_tanh_grad(float x) {
    float x3 = x * x * x;
    float u = GELU_TANH_COEF * (x + 0.044715f * x3);
    float t = (u > 8.0f) ? 1.0f : (u < -8.0f) ? -1.0f : tanh(u);
    float du = GELU_TANH_COEF * (1.0f + 0.134145f * x * x);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * du;
}

kernel void gelu_backward(
    device const float *g [[buffer(0)]],
    device const float *dy [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    out[gid] = dy[gid] * gelu_tanh_grad(g[gid]);
}
