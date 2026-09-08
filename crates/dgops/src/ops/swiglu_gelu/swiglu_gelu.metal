#include <metal_stdlib>
using namespace metal;

constant float GELU_TANH_COEF = 0.7978846f;

inline float gelu_tanh(float x) {
    float x3 = x * x * x;
    float u = GELU_TANH_COEF * (x + 0.044715f * x3);
    float t = (u > 8.0f) ? 1.0f : (u < -8.0f) ? -1.0f : tanh(u);
    return 0.5f * x * (1.0f + t);
}

/// out[i] = gelu_tanh(gate[i]) * up[i].
kernel void swiglu_gelu(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    out[gid] = gelu_tanh(gate[gid]) * up[gid];
}
