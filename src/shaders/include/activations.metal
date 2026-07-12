#ifndef DGQ_INCLUDE_ACTIVATIONS_METAL
#define DGQ_INCLUDE_ACTIVATIONS_METAL

#include <metal_stdlib>
using namespace metal;

constant float GELU_TANH_COEF = 0.7978845608028654;

/// PyTorch tanh GELU approximation (matches tier-1 subkernel oracle).
inline float gelu_tanh(float x) {
    float x3 = x * x * x;
    float u = GELU_TANH_COEF * (x + 0.044715f * x3);
    float t = (u > 8.0f) ? 1.0f : (u < -8.0f) ? -1.0f : tanh(u);
    return 0.5f * x * (1.0f + t);
}

inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

inline float swiglu_gelu(float gate, float up) {
    return gelu_tanh(gate) * up;
}

inline float swiglu_silu(float gate, float up) {
    return silu(gate) * up;
}

#endif
