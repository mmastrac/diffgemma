#ifndef DGQ_KERNEL_ACTIVATIONS_METAL
#define DGQ_KERNEL_ACTIVATIONS_METAL

#include <metal_stdlib>
using namespace metal;

constant float GELU_TANH_COEF = 0.7978845608028654;

inline float gelu_tanh(float x) {
    float x3 = x * x * x;
    float u = GELU_TANH_COEF * (x + 0.044715f * x3);
    float t = (u > 8.0f) ? 1.0f : (u < -8.0f) ? -1.0f : tanh(u);
    return 0.5f * x * (1.0f + t);
}

#endif
