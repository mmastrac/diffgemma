// Backward pass of tanh-approximation GELU. Mirrors gelu_backward.metal.
#include <cmath>

__device__ __forceinline__ float gelu_tanh_grad(float x) {
    float x3 = x * x * x;
    float u = 0.7978846f * (x + 0.044715f * x3);
    float t = (u > 8.0f) ? 1.0f : (u < -8.0f) ? -1.0f : tanhf(u);
    float du = 0.7978846f * (1.0f + 0.134145f * x * x);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * du;
}

extern "C" __global__ void gelu_backward(
    const float *g,
    const float *dy,
    float *out,
    unsigned len
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) {
        return;
    }
    out[i] = dy[i] * gelu_tanh_grad(g[i]);
}
