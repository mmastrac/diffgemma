// out[i] = gelu_tanh(gate[i]) * up[i]. Mirrors swiglu_gelu.metal.

__device__ __forceinline__ float gelu_tanh(float x) {
    float x3 = x * x * x;
    float u = 0.7978846f * (x + 0.044715f * x3);
    float t = (u > 8.0f) ? 1.0f : (u < -8.0f) ? -1.0f : tanhf(u);
    return 0.5f * x * (1.0f + t);
}

extern "C" __global__ void swiglu_gelu(
    const float *gate,
    const float *up,
    float *out,
    unsigned len
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) {
        return;
    }
    out[i] = gelu_tanh(gate[i]) * up[i];
}
