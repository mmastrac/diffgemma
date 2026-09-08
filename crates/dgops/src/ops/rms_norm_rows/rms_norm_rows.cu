// Per-row Gemma RMSNorm: one thread block per row, f32 I/O, optional affine
// scale. Mirrors rms_norm_rows.metal; rsqrtf/__syncthreads are device builtins.

extern "C" __global__ void rms_norm_rows(
    const float *x,
    const float *weight,
    float *out,
    unsigned seq_len,
    unsigned hidden,
    float eps
) {
    const unsigned row = blockIdx.x;
    if (row >= seq_len) {
        return;
    }
    const float *xr = x + (size_t)row * hidden;
    float *orow = out + (size_t)row * hidden;

    const unsigned TG = blockDim.x;
    __shared__ float scratch[256];

    float acc = 0.0f;
    for (unsigned i = threadIdx.x; i < hidden; i += TG) {
        float v = xr[i];
        acc += v * v;
    }
    scratch[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned s = TG / 2u; s > 0u; s >>= 1u) {
        if (threadIdx.x < s) {
            scratch[threadIdx.x] += scratch[threadIdx.x + s];
        }
        __syncthreads();
    }
    const float rms_inv = rsqrtf(scratch[0] / (float)hidden + eps);
    __syncthreads();

    for (unsigned i = threadIdx.x; i < hidden; i += TG) {
        orow[i] = xr[i] * rms_inv * weight[i];
    }
}
