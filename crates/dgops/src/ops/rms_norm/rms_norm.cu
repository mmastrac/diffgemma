// Per-row RMSNorm. Mirrors rms_norm.metal.

extern "C" __global__ void rms_norm_rows(
    const float *x,
    const float *weight,
    float *out,
    unsigned rows,
    unsigned hidden,
    float eps
) {
    const unsigned TG = 256u;
    unsigned row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    __shared__ float scratch[256];

    const float *xr = x + (size_t)row * hidden;
    float *orow = out + (size_t)row * hidden;

    float sum = 0.0f;
    for (unsigned i = threadIdx.x; i < hidden; i += TG) {
        float v = xr[i];
        sum += v * v;
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned s = TG / 2u; s > 0u; s >>= 1u) {
        if (threadIdx.x < s) {
            scratch[threadIdx.x] += scratch[threadIdx.x + s];
        }
        __syncthreads();
    }
    float inv = rsqrtf(scratch[0] / (float)hidden + eps);
    for (unsigned i = threadIdx.x; i < hidden; i += TG) {
        orow[i] = xr[i] * inv * weight[i];
    }
}
