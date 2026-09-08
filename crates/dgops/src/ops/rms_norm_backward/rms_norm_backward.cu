// Backward pass of per-row RMSNorm. Mirrors rms_norm_backward.metal.

extern "C" __global__ void rms_norm_backward(
    const float *x,
    const float *weight,
    const float *dy,
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
    __shared__ float scratch_sq[256];
    __shared__ float scratch_dot[256];

    const float *xr = x + (size_t)row * hidden;
    const float *dr = dy + (size_t)row * hidden;
    float *orow = out + (size_t)row * hidden;

    float sum_sq = 0.0f;
    float dot = 0.0f;
    for (unsigned i = threadIdx.x; i < hidden; i += TG) {
        float xv = xr[i];
        sum_sq += xv * xv;
        dot += dr[i] * weight[i] * xv;
    }
    scratch_sq[threadIdx.x] = sum_sq;
    scratch_dot[threadIdx.x] = dot;
    __syncthreads();
    for (unsigned s = TG / 2u; s > 0u; s >>= 1u) {
        if (threadIdx.x < s) {
            scratch_sq[threadIdx.x] += scratch_sq[threadIdx.x + s];
            scratch_dot[threadIdx.x] += scratch_dot[threadIdx.x + s];
        }
        __syncthreads();
    }
    float inv = rsqrtf(scratch_sq[0] / (float)hidden + eps);
    float inv3 = inv * inv * inv;
    float coef = inv3 * scratch_dot[0] / (float)hidden;
    for (unsigned i = threadIdx.x; i < hidden; i += TG) {
        orow[i] = inv * dr[i] * weight[i] - coef * xr[i];
    }
}
