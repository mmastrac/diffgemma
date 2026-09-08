// Numerically stable row softmax. Mirrors softmax.metal.
#include <cmath>

extern "C" __global__ void softmax_rows(float *x, unsigned rows, unsigned cols) {
    const unsigned TG = 256u;
    unsigned row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    float *r = x + (size_t)row * cols;
    __shared__ float scratch[256];

    float local_max = -1e30f;
    for (unsigned c = threadIdx.x; c < cols; c += TG) {
        local_max = fmaxf(local_max, r[c]);
    }
    scratch[threadIdx.x] = local_max;
    __syncthreads();
    for (unsigned s = TG / 2u; s > 0u; s >>= 1u) {
        if (threadIdx.x < s) {
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float row_max = scratch[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (unsigned c = threadIdx.x; c < cols; c += TG) {
        float e = expf(r[c] - row_max);
        r[c] = e;
        local_sum += e;
    }
    scratch[threadIdx.x] = local_sum;
    __syncthreads();
    for (unsigned s = TG / 2u; s > 0u; s >>= 1u) {
        if (threadIdx.x < s) {
            scratch[threadIdx.x] += scratch[threadIdx.x + s];
        }
        __syncthreads();
    }
    float inv = 1.0f / scratch[0];
    __syncthreads();
    for (unsigned c = threadIdx.x; c < cols; c += TG) {
        r[c] *= inv;
    }
}
