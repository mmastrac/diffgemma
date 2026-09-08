// Backward pass of row softmax. Mirrors softmax_backward.metal.

extern "C" __global__ void softmax_backward(
    const float *probs,
    const float *dp,
    float *out,
    unsigned rows,
    unsigned cols
) {
    const unsigned TG = 256u;
    unsigned row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    __shared__ float scratch[256];

    const float *pr = probs + (size_t)row * cols;
    const float *dr = dp + (size_t)row * cols;
    float *orow = out + (size_t)row * cols;

    float s = 0.0f;
    for (unsigned i = threadIdx.x; i < cols; i += TG) {
        s += pr[i] * dr[i];
    }
    scratch[threadIdx.x] = s;
    __syncthreads();
    for (unsigned step = TG / 2u; step > 0u; step >>= 1u) {
        if (threadIdx.x < step) {
            scratch[threadIdx.x] += scratch[threadIdx.x + step];
        }
        __syncthreads();
    }
    float row_sum = scratch[0];
    for (unsigned i = threadIdx.x; i < cols; i += TG) {
        orow[i] = pr[i] * (dr[i] - row_sum);
    }
}
