// Scatter-add whole rows with atomic adds. Mirrors scatter_add_rows.metal.

extern "C" __global__ void scatter_add_rows(
    float *dst,
    const unsigned *indices,
    const float *src,
    unsigned num_indices,
    unsigned hidden
) {
    unsigned t = blockIdx.x;
    if (t >= num_indices) {
        return;
    }
    unsigned row = indices[t];
    float *d = dst + (size_t)row * hidden;
    const float *s = src + (size_t)t * hidden;
    for (unsigned i = threadIdx.x; i < hidden; i += blockDim.x) {
        atomicAdd(&d[i], s[i]);
    }
}
