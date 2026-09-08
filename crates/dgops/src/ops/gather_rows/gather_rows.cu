// Gather rows by index. Mirrors gather_rows.metal.

extern "C" __global__ void gather_rows(
    float *out,
    const float *src,
    const unsigned *indices,
    unsigned num_indices,
    unsigned hidden
) {
    unsigned t = blockIdx.x;
    if (t >= num_indices) {
        return;
    }
    unsigned row = indices[t];
    const float *s = src + (size_t)row * hidden;
    float *d = out + (size_t)t * hidden;
    for (unsigned i = threadIdx.x; i < hidden; i += blockDim.x) {
        d[i] = s[i];
    }
}
