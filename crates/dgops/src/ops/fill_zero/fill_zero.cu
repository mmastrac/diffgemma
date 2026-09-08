// Zero a range of an f32 buffer. Mirrors fill_zero.metal.

extern "C" __global__ void vec_fill_zero(float *x, unsigned base, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    x[base + i] = 0.0f;
}
