// x *= scale, elementwise. Mirrors vec_scale.metal.

extern "C" __global__ void vec_scale_inplace(float *x, float scale, unsigned len) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) {
        return;
    }
    x[i] *= scale;
}
