// out += addend, elementwise. Mirrors vec_add.metal.

extern "C" __global__ void vec_add_inplace(float *out, const float *addend, unsigned len) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) {
        return;
    }
    out[i] += addend[i];
}
