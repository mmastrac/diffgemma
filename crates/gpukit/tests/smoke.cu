// Trivial CUDA smoke kernel for the gpukit driver-FFI test.
// y[i] = a * x[i] + y[i]
extern "C" __global__ void saxpy(float *y, const float *x, float a, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
