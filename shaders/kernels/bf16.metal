#ifndef DGQ_KERNEL_BF16_METAL
#define DGQ_KERNEL_BF16_METAL

#include <metal_stdlib>
using namespace metal;

inline float bf16_bytes(device const uchar *p) {
    return as_type<float>((uint(p[0]) | (uint(p[1]) << 8)) << 16);
}

#endif
