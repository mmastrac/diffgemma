#ifndef DGQ_INCLUDE_COMMON_METAL
#define DGQ_INCLUDE_COMMON_METAL

#include <metal_stdlib>
using namespace metal;

inline float bf16_bytes(device const uchar *p) {
    return as_type<float>((uint(p[0]) | (uint(p[1]) << 8)) << 16);
}

inline float bf16_to_f32(ushort b) {
    return as_type<float>(uint(b) << 16);
}

#endif
