#include <metal_stdlib>
using namespace metal;

/// Copy `len` f32 elements device-to-device starting at `base`.
kernel void copy_f32(
    device const float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant uint2 &range [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint base = range.x;
    uint len = range.y;
    uint i = base + gid;
    if (gid >= len) {
        return;
    }
    dst[i] = src[i];
}
