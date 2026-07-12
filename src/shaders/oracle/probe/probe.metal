#include <metal_stdlib>
using namespace metal;

/// Elementwise copy for bandwidth measurement.
kernel void memcpy_f32(
    device const float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        dst[gid] = src[gid];
    }
}
