#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_KERNEL_COMMON_METAL
#include "common.metal"
#endif

kernel void vec_mul_inplace(
    device float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    constant uint &len [[buffer(2)]],
    device float *dump [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (K_SHAPE_ASSERT && len == 0u) return;
    if (gid >= len) return;
    (void)K_USE_FP4;
    if (K_DUMP_STAGE >= 1u) dump[gid] = b[gid];
    a[gid] *= b[gid];
}
