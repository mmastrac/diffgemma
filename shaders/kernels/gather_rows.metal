#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_KERNEL_COMMON_METAL
#include "common.metal"
#endif

kernel void gather_rows(
    device const float *src [[buffer(0)]],
    device const uint *indices [[buffer(1)]],
    device float *dst [[buffer(2)]],
    constant uint2 &dims [[buffer(3)]],
    constant uint &batch_size [[buffer(4)]],
    device float *dump [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint hidden = dims.y;
    uint bi = gid / hidden;
    uint h = gid % hidden;
    if (K_SHAPE_ASSERT && (hidden == 0u || batch_size == 0u)) return;
    if (bi >= batch_size) return;
    (void)K_USE_FP4;
    uint tok = indices[bi];
    float v = src[tok * hidden + h];
    if (K_DUMP_STAGE >= 1u) dump[gid] = float(tok);
    dst[bi * hidden + h] = v;
}
