#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_FC_AXES_METAL
#include "fc_axes.metal"
#endif

kernel void memzero_bytes(
    device uchar4 *p [[buffer(0)]],
    device float *dump [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    K_ELEMENTWISE_GUARD();
    if (K_DUMP_STAGE >= 1u) dump[i] = 0.0f;
    p[i] = uchar4(0);
}
