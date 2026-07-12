#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"

kernel void router_scale_rows(
    device float *x [[buffer(0)]],
    device const float *scale [[buffer(1)]],
    constant uint2 &dims [[buffer(2)]],
    constant float &root [[buffer(3)]],
    device float *dump [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint seq_len = dims.x;
    uint hidden = dims.y;
    uint s = gid;
    if (K_SHAPE_ASSERT && (hidden == 0u || seq_len == 0u)) return;
    if (s >= seq_len) return;
    K_ELEMENTWISE_GUARD();
    uint off = s * hidden;
    if (K_DUMP_STAGE >= 1u) dump[s] = root;
    for (uint i = 0; i < hidden; i++) {
        x[off + i] *= scale[i] * root;
    }
}
