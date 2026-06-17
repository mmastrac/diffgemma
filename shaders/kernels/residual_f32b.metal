#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "arena.metal"

kernel void residual_f32b(
    device const ushort *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device ushort *y [[buffer(2)]],
    device float *dump [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    float v = arena_load(a, i) + b[i];
    if (K_DUMP_STAGE >= 1u) dump[i] = v;
    K_ELEMENTWISE_GUARD();
    arena_store(y, i, v);
}
