#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "arena.metal"

/// y[base+gid] (bf16 arena) = x[gid] (f32) * scale.
/// Fused convert+scale for chunked SC softembed f32 accumulator → half arena.
kernel void f32_to_half_scale(
    device const float *x [[buffer(0)]],
    device ushort *y [[buffer(1)]],
    constant uint &base [[buffer(2)]],
    constant uint &len [[buffer(3)]],
    constant float &scale [[buffer(4)]],
    device float *dump [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (K_SHAPE_ASSERT && len == 0u) return;
    if (gid >= len) return;
    K_ELEMENTWISE_GUARD();
    uint i = base + gid;
    float v = x[gid] * scale;
    if (K_DUMP_STAGE >= 1u) dump[gid] = v;
    arena_store(y, i, v);
}
