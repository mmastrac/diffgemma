#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "common.metal"
#include "arena.metal"

/// In-place bf16 scale (logits buffer).
kernel void half_scale_fp16(
    device ushort *y [[buffer(0)]],
    constant uint &n [[buffer(1)]],
    constant float &scale [[buffer(2)]],
    device float *dump [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (K_SHAPE_ASSERT && n == 0u) return;
    if (gid >= n) return;
    float v = arena_load(y, gid) * scale;
    if (K_DUMP_STAGE >= 1u) dump[gid] = v;
    K_ELEMENTWISE_GUARD();
    arena_store(y, gid, v);
}
