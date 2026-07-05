#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "arena.metal"

/// `y = (a + b) * scalar` over bf16 arena planes.
kernel void residual_half(
    device const ushort *a [[buffer(0)]],
    device const ushort *b [[buffer(1)]],
    device ushort *y [[buffer(2)]],
    device const uchar *blob [[buffer(3)]],
    constant ulong &scal_off [[buffer(4)]],
    device float *dump [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
    float s = scal_off ? bf16_bytes(blob + scal_off) : 1.0f;
    float v = (arena_load(a, i) + arena_load(b, i)) * s;
    if (K_DUMP_STAGE >= 1u) dump[i] = v;
    K_ELEMENTWISE_GUARD();
    arena_store(y, i, v);
}
