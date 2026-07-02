#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "arena.metal"
#include "hidden_fc.metal"

/// `y = (a + b) * scalar`. Under DGQ_HIDDEN_F32, `a` and `y` are the f32
/// hidden/stream planes (K_HIDDEN_A_F32 / K_HIDDEN_Y_F32); `b` (the bf16
/// branch being added) stays bf16 in every configuration.
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
    float av = K_HIDDEN_A_F32 ? ((device const float *)a)[i] : arena_load(a, i);
    float v = (av + arena_load(b, i)) * s;
    if (K_DUMP_STAGE >= 1u) dump[i] = v;
    K_ELEMENTWISE_GUARD();
    if (K_HIDDEN_Y_F32) {
        ((device float *)y)[i] = v;
    } else {
        arena_store(y, i, v);
    }
}
