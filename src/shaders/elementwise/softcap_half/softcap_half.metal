#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "arena.metal"

constant float SOFTCAP = 30.0f;

kernel void softcap_half(
    device ushort *logits [[buffer(0)]],
    constant uint &base [[buffer(1)]],
    constant uint &len [[buffer(2)]],
    device float *dump [[buffer(3)]],
    device DebugStatus *dbg [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (K_SHAPE_ASSERT && len == 0u) return;
    if (gid >= len) return;
    K_ELEMENTWISE_GUARD();
    uint i = base + gid;
    float v = arena_load_bf16(logits, i);
    float x = clamp(v / SOFTCAP, -20.0f, 20.0f);
    float out = tanh(x) * SOFTCAP;
    if (K_DUMP_STAGE >= 1u) dump[gid] = out;
    dgq_assert_finite_f32(dbg, DbgKernelSoftcapHalf, out, gid);
    arena_store_bf16(logits, i, out);
}
