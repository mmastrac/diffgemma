#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "arena.metal"

/// Elementwise convert + scale, generic over src/dst element dtype (FC4/FC5)
/// tunable-style — one kernel subsuming the whole converter family:
///   dst[dst_base + gid] = to_dst( scale * from_src(src[src_base + gid]) )
///   K_SRC_F32 / K_DST_F32: true = f32 (4-byte), false = activation-arena
///   2-byte slot (bf16, or fp16 under K_ARENA_F16).
/// The four historical kernels are exactly the (src_f32, dst_f32) corners:
///   (false, true ) = half_to_f32        (arena -> f32,   no scale)
///   (false, false) = half_scale         (arena -> arena, in-place scale)
///   (true,  true ) = copy_f32           (f32   -> f32,   no scale)
///   (true,  false) = f32_to_half_scale  (f32   -> arena, scale)
/// Scale is a runtime param (1.0 for the copy/convert cases -> f32 identity,
/// bit-exact). Separate src/dst offsets cover f32_to_half_scale's asymmetry.
constant bool K_SRC_F32 [[function_constant(4)]];
constant bool K_DST_F32 [[function_constant(5)]];

kernel void convert_scale(
    device const uchar *src [[buffer(0)]],
    device uchar *dst [[buffer(1)]],
    constant uint &src_base [[buffer(2)]],
    constant uint &dst_base [[buffer(3)]],
    constant uint &len [[buffer(4)]],
    constant float &scale [[buffer(5)]],
    device float *dump [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (K_SHAPE_ASSERT && len == 0u) {
        return;
    }
    if (gid >= len) {
        return;
    }
    K_ELEMENTWISE_GUARD();
    const uint si = src_base + gid;
    const uint di = dst_base + gid;
    float v = (K_SRC_F32 ? ((device const float *)src)[si]
                         : arena_load((device const ushort *)src, si))
        * scale;
    if (K_DUMP_STAGE >= 1u) {
        dump[gid] = v;
    }
    if (K_DST_F32) {
        ((device float *)dst)[di] = v;
    } else {
        arena_store((device ushort *)dst, di, v);
    }
}
