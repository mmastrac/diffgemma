#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "arena.metal"

// Source / destination element format (per-kernel axes, FC4/FC5):
//   true  = f32 (4-byte) buffer
//   false = activation-arena 2-byte slot (bf16, or fp16 under K_ARENA_F16)
// The three historical entry points collapse into this one generic kernel,
// specialized by (K_SRC_F32, K_DST_F32):
//   (true,  true)  = gather_rows            (f32   -> f32,   batched-prefill gather)
//   (false, false) = gather_rows_bf16       (arena -> arena, sampler compaction)
//   (false, true)  = gather_rows_bf16_to_f32(arena -> f32,   fused MoE input)
// The bf16->bf16 case is bit-identical to the old raw ushort copy: arena_load
// then arena_store round-trips an already-representable bf16/f16 value exactly.
constant bool K_SRC_F32 [[function_constant(4)]];
constant bool K_DST_F32 [[function_constant(5)]];

/// Gather rows by index from a row-major `[tokens, hidden]` source into
/// `[batch, hidden]`, generic over src/dst element format.
kernel void gather_rows(
    device const uchar *src [[buffer(0)]],
    device const uint *indices [[buffer(1)]],
    device uchar *dst [[buffer(2)]],
    constant uint2 &dims [[buffer(3)]],
    constant uint &batch_size [[buffer(4)]],
    device float *dump [[buffer(5)]],
    constant uint &elem_base [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint hidden = dims.y;
    uint elem = elem_base + gid;
    uint bi = elem / hidden;
    uint h = elem % hidden;
    if (K_SHAPE_ASSERT && (hidden == 0u || batch_size == 0u)) {
        return;
    }
    if (bi >= batch_size) {
        return;
    }
    K_ELEMENTWISE_GUARD();
    uint tok = indices[bi];
    ulong sidx = (ulong)tok * hidden + h;
    float v = K_SRC_F32 ? ((device const float *)src)[sidx]
                        : arena_load((device const ushort *)src, sidx);
    if (K_DUMP_STAGE >= 1u) {
        dump[gid] = float(tok);
    }
    if (K_DST_F32) {
        ((device float *)dst)[elem] = v;
    } else {
        arena_store((device ushort *)dst, elem, v);
    }
}
