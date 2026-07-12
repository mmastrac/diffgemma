#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "arena.metal"
#include "common.metal"
#include "attention_device.metal"

// Session-wide KV storage format (must match the attention kernels).
// Unset (oracle/test compiles) = f16.
constant uint KV_FMT_FC [[function_constant(4)]];
constant uint KV_FMT = is_function_constant_defined(KV_FMT_FC) ? KV_FMT_FC : 0u;
constant bool KV_Q8 = (KV_FMT == 1u);

/// Inverse of pack_encoder_kv: hydrate engine f32 K/V from a monolithic b4
/// layer region. Ring-aware on the SOURCE side (sliding layers store slot =
/// pos & mask); the engine destination is linear absolute positions (the
/// extend attention mask indexes prefix keys by buffer index == position).
/// Grid z is head_dim ELEMENTS for both formats (q8 dequant per element —
/// hydrate runs once per delta, not per step; simplicity over gather speed).
kernel void unpack_encoder_kv(
    device const uchar *src [[buffer(0)]],
    device float *keys [[buffer(1)]],
    device float *values [[buffer(2)]],
    constant uint4 &shape [[buffer(3)]],
    constant ulong &kv_region_bytes [[buffer(4)]],
    constant uint &src_pos [[buffer(5)]],
    constant uint &kv_ring_mask [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint pos = gid.x;
    const uint hh = gid.y;
    const uint d = gid.z;
    const uint token_count = shape.x;
    const uint dst_pos = shape.y;
    const uint nkv = shape.z;
    const uint hd = shape.w;
    if (K_SHAPE_ASSERT && (token_count == 0u || nkv == 0u || hd == 0u)) {
        return;
    }
    if (pos >= token_count || hh >= nkv || d >= hd) {
        return;
    }
    K_ELEMENTWISE_GUARD();
    const uint per_token = nkv * hd;
    const uint abs_pos = src_pos + pos;
    const uint slot = (kv_ring_mask != 0u) ? (abs_pos & kv_ring_mask) : abs_pos;
    const uint dst_i = (dst_pos + pos) * per_token + hh * hd + d;

    device const uchar *slot_base = src + kv_region_bytes
        + ulong(slot) * kv_slot_stride_bytes(nkv, hd, KV_FMT);
    device const uchar *krow = slot_base + ulong(hh) * kv_row_bytes(hd, KV_FMT);
    device const uchar *vrow = slot_base + ((ulong)nkv + hh) * kv_row_bytes(hd, KV_FMT);

    if (KV_Q8) {
        keys[dst_i] = kv_q8_load(krow, d, hd);
        values[dst_i] = kv_q8_load(vrow, d, hd);
        return;
    }
    keys[dst_i] = float(*(device const half *)(krow + ulong(d) * 2ul));
    values[dst_i] = float(*(device const half *)(vrow + ulong(d) * 2ul));
}
