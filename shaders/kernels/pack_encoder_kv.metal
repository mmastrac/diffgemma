#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "arena.metal"

/// Pack engine f32 K/V prefix into monolithic b4 layer region (matches CPU pack layout).
kernel void pack_encoder_kv(
    device const float *keys [[buffer(0)]],
    device const float *values [[buffer(1)]],
    device uchar *dst [[buffer(2)]],
    constant uint4 &shape [[buffer(3)]],
    constant ulong &kv_region_bytes [[buffer(4)]],
    constant uint &src_pos [[buffer(5)]],
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
    const uint src_i = (src_pos + pos) * per_token + hh * hd + d;
    const uint token_stride = nkv * hd * 2u;
    const ulong base_idx = kv_region_bytes / 2ul + ulong(dst_pos + pos) * ulong(token_stride);
    const ulong k_idx = base_idx + ulong(hh * hd + d);
    const ulong v_idx = base_idx + ulong(nkv * hd + hh * hd + d);
    // KV cache is always bf16 (shared with the bf16 encoder + read as bf16 in
    // attention); pack as bf16, not the toggleable arena precision.
    const ushort kb = arena_bf16_bits(keys[src_i]);
    const ushort vb = arena_bf16_bits(values[src_i]);
    device ushort *k_dst = (device ushort *)(dst + k_idx * 2ul);
    device ushort *v_dst = (device ushort *)(dst + v_idx * 2ul);
    k_dst[0] = kb;
    v_dst[0] = vb;
}
