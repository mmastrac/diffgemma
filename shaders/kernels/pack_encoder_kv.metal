#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"

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
    const uint token_stride_half = nkv * hd * 2u;
    const ulong half_base = kv_region_bytes / 2ul + ulong(dst_pos + pos) * ulong(token_stride_half);
    const ulong k_half = half_base + ulong(hh * hd + d);
    const ulong v_half = half_base + ulong(nkv * hd + hh * hd + d);
    const half kf = half(keys[src_i]);
    const half vf = half(values[src_i]);
    const ushort kb = as_type<ushort>(kf);
    const ushort vb = as_type<ushort>(vf);
    device uchar *k_dst = dst + k_half * 2ul;
    device uchar *v_dst = dst + v_half * 2ul;
    k_dst[0] = uchar(kb & 0xffu);
    k_dst[1] = uchar(kb >> 8);
    v_dst[0] = uchar(vb & 0xffu);
    v_dst[1] = uchar(vb >> 8);
}
