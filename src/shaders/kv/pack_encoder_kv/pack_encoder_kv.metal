#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "arena.metal"
#include "common.metal"
#include "attention_device.metal"

// Session-wide KV storage format (must match the attention kernels).
// Unset (oracle/test compiles) = f16. Under q8 the grid's z axis is
// head_dim/32 GROUPS (one thread quantizes a 32-group of K and of V);
// under f16 it is head_dim elements.
constant uint KV_FMT_FC [[function_constant(4)]];
constant uint KV_FMT = is_function_constant_defined(KV_FMT_FC) ? KV_FMT_FC : 0u;
constant bool KV_Q8 = (KV_FMT == 1u);
constant bool KV_Q4 = (KV_FMT == 2u);

/// Pack engine f32 K/V prefix into monolithic b4 layer region (matches CPU pack layout).
kernel void pack_encoder_kv(
    device const float *keys [[buffer(0)]],
    device const float *values [[buffer(1)]],
    device uchar *dst [[buffer(2)]],
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
    // Sliding layers store KV in a power-of-two ring (slot = pos & mask);
    // full layers are linear (mask 0).
    const uint abs_pos = dst_pos + pos;
    const uint slot = (kv_ring_mask != 0u) ? (abs_pos & kv_ring_mask) : abs_pos;

    if (KV_Q8) {
        // gid.z = group index; quantize one 32-group of K and of V.
        const uint g = d;
        device uchar *slot_base = dst + kv_region_bytes
            + ulong(slot) * kv_slot_stride_bytes(nkv, hd, KV_FMT);
        device uchar *krow = slot_base + hh * kv_row_bytes(hd, KV_FMT);
        device uchar *vrow = slot_base + ((ulong)nkv + hh) * kv_row_bytes(hd, KV_FMT);
        const uint src_base = (src_pos + pos) * per_token + hh * hd + g * KV_Q8_GROUP;
        float kg[KV_Q8_GROUP];
        float vg[KV_Q8_GROUP];
        for (uint j = 0u; j < KV_Q8_GROUP; ++j) {
            kg[j] = keys[src_base + j];
            vg[j] = values[src_base + j];
        }
        kv_q8_store_group(krow, g, kg, hd);
        kv_q8_store_group(vrow, g, vg, hd);
        return;
    }

    const uint src_i = (src_pos + pos) * per_token + hh * hd + d;
    const uint token_stride = nkv * hd * 2u;
    const ulong base_idx = kv_region_bytes / 2ul + ulong(slot) * ulong(token_stride);
    const ulong k_idx = base_idx + ulong(hh * hd + d);
    const ulong v_idx = base_idx + ulong(nkv * hd + hh * hd + d);
    // KV cache is f16 (see attention_device.metal kv_store): pack engine f32
    // KV as f16, not the toggleable arena precision.
    const ushort kb = as_type<ushort>(half(keys[src_i]));
    const ushort vb = as_type<ushort>(half(values[src_i]));
    device ushort *k_dst = (device ushort *)(dst + k_idx * 2ul);
    device ushort *v_dst = (device ushort *)(dst + v_idx * 2ul);
    k_dst[0] = kb;
    v_dst[0] = vb;
}
