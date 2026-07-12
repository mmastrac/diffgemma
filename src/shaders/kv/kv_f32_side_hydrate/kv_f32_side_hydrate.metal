#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "attention_device.metal"

// Session-wide KV storage format of the SOURCE monolithic cache (must match
// the attention kernels). Unset (oracle/test compiles) = f16.
constant uint KV_FMT_FC [[function_constant(4)]];
constant uint KV_FMT = is_function_constant_defined(KV_FMT_FC) ? KV_FMT_FC : 0u;
constant bool KV_Q8 = (KV_FMT == 1u);

/// E14: (re)hydrate a sliding layer's f32 side ring from the monolithic
/// cache for positions [pos0, pos0+count) — used when a prefill resumes at
/// an offset the side ring is not valid for (session restore, rollback /
/// truncate, conversation swap). Widening f16 -> f32 bakes in the single
/// rounding those values already carry; it does NOT compound further.
/// shape = (count, pos0, nkv, hd); slot = pos & ring_mask on BOTH sides.
kernel void kv_f32_side_hydrate(
    device const uchar *kvcache [[buffer(0)]],
    device float *kv_f32_side [[buffer(1)]],
    constant uint4 &shape [[buffer(2)]],
    constant ulong &kv_region [[buffer(3)]],
    constant uint &kv_ring_mask [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint count = shape.x;
    const uint pos0 = shape.y;
    const uint nkv = shape.z;
    const uint hd = shape.w;
    const uint i = gid.x;
    const uint hh = gid.y;
    const uint d = gid.z;
    if (i >= count || hh >= nkv || d >= hd) {
        return;
    }
    K_ELEMENTWISE_GUARD();
    const uint p = pos0 + i;
    const uint slot = (kv_ring_mask != 0u) ? (p & kv_ring_mask) : p;
    device float *dst = kv_f32_side + (ulong)slot * nkv * hd * 2u;

    if (KV_Q8) {
        device const uchar *slot_base = kvcache + kv_region
            + (ulong)slot * kv_slot_stride_bytes(nkv, hd, KV_FMT);
        device const uchar *krow = slot_base + hh * kv_row_bytes(hd, KV_FMT);
        device const uchar *vrow =
            slot_base + ((ulong)nkv + hh) * kv_row_bytes(hd, KV_FMT);
        dst[hh * hd + d] = kv_q8_load(krow, d, hd);
        dst[(ulong)nkv * hd + hh * hd + d] = kv_q8_load(vrow, d, hd);
        return;
    }
    device const half *src = (device const half *)(kvcache + kv_region)
        + (ulong)slot * nkv * hd * 2u;
    dst[hh * hd + d] = float(src[hh * hd + d]);
    dst[(ulong)nkv * hd + hh * hd + d] = float(src[(ulong)nkv * hd + hh * hd + d]);
}
