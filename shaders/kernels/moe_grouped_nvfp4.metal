#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_KERNEL_COMMON_METAL
#include "common.metal"
#endif
#ifndef DGQ_INCLUDE_DEQUANT_METAL
// dequant_nvfp4_tile, nvfp4_row_bytes from include/dequant.metal (Rust concat)
#endif
#ifndef DGQ_INCLUDE_ACTIVATIONS_METAL
// gelu_tanh from include/activations.metal (Rust concat)
#endif
#ifndef DGQ_INCLUDE_ATTENTION_METAL
// LayerOffsets from include/attention.metal (Rust concat)
#endif
#ifndef DGQ_INCLUDE_MOE_ROUTER_METAL
// RouteScratch from include/moe_router.metal (Rust concat)
#endif
#ifndef DGQ_INCLUDE_MOE_GROUPED_METAL
// MoeGroupedDims + atomic_add_f32 from include/moe_grouped.metal (Rust concat)
#endif

/// Grouped MoE expert forward (NVFP4): gate||up → GELU×up → down, weighted scatter to moe_out.
kernel void moe_grouped_nvfp4(
    device const half* moe_in [[buffer(0)]],
    device float* moe_out [[buffer(1)]],
    device const uchar* blob [[buffer(2)]],
    device const LayerOffsets* L [[buffer(3)]],
    device const RouteScratch* R [[buffer(4)]],
    constant MoeGroupedDims& dims [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tpg [[threads_per_threadgroup]]
) {
    if (K_SHAPE_ASSERT
        && (dims.hidden == 0u || dims.moe_ff == 0u || dims.n_experts == 0u)) {
        return;
    }
    K_ELEMENTWISE_GUARD();
    if (dims.moe_ff > MOE_MAX_FF || dims.hidden > MOE_MAX_HIDDEN
        || dims.n_experts > MOE_MAX_EXPERTS) {
        return;
    }

    const uint e = tgid.y;
    const uint ltid = lid.x, tpg_w = tpg.x;
    const uint end = (e + 1u < dims.n_experts) ? R->offset[e + 1u] : R->num_slots;
    const uint n_tok = end - R->offset[e];
    if (tgid.x >= n_tok) {
        return;
    }
    const uint slot = R->offset[e] + tgid.x;
    const uint tok = R->token_list[slot];
    const float w = float(R->weight[tok][R->slot_list[slot]]);
    device const half* x = moe_in + (ulong)tok * dims.hidden;
    const ulong gu = L->experts_gate_up
        + (ulong)e * nvfp4_matrix_bytes(dims.moe_ff * 2u, dims.hidden);
    const ulong dn = L->experts_down + (ulong)e * nvfp4_matrix_bytes(dims.hidden, dims.moe_ff);
    float gu_scale = as_type<float>(*(device const uint*)(blob + gu));
    float dn_scale = as_type<float>(*(device const uint*)(blob + dn));
    const ulong gu_body = gu + 4ul;
    const ulong dn_body = dn + 4ul;
    const ulong hid_row = nvfp4_row_bytes(dims.hidden);
    const ulong ff_row = nvfp4_row_bytes(dims.moe_ff);
    threadgroup float act[704];
    for (uint r = ltid; r < dims.moe_ff; r += tpg_w) {
        float g = 0.f, u = 0.f;
        device const uchar* grow = blob + gu_body + (ulong)r * hid_row;
        device const uchar* urow = blob + gu_body + (ulong)(r + dims.moe_ff) * hid_row;
        for (uint k0 = 0; k0 < dims.hidden; k0 += 32u) {
            float wg[32], wu[32];
            dequant_nvfp4_tile(grow, dims.hidden, k0, wg, gu_scale);
            dequant_nvfp4_tile(urow, dims.hidden, k0, wu, gu_scale);
            for (uint i = 0; i < 32u; ++i) {
                float xv = float(x[k0 + i]);
                g += wg[i] * xv;
                u += wu[i] * xv;
            }
        }
        act[r] = gelu_tanh(g) * u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = ltid; d < dims.hidden; d += tpg_w) {
        float o = 0.f;
        device const uchar* drow = blob + dn_body + (ulong)d * ff_row;
        for (uint k0 = 0; k0 < dims.moe_ff; k0 += 32u) {
            float wd[32];
            dequant_nvfp4_tile(drow, dims.moe_ff, k0, wd, dn_scale);
            for (uint i = 0; i < 32u; ++i) {
                o += wd[i] * act[k0 + i];
            }
        }
        atomic_add_f32((device atomic_uint*)&moe_out[(ulong)tok * dims.hidden + d], w * o);
    }
}
