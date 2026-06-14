#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_KERNEL_COMMON_METAL
#include "common.metal"
#endif
#ifndef DGQ_INCLUDE_DEQUANT_METAL
// dequant_q4_group, q4_row_bytes from include/dequant.metal (Rust concat)
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

/// Grouped MoE expert forward (Q4): gate||up → GELU×up → down, weighted scatter to moe_out.
/// K_DUMP_STAGE >= 1: write act/metadata to dump [[buffer(6)]] and direct-store moe_out
/// (single-writer debug path; production uses atomic scatter).
kernel void moe_grouped(
    device const half* moe_in [[buffer(0)]],
    device float* moe_out [[buffer(1)]],
    device const uchar* blob [[buffer(2)]],
    device const LayerOffsets* L [[buffer(3)]],
    device const RouteScratch* R [[buffer(4)]],
    constant MoeGroupedDims& dims [[buffer(5)]],
    device float* dump [[buffer(6)]],
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
    const bool probe = (K_DUMP_STAGE >= 1u);
    if (probe && ltid == 0u) {
        const uint meta = dims.moe_ff * 2u;
        dump[meta + 0] = float(tok);
        dump[meta + 1] = float(slot);
        dump[meta + 2] = float(e);
        dump[meta + 3] = w;
        for (uint i = 0u; i < 8u; ++i) {
            dump[meta + 4u + i] = float(x[i]);
        }
        for (uint i = 0u; i < 8u; ++i) {
            dump[meta + 12u + i] = float(moe_in[i]);
        }
    }
    const ulong gu = L->experts_gate_up
        + (ulong)e * (ulong)(dims.moe_ff * 2u) * q4_row_bytes(dims.hidden);
    const ulong dn = L->experts_down + (ulong)e * (ulong)dims.hidden * q4_row_bytes(dims.moe_ff);
    threadgroup float act[704];
    for (uint r = ltid; r < dims.moe_ff; r += tpg_w) {
        float g = 0.f, u = 0.f;
        device const uchar* grow = blob + gu + (ulong)r * q4_row_bytes(dims.hidden);
        device const uchar* urow = blob + gu + (ulong)(r + dims.moe_ff) * q4_row_bytes(dims.hidden);
        for (uint k0 = 0; k0 < dims.hidden; k0 += 32u) {
            float wg[32], wu[32];
            dequant_q4_group(grow + (k0 / 32u) * 20ul, wg);
            dequant_q4_group(urow + (k0 / 32u) * 20ul, wu);
            for (uint i = 0; i < 32u; ++i) {
                float xv = float(x[k0 + i]);
                g += wg[i] * xv;
                u += wu[i] * xv;
            }
        }
        act[r] = gelu_tanh(g) * u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (probe && ltid == 0u) {
        for (uint r = 0u; r < dims.moe_ff; ++r) {
            dump[r] = act[r];
        }
    }
    for (uint d = ltid; d < dims.hidden; d += tpg_w) {
        if (probe && d == 0u && ltid == 0u) {
            for (uint r = 0u; r < dims.moe_ff; ++r) {
                dump[dims.moe_ff + r] = act[r];
            }
        }
        float o = 0.f;
        device const uchar* drow = blob + dn + (ulong)d * q4_row_bytes(dims.moe_ff);
        for (uint k0 = 0; k0 < dims.moe_ff; k0 += 32u) {
            float wd[32];
            dequant_q4_group(drow + (k0 / 32u) * 20ul, wd);
            for (uint i = 0; i < 32u; ++i) {
                o += wd[i] * act[k0 + i];
            }
        }
        if (probe) {
            if (ltid == 0u && d < 8u) {
                const uint meta = dims.moe_ff * 2u;
                dump[meta + 20u + d] = w * o;
            }
            moe_out[(ulong)tok * dims.hidden + d] = w * o;
        } else {
            atomic_add_f32((device atomic_uint*)&moe_out[(ulong)tok * dims.hidden + d], w * o);
        }
    }
    if (probe) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ltid == 0u) {
            const uint meta = dims.moe_ff * 2u;
            for (uint i = 0u; i < 8u; ++i) {
                dump[meta + 28u + i] = moe_out[(ulong)tok * dims.hidden + i];
            }
        }
    }
}
