#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "dequant.metal"
#include "activations.metal"
#include "attention_device.metal"
#include "moe_router_device.metal"
#include "moe_grouped_device.metal"

/// Grouped MoE expert forward: gate||up → GELU×up → down, weighted scatter to moe_out.
/// Format selected by K_QUANT_FORMAT (FC3): Q4 affine vs NVFP4.
/// K_DUMP_STAGE >= 1: write act/metadata to dump [[buffer(6)]] (Q4 debug path).
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
    if (K_SHAPE_ASSERT
        && K_QUANT_FORMAT != QUANT_Q4_AFFINE && K_QUANT_FORMAT != QUANT_NVFP4) {
        return;
    }
    if (dims.moe_ff > MOE_MAX_FF || dims.hidden > MOE_MAX_HIDDEN
        || dims.n_experts > MOE_MAX_EXPERTS) {
        return;
    }

    const uint e = tgid.y;
    const uint ltid = lid.x, tpg_w = tpg.x;
    const uint end = R->row_start[e + 1u];
    const uint n_tok = end - R->row_start[e];
    if (tgid.x >= n_tok) {
        return;
    }
    const uint slot = R->row_start[e] + tgid.x;
    const uint tok = R->token_list[slot];
    const float w = float(R->weight[tok][R->slot_list[slot]]);
    device const half* x = moe_in + (ulong)tok * dims.hidden;
    const bool is_nvfp4 = (K_QUANT_FORMAT == QUANT_NVFP4);
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

    float gu_scale = 0.f;
    float dn_scale = 0.f;
    ulong gu = 0ul;
    ulong dn = 0ul;
    ulong gu_body = 0ul;
    ulong dn_body = 0ul;
    ulong hid_row = 0ul;
    ulong ff_row = 0ul;
    if (is_nvfp4) {
        gu = L->experts_gate_up
            + (ulong)e * nvfp4_matrix_bytes(dims.moe_ff * 2u, dims.hidden);
        dn = L->experts_down + (ulong)e * nvfp4_matrix_bytes(dims.hidden, dims.moe_ff);
        gu_scale = as_type<float>(*(device const uint*)(blob + gu));
        dn_scale = as_type<float>(*(device const uint*)(blob + dn));
        gu_body = gu + 4ul;
        dn_body = dn + 4ul;
        hid_row = nvfp4_row_bytes(dims.hidden);
        ff_row = nvfp4_row_bytes(dims.moe_ff);
    } else {
        gu = L->experts_gate_up
            + (ulong)e * (ulong)(dims.moe_ff * 2u) * q4_row_bytes(dims.hidden);
        dn = L->experts_down + (ulong)e * (ulong)dims.hidden * q4_row_bytes(dims.moe_ff);
        gu_body = gu;
        dn_body = dn;
    }

    threadgroup float act[704];
    for (uint r = ltid; r < dims.moe_ff; r += tpg_w) {
        float g = 0.f, u = 0.f;
        if (is_nvfp4) {
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
        } else {
            device const uchar* grow = blob + gu_body + (ulong)r * q4_row_bytes(dims.hidden);
            device const uchar* urow = blob + gu_body
                + (ulong)(r + dims.moe_ff) * q4_row_bytes(dims.hidden);
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
        if (is_nvfp4) {
            device const uchar* drow = blob + dn_body + (ulong)d * ff_row;
            for (uint k0 = 0; k0 < dims.moe_ff; k0 += 32u) {
                float wd[32];
                dequant_nvfp4_tile(drow, dims.moe_ff, k0, wd, dn_scale);
                for (uint i = 0; i < 32u; ++i) {
                    o += wd[i] * act[k0 + i];
                }
            }
        } else {
            device const uchar* drow = blob + dn_body + (ulong)d * q4_row_bytes(dims.moe_ff);
            for (uint k0 = 0; k0 < dims.moe_ff; k0 += 32u) {
                float wd[32];
                dequant_q4_group(drow + (k0 / 32u) * 20ul, wd);
                for (uint i = 0; i < 32u; ++i) {
                    o += wd[i] * act[k0 + i];
                }
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
