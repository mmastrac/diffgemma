#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "common.metal"
#include "attention_device.metal"
#include "moe_router_device.metal"
#include "arena.metal"

/// RMSNorm → router scale → linear → softmax → top-k (monolith k_router).
kernel void moe_router(
    device const ushort *stream [[buffer(0)]],
    device const uchar *blob [[buffer(1)]],
    device const LayerOffsets *L [[buffer(2)]],
    device RouteScratch *R [[buffer(3)]],
    constant RouterDims &dims [[buffer(4)]],
    device DebugStatus *dbg [[buffer(5)]],
    uint tok [[threadgroup_position_in_grid]],
    uint e [[thread_position_in_threadgroup]]
) {
    if (K_SHAPE_ASSERT
        && (dims.canvas == 0u || dims.hidden == 0u || dims.n_experts == 0u || dims.top_k == 0u)) {
        return;
    }
    K_ELEMENTWISE_GUARD();
    if (tok >= dims.canvas || dims.n_experts > MOE_MAX_EXPERTS || dims.top_k > MOE_MAX_TOP_K) {
        return;
    }

    threadgroup float logits[128];
    threadgroup float red[4];
    device const ushort *x = stream + (ulong)tok * dims.hidden;
    // f32 stream plane (DGQ_HIDDEN_F32): index from the buffer BASE as float —
    // ushort pointer arithmetic would scale the row offset by 2, not 4.
    device const float *xf = ((device const float *)stream) + (ulong)tok * dims.hidden;
    float ss = 0.f;
    for (uint i = e; i < dims.hidden; i += 128u) {
        float t = arena_load(x, i);
        ss += t * t;
    }
    ss = simd_sum(ss);
    if ((e & 31u) == 0u) {
        red[e / 32u] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == 0u) {
        red[0] = rsqrt((red[0] + red[1] + red[2] + red[3]) / float(dims.hidden) + MOE_ROUTER_RMS_EPS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float norm_inv = red[0];
    if (e < dims.n_experts) {
        device const uchar *rs = blob + L->router_scale;
        device const uchar *wr = blob + L->router_proj + (ulong)e * dims.hidden * 2ul;
        float acc = 0.f;
        for (uint d = 0u; d < dims.hidden; ++d) {
            float xv = arena_load(x, d);
            float xn = xv * norm_inv * bf16_bytes(rs + 2ul * d) * dims.router_hscale;
            acc += xn * bf16_bytes(wr + 2ul * d);
        }
        logits[e] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == 0u) {
        uint pick[8];
        for (uint kk = 0u; kk < dims.top_k; ++kk) {
            float best = -1e30f;
            uint bi = 0u;
            for (uint i = 0u; i < dims.n_experts; ++i) {
                bool taken = false;
                for (uint p = 0u; p < kk; ++p) {
                    taken = taken || (pick[p] == i);
                }
                if (!taken && logits[i] > best) {
                    best = logits[i];
                    bi = i;
                }
            }
            pick[kk] = bi;
        }
        float mx = logits[pick[0]];
        for (uint kk = 1u; kk < dims.top_k; ++kk) {
            mx = max(mx, logits[pick[kk]]);
        }
        float sum = 0.f;
        for (uint kk = 0u; kk < dims.top_k; ++kk) {
            sum += exp(logits[pick[kk]] - mx);
        }
        device const uchar *pes = blob + L->per_expert_scale;
        float inv = (sum > 0.f) ? (1.f / sum) : 0.f;
        for (uint kk = 0u; kk < dims.top_k; ++kk) {
            float w = exp(logits[pick[kk]] - mx) * inv * bf16_bytes(pes + 2ul * pick[kk]);
            dgq_assert_index(dbg, DbgKernelMoeRouter, pick[kk], dims.n_experts);
            R->expert[tok][kk] = pick[kk];
            // Router weight is stored bf16 and read back as bf16 by the scatter
            // (bf16_to_f32(as_type<ushort>(...))). Must use the ALWAYS-bf16 packer,
            // the bf16 one — the router writes
            // f16 bits that the scatter misreads as bf16 -> ~zero weights -> MoE=0.
            R->weight[tok][kk] = as_type<half>(arena_bf16_bits(w));
        }
    }
}
