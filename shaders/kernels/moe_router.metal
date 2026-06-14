#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_FC_AXES_METAL
#include "fc_axes.metal"
#endif
#ifndef DGQ_INCLUDE_COMMON_METAL
// bf16_bytes from include/common.metal (Rust concat)
#endif
#ifndef DGQ_INCLUDE_ATTENTION_DEVICE_METAL
// LayerOffsets from include/attention_device.metal (Rust concat)
#endif
#ifndef DGQ_INCLUDE_MOE_ROUTER_METAL
// RouteScratch + RouterDims from include/moe_router_device.metal (Rust concat)
#endif

/// RMSNorm → router scale → linear → softmax → top-k (monolith k_router).
kernel void moe_router(
    device const half *stream [[buffer(0)]],
    device const uchar *blob [[buffer(1)]],
    device const LayerOffsets *L [[buffer(2)]],
    device RouteScratch *R [[buffer(3)]],
    constant RouterDims &dims [[buffer(4)]],
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
    device const half *x = stream + (ulong)tok * dims.hidden;
    float ss = 0.f;
    for (uint i = e; i < dims.hidden; i += 128u) {
        float t = float(x[i]);
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
            float xn = float(x[d]) * norm_inv * bf16_bytes(rs + 2ul * d) * dims.router_hscale;
            acc += xn * bf16_bytes(wr + 2ul * d);
        }
        logits[e] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == 0u) {
        float mx = logits[0];
        for (uint i = 1u; i < dims.n_experts; ++i) {
            mx = max(mx, logits[i]);
        }
        float sum = 0.f;
        for (uint i = 0u; i < dims.n_experts; ++i) {
            logits[i] = exp(logits[i] - mx);
            sum += logits[i];
        }
        float wsum = 0.f;
        uint pick[8];
        for (uint kk = 0u; kk < dims.top_k; ++kk) {
            float best = -1.f;
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
            wsum += logits[bi];
        }
        device const uchar *pes = blob + L->per_expert_scale;
        for (uint kk = 0u; kk < dims.top_k; ++kk) {
            R->expert[tok][kk] = pick[kk];
            R->weight[tok][kk] = half((logits[pick[kk]] / wsum) * bf16_bytes(pes + 2ul * pick[kk]));
        }
    }
}
