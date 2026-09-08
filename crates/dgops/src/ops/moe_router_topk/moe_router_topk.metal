#include <metal_stdlib>
using namespace metal;

constant float ROUTER_RMS_EPS = 1e-6f;
constant uint MAX_EXPERTS = 256u;

/// Must stay layout-identical to MoeRouterParams in mod.rs and .cu.
struct MoeRouterParams {
    uint canvas;
    uint hidden;
    uint n_experts;
    uint top_k;
    float router_hscale;
    uint top_k2;
};

/// One threadgroup per canvas row: normalize, project, then one thread selects
/// the top-k (ties toward the lower expert index) and softmaxes over them.
kernel void moe_router_topk(
    device const float *stream [[buffer(0)]],
    device const float *router_scale [[buffer(1)]],
    device const float *router_proj [[buffer(2)]],
    device const float *per_expert_scale [[buffer(3)]],
    device uint *out_idx [[buffer(4)]],
    device float *out_w [[buffer(5)]],
    constant MoeRouterParams &p [[buffer(6)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint3 tpg3 [[threads_per_threadgroup]]
) {
    const uint tpg = tpg3.x;
    const uint hidden = p.hidden;
    const uint tok = tgp.y;
    if (tok >= p.canvas || p.n_experts > MAX_EXPERTS) {
        return;
    }

    threadgroup float red[256];
    float acc = 0.0f;
    for (uint d = lid; d < hidden; d += tpg) {
        float v = stream[tok * p.hidden + d];
        acc += v * v;
    }
    red[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            red[lid] += red[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float rms_inv = rsqrt(red[0] / float(p.hidden) + ROUTER_RMS_EPS);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float logits[MAX_EXPERTS];
    for (uint e = lid; e < p.n_experts; e += tpg) {
        float dot = 0.0f;
        const device float *proj = router_proj + (ulong)e * p.hidden;
        for (uint d = 0; d < p.hidden; d++) {
            dot += stream[tok * p.hidden + d] * rms_inv * router_scale[d] * p.router_hscale * proj[d];
        }
        logits[e] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0u) {
        // Insertion-sorted top-k, best first; ties keep the lower index.
        threadgroup float top_val[64];
        threadgroup uint top_idx[64];
        const uint k = p.top_k;
        for (uint i = 0; i < k; i++) {
            top_val[i] = -1.0e30f;
            top_idx[i] = 0xFFFFFFFFu;
        }
        for (uint e = 0; e < p.n_experts; e++) {
            const float v = logits[e];
            if (v > top_val[k - 1u] ||
                (v == top_val[k - 1u] && e < top_idx[k - 1u])) {
                uint j = k - 1u;
                while (j > 0u &&
                       (top_val[j - 1u] < v ||
                        (top_val[j - 1u] == v && top_idx[j - 1u] > e))) {
                    top_val[j] = top_val[j - 1u];
                    top_idx[j] = top_idx[j - 1u];
                    j--;
                }
                top_val[j] = v;
                top_idx[j] = e;
            }
        }
        float mx = -1.0e30f;
        for (uint i = 0; i < k; i++) {
            mx = max(mx, top_val[i]);
        }
        float sum = 0.0f;
        for (uint i = 0; i < k; i++) {
            top_val[i] = exp(top_val[i] - mx);
            sum += top_val[i];
        }
        for (uint i = 0; i < k; i++) {
            float w = (sum > 0.0f) ? (top_val[i] / sum) : (1.0f / float(k));
            w *= per_expert_scale[top_idx[i]];
            out_idx[tok * k + i] = top_idx[i];
            out_w[tok * k + i] = as_type<float>(as_type<uint>(w) & 0xFFFF0000u);
        }
    }
}
