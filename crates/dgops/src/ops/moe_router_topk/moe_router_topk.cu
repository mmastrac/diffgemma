// MoE router: RMSNorm -> scaled linear -> top-k softmax. Mirrors
// moe_router_topk.metal. rsqrtf/expf are device builtins: NVRTC has no <cmath>.

struct MoeRouterParams {
    unsigned canvas;
    unsigned hidden;
    unsigned n_experts;
    unsigned top_k;
    float router_hscale;
    unsigned top_k2;
};

extern "C" __global__ void moe_router_topk(
    const float *stream,
    const float *router_scale,
    const float *router_proj,
    const float *per_expert_scale,
    unsigned *out_idx,
    float *out_w,
    MoeRouterParams p
) {
    const unsigned tok = blockIdx.x;
    const unsigned hidden = p.hidden;
    if (tok >= p.canvas || p.n_experts > 256u || p.top_k > 64u) {
        return;
    }
    __shared__ float red[256];
    __shared__ float logits[256];
    __shared__ float top_val[64];
    __shared__ unsigned top_idx[64];

    float acc = 0.0f;
    for (unsigned d = threadIdx.x; d < hidden; d += blockDim.x) {
        float v = stream[tok * p.hidden + d];
        acc += v * v;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned s = blockDim.x / 2u; s > 0u; s >>= 1u) {
        if (threadIdx.x < s) {
            red[threadIdx.x] += red[threadIdx.x + s];
        }
        __syncthreads();
    }
    const float rms_inv = rsqrtf(red[0] / (float)p.hidden + 1.0e-6f);
    __syncthreads();

    for (unsigned e = threadIdx.x; e < p.n_experts; e += blockDim.x) {
        float dot = 0.0f;
        const float *proj = router_proj + (unsigned long long)e * p.hidden;
        for (unsigned d = 0; d < p.hidden; d++) {
            dot += stream[tok * p.hidden + d] * rms_inv * router_scale[d] * p.router_hscale * proj[d];
        }
        logits[e] = dot;
    }
    __syncthreads();

    if (threadIdx.x == 0u) {
        const unsigned k = p.top_k;
        for (unsigned i = 0; i < k; i++) {
            top_val[i] = -1.0e30f;
            top_idx[i] = 0xFFFFFFFFu;
        }
        for (unsigned e = 0; e < p.n_experts; e++) {
            const float v = logits[e];
            if (v > top_val[k - 1u] ||
                (v == top_val[k - 1u] && e < top_idx[k - 1u])) {
                unsigned j = k - 1u;
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
        for (unsigned i = 0; i < k; i++) {
            mx = fmaxf(mx, top_val[i]);
        }
        float sum = 0.0f;
        for (unsigned i = 0; i < k; i++) {
            top_val[i] = expf(top_val[i] - mx);
            sum += top_val[i];
        }
        for (unsigned i = 0; i < k; i++) {
            float w = (sum > 0.0f) ? (top_val[i] / sum) : (1.0f / (float)k);
            w *= per_expert_scale[top_idx[i]];
            out_idx[tok * k + i] = top_idx[i];
            out_w[tok * k + i] = __uint_as_float(__float_as_uint(w) & 0xFFFF0000u);
        }
    }
}
