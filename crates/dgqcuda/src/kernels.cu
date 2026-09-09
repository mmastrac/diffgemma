// Diffusion forward-pass kernels. All f32, device-resident buffers; the
// pipeline dispatches these through gpukit::cuda directly.
// rsqrtf/expf/tanhf/sqrtf are device builtins: NVRTC has no <cmath>.

extern "C" __global__ void dgq_rms_norm(
    const float *x, const float *weight, float *out,
    unsigned seq, unsigned hidden, float eps
) {
    const unsigned row = blockIdx.x;
    if (row >= seq) return;
    const float *xr = x + (size_t)row * hidden;
    float *o = out + (size_t)row * hidden;
    __shared__ float red[256];
    float acc = 0.0f;
    for (unsigned i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = xr[i];
        acc += v * v;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned s = blockDim.x / 2u; s > 0u; s >>= 1u) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float)hidden + eps);
    __syncthreads();
    for (unsigned i = threadIdx.x; i < hidden; i += blockDim.x) {
        o[i] = xr[i] * inv * weight[i];
    }
}

// Per-head QK-norm: one block per (row, head); weight may be null (V).
extern "C" __global__ void dgq_rms_norm_heads(
    const float *x, const float *weight, float *out,
    unsigned rows, unsigned heads, unsigned head_dim, float eps
) {
    const unsigned r = blockIdx.y;
    const unsigned h = blockIdx.x;
    if (r >= rows || h >= heads) return;
    const float *xr = x + ((size_t)r * heads + h) * head_dim;
    float *o = out + ((size_t)r * heads + h) * head_dim;
    __shared__ float red[256];
    float acc = 0.0f;
    for (unsigned i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = xr[i];
        acc += v * v;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned s = blockDim.x / 2u; s > 0u; s >>= 1u) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float)head_dim + eps);
    __syncthreads();
    for (unsigned i = threadIdx.x; i < head_dim; i += blockDim.x) {
        o[i] = xr[i] * inv * (weight ? weight[i] : 1.0f);
    }
}

// Split-half / proportional RoPE over [seq, heads, head_dim], in place.
extern "C" __global__ void dgq_rope(
    float *x, const float *freqs,
    unsigned seq, unsigned heads, unsigned head_dim, unsigned rotary_dim
) {
    const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned total = seq * heads;
    if (gid >= total) return;
    const unsigned s = gid / heads;
    float *head = x + (size_t)gid * head_dim;
    const float *f = freqs + (size_t)s * rotary_dim;
    const unsigned half = rotary_dim / 2u;
    const unsigned half_head = head_dim / 2u;
    const bool proportional = rotary_dim < head_dim;
    for (unsigned d = 0; d < half; d++) {
        const float c = f[2u * d];
        const float sn = f[2u * d + 1u];
        const unsigned i1 = proportional ? (half_head + d) : (d + half);
        const float x0 = head[d];
        const float x1 = head[i1];
        head[d] = x0 * c - x1 * sn;
        head[i1] = x0 * sn + x1 * c;
    }
}

// Causal (+ optional sliding window) GQA attention, one block per query head.
// kv layout: [total_kv, n_kv_heads, 2*head_dim] = K then V per head.
extern "C" __global__ void dgq_attention_v2(
    const float *q, const float *kv, float *out,
    unsigned seq, unsigned n_heads, unsigned n_kv_heads,
    unsigned head_dim, unsigned total_kv, unsigned window
) {
    const unsigned row = blockIdx.x;
    const unsigned tok = row / n_heads;
    const unsigned qh = row % n_heads;
    if (tok >= seq) return;
    const unsigned n_groups = n_heads / n_kv_heads;
    const unsigned kvh = qh / n_groups;
    const float *qv = q + (size_t)row * head_dim;
    float *ov = out + (size_t)row * head_dim;
    float m = -1.0e30f, l = 0.0f;
    float acc[512];
    for (unsigned d = 0; d < head_dim; d++) acc[d] = 0.0f;
    for (unsigned t = 0; t < total_kv; t++) {
        if (t > tok) break;
        if (window > 0u && t + window <= tok) continue;
        // K for this position/head; V sits in the same position's V block,
        // which starts n_kv_heads*head_dim past the position's K block.
        const size_t pos_base = (size_t)t * 2u * n_kv_heads * head_dim;
        const float *k = kv + pos_base + (size_t)kvh * head_dim;
        float dot = 0.0f;
        for (unsigned d = 0; d < head_dim; d++) dot += qv[d] * k[d];
        const float mn = fmaxf(m, dot);
        const float corr = expf(m - mn);
        const float p = expf(dot - mn);
        m = mn;
        l = l * corr + p;
        const float *v = kv + pos_base + (size_t)n_kv_heads * head_dim + (size_t)kvh * head_dim;
        for (unsigned d = 0; d < head_dim; d++) acc[d] = acc[d] * corr + p * v[d];
    }
    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (unsigned d = 0; d < head_dim; d++) ov[d] = acc[d] * inv;
}

extern "C" __global__ void dgq_vec_add(float *x, const float *y, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

extern "C" __global__ void dgq_scale(float *x, float s, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

// out[i] = gelu_tanh(gate[i]) * up[i] * w  (w = per-token expert weight)
extern "C" __global__ void dgq_swiglu_weighted(
    const float *gate, const float *up, float w, float *out, unsigned n
) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = gate[i];
    float x3 = x * x * x;
    float u = 0.7978846f * (x + 0.044715f * x3);
    float t = (u > 8.0f) ? 1.0f : (u < -8.0f) ? -1.0f : tanhf(u);
    float g = 0.5f * x * (1.0f + t);
    out[i] = g * up[i] * w;
}

extern "C" __global__ void dgq_softcap(float *x, float cap, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = tanhf(x[i] / cap) * cap;
}

// Router input: rms_norm_no_scale(residual) * scale * hidden^-0.5
extern "C" __global__ void dgq_router_input(
    const float *residual, const float *scale, float *out,
    unsigned seq, unsigned hidden, float eps, float root
) {
    const unsigned row = blockIdx.x;
    if (row >= seq) return;
    const float *xr = residual + (size_t)row * hidden;
    float *o = out + (size_t)row * hidden;
    __shared__ float red[256];
    float acc = 0.0f;
    for (unsigned i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = xr[i];
        acc += v * v;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned s = blockDim.x / 2u; s > 0u; s >>= 1u) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float)hidden + eps);
    __syncthreads();
    for (unsigned i = threadIdx.x; i < hidden; i += blockDim.x) {
        o[i] = xr[i] * inv * scale[i] * root;
    }
}

// Router top-k: rank the (already projected) logits, softmax over the
// selected set, multiply by the per-expert scale. Ties keep the lower index.
extern "C" __global__ void dgq_router_topk(
    const float *logits, const float *per_expert_scale,
    unsigned *out_idx, float *out_w,
    unsigned seq, unsigned n_experts, unsigned top_k
) {
    const unsigned tok = blockIdx.x;
    if (tok >= seq) return;
    __shared__ float top_val[64];
    __shared__ unsigned top_idx[64];
    if (threadIdx.x != 0u) return;
    const float *row = logits + (size_t)tok * n_experts;
    const unsigned k = top_k;
    for (unsigned i = 0; i < k; i++) { top_val[i] = -1.0e30f; top_idx[i] = 0xFFFFFFFFu; }
    for (unsigned e = 0; e < n_experts; e++) {
        const float v = row[e];
        if (v > top_val[k - 1u] || (v == top_val[k - 1u] && e < top_idx[k - 1u])) {
            unsigned j = k - 1u;
            while (j > 0u && (top_val[j - 1u] < v || (top_val[j - 1u] == v && top_idx[j - 1u] > e))) {
                top_val[j] = top_val[j - 1u];
                top_idx[j] = top_idx[j - 1u];
                j--;
            }
            top_val[j] = v;
            top_idx[j] = e;
        }
    }
    float mx = -1.0e30f;
    for (unsigned i = 0; i < k; i++) mx = fmaxf(mx, top_val[i]);
    float sum = 0.0f;
    for (unsigned i = 0; i < k; i++) { top_val[i] = expf(top_val[i] - mx); sum += top_val[i]; }
    for (unsigned i = 0; i < k; i++) {
        float w = (sum > 0.0f) ? (top_val[i] / sum) : (1.0f / (float)k);
        w *= per_expert_scale[top_idx[i]];
        out_idx[tok * k + i] = top_idx[i];
        out_w[tok * k + i] = w;
    }
}

// dst += src * w
extern "C" __global__ void dgq_accum(float *dst, const float *src, float w, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i] * w;
}

