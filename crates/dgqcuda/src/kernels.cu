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
    // Every thread must see the finished reduction before it reads red[0]:
    // without this the threads that skipped the last tree levels read a
    // partial sum, so the row scale depends on scheduling (the prompt path
    // and the denoise step disagreed on the same row).
    __syncthreads();
    const float inv = rsqrtf(red[0] / (float)hidden + eps);
    for (unsigned i = threadIdx.x; i < hidden; i += blockDim.x) {
        o[i] = xr[i] * inv * weight[i];
    }
}


// Scale-free row norm (rms_norm_no_scale): out = x * rsqrt(mean(x^2) + eps).
// A separate entry rather than a null weight pointer, which the driver
// rejects as an illegal address.
extern "C" __global__ void dgq_rms_norm_ns(
    const float *x, float *out, unsigned seq, unsigned hidden, float eps
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
    // Same barrier as dgq_rms_norm: red[0] is only final once every thread
    // has left the tree reduction.
    __syncthreads();
    const float inv = rsqrtf(red[0] / (float)hidden + eps);
    for (unsigned i = threadIdx.x; i < hidden; i += blockDim.x) {
        o[i] = xr[i] * inv;
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
    // Barrier before reading the reduced value (see dgq_rms_norm).
    __syncthreads();
    const float inv = rsqrtf(red[0] / (float)head_dim + eps);
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

// GQA attention, one block per query head.
// kv layout: [total_kv, n_kv_heads, 2*head_dim] = K then V per head.
//
// pos0 is the absolute position of query row 0 (the denoise pass runs the
// prompt and the canvas in one sequence, so the canvas starts at pos0 = prompt
// length). causal_split is the number of leading rows that attend causally:
// rows tok < causal_split see only positions <= their own, every other row
// sees the whole sequence (the diffusion canvas is bidirectional).
// causal_split = seq is a plain causal prefill; 0 is fully bidirectional.
extern "C" __global__ void dgq_attention_v2(
    const float *q, const float *kv, float *out,
    unsigned seq, unsigned n_heads, unsigned n_kv_heads,
    unsigned head_dim, unsigned total_kv, unsigned window,
    unsigned pos0, unsigned causal_split
) {
    const unsigned row = blockIdx.x;
    const unsigned tok = row / n_heads;
    const unsigned qh = row % n_heads;
    if (tok >= seq) return;
    const unsigned n_groups = n_heads / n_kv_heads;
    const unsigned kvh = qh / n_groups;
    const float *qv = q + (size_t)row * head_dim;
    float *ov = out + (size_t)row * head_dim;
    const unsigned abs_pos = pos0 + tok;
    // Exclusive upper bound on the attended positions. Clamped to the KV
    // length: with a non-zero pos0 a small sequence can put abs_pos + 1 past
    // the buffer, and reading there is an out-of-bounds fault (or, worse,
    // silently non-deterministic attention).
    const unsigned causal_end = (abs_pos + 1u < total_kv) ? (abs_pos + 1u) : total_kv;
    const unsigned kv_end = (tok < causal_split) ? causal_end : total_kv;
    float m = -1.0e30f, l = 0.0f;
    float acc[512];
    for (unsigned d = 0; d < head_dim; d++) acc[d] = 0.0f;
    for (unsigned t = 0; t < kv_end; t++) {
        if (window > 0u && t + window <= abs_pos) continue;
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
    // Barrier before reading the reduced value (see dgq_rms_norm).
    __syncthreads();
    const float inv = rsqrtf(red[0] / (float)hidden + eps);
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
// Sampler row stats for one canvas position: tempered softmax max/sum, the
// natural-log entropy (nats), and the argmax of the tempered logits.
//   entropy = ln(Z) - sum(e_i * x_i) / Z,  x_i = logit_i / t,  e_i = exp(x_i - mx)
// Ties in the argmax resolve to the lower id, like the CPU oracle.
extern "C" __global__ void dgq_row_stats(
    const float *logits, float *rowstat, float *entropy, unsigned *argmax,
    unsigned rows, unsigned cols, float t
) {
    const unsigned row = blockIdx.x;
    if (row >= rows) return;
    const float *lr = logits + (size_t)row * cols;
    __shared__ float r_mx[256];
    __shared__ float r_sum[256];
    __shared__ float r_ent[256];
    __shared__ unsigned r_am[256];
    __shared__ float r_amv[256];

    float mx = -1.0e30f;
    float amv = -1.0e30f;
    unsigned am = 0u;
    for (unsigned v = threadIdx.x; v < cols; v += blockDim.x) {
        const float x = lr[v] / t;
        if (x > amv || (x == amv && v < am)) { amv = x; am = v; }
        if (x > mx) mx = x;
    }
    r_mx[threadIdx.x] = mx;
    r_am[threadIdx.x] = am;
    r_amv[threadIdx.x] = amv;
    __syncthreads();
    for (unsigned s = blockDim.x / 2u; s > 0u; s >>= 1u) {
        if (threadIdx.x < s) {
            if (r_mx[threadIdx.x + s] > r_mx[threadIdx.x]) r_mx[threadIdx.x] = r_mx[threadIdx.x + s];
            if (r_amv[threadIdx.x + s] > r_amv[threadIdx.x]
                || (r_amv[threadIdx.x + s] == r_amv[threadIdx.x]
                    && r_am[threadIdx.x + s] < r_am[threadIdx.x])) {
                r_amv[threadIdx.x] = r_amv[threadIdx.x + s];
                r_am[threadIdx.x] = r_am[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    mx = r_mx[0];
    float sum = 0.0f, ent = 0.0f;
    for (unsigned v = threadIdx.x; v < cols; v += blockDim.x) {
        const float x = lr[v] / t;
        const float e = expf(x - mx);
        sum += e;
        ent += e * (x - mx);
    }
    r_sum[threadIdx.x] = sum;
    r_ent[threadIdx.x] = ent;
    __syncthreads();
    for (unsigned s = blockDim.x / 2u; s > 0u; s >>= 1u) {
        if (threadIdx.x < s) {
            r_sum[threadIdx.x] += r_sum[threadIdx.x + s];
            r_ent[threadIdx.x] += r_ent[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        const float z = r_sum[0];
        rowstat[row * 2u] = mx;
        rowstat[row * 2u + 1u] = z;
        entropy[row] = logf(z) - r_ent[0] / z;
        argmax[row] = r_am[0];
    }
}

// Sparse self-conditioning soft embedding, one block per canvas row: the soft
// embedding is sum_v p_v * embed[v] * scale with p = softmax(logits), but the
// diffusion distribution sharpens, so only entries within e^-THRESH of the row
// max are read (the tail contributes < 1e-4 of the row mass).
//
// Each thread scans its strided slice of the row and keeps up to MAXK
// survivors in shared memory (the engine sc_sparse_select uses the same
// per-thread compaction; survivors past the budget drop, the same
// approximation the engine documents). The hidden-dimension loop then walks
// the survivor lists and touches each surviving embed row once.
#define MAXK 16

extern "C" __global__ void dgq_soft_embed(
    const float *logits, const unsigned short *embed, float *out,
    unsigned rows, unsigned cols, unsigned hidden, float thresh, float scale
) {
    const unsigned row = blockIdx.x;
    if (row >= rows) return;
    const unsigned tid = threadIdx.x;
    const float *lr = logits + (size_t)row * cols;
    float *orow = out + (size_t)row * hidden;
    __shared__ unsigned s_idx[256 * MAXK];
    __shared__ float s_prob[256 * MAXK];
    __shared__ unsigned s_cnt[256];
    __shared__ float r_mx[256];

    float mx = -1.0e30f;
    for (unsigned v = tid; v < cols; v += blockDim.x) {
        if (lr[v] > mx) mx = lr[v];
    }
    r_mx[tid] = mx;
    __syncthreads();
    for (unsigned s = blockDim.x / 2u; s > 0u; s >>= 1u) {
        if (tid < s && r_mx[tid + s] > r_mx[tid]) r_mx[tid] = r_mx[tid + s];
        __syncthreads();
    }
    mx = r_mx[0];
    float z = 0.0f;
    for (unsigned v = tid; v < cols; v += blockDim.x) {
        if (lr[v] - mx >= thresh) z += expf(lr[v] - mx);
    }
    r_mx[tid] = z;
    __syncthreads();
    for (unsigned s = blockDim.x / 2u; s > 0u; s >>= 1u) {
        if (tid < s) r_mx[tid] += r_mx[tid + s];
        __syncthreads();
    }
    z = r_mx[0];

    unsigned cnt = 0u;
    for (unsigned v = tid; v < cols; v += blockDim.x) {
        if (lr[v] - mx >= thresh && cnt < MAXK) {
            s_idx[tid * MAXK + cnt] = v;
            s_prob[tid * MAXK + cnt] = expf(lr[v] - mx) / z;
            cnt++;
        }
    }
    s_cnt[tid] = cnt;
    __syncthreads();
    for (unsigned d = tid; d < hidden; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned t = 0u; t < blockDim.x; t++) {
            const unsigned c = s_cnt[t];
            const unsigned base = t * MAXK;
            for (unsigned i = 0u; i < c; i++) {
                const unsigned short bits = embed[(size_t)s_idx[base + i] * hidden + d];
                acc += s_prob[base + i] * __uint_as_float((unsigned)bits << 16u);
            }
        }
        orow[d] = acc * scale;
    }
}
