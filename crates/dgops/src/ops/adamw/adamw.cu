// AdamW update, [p_new, m_new, v_new]. Mirrors adamw.metal.
#include <cmath>

struct AdamwParams {
    unsigned step;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
};

extern "C" __global__ void adamw(
    const float *p,
    const float *g,
    const float *m,
    const float *v,
    float *out,
    AdamwParams params,
    unsigned len
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) {
        return;
    }
    float bc1 = 1.0f - powf(params.beta1, (float)params.step);
    float bc2 = 1.0f - powf(params.beta2, (float)params.step);
    float m_new = params.beta1 * m[i] + (1.0f - params.beta1) * g[i];
    float v_new = params.beta2 * v[i] + (1.0f - params.beta2) * g[i] * g[i];
    float m_hat = m_new / bc1;
    float v_hat = v_new / bc2;
    float update = m_hat / (sqrtf(v_hat) + params.eps) + params.weight_decay * p[i];
    out[i] = p[i] - params.lr * update;
    out[len + i] = m_new;
    out[2u * len + i] = v_new;
}
