#include <metal_stdlib>
using namespace metal;

struct AdamwParams {
    uint step;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
};

/// out = [p_new, m_new, v_new], each len elements.
kernel void adamw(
    device const float *p [[buffer(0)]],
    device const float *g [[buffer(1)]],
    device const float *m [[buffer(2)]],
    device const float *v [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant AdamwParams &params [[buffer(5)]],
    constant uint &len [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    float bc1 = 1.0f - pow(params.beta1, float(params.step));
    float bc2 = 1.0f - pow(params.beta2, float(params.step));
    float m_new = params.beta1 * m[gid] + (1.0f - params.beta1) * g[gid];
    float v_new = params.beta2 * v[gid] + (1.0f - params.beta2) * g[gid] * g[gid];
    float m_hat = m_new / bc1;
    float v_hat = v_new / bc2;
    float update = m_hat / (sqrt(v_hat) + params.eps) + params.weight_decay * p[gid];
    out[gid] = p[gid] - params.lr * update;
    out[len + gid] = m_new;
    out[2u * len + gid] = v_new;
}
