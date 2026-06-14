#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_KERNEL_COMMON_METAL
#include "common.metal"
#endif
#ifndef DGQ_INCLUDE_COMMON_METAL
#error "sc_softembed: bundle must include shaders/include/common.metal"
#endif
#ifndef DGQ_INCLUDE_DEQUANT_METAL
#error "sc_softembed: bundle must include shaders/include/dequant.metal"
#endif

/// soft[tok,d] = sum_v softmax(logits[tok,v]) * dequant(embed[v,d]) * embed_scale
/// first_step != 0 -> zero output (SC MLP still runs on CPU path).
kernel void sc_softembed(
    device const half *logits [[buffer(0)]],
    device const float *rowstat [[buffer(1)]],
    device const uchar *blob [[buffer(2)]],
    device half *soft [[buffer(3)]],
    constant ulong &w_off [[buffer(4)]],
    constant uint &first_step [[buffer(5)]],
    constant uint3 &dims [[buffer(6)]],
    constant float &embed_scale [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    const uint hidden = dims.x;
    const uint num_tokens = dims.y;
    const uint vocab = dims.z;
    const uint tok = tgid.y;
    const uint d = tgid.x * 64u + lid.x;
    if (K_SHAPE_ASSERT && (hidden == 0u || num_tokens == 0u || vocab == 0u)) {
        return;
    }
    if (tok >= num_tokens || d >= hidden) {
        return;
    }
    K_ELEMENTWISE_GUARD();
    if (first_step != 0u) {
        soft[(ulong)tok * hidden + d] = half(0);
        return;
    }
    float mx = rowstat[tok * 2u];
    float sum = rowstat[tok * 2u + 1u];
    device const half *lr = logits + (ulong)tok * vocab;
    float acc = 0.f;
    for (uint v = 0; v < vocab; ++v) {
        float p = exp(float(lr[v]) - mx) / sum;
        device const uchar *row = blob + w_off + (ulong)v * q8_row_bytes(hidden);
        acc += p * q8_at(row, d, bf16_bytes(row));
    }
    soft[(ulong)tok * hidden + d] = half(acc * embed_scale);
}
