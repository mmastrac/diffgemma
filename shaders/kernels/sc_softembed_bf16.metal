#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "common.metal"
#include "dequant.metal"
#include "arena.metal"

/// soft[tok,d] = sum_v softmax(logits[tok,v]) * embed[v,d] * embed_scale, bf16 embed.
/// bf16-embed variant of `sc_softembed` (slow O(vocab*hidden) reference path).
kernel void sc_softembed_bf16(
    device const ushort *logits [[buffer(0)]],
    device const float *rowstat [[buffer(1)]],
    device const uchar *blob [[buffer(2)]],
    device ushort *soft [[buffer(3)]],
    constant ulong &w_off [[buffer(4)]],
    constant uint &first_step [[buffer(5)]],
    constant uint3 &dims [[buffer(6)]],
    constant float &embed_scale [[buffer(7)]],
    device DebugStatus *dbg [[buffer(8)]],
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
        arena_store(soft, (ulong)tok * hidden + d, 0.f);
        return;
    }
    float mx = rowstat[tok * 2u];
    float sum = rowstat[tok * 2u + 1u];
    if (lid.x == 0u && d == 0u) {
        dgq_assert_positive_f32(dbg, DbgKernelScSoftembed, sum, tok);
    }
    device const ushort *lr = logits + (ulong)tok * vocab;
    device const ushort *emb = (device const ushort *)(blob + w_off);
    float acc = 0.f;
    for (uint v = 0; v < vocab; ++v) {
        float p = exp(arena_load(lr, v) - mx) / sum;
        acc += p * bf16_to_f32(emb[(ulong)v * hidden + d]);
    }
    arena_store(soft, (ulong)tok * hidden + d, acc * embed_scale);
}
