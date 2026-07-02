#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "arena.metal"
#include "dequant.metal"
#include "hidden_fc.metal"

/// Gather Q8 embed rows by token id: out[tok,d] = dequant(embed[id], d) * embed_scale.
kernel void embed_gather(
    device const uchar *blob [[buffer(0)]],
    device const uint *ids [[buffer(1)]],
    device ushort *out [[buffer(2)]],
    constant ulong &w_off [[buffer(3)]],
    constant uint2 &dims [[buffer(4)]],
    constant float &embed_scale [[buffer(5)]],
    constant uint &vocab [[buffer(6)]],
    device DebugStatus *dbg [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint hidden = dims.x;
    const uint num_tokens = dims.y;
    const uint tok = gid.y;
    const uint d = gid.x;
    dgq_assert_dims_nonzero(dbg, DbgKernelEmbedGather, hidden, num_tokens);
    if (tok >= num_tokens || d >= hidden) {
        if (dgq_debug_fast_enabled()) {
            dgq_assert_index(dbg, DbgKernelEmbedGather, tok, num_tokens);
            dgq_assert_index(dbg, DbgKernelEmbedGather, d, hidden);
        }
        return;
    }
    K_ELEMENTWISE_GUARD();
    const uint id = ids[tok];
    if (d == 0u) {
        dgq_assert_token_id(dbg, DbgKernelEmbedGather, id, vocab);
    }
    device const uchar *row = blob + w_off + (ulong)id * q8_row_bytes(hidden);
    float v = q8_at(row, d, bf16_bytes(row)) * embed_scale;
    if (dgq_debug_deep_enabled() && d == 0u) {
        dgq_assert_finite_f32(dbg, DbgKernelEmbedGather, v, tok);
    }
    if (K_HIDDEN_Y_F32) {
        ((device float *)out)[(ulong)tok * hidden + d] = v;
    } else {
        arena_store(out, (ulong)tok * hidden + d, v);
    }
}
