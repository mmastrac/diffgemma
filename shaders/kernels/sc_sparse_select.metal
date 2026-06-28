#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "arena.metal"
#include "sc_prob_scale.metal"

// SC sparse softembed, pass 1: per row (token), scan the vocab and compact the
// "survivor" entries whose softmax prob is within e^THRESH of the row max into a
// per-row list (idx + fp16 prob). The distribution sharpens during diffusion, so
// most rows keep few entries; THRESH=-10 was smoketest-validated (16/16) as
// quality-neutral. count[row] is the FULL survivor count (may exceed maxk →
// overflow, monitored by the caller). One threadgroup (256 threads) per row.
constant float SC_SPARSE_THRESH = -10.0f;

kernel void sc_sparse_select(
    device const ushort *logits [[buffer(0)]],   // [rows, vocab] bf16
    device const float *rowstat [[buffer(1)]],   // [rows][2] = (max, sum)
    device uint *out_idx [[buffer(2)]],          // [rows, maxk]
    device half *out_prob [[buffer(3)]],         // [rows, maxk]
    device uint *out_cnt [[buffer(4)]],          // [rows]
    constant uint4 &params [[buffer(5)]],        // rows, vocab, maxk, _
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    const uint row = tgid.x;
    const uint rows = params.x, vocab = params.y, maxk = params.z;
    if (row >= rows) {
        return;
    }
    const float mx = rowstat[row * 2u];
    const float sum = rowstat[row * 2u + 1u];

    threadgroup atomic_uint cnt;
    if (lid.x == 0u) {
        atomic_store_explicit(&cnt, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint v = lid.x; v < vocab; v += 256u) {
        const float x = arena_load_bf16(logits, (ulong)row * vocab + v);
        if (x - mx >= SC_SPARSE_THRESH) {
            const uint slot = atomic_fetch_add_explicit(&cnt, 1u, memory_order_relaxed);
            if (slot < maxk) {
                out_idx[(ulong)row * maxk + slot] = v;
                out_prob[(ulong)row * maxk + slot] =
                    half((exp(x - mx) / sum) * SC_PROB_GEMM_SCALE);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid.x == 0u) {
        out_cnt[row] = atomic_load_explicit(&cnt, memory_order_relaxed);
    }
}
