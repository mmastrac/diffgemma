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
//
// DETERMINISTIC ORDER: slots are assigned via a threadgroup count + exclusive
// prefix-sum, NOT an atomic_fetch_add race. The old atomic produced a
// run-to-run-varying survivor order, so the pass-2 gather's non-associative f32
// sum (sum_j prob[j]*embed[idx[j]]) rounded differently each run — tiny diffs
// that flipped borderline-convergence prompts (smoketest 14/15/16 jitter). With
// a fixed order (thread-major: thread t's strided survivors, ascending v) the
// gather sum is reproducible. Two vocab scans (count, then write); the extra
// read is negligible vs the gather.
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
    const uint tid = lid.x;
    const float mx = rowstat[row * 2u];
    const float sum = rowstat[row * 2u + 1u];

    // Pass 1: count this thread's survivors over its strided vocab subset.
    threadgroup uint counts[256];
    uint local = 0u;
    for (uint v = tid; v < vocab; v += 256u) {
        const float x = arena_load_bf16(logits, (ulong)row * vocab + v);
        if (x - mx >= SC_SPARSE_THRESH) {
            local += 1u;
        }
    }
    counts[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Exclusive prefix-sum -> deterministic base slot for this thread.
    uint base = 0u;
    for (uint i = 0u; i < tid; ++i) {
        base += counts[i];
    }

    // Pass 2: write survivors at base + running index (deterministic order).
    uint w = base;
    for (uint v = tid; v < vocab; v += 256u) {
        const float x = arena_load_bf16(logits, (ulong)row * vocab + v);
        if (x - mx >= SC_SPARSE_THRESH) {
            if (w < maxk) {
                out_idx[(ulong)row * maxk + w] = v;
                out_prob[(ulong)row * maxk + w] =
                    half((exp(x - mx) / sum) * SC_PROB_GEMM_SCALE);
            }
            w += 1u;
        }
    }

    // out_cnt = full survivor count (sum over threads); may exceed maxk.
    if (tid == 0u) {
        uint total = 0u;
        for (uint i = 0u; i < 256u; ++i) {
            total += counts[i];
        }
        out_cnt[row] = total;
    }
}
