#include <metal_stdlib>
using namespace metal;

/// Scatter `[seq, chunk]` GEMM output into `[seq, vocab]` logits at column offset `v0`.
kernel void scatter_vocab_chunk(
    device const float *chunk [[buffer(0)]],
    device float *logits [[buffer(1)]],
    constant uint4 &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint seq_len = params.x;
    uint chunk_cols = params.y;
    uint v0 = params.z;
    uint vocab = params.w;
    uint row = gid.y;
    uint col = gid.x;
    if (row >= seq_len || col >= chunk_cols) {
        return;
    }
    logits[row * vocab + v0 + col] = chunk[row * chunk_cols + col];
}
