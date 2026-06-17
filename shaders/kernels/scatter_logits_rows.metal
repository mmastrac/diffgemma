#include <metal_stdlib>
using namespace metal;

/// Scatter compact `[num_active, vocab]` logits rows into full `[canvas, vocab]` buffer.
kernel void scatter_logits_rows(
    device const half *compact [[buffer(0)]],
    device half *logits [[buffer(1)]],
    device const uint *row_map [[buffer(2)]],
    constant uint2 &dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint num_active = dims.x;
    uint vocab = dims.y;
    uint bi = gid.y;
    uint col = gid.x;
    if (bi >= num_active || col >= vocab) {
        return;
    }
    uint dst_row = row_map[bi];
    logits[(ulong)dst_row * vocab + col] = compact[(ulong)bi * vocab + col];
}
