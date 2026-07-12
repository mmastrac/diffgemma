#include <metal_stdlib>
using namespace metal;

#include "sampler_device.metal"

/// Write unfrozen canvas row indices into `indices[0..num_active)`; `num_active[0]` is the count.
kernel void compact_active_rows(
    device const CanvasState *S [[buffer(0)]],
    device uint *indices [[buffer(1)]],
    device uint *num_active [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]]
) {
    if (lid == 0u) {
        uint n = 0u;
        for (uint i = 0u; i < DGQ_SAMPLER_MAX_CANVAS; ++i) {
            if (!frozen_at(S, i)) {
                indices[n++] = i;
            }
        }
        num_active[0] = n;
    }
}
