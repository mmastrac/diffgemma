#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_KERNEL_COMMON_METAL
#include "common.metal"
#endif
#ifndef DGQ_INCLUDE_SAMPLER_METAL
#include "sampler.metal"
#endif

/// Accepted positions -> new_sample; rejected -> fresh uniform id; updates rng_state.
kernel void sample_write(
    device CanvasState *S [[buffer(0)]],
    constant uint &canvas_size [[buffer(1)]],
    constant uint &vocab_size [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]]
) {
    if (K_SHAPE_ASSERT && (canvas_size == 0u || canvas_size > DGQ_SAMPLER_MAX_CANVAS || vocab_size == 0u)) {
        return;
    }
    K_ELEMENTWISE_GUARD();

    if (lid == 0u) {
        ulong st = S->rng_state;
        for (uint i = 0u; i < canvas_size; ++i) {
            if (S->accept[i] != 0u) {
                S->ids[i] = S->new_sample[i];
            } else {
                st = lcg_next(st);
                S->ids[i] = uint(st >> 32) % vocab_size;
            }
        }
        S->rng_state = st;
    }
}
