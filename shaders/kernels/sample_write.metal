#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "sampler_device.metal"

/// Accepted positions -> new_sample; rejected -> fresh uniform id; updates rng_state.
kernel void sample_write(
    device CanvasState *S [[buffer(0)]],
    constant uint &canvas_size [[buffer(1)]],
    constant uint &vocab_size [[buffer(2)]],
    device DebugStatus *dbg [[buffer(3)]],
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
                uint t = S->new_sample[i];
                dgq_assert_token_id(dbg, DbgKernelSampleWrite, t, vocab_size);
                S->ids[i] = t;
            } else {
                st = lcg_next(st);
                uint t = uint(st >> 32) % vocab_size;
                dgq_assert_token_id(dbg, DbgKernelSampleWrite, t, vocab_size);
                S->ids[i] = t;
            }
        }
        S->rng_state = st;
    }
}
