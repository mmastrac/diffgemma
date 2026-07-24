#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "sampler_device.metal"

/// Accepted positions -> new_sample; rejected -> fresh uniform id; updates rng_state.
///
/// `freeze_enable=0` (default; DGQ_FREEZE=1 re-enables) skips the frozen bit
/// entirely: every row stays live, the accept set is re-decided from fresh
/// entropies each step and accepted rows take that step's fresh denoiser
/// token — the MLX/HF reference semantics (our CPU sampler). The legacy
/// freeze was proven to be the flat-row wart driver.
/// `use_argmax!=0` (default; DGQ_DENOISER_ARGMAX=0 restores HF categorical)
/// commits the row argmax instead of the tempered categorical sample,
/// matching MLX's default temperature=0 denoiser (the schedule temp then
/// only shapes entropy/SC, never the token choice).
kernel void sample_write(
    device CanvasState *S [[buffer(0)]],
    constant uint &canvas_size [[buffer(1)]],
    constant uint &vocab_size [[buffer(2)]],
    device DebugStatus *dbg [[buffer(3)]],
    constant uint &freeze_enable [[buffer(4)]],
    constant uint &use_argmax [[buffer(5)]],
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
                uint t = use_argmax != 0u ? S->prev_argmax[i] : S->new_sample[i];
                dgq_assert_token_id(dbg, DbgKernelSampleWrite, t, vocab_size);
                S->ids[i] = t;
                if (freeze_enable != 0u) {
                    set_frozen(S, i);
                }
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
