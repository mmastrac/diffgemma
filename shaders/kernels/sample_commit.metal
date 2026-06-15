#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "sampler_device.metal"

/// LCG u_cat draws, entropy sort, accept mask, stability / early stop, step++.
kernel void sample_commit(
    device CanvasState *S [[buffer(0)]],
    constant StepParams &P [[buffer(1)]],
    constant uint &canvas_size [[buffer(2)]],
    constant uint &pad_token [[buffer(3)]],
    constant uint &filler_token [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]]
) {
    if (K_SHAPE_ASSERT && (canvas_size == 0u || canvas_size > DGQ_SAMPLER_MAX_CANVAS)) {
        return;
    }
    K_ELEMENTWISE_GUARD();

    threadgroup float ent[256];
    if (lid < canvas_size) {
        ent[lid] = S->entropy[lid];
        S->sorted_idx[lid] = lid;
        S->accept[lid] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        ulong st = S->rng_state;
        for (uint i = 0u; i < canvas_size; ++i) {
            st = lcg_next(st);
            S->u_cat[i] = lcg_f32(st);
        }
        S->rng_state = st;
        for (uint i = 1u; i < canvas_size; ++i) {
            uint id = S->sorted_idx[i];
            float e = ent[id];
            int j = int(i) - 1;
            while (j >= 0 && ent[S->sorted_idx[j]] > e) {
                S->sorted_idx[j + 1u] = S->sorted_idx[j];
                --j;
            }
            S->sorted_idx[j + 1u] = id;
        }
        bool final_step = (S->step + 1u >= P.max_steps);
        float prefix = 0.f;
        if (!final_step) {
            for (uint i = 0u; i < canvas_size; ++i) {
                uint id = S->sorted_idx[i];
                if (prefix <= P.entropy_bound) {
                    S->accept[id] = 1u;
                    prefix += ent[id];
                } else {
                    break;
                }
            }
        }
        float mean = 0.f;
        for (uint i = 0u; i < canvas_size; ++i) {
            mean += ent[i];
        }
        S->mean_entropy = mean / float(canvas_size);
        uint changed = atomic_load_explicit((device atomic_uint *)&S->argmax_changed,
                                            memory_order_relaxed);
        S->argmax_stable = changed ? 0u : (S->argmax_stable + 1u);
        atomic_store_explicit((device atomic_uint *)&S->argmax_changed, 0u,
                              memory_order_relaxed);
        S->step += 1u;
        bool degenerate = true;
        uint real_count = 0u;
        for (uint i = 0u; i < canvas_size; ++i) {
            uint t = S->prev_argmax[i];
            if (t != pad_token && t != filler_token) {
                degenerate = false;
                real_count++;
            }
        }
        bool confident_stable = S->mean_entropy < P.conf_threshold
            && S->argmax_stable >= P.stability_threshold;
        bool allowed = !degenerate
            && (S->step >= P.min_early_stop_steps || real_count >= 8u);
        if (confident_stable && allowed) {
            S->stop_flag = 1u;
        }
        if (S->step >= P.max_steps) {
            S->stop_flag = 1u;
        }
    }
}
