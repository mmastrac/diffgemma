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
    constant uint &eos_token [[buffer(5)]],
    device DebugStatus *dbg [[buffer(6)]],
    constant float &early_stop_mean_ent [[buffer(7)]],
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
        dgq_assert_entropy_nonneg(dbg, DbgKernelSampleCommit, ent[lid], lid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        ulong st = S->rng_state;
        for (uint i = 0u; i < canvas_size; ++i) {
            st = lcg_next(st);
            float u = lcg_f32(st);
            dgq_assert_fast(u >= 0.f && u < 1.f, dbg, DbgErrNonFinite, DbgKernelSampleCommit, i);
            S->u_cat[i] = u;
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
                if (prefix > P.entropy_bound) {
                    break;
                }
                S->accept[id] = 1u;
                prefix += ent[id];
            }
        }
        uint sig = 0u;
        for (uint i = 0u; i < canvas_size; ++i) {
            sig = sig * 31u + S->accept[i];
        }
        if (sig == S->prev_accept_sig) {
            S->accept_plateau += 1u;
        } else {
            S->accept_plateau = 0u;
        }
        S->prev_accept_sig = sig;

        float mean = 0.f;
        for (uint i = 0u; i < canvas_size; ++i) {
            mean += ent[i];
        }
        S->mean_entropy = mean / float(canvas_size);
        dgq_assert_entropy_nonneg(dbg, DbgKernelSampleCommit, S->mean_entropy, canvas_size);

        // MLX `_diffusion_stable_and_confident`: full-canvas argmax history + mean entropy.
        bool canvas_stable = false;
        uint thresh = P.stability_threshold;
        if (thresh == 0u) {
            canvas_stable = true;
        } else if (S->argmax_hist_len == thresh) {
            canvas_stable = true;
            uint ring_cap = min(thresh, DGQ_SAMPLER_ARGMAX_HIST_MAX);
            for (uint s = 0u; s < thresh && canvas_stable; ++s) {
                uint slot = (S->argmax_hist_base + s) % ring_cap;
                ulong off = ulong(slot) * ulong(DGQ_SAMPLER_MAX_CANVAS);
                for (uint i = 0u; i < canvas_size; ++i) {
                    if (S->prev_argmax[i] != S->argmax_hist[off + ulong(i)]) {
                        canvas_stable = false;
                        break;
                    }
                }
            }
        }
        S->canvas_stable = canvas_stable ? 1u : 0u;

        if (thresh > 0u) {
            uint ring_cap = min(thresh, DGQ_SAMPLER_ARGMAX_HIST_MAX);
            if (S->argmax_hist_len < thresh) {
                uint slot = (S->argmax_hist_base + S->argmax_hist_len) % ring_cap;
                ulong off = ulong(slot) * ulong(DGQ_SAMPLER_MAX_CANVAS);
                for (uint i = 0u; i < canvas_size; ++i) {
                    S->argmax_hist[off + ulong(i)] = S->prev_argmax[i];
                }
                S->argmax_hist_len += 1u;
            } else {
                ulong off = ulong(S->argmax_hist_base) * ulong(DGQ_SAMPLER_MAX_CANVAS);
                for (uint i = 0u; i < canvas_size; ++i) {
                    S->argmax_hist[off + ulong(i)] = S->prev_argmax[i];
                }
                S->argmax_hist_base = (S->argmax_hist_base + 1u) % ring_cap;
            }
        }

        S->step += 1u;

        // Pad-aware gate: an all-pad/filler argmax is a degenerate forward,
        // not a converged answer — suppress confident/plateau stops so the
        // denoiser gets more steps to recover (mirrors the CPU stopper).
        bool degenerate = true;
        for (uint i = 0u; i < canvas_size && degenerate; ++i) {
            uint t = S->prev_argmax[i];
            if (t != pad_token && t != filler_token) {
                degenerate = false;
            }
        }

        bool confident_stable = !degenerate && canvas_stable && S->mean_entropy < P.conf_threshold;
        bool plateau_stop = !degenerate && S->accept_plateau >= P.accept_plateau_threshold
            && S->mean_entropy < P.plateau_prefix_mean_max;
        // Entropy-only early stop (DGQ_EARLY_STOP_MEAN_ENT): stop once the mean
        // entropy is low enough, without waiting for full argmax stability —
        // the answer text is settled well before the argmax fully locks.
        bool ent_stop = !degenerate && early_stop_mean_ent > 0.f
            && S->step >= P.min_early_stop_steps
            && S->mean_entropy < early_stop_mean_ent;
        S->stop_flag = 0u;
        if (confident_stable || ent_stop) {
            S->stop_flag = 1u;
        } else if (plateau_stop) {
            S->stop_flag = 2u;
        } else if (S->step >= P.max_steps) {
            S->stop_flag = 3u;
        }

        (void)eos_token;
    }
}
