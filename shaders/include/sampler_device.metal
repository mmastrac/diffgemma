#ifndef DGQ_INCLUDE_SAMPLER_METAL
#define DGQ_INCLUDE_SAMPLER_METAL

/// Shared sampler state layouts (must match `step_kernel.rs` CanvasState / StepParams).
constant uint DGQ_SAMPLER_MAX_CANVAS = 256u;

struct StepParams {
    uint kv_len;
    uint max_steps;
    float entropy_bound;
    float t_min;
    float t_max;
    float conf_threshold;
    uint stability_threshold;
    uint min_early_stop_steps;
    uint accept_plateau_threshold;
    float plateau_prefix_mean_max;
};

struct CanvasState {
    uint ids[256];
    uint prev_argmax[256];
    uint new_sample[256];
    float entropy[256];
    uint sorted_idx[256];
    uint accept[256];
    float u_cat[256];
    ulong rng_state;
    uint step;
    uint stop_flag;
    uint argmax_stable;
    uint argmax_changed;
    float mean_entropy;
    uint accept_plateau;
    uint prev_accept_sig;
};

inline ulong lcg_next(ulong s) { return s * 6966169279ul + 1039523323ul; }
inline float lcg_f32(ulong s) { return float(uint(s >> 32)) * (1.0f / 4294967296.0f); }

/// Temperature at denoise step: `steps_done` completed before this rowstats pass.
/// CPU: cur_step counts max..1; t = t_min + (t_max - t_min) * (cur / n).
inline float temp_at(uint steps_done, constant StepParams &P) {
    float cur = float(P.max_steps - steps_done);
    return P.t_min + (P.t_max - P.t_min) * (cur / float(P.max_steps));
}

#endif
