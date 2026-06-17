#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "sampler_device.metal"
#include "arena.metal"

/// Categorical inverse-CDF per row (tempered); tpg MUST divide cols evenly.
kernel void sample_apply(
    device const ushort *logits [[buffer(0)]],
    device const float *rowstat [[buffer(1)]],
    device CanvasState *S [[buffer(2)]],
    constant StepParams &P [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    device DebugStatus *dbg [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    if (K_SHAPE_ASSERT && (cols == 0u || (cols % tpg) != 0u)) {
        return;
    }
    K_ELEMENTWISE_GUARD();

    if (frozen_at(S, row)) {
        return;
    }

    float t = temp_at(S->step - 1u, P);
    float mx = rowstat[row * 2u], Z = rowstat[row * 2u + 1u];
    dgq_assert_positive_f32(dbg, DbgKernelSampleApply, Z, row);
    float target = S->u_cat[row] * Z;
    device const ushort *lr = logits + (ulong)row * cols;
    threadgroup float chunk[256];
    uint per = cols / tpg;
    float local = 0.f;
    for (uint v = lid * per; v < (lid + 1u) * per; ++v) {
        local += exp(arena_load(lr, v) / t - mx);
    }
    chunk[lid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float cum = 0.f;
        uint pick = cols - 1u;
        for (uint c = 0u; c < tpg; ++c) {
            if (cum + chunk[c] >= target) {
                for (uint v = c * per; v < (c + 1u) * per; ++v) {
                    cum += exp(arena_load(lr, v) / t - mx);
                    if (cum >= target) {
                        pick = v;
                        break;
                    }
                }
                break;
            }
            cum += chunk[c];
        }
        dgq_assert_index(dbg, DbgKernelSampleApply, pick, cols);
        S->new_sample[row] = pick;
    }
}
