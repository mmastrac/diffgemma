#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "sampler_device.metal"

/// Tempered row stats -> rowstat {mx, sum}; per-row entropy (nats); argmax.
kernel void sample_rowstats(
    device const half *logits [[buffer(0)]],
    device float *rowstat [[buffer(1)]],
    device CanvasState *S [[buffer(2)]],
    constant StepParams &P [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    constant uint &pad_token [[buffer(5)]],
    constant uint &filler_token [[buffer(6)]],
    constant uint &eos_token [[buffer(7)]],
    device DebugStatus *dbg [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    if (K_SHAPE_ASSERT && cols == 0u) {
        return;
    }
    K_ELEMENTWISE_GUARD();

    float t = temp_at(S->step, P);

    device const half *lr = logits + (ulong)row * cols;
    threadgroup float r_mx[8];
    threadgroup float r_sum[8];
    threadgroup float r_ent[8];
    threadgroup uint r_am[8];
    threadgroup float r_amv[8];
    float mx = -INFINITY;
    uint am = 0u;
    float amv = -INFINITY;
    for (uint v = lid; v < cols; v += tpg) {
        float x = float(lr[v]) / t;
        if (x > amv) {
            amv = x;
            am = v;
        }
        mx = max(mx, x);
    }
    mx = simd_max(mx);
    for (uint o = 16u; o > 0u; o >>= 1u) {
        float ov = simd_shuffle_down(amv, o);
        uint oi = simd_shuffle_down(am, o);
        if (ov > amv || (ov == amv && oi < am)) {
            amv = ov;
            am = oi;
        }
    }
    uint sg = lid / 32u, nsg = (tpg + 31u) / 32u;
    if ((lid & 31u) == 0u) {
        r_mx[sg] = mx;
        r_am[sg] = am;
        r_amv[sg] = amv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        for (uint i = 1u; i < nsg; ++i) {
            r_mx[0] = max(r_mx[0], r_mx[i]);
            if (r_amv[i] > r_amv[0] || (r_amv[i] == r_amv[0] && r_am[i] < r_am[0])) {
                r_amv[0] = r_amv[i];
                r_am[0] = r_am[i];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = r_mx[0];
    float sum = 0.f, ent = 0.f;
    for (uint v = lid; v < cols; v += tpg) {
        float x = float(lr[v]) / t;
        float e = exp(x - mx);
        sum += e;
        ent += e * (x - mx);
    }
    sum = simd_sum(sum);
    ent = simd_sum(ent);
    if ((lid & 31u) == 0u) {
        r_sum[sg] = sum;
        r_ent[sg] = ent;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float ts = 0.f, te = 0.f;
        for (uint i = 0u; i < nsg; ++i) {
            ts += r_sum[i];
            te += r_ent[i];
        }
        dgq_assert_positive_f32(dbg, DbgKernelSampleRowstats, ts, row);
        float ent_val = log(ts) - te / ts;
        dgq_assert_entropy_nonneg(dbg, DbgKernelSampleRowstats, ent_val, row);
        dgq_assert_index(dbg, DbgKernelSampleRowstats, r_am[0], cols);
        S->entropy[row] = ent_val;
        rowstat[row * 2u] = mx;
        rowstat[row * 2u + 1u] = ts;
        S->prev_argmax[row] = r_am[0];
        (void)pad_token;
        (void)filler_token;
        (void)eos_token;
    }
}
