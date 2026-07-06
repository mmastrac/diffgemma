#include <metal_stdlib>
using namespace metal;
#include <metal_simdgroup_matrix>

#include "fc_axes.metal"
#include "debug_status.metal"
#include "common.metal"
#include "attention_device.metal"
#include "arena.metal"
#include "sampler_device.metal"

// GQA-grouped matrix-unit attention: one threadgroup (QG=2 simdgroups, 64 lanes)
// per (query-tile of MT=8 rows, KV head). The two Q heads that share a KV head
// (16 Q / 8 KV = group size 2) stage K/V *once* and reuse it — halving the
// bf16->half staging tax that sank the 1-head MMA path, and the staging itself
// runs on 64 lanes. Each simdgroup owns one Q head's QK / softmax / O.
//
// Only valid for group size 2 and hd <= 256 (sliding layers). Full-attention
// layers use `attention_mma_full`. Same semantics as `attention` (online
// softmax, all-valid, no 1/sqrt(d) scale); `attention` stays the oracle.
//
// The O accumulator is REGISTER-RESIDENT (oreg, mma_full's pattern): the old
// 16 KiB threadgroup O tile put the whole kernel at ~26 KiB tgmem = 1 resident
// threadgroup/core, so nothing hid the per-chunk barrier+load latencies (the
// term that grows with kv_len). Register O drops tgmem to ~10 KiB (3x
// occupancy). Loops touching oreg are compile-time bounded (NCH_MAX) with an
// `8c < hd` guard so oreg never spills for runtime hd <= 256.

constant uint HD_MAX = 256u;
constant uint NCH_MAX = HD_MAX / 8u;  // 32 head-dim chunks of 8
constant uint MT = 8u;
constant uint QG = 2u;

kernel void attention_mma2(
    device const ushort *q [[buffer(0)]],
    device const ushort *kvcache [[buffer(1)]],
    device ushort *out [[buffer(2)]],
    device const LayerOffsets *L [[buffer(3)]],
    constant StepParams &P [[buffer(4)]],
    constant AttnDims &dims [[buffer(5)]],
    device DebugStatus *dbg [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    const uint hd = L->head_dim;
    const uint nkv = L->n_kv_heads;
    const uint kvh = tgid.y;
    const uint tok0 = tgid.x * MT;
    if (kvh >= nkv || tok0 >= dims.canvas) {
        return;
    }
    const uint sg = lid.x / 32u;    // which Q head in the group (0..QG-1)
    const uint lane = lid.x % 32u;  // lane within simdgroup
    const uint qh = kvh * QG + sg;  // model group size == QG
    const uint T = P.kv_len + dims.canvas;
    // Causal (prefill): query row r is at absolute pos kv_len+tok0+r and attends
    // only [0..pos]. Skip key-tiles fully past the tile's max query pos, and mask
    // within-tile keys per row in the softmax.
    const bool causal = dims.causal != 0u;
    const uint T_eff = causal ? min(T, P.kv_len + tok0 + MT) : T;
    // Sliding window (dims.window != 0): keys below the window start are masked.
    // Denoise: canvas attends the last window-1 encoder positions + all canvas
    // (uniform across rows). Causal: per-row [qpos-(window-1), qpos]; the tile
    // loop starts at the OLDEST row's window start, per-row softmax masking does
    // the rest. t_lo = 0 (bit-identical) until the context outgrows the window.
    const uint wm1 = (dims.window != 0u) ? (dims.window - 1u) : 0u;
    uint t_lo = 0u;
    if (dims.window != 0u) {
        const uint qpos0 = causal ? (P.kv_len + tok0) : P.kv_len;
        t_lo = (qpos0 > wm1) ? (qpos0 - wm1) : 0u;
    }
    const uint t_start = (t_lo / 8u) * 8u;  // tile-aligned; softmax masks the edge
    device const ushort *base = kvcache + L->kv_region / 2;

    threadgroup half qs[QG][MT][HD_MAX];   // staged Q per head
    threadgroup half ks[MT][8];            // shared K chunk [key][d]
    threadgroup half vs[MT][8];            // shared V chunk [key][d]
    threadgroup half ph[QG][MT][8];        // softmax probs per head
    threadgroup float st[QG][MT][8];       // QK scores per head
    threadgroup float pvt[QG][MT][8];      // P·V chunk per head
    threadgroup float mrow[QG][MT];
    threadgroup float lrow[QG][MT];
    threadgroup float corr[QG][MT];

    // Register-resident O accumulator (mma_full's pattern): lane owns rows
    // {r0, r1} at column dcol of every 8-wide head-dim chunk.
    const uint dcol = lane % 8u;
    const uint r0 = lane / 8u;          // 0..3
    const uint r1 = r0 + 4u;            // 4..7
    float oreg[2u * NCH_MAX];
    for (uint j = 0u; j < 2u * NCH_MAX; ++j) {
        oreg[j] = 0.f;
    }

    // Stage Q for both heads (all 64 lanes).
    for (uint i = lid.x; i < QG * MT * hd; i += 64u) {
        uint h = i / (MT * hd);
        uint rem = i % (MT * hd);
        uint r = rem / hd, d = rem % hd;
        uint tok = tok0 + r;
        uint qhh = kvh * QG + h;
        qs[h][r][d] = (tok < dims.canvas)
            ? half(arena_load(q + (ulong)tok * dims.n_q_heads * hd + qhh * hd, d))
            : half(0);
    }
    if (lid.x < QG * MT) {
        uint h = lid.x / MT, r = lid.x % MT;
        mrow[h][r] = -INFINITY;
        lrow[h][r] = 0.f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t0 = t_start; t0 < T_eff; t0 += 8u) {
        // ---- S[MT x 8] = Q . K^T over head_dim chunks (K shared by both heads) ----
        simdgroup_float8x8 sacc(0.f);
        for (uint kd = 0u; kd < hd; kd += 8u) {
            for (uint i = lid.x; i < 8u * 8u; i += 64u) {
                uint key = i / 8u, d = i % 8u;
                uint t = t0 + key;
                ks[key][d] = (t < T)
                    ? half(arena_load_bf16(base + (ulong)t * nkv * hd * 2u + kvh * hd, kd + d))
                    : half(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_half8x8 a, b;
            simdgroup_load(a, &qs[sg][0][kd], HD_MAX);
            simdgroup_load(b, &ks[0][0], 8, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(sacc, a, b, sacc);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        simdgroup_store(sacc, &st[sg][0][0], 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- online softmax over this 8-key tile (per head) ----
        if (lane < MT) {
            const uint qpos = P.kv_len + tok0 + lane;  // causal cutoff for this row
            // Per-row window start: causal rows each have their own; denoise is uniform.
            const uint row_lo = (dims.window == 0u)
                ? 0u
                : (causal ? ((qpos > wm1) ? qpos - wm1 : 0u) : t_lo);
            float tmax = -INFINITY;
            for (uint t = 0u; t < 8u; ++t) {
                bool valid = (t0 + t < T) && (t0 + t >= row_lo) && (!causal || t0 + t <= qpos);
                if (valid) {
                    tmax = max(tmax, st[sg][lane][t]);
                }
            }
            float mnew = max(mrow[sg][lane], tmax);
            float c = isinf(mrow[sg][lane]) ? 0.f : exp(mrow[sg][lane] - mnew);
            corr[sg][lane] = c;
            float lsum = 0.f;
            for (uint t = 0u; t < 8u; ++t) {
                bool valid = (t0 + t < T) && (t0 + t >= row_lo) && (!causal || t0 + t <= qpos);
                float p = valid ? exp(st[sg][lane][t] - mnew) : 0.f;
                ph[sg][lane][t] = half(p);
                lsum += p;
            }
            lrow[sg][lane] = lrow[sg][lane] * c + lsum;
            mrow[sg][lane] = mnew;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float cr0 = corr[sg][r0];
        const float cr1 = corr[sg][r1];

        // ---- O = O*corr + P . V over head_dim chunks (into registers) ----
        for (uint c = 0u; c < NCH_MAX; ++c) {
            const uint kd = c * 8u;
            if (kd >= hd) {
                break;
            }
            for (uint i = lid.x; i < 8u * 8u; i += 64u) {
                uint key = i / 8u, d = i % 8u;
                uint t = t0 + key;
                vs[key][d] = (t < T)
                    ? half(arena_load_bf16(
                          base + (ulong)t * nkv * hd * 2u + (ulong)nkv * hd + kvh * hd, kd + d))
                    : half(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_float8x8 pvacc(0.f);
            simdgroup_half8x8 a, b;
            simdgroup_load(a, &ph[sg][0][0], 8);
            simdgroup_load(b, &vs[0][0], 8);
            simdgroup_multiply_accumulate(pvacc, a, b, pvacc);
            simdgroup_store(pvacc, &pvt[sg][0][0], 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            oreg[2u * c] = oreg[2u * c] * cr0 + pvt[sg][r0][dcol];
            oreg[2u * c + 1u] = oreg[2u * c + 1u] * cr1 + pvt[sg][r1][dcol];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (lane < MT) {
        dgq_assert_positive_f32(dbg, DbgKernelAttention, lrow[sg][lane], (tok0 << 16u) | qh);
    }

    // Store O (register accumulator / denom) for rows r0, r1.
    const float l0 = lrow[sg][r0];
    const float l1 = lrow[sg][r1];
    const uint t_r0 = tok0 + r0;
    const uint t_r1 = tok0 + r1;
    for (uint c = 0u; c < NCH_MAX; ++c) {
        uint d = c * 8u + dcol;
        if (d >= hd) {
            break;
        }
        if (t_r0 < dims.canvas) {
            float y = (l0 > 0.f) ? oreg[2u * c] / l0 : 0.f;
            dgq_assert_finite_f32(dbg, DbgKernelAttention, y, d);
            arena_store(out + (ulong)t_r0 * dims.n_q_heads * hd + qh * hd, d, y);
        }
        if (t_r1 < dims.canvas) {
            float y = (l1 > 0.f) ? oreg[2u * c + 1u] / l1 : 0.f;
            dgq_assert_finite_f32(dbg, DbgKernelAttention, y, d);
            arena_store(out + (ulong)t_r1 * dims.n_q_heads * hd + qh * hd, d, y);
        }
    }
}
