#include <metal_stdlib>
using namespace metal;
#include <metal_simdgroup_matrix>

#include "fc_axes.metal"
#include "debug_status.metal"
#include "common.metal"
#include "attention_device.metal"
#include "arena.metal"
#include "sampler_device.metal"

// Flash-style GQA matrix-unit attention for FULL/GLOBAL layers (hd=512, nkv=2,
// GQA group 8). Same semantics as `attention` (all-valid bidirectional, online
// softmax, no 1/sqrt(d) — folded into QK-norm upstream); `attention` stays the
// oracle.
//
// vs `attention_mma` (QG=1, 16 KiB tgmem O tile, no K/V sharing — ties scalar):
//   (1) O accumulator is REGISTER-RESIDENT (oreg[2*NCH] per lane), not a tgmem
//       tile — frees ~16 KiB tgmem, lifts concurrent-threadgroup occupancy.
//   (2) QG simdgroups (one Q head each) share K/V staging per threadgroup, so
//       each KV element is read from device once per QG heads (not once per
//       head) — the bandwidth that grows with kv_len. Mirrors attention_mma2's
//       group-sharing, but at hd=512 the per-head O can't live in tgmem, hence
//       the register accumulator.
//
// hd is COMPILE-TIME (HD=512) so the head-dim chunk loops unroll and oreg stays
// in registers. nkv / n_q_heads stay runtime (KV addressing only). Full layers
// only — assert head_dim==512.

constant uint HD = 512u;       // full-layer head_dim (compile-time)
constant uint NCH = HD / 8u;   // 64 head-dim chunks of 8
constant uint MT = 8u;         // query rows per tile
constant uint QG = 2u;         // Q heads (simdgroups) per threadgroup; share K/V

kernel void attention_mma_full(
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
    if (K_SHAPE_ASSERT && hd != HD) {
        if (lid.x == 0u) {
            arena_store(out, 0, as_type<float>(0x7fc00000u));  // NaN: shape misuse
        }
        return;
    }

    const uint group = dims.n_q_heads / nkv;   // 8 for full layers
    const uint kvh = tgid.y;                    // KV head
    const uint sub = tgid.z;                    // sub-group of QG heads within this kvh
    const uint tok0 = tgid.x * MT;
    if (kvh >= nkv || tok0 >= dims.canvas) {
        return;
    }
    const uint sg = lid.x / 32u;                // simdgroup index -> local head in [0,QG)
    const uint lane = lid.x % 32u;
    const uint nlanes = QG * 32u;
    const uint qh = kvh * group + sub * QG + sg;  // global Q head for this simdgroup
    const uint T = P.kv_len + dims.canvas;
    // Causal (prefill): row r is at abs pos kv_len+tok0+r, attends only [0..pos].
    const bool causal = dims.causal != 0u;
    const uint T_eff = causal ? min(T, P.kv_len + tok0 + MT) : T;
    device const ushort *base = kvcache + L->kv_region / 2;

    // Shared K/V staging (one chunk, reused by all QG simdgroups).
    threadgroup half ks[MT][8];
    threadgroup half vs[MT][8];
    // Per-head (per-simdgroup) staging / softmax scratch.
    threadgroup half qs[QG][MT][HD];   // staged Q (bf16 -> half), QG*8 KiB
    threadgroup float st[QG][MT][8];   // QK scores S[row][key]
    threadgroup half ph[QG][MT][8];    // softmax probs P[row][key]
    threadgroup float pvt[QG][MT][8];  // P.V chunk [row][d]
    threadgroup float mrow[QG][MT];    // running max per row
    threadgroup float lrow[QG][MT];    // running denom per row
    threadgroup float corr[QG][MT];    // rescale per row for this key-tile

    // Register-resident O accumulator. Lane owns rows {r0, r1} and column dcol of
    // every 8-wide head-dim chunk: oreg[2c] = O[r0][8c+dcol], oreg[2c+1] = O[r1][..].
    const uint dcol = lane % 8u;
    const uint r0 = lane / 8u;          // 0..3
    const uint r1 = r0 + 4u;            // 4..7
    float oreg[2u * NCH];
    for (uint j = 0u; j < 2u * NCH; ++j) {
        oreg[j] = 0.f;
    }

    // Stage Q[QG][MT x HD] -> half (all lanes cooperate).
    for (uint i = lid.x; i < QG * MT * HD; i += nlanes) {
        uint h = i / (MT * HD);
        uint rem = i % (MT * HD);
        uint r = rem / HD, d = rem % HD;
        uint tok = tok0 + r;
        uint qhh = kvh * group + sub * QG + h;
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

    for (uint t0 = 0u; t0 < T_eff; t0 += 8u) {
        // ---- S[MT x 8] = Q . K^T over head_dim chunks (per simdgroup) ----
        simdgroup_float8x8 sacc(0.f);
        for (uint c = 0u; c < NCH; ++c) {
            uint kd = c * 8u;
            for (uint i = lid.x; i < 8u * 8u; i += nlanes) {
                uint key = i / 8u, d = i % 8u;
                uint t = t0 + key;
                ks[key][d] = (t < T)
                    ? half(arena_load_bf16(base + (ulong)t * nkv * hd * 2u + kvh * hd, kd + d))
                    : half(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_half8x8 a, b;
            simdgroup_load(a, &qs[sg][0][kd], HD);
            simdgroup_load(b, &ks[0][0], 8, ulong2(0, 0), true);  // -> b[d][key]
            simdgroup_multiply_accumulate(sacc, a, b, sacc);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        simdgroup_store(sacc, &st[sg][0][0], 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- online softmax over this 8-key tile (per row) ----
        if (lane < MT) {
            const uint qpos = P.kv_len + tok0 + lane;  // causal cutoff for this row
            float tmax = -INFINITY;
            for (uint t = 0u; t < 8u; ++t) {
                bool valid = (t0 + t < T) && (!causal || t0 + t <= qpos);
                if (valid) {
                    tmax = max(tmax, st[sg][lane][t]);
                }
            }
            float mnew = max(mrow[sg][lane], tmax);
            float cc = isinf(mrow[sg][lane]) ? 0.f : exp(mrow[sg][lane] - mnew);
            corr[sg][lane] = cc;
            float lsum = 0.f;
            for (uint t = 0u; t < 8u; ++t) {
                bool valid = (t0 + t < T) && (!causal || t0 + t <= qpos);
                float p = valid ? exp(st[sg][lane][t] - mnew) : 0.f;
                ph[sg][lane][t] = half(p);
                lsum += p;
            }
            lrow[sg][lane] = lrow[sg][lane] * cc + lsum;
            mrow[sg][lane] = mnew;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float cr0 = corr[sg][r0];
        const float cr1 = corr[sg][r1];

        // ---- O = O*corr + P . V over head_dim chunks (into registers) ----
        for (uint c = 0u; c < NCH; ++c) {
            uint kd = c * 8u;
            for (uint i = lid.x; i < 8u * 8u; i += nlanes) {
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
            simdgroup_load(a, &ph[sg][0][0], 8);   // P[row][key]
            simdgroup_load(b, &vs[0][0], 8);       // V[key][d]
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
    for (uint c = 0u; c < NCH; ++c) {
        uint d = c * 8u + dcol;
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
