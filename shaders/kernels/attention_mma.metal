#include <metal_stdlib>
using namespace metal;
#include <metal_simdgroup_matrix>

#include "fc_axes.metal"
#include "debug_status.metal"
#include "common.metal"
#include "attention_device.metal"
#include "arena.metal"
#include "sampler_device.metal"

// Flash-style GQA attention on the matrix units (simdgroup_float8x8).
// Same semantics as `attention` (all-valid bidirectional, online softmax, no
// 1/sqrt(d) scale — folded into QK-norm upstream); `attention` stays the oracle.
//
// Tiling: one threadgroup (1 simdgroup, 32 lanes) per (query-tile of MT=8 rows,
// q_head). Q[MT x hd] staged to half once; running O[MT x hd] accumulator and
// per-row m/l in threadgroup f32. Keys stream in tiles of NT=8: QK^T and P·V run
// as 8x8 MMA over head_dim chunks; online softmax per key-tile. Ragged T / canvas
// tails masked (-inf) or skipped at store.
//
// Tile size: NT=8 measured best. Larger NT grows threadgroup memory and, with the
// 16 KiB O accumulator already resident at hd=512, drops concurrent-threadgroup
// occupancy faster than it saves softmax cycles (NT=64 regressed both shapes).
// Net result: this path beats scalar only at hd=512 (full layers); the well-tuned
// scalar kernel (64 lanes, direct reads, tiny TG mem, high occupancy) wins at
// hd=256. See NOTES — wire MMA for is_full layers only.

constant uint HD_MAX = 512u;
constant uint MT = 8u;

kernel void attention_mma(
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
    const uint qh = tgid.y;
    const uint tok0 = tgid.x * MT;
    if (qh >= dims.n_q_heads || tok0 >= dims.canvas) {
        return;
    }
    const uint lane = lid.x;
    const uint kvh = qh / (dims.n_q_heads / nkv);
    const uint T = P.kv_len + dims.canvas;
    device const ushort *base = kvcache + L->kv_region / 2;

    threadgroup half qs[MT][HD_MAX];   // staged Q (bf16 -> half)
    threadgroup float ot[MT][HD_MAX];  // running O accumulator
    threadgroup half ks[MT][8];        // K key-tile chunk [key][d]
    threadgroup half vs[MT][8];        // V key-tile chunk [key][d]
    threadgroup half ph[MT][8];        // softmax probs P [row][key]
    threadgroup float st[MT][8];       // QK scores S [row][key]
    threadgroup float pvt[MT][8];      // P·V chunk [row][d]
    threadgroup float mrow[MT];        // running max per row
    threadgroup float lrow[MT];        // running denom per row
    threadgroup float corr[MT];        // rescale per row for this key-tile

    // Stage Q[MT x hd] -> half and zero the O accumulator.
    for (uint i = lane; i < MT * hd; i += 32u) {
        uint r = i / hd, d = i % hd;
        uint tok = tok0 + r;
        qs[r][d] = (tok < dims.canvas)
            ? half(arena_load(q + (ulong)tok * dims.n_q_heads * hd + qh * hd, d))
            : half(0);
        ot[r][d] = 0.f;
    }
    if (lane < MT) {
        mrow[lane] = -INFINITY;
        lrow[lane] = 0.f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t0 = 0u; t0 < T; t0 += 8u) {
        // ---- S[MT x 8] = Q . K^T over head_dim chunks ----
        simdgroup_float8x8 sacc(0.f);
        for (uint kd = 0u; kd < hd; kd += 8u) {
            for (uint i = lane; i < 8u * 8u; i += 32u) {
                uint key = i / 8u, d = i % 8u;
                uint t = t0 + key;
                ks[key][d] = (t < T)
                    ? half(arena_load(base + (ulong)t * nkv * hd * 2u + kvh * hd, kd + d))
                    : half(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_half8x8 a, b;
            simdgroup_load(a, &qs[0][kd], HD_MAX);
            simdgroup_load(b, &ks[0][0], 8, ulong2(0, 0), true);  // -> b[d][key]
            simdgroup_multiply_accumulate(sacc, a, b, sacc);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        simdgroup_store(sacc, &st[0][0], 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- online softmax over this 8-key tile ----
        if (lane < MT) {
            float tmax = -INFINITY;
            for (uint t = 0u; t < 8u; ++t) {
                if (t0 + t < T) {
                    tmax = max(tmax, st[lane][t]);
                }
            }
            float mnew = max(mrow[lane], tmax);
            float c = isinf(mrow[lane]) ? 0.f : exp(mrow[lane] - mnew);
            corr[lane] = c;
            float lsum = 0.f;
            for (uint t = 0u; t < 8u; ++t) {
                float p = (t0 + t < T) ? exp(st[lane][t] - mnew) : 0.f;
                ph[lane][t] = half(p);
                lsum += p;
            }
            lrow[lane] = lrow[lane] * c + lsum;
            mrow[lane] = mnew;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- O = O*corr + P . V over head_dim chunks ----
        for (uint kd = 0u; kd < hd; kd += 8u) {
            for (uint i = lane; i < 8u * 8u; i += 32u) {
                uint key = i / 8u, d = i % 8u;
                uint t = t0 + key;
                vs[key][d] = (t < T)
                    ? half(arena_load(
                          base + (ulong)t * nkv * hd * 2u + (ulong)nkv * hd + kvh * hd, kd + d))
                    : half(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_float8x8 pvacc(0.f);
            simdgroup_half8x8 a, b;
            simdgroup_load(a, &ph[0][0], 8);   // P[row][key]
            simdgroup_load(b, &vs[0][0], 8);   // V[key][d]
            simdgroup_multiply_accumulate(pvacc, a, b, pvacc);
            simdgroup_store(pvacc, &pvt[0][0], 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lane; i < 8u * 8u; i += 32u) {
                uint r = i / 8u, d = i % 8u;
                ot[r][kd + d] = ot[r][kd + d] * corr[r] + pvt[r][d];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (lane < MT) {
        dgq_assert_positive_f32(dbg, DbgKernelAttention, lrow[lane], (tok0 << 16u) | qh);
    }
    for (uint i = lane; i < MT * hd; i += 32u) {
        uint r = i / hd, d = i % hd;
        uint tok = tok0 + r;
        if (tok < dims.canvas) {
            float l = lrow[r];
            float y = (l > 0.f) ? ot[r][d] / l : 0.f;
            dgq_assert_finite_f32(dbg, DbgKernelAttention, y, d);
            arena_store(out + (ulong)tok * dims.n_q_heads * hd + qh * hd, d, y);
        }
    }
}
