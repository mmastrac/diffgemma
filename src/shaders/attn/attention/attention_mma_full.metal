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
// One threadgroup per (8-query tile, KV head, Q head); the two simdgroups
// SPLIT head_dim (each owns a 256-wide half). vs the previous
// two-heads-per-tg layout this halves the per-lane O register footprint
// (oreg[64], the profile that made attention_mma2 4-7x scalar) and drops
// threadgroup memory to ~10 KiB, so several threadgroups stay resident per
// core and the per-chunk staging latencies — the term that grows with
// kv_len — overlap across threadgroups. Q/K/V are still read from device
// once per (tile, head): the halves partition d, they don't duplicate it.
// The QK dot needs a cross-simdgroup partial-sum per key-tile (st[0]+st[1]);
// PV/O run entirely simdgroup-local.
//
// hd is COMPILE-TIME (HD=512) so the head-dim chunk loops unroll and oreg
// stays in registers. nkv / n_q_heads stay runtime (KV addressing only).
// Full layers only — assert head_dim==512.

#ifndef MMA_FULL_HD
#define MMA_FULL_HD 512u
#endif
constant uint HD = MMA_FULL_HD; // full-layer head_dim (compile-time)
constant uint NCH = HD / 8u;    // 64 head-dim chunks of 8
constant uint MT = 8u;          // query rows per tile
constant uint QG = 2u;          // simdgroups per tg = d-halves (tg width 64)
constant uint NCH_H = NCH / QG; // chunks per d-half (32)

// Session-wide KV storage format: q8 (group-32) for long-context sessions,
// f16 otherwise. Unset (oracle/test compiles) = f16. The q8 path stages
// dequantized half tiles through tgmem (simdgroup_load can't read i8).
constant uint KV_FMT_FC [[function_constant(4)]];
constant uint KV_FMT = is_function_constant_defined(KV_FMT_FC) ? KV_FMT_FC : 0u;
constant bool KV_Q8 = (KV_FMT == 1u);
constant bool KV_Q4 = (KV_FMT == 2u);

// Prefill variant: K/V come from the f32 side cache (buffer 9, bound at
// this layer's side offset; full layers are LINEAR — index by absolute t0).
// Q is staged f32 and the MMA runs all-float. See attention_mma2.metal.
constant bool KV_F32_SIDE_FC [[function_constant(30)]];
constant bool KV_F32_SIDE =
    is_function_constant_defined(KV_F32_SIDE_FC) && KV_F32_SIDE_FC;

// QK-ILP2 chain-split (FC31, opt-in `DGQ_ATTN_MMA_FULL_QK_ILP2`): split the
// 32-deep serial QK dot into two interleaved 16-deep accumulator chains (even/
// odd chunks), summed at the end. The two chains have no data dependency on
// each other, so the GPU issues both MMAs concurrently — halving the
// serial-dependency depth of the QK stage. PV is already per-chunk
// independent (no change). Non-bit-identical (different FP-associativity);
// quality-gated. Default OFF.
constant bool QK_ILP2_FC [[function_constant(31)]];
constant bool QK_ILP2 =
    is_function_constant_defined(QK_ILP2_FC) && QK_ILP2_FC;

kernel void attention_mma_full(
    device const ushort *q [[buffer(0)]],
    device const ushort *kvcache [[buffer(1)]],
    device ushort *out [[buffer(2)]],
    device const LayerOffsets *L [[buffer(3)]],
    constant StepParams &P [[buffer(4)]],
    constant AttnDims &dims [[buffer(5)]],
    device DebugStatus *dbg [[buffer(6)]],
    constant AttnBlockRange &blk [[buffer(7)]],
    device const float *kv_f32_side [[buffer(9), function_constant(KV_F32_SIDE_FC)]],
    // Per-(head, row) online-softmax state across kv-block dispatches:
    // layout [qh][tok] -> {m, l} in the first 2*NQ*CANVAS floats, then
    // [qh][tok][d] unnormalized O. f32 round-trip = bit-identical.
    device float *attn_state [[buffer(8)]],
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
    const uint sub = tgid.z;                    // Q head within this kvh's group
    const uint tok0 = tgid.x * MT;
    if (kvh >= nkv || tok0 >= dims.canvas) {
        return;
    }
    const uint sg = lid.x / 32u;                // simdgroup index -> d-half
    const uint lane = lid.x % 32u;
    const uint nlanes = QG * 32u;
    const uint dlo = sg * (HD / QG);            // this simdgroup's d-half base
    const uint qh = kvh * group + sub;          // global Q head for this tg
    const uint T = P.kv_len + dims.canvas;
    // Causal (prefill): row r is at abs pos kv_len+tok0+r, attends only [0..pos].
    const bool causal = dims.causal != 0u;
    const uint T_eff = causal ? min(T, P.kv_len + tok0 + MT) : T;

    // Middle block with no keys for this query tile (causal cutoff before the
    // block): state passes through untouched, nothing to do. The LAST block
    // must still run to normalize + write `out`.
    if (blk.is_first == 0u && blk.is_last == 0u && blk.t_begin >= T_eff) {
        return;
    }
    // State planes: {m, l} per (head, row), then unnormalized O.
    const uint ml_base = (qh * dims.canvas + tok0) * 2u;
    device float *st_ml = attn_state + ml_base;
    device float *st_o = attn_state
        + (ulong)dims.n_q_heads * dims.canvas * 2u
        + ((ulong)qh * dims.canvas + tok0) * HD;
    // KV cache is f16: K/V tiles are simdgroup_load'ed STRAIGHT from device
    // memory (row stride = one key), no bf16->half staging pass at all. Tail
    // keys of a ragged tile read in-bounds pad/stale slots (finite; the layer
    // region is over-allocated by 8 rows) and are zeroed by the softmax mask.
    device const half *kv16 = (device const half *)(kvcache + L->kv_region / 2);
    const ulong kstride = (ulong)nkv * hd * 2u;  // elements between key rows

    // Q tile, shared by both d-halves (staged once).
    threadgroup half qs[MT][HD];        // 8 KiB
    // f32-side path only: float Q stage + float probs (dead-stripped from
    // the f16/q8 pipelines, like kq8 below).
    threadgroup float qsf[MT][HD];
    threadgroup float phf[MT][8];
    // q8 path only: dequantized K (then V) tile, blocked 8x8 (8 KiB;
    // dead-stripped from f16 pipelines).
    threadgroup half kq8[NCH][8][8];
    threadgroup float st[QG][MT][8];    // per-half partial QK scores
    // ILP2 only: second QK partial per half (even/odd chunk split).
    // Dead-stripped from the non-ILP2 pipeline by QK_ILP2.
    threadgroup float st_ilp[QG][MT][8];
    threadgroup half ph[MT][8];         // softmax probs (shared)
    threadgroup float pvt[QG][MT][8];   // P.V chunk per half
    threadgroup float mrow[MT];         // running max per row (shared)
    threadgroup float lrow[MT];         // running denom per row (shared)
    threadgroup float corr[MT];         // rescale per row for this key-tile

    // Register-resident O accumulator for THIS d-half: lane owns rows {r0, r1}
    // at column dcol of each 8-wide chunk in [dlo, dlo+256).
    const uint dcol = lane % 8u;
    const uint r0 = lane / 8u;          // 0..3
    const uint r1 = r0 + 4u;            // 4..7
    float oreg[2u * NCH_H];
    if (blk.is_first != 0u) {
        for (uint j = 0u; j < 2u * NCH_H; ++j) {
            oreg[j] = 0.f;
        }
    } else {
        // Resume: reload this lane's O slice from the state buffer (f32
        // round-trip, exact).
        for (uint c = 0u; c < NCH_H; ++c) {
            uint d = dlo + c * 8u + dcol;
            oreg[2u * c] = st_o[(ulong)r0 * HD + d];
            oreg[2u * c + 1u] = st_o[(ulong)r1 * HD + d];
        }
    }

    // Stage Q[MT x HD] (all lanes cooperate).
    for (uint i = lid.x; i < MT * HD; i += nlanes) {
        uint r = i / HD, d = i % HD;
        uint tok = tok0 + r;
        const float qv = (tok < dims.canvas)
            ? arena_load(q + (ulong)tok * dims.n_q_heads * hd + qh * hd, d)
            : 0.f;
        if (KV_F32_SIDE) {
            qsf[r][d] = qv;
        } else {
            qs[r][d] = half(qv);
        }
    }
    if (lid.x < MT) {
        if (blk.is_first != 0u) {
            mrow[lid.x] = -INFINITY;
            lrow[lid.x] = 0.f;
        } else {
            mrow[lid.x] = st_ml[lid.x * 2u];
            lrow[lid.x] = st_ml[lid.x * 2u + 1u];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const uchar *kvb = (device const uchar *)kvcache + L->kv_region;
    const ulong q8_stride = kv_slot_stride_bytes(nkv, hd, KV_FMT);
    const ulong q8_row = kv_row_bytes(hd, KV_FMT);

    const uint t_stop = min(T_eff, blk.t_end);
    for (uint t0 = blk.t_begin; t0 < t_stop; t0 += 8u) {
        // ---- partial S = Q . K^T over THIS half's chunks ----
        device const half *kb = kv16 + (ulong)t0 * kstride + kvh * hd;
        device const float *kbf =
            KV_F32_SIDE ? (kv_f32_side + (ulong)t0 * kstride + kvh * hd)
                        : (device const float *)nullptr;
        if (!KV_F32_SIDE && KV_Q8) {
            // Dequantize the full K tile into tgmem: one (key, 32-group) per
            // work item, vectorized (2 items/lane at HD=512).
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lid.x; i < 8u * (HD / 32u); i += nlanes) {
                const uint key = i >> 4u;      // / (HD/32 = 16)
                const uint g = i & 15u;
                device const uchar *row =
                    kvb + (ulong)(t0 + key) * q8_stride + kvh * q8_row;
                kv_q8_dequant_group_blocked(row, g, HD, key, &kq8[0][0][0]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        simdgroup_float8x8 sacc(0.f);
        if (QK_ILP2) {
            // ILP2: two independent 16-deep chains (even/odd chunks), no data
            // dependency between them -> the GPU issues both MMAs concurrently,
            // halving the QK serial-dependency depth. Store each chain to its
            // own tgmem slot; the cross-half softmax sums all four partials.
            simdgroup_float8x8 sacc0(0.f), sacc1(0.f);
            for (uint c = 0u; c < NCH_H; c += 2u) {
                uint kd0 = dlo + c * 8u;
                uint kd1 = dlo + (c + 1u) * 8u;
                if (KV_F32_SIDE) {
                    simdgroup_float8x8 a0, b0, a1, b1;
                    simdgroup_load(a0, &qsf[0][kd0], HD);
                    simdgroup_load(b0, kbf + kd0, kstride, ulong2(0, 0), true);
                    simdgroup_load(a1, &qsf[0][kd1], HD);
                    simdgroup_load(b1, kbf + kd1, kstride, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(sacc0, a0, b0, sacc0);
                    simdgroup_multiply_accumulate(sacc1, a1, b1, sacc1);
                } else {
                    simdgroup_half8x8 a0, b0, a1, b1;
                    simdgroup_load(a0, &qs[0][kd0], HD);
                    simdgroup_load(a1, &qs[0][kd1], HD);
                    if (KV_Q8) {
                        simdgroup_load(b0, &kq8[kd0 / 8u][0][0], 8, ulong2(0, 0), true);
                        simdgroup_load(b1, &kq8[kd1 / 8u][0][0], 8, ulong2(0, 0), true);
                    } else {
                        simdgroup_load(b0, kb + kd0, kstride, ulong2(0, 0), true);
                        simdgroup_load(b1, kb + kd1, kstride, ulong2(0, 0), true);
                    }
                    simdgroup_multiply_accumulate(sacc0, a0, b0, sacc0);
                    simdgroup_multiply_accumulate(sacc1, a1, b1, sacc1);
                }
            }
            simdgroup_store(sacc0, &st[sg][0][0], 8);
            simdgroup_store(sacc1, &st_ilp[sg][0][0], 8);
        } else {
            for (uint c = 0u; c < NCH_H; ++c) {
                uint kd = dlo + c * 8u;
                if (KV_F32_SIDE) {
                    simdgroup_float8x8 af, bf;
                    simdgroup_load(af, &qsf[0][kd], HD);
                    simdgroup_load(bf, kbf + kd, kstride, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(sacc, af, bf, sacc);
                    continue;
                }
                simdgroup_half8x8 a, b;
                simdgroup_load(a, &qs[0][kd], HD);
                if (KV_Q8) {
                    simdgroup_load(b, &kq8[kd / 8u][0][0], 8, ulong2(0, 0), true);
                } else {
                    simdgroup_load(b, kb + kd, kstride, ulong2(0, 0), true);  // -> b[d][key]
                }
                simdgroup_multiply_accumulate(sacc, a, b, sacc);
            }
            simdgroup_store(sacc, &st[sg][0][0], 8);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- cross-half sum + online softmax over this 8-key tile (sg0) ----
        if (lid.x < MT) {
            const uint row = lid.x;
            const uint qpos = P.kv_len + tok0 + row;  // causal cutoff for this row
            float s[8];
            float tmax = -INFINITY;
            for (uint t = 0u; t < 8u; ++t) {
                bool valid = (t0 + t < T) && (!causal || t0 + t <= qpos);
                if (QK_ILP2) {
                    s[t] = st[0][row][t] + st_ilp[0][row][t]
                         + st[1][row][t] + st_ilp[1][row][t];
                } else {
                    s[t] = st[0][row][t] + st[1][row][t];
                }
                if (valid) {
                    tmax = max(tmax, s[t]);
                }
            }
            float mnew = max(mrow[row], tmax);
            float cc = isinf(mrow[row]) ? 0.f : exp(mrow[row] - mnew);
            corr[row] = cc;
            float lsum = 0.f;
            for (uint t = 0u; t < 8u; ++t) {
                bool valid = (t0 + t < T) && (!causal || t0 + t <= qpos);
                float p = valid ? exp(s[t] - mnew) : 0.f;
                if (KV_F32_SIDE) {
                    phf[row][t] = p;
                } else {
                    ph[row][t] = half(p);
                }
                lsum += p;
            }
            lrow[row] = lrow[row] * cc + lsum;
            mrow[row] = mnew;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float cr0 = corr[r0];
        const float cr1 = corr[r1];

        // ---- O = O*corr + P . V over THIS half's chunks ----
        device const half *vb = kv16 + (ulong)t0 * kstride + (ulong)nkv * hd + kvh * hd;
        device const float *vbf = KV_F32_SIDE
            ? (kv_f32_side + (ulong)t0 * kstride + (ulong)nkv * hd + kvh * hd)
            : (device const float *)nullptr;
        if (!KV_F32_SIDE && KV_Q8) {
            // Both halves are done with K; reuse kq8 for the V tile.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lid.x; i < 8u * (HD / 32u); i += nlanes) {
                const uint key = i >> 4u;
                const uint g = i & 15u;
                device const uchar *row = kvb + (ulong)(t0 + key) * q8_stride
                    + ((ulong)nkv + kvh) * q8_row;
                kv_q8_dequant_group_blocked(row, g, HD, key, &kq8[0][0][0]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint c = 0u; c < NCH_H; ++c) {
            uint kd = dlo + c * 8u;
            simdgroup_float8x8 pvacc(0.f);
            if (KV_F32_SIDE) {
                simdgroup_float8x8 af, bf;
                simdgroup_load(af, &phf[0][0], 8);
                simdgroup_load(bf, vbf + kd, kstride);
                simdgroup_multiply_accumulate(pvacc, af, bf, pvacc);
            } else {
                simdgroup_half8x8 a, b;
                simdgroup_load(a, &ph[0][0], 8);       // P[row][key]
                if (KV_Q8) {
                    simdgroup_load(b, &kq8[kd / 8u][0][0], 8);
                } else {
                    simdgroup_load(b, vb + kd, kstride);   // V[key][d]
                }
                simdgroup_multiply_accumulate(pvacc, a, b, pvacc);
            }
            simdgroup_store(pvacc, &pvt[sg][0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            oreg[2u * c] = oreg[2u * c] * cr0 + pvt[sg][r0][dcol];
            oreg[2u * c + 1u] = oreg[2u * c + 1u] * cr1 + pvt[sg][r1][dcol];
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        // ph/corr are rewritten only after the next tile's tg barrier.
    }

    if (blk.is_last == 0u) {
        // Persist state for the next kv-block dispatch (f32, exact).
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid.x < MT) {
            st_ml[lid.x * 2u] = mrow[lid.x];
            st_ml[lid.x * 2u + 1u] = lrow[lid.x];
        }
        for (uint c = 0u; c < NCH_H; ++c) {
            uint d = dlo + c * 8u + dcol;
            st_o[(ulong)r0 * HD + d] = oreg[2u * c];
            st_o[(ulong)r1 * HD + d] = oreg[2u * c + 1u];
        }
        return;
    }

    if (lid.x < MT) {
        dgq_assert_positive_f32(dbg, DbgKernelAttention, lrow[lid.x], (tok0 << 16u) | qh);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Store O (register accumulator / denom) for rows r0, r1, this d-half.
    const float l0 = lrow[r0];
    const float l1 = lrow[r1];
    const uint t_r0 = tok0 + r0;
    const uint t_r1 = tok0 + r1;
    for (uint c = 0u; c < NCH_H; ++c) {
        uint d = dlo + c * 8u + dcol;
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
