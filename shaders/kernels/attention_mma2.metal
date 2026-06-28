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
// Only valid for group size 2 and hd <= 256 (sliding layers): O for two heads is
// 2*8*256*4 = 16 KiB; at hd=512 it would be 32 KiB and blow the threadgroup limit.
// Full-attention layers keep the scalar/1-head path. Same semantics as `attention`
// (online softmax, all-valid, no 1/sqrt(d) scale); `attention` stays the oracle.

constant uint HD_MAX = 256u;
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
    device const ushort *base = kvcache + L->kv_region / 2;

    threadgroup half qs[QG][MT][HD_MAX];   // staged Q per head
    threadgroup float ot[QG][MT][HD_MAX];  // running O per head
    threadgroup half ks[MT][8];            // shared K chunk [key][d]
    threadgroup half vs[MT][8];            // shared V chunk [key][d]
    threadgroup half ph[QG][MT][8];        // softmax probs per head
    threadgroup float st[QG][MT][8];       // QK scores per head
    threadgroup float pvt[QG][MT][8];      // P·V chunk per head
    threadgroup float mrow[QG][MT];
    threadgroup float lrow[QG][MT];
    threadgroup float corr[QG][MT];

    // Stage Q for both heads (all 64 lanes) and zero O.
    for (uint i = lid.x; i < QG * MT * hd; i += 64u) {
        uint h = i / (MT * hd);
        uint rem = i % (MT * hd);
        uint r = rem / hd, d = rem % hd;
        uint tok = tok0 + r;
        uint qhh = kvh * QG + h;
        qs[h][r][d] = (tok < dims.canvas)
            ? half(arena_load(q + (ulong)tok * dims.n_q_heads * hd + qhh * hd, d))
            : half(0);
        ot[h][r][d] = 0.f;
    }
    if (lid.x < QG * MT) {
        uint h = lid.x / MT, r = lid.x % MT;
        mrow[h][r] = -INFINITY;
        lrow[h][r] = 0.f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t0 = 0u; t0 < T; t0 += 8u) {
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
            float tmax = -INFINITY;
            for (uint t = 0u; t < 8u; ++t) {
                if (t0 + t < T) {
                    tmax = max(tmax, st[sg][lane][t]);
                }
            }
            float mnew = max(mrow[sg][lane], tmax);
            float c = isinf(mrow[sg][lane]) ? 0.f : exp(mrow[sg][lane] - mnew);
            corr[sg][lane] = c;
            float lsum = 0.f;
            for (uint t = 0u; t < 8u; ++t) {
                float p = (t0 + t < T) ? exp(st[sg][lane][t] - mnew) : 0.f;
                ph[sg][lane][t] = half(p);
                lsum += p;
            }
            lrow[sg][lane] = lrow[sg][lane] * c + lsum;
            mrow[sg][lane] = mnew;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- O = O*corr + P . V over head_dim chunks (V shared by both heads) ----
        for (uint kd = 0u; kd < hd; kd += 8u) {
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
            for (uint i = lane; i < MT * 8u; i += 32u) {
                uint r = i / 8u, d = i % 8u;
                ot[sg][r][kd + d] = ot[sg][r][kd + d] * corr[sg][r] + pvt[sg][r][d];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (lane < MT) {
        dgq_assert_positive_f32(dbg, DbgKernelAttention, lrow[sg][lane], (tok0 << 16u) | qh);
    }
    for (uint i = lane; i < MT * hd; i += 32u) {
        uint r = i / hd, d = i % hd;
        uint tok = tok0 + r;
        if (tok < dims.canvas) {
            float l = lrow[sg][r];
            float y = (l > 0.f) ? ot[sg][r][d] / l : 0.f;
            dgq_assert_finite_f32(dbg, DbgKernelAttention, y, d);
            arena_store(out + (ulong)tok * dims.n_q_heads * hd + qh * hd, d, y);
        }
    }
}
