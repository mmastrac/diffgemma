#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#include "fc_axes.metal"
#include "common.metal"
#include "arena.metal"

// E18: FUSED FLASH attention for FULL-layer PREFILL (device-load rewrite).
//
// E17 (attention_gemm) materializes S=Q.K^T and P to device (~4 canvas x T
// round-trips/head). Flash keeps the O accumulator resident and streams the
// keys, combining each block into the running (row-max m, denom l, O) with the
// online softmax rescale -- S/P never touch device.
//
// The v1 staged flash was 5.8x SLOWER than E17 at hd=512: staging K/V to tgmem
// in hd-chunks / key-sub-blocks (forced by the 32 KiB limit) exploded the
// barrier count (~80k/head-tile @100k). This rewrite loads K and V 8x8 tiles
// DIRECTLY from device via simdgroup_load in the MMA -- no tgmem staging, no
// staging barriers. Only Q is staged (once; it is bf16 and needs conversion) and
// the small S/P planes. f16 KV only (production prefill main path); the f32-side
// ring can't simdgroup_load-convert and would blow tgmem at hd=512.
//
// Layout: 256 threads = 8 simdgroups; BQ=16 query rows/threadgroup; O split
// across simdgroups by head-dim (ND = hd/8 = 64). QK: simdgroup sg computes S
// columns [sg*KPS, +KPS) via device-load K transposed (Q.K^T). PV: simdgroup sg
// accumulates O[:, sg*ND..] via device-load V untransposed. The KV buffer must
// be padded +BK rows so tile-loads past T stay in bounds (masked cols are
// ignored by softmax / zeroed in P).
// Grid: (ceil(M/BQ) row tiles, 1, HC heads); head_base like E17a.

#ifndef FL_BQ
#define FL_BQ 16
#endif
#ifndef FL_BK
#define FL_BK 64
#endif
// Compile-time head_dim (must equal d.hd at dispatch): sizes the resident Q
// stage and O accumulator. Lower hd relaxes the O register budget → larger BQ.
#ifndef FL_HD
#define FL_HD 512
#endif

constant uint BQ = FL_BQ;          // query rows per threadgroup
constant uint BK = FL_BK;          // keys streamed per block
constant uint NSG = 8u;            // simdgroups (256 threads)
constant uint TPG = NSG * 32u;
constant uint QPAD = 8u;           // bank pad on the hd axis of Qs

struct FlashDims {
    uint m;             // query rows this dispatch (active_canvas, <= 256)
    uint t_total;       // T (total keys)
    uint hd;            // head_dim (512 full / 256 sliding)
    uint a_row_stride;  // Q arena row stride (n_q_heads*hd)
    uint b_row_stride;  // native KV row stride (nkv*hd*2)
    uint out_row_stride;// attno arena row stride (n_q_heads*hd)
    uint kv_len;        // causal cutoff base: valid t <= kv_len + row
    uint group;         // GQA group (n_q_heads / nkv): kvh = qh / group
    uint nkv;           // KV heads: V base = (nkv + kvh) * hd
    uint head_base;     // global Q head = head_base + tgid.z
    uint causal;        // 1 = causal prefill
};

kernel void attn_flash(
    device const ushort *q [[buffer(0)]],     // Q plane [canvas][n_q_heads][hd]
    device const half *kcache [[buffer(1)]],  // kv16 base (K then V regions)
    device ushort *out [[buffer(2)]],         // attno arena [canvas][n_q_heads][hd]
    constant FlashDims &d [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint hd = d.hd;
    const uint ND = hd / NSG;              // O cols this simdgroup owns
    const uint TM = BQ / 8u;               // O fragment rows
    const uint TN = ND / 8u;               // O fragment cols (this simdgroup)
    const uint KPS = BK / NSG;             // key cols per simdgroup (QK)
    const uint TK = KPS / 8u;              // key fragments per simdgroup
    const uint m0 = tgid.x * BQ;           // first query row
    const uint qh = d.head_base + tgid.z;  // global data head
    const uint kvh = qh / d.group;
    const uint tid = lid.x;
    const uint bstride = d.b_row_stride;

    threadgroup half Qs[FL_BQ * (FL_HD + 8u)];   // Q staged once
    threadgroup float Sf[FL_BQ * FL_BK];        // scores plane
    threadgroup half Ph[FL_BQ * FL_BK];         // probs (half) for PV
    threadgroup float rm[FL_BQ];                // running row max
    threadgroup float rl[FL_BQ];                // running row denom
    threadgroup float bmax[FL_BQ];             // block max / corr scratch

    const uint qpad_hd = hd + QPAD;

    for (uint i = tid; i < BQ * hd; i += TPG) {
        const uint row = i / hd;
        const uint kk = i % hd;
        half v = 0.h;
        if (m0 + row < d.m) {
            const ushort u = q[(ulong)(m0 + row) * d.a_row_stride + (ulong)qh * hd + kk];
            v = K_ARENA_F16 ? as_type<half>(u) : half(as_type<float>(uint(u) << 16u));
        }
        Qs[row * qpad_hd + kk] = v;
    }
    for (uint r = tid; r < BQ; r += TPG) {
        rm[r] = -INFINITY;
        rl[r] = 0.f;
    }
    simdgroup_float8x8 O[FL_BQ / 8u][(FL_HD / 8u) / 8u];   // [TM][TN]
    for (uint i = 0u; i < TM; ++i) {
        for (uint j = 0u; j < TN; ++j) {
            O[i][j] = simdgroup_float8x8(0.f);
        }
    }
    uint fm, fn;
    {
        const uint qid = lane / 4u;
        fm = (qid & 4u) + ((lane / 2u) % 4u);
        fn = (qid & 2u) * 2u + (lane % 2u) * 2u;
    }

    const uint t_stop = d.causal ? min(d.kv_len + (m0 + BQ - 1u), d.t_total - 1u)
                                 : d.t_total - 1u;
    device const half *kb = kcache + (ulong)kvh * hd;               // K base
    device const half *vb = kcache + (ulong)(d.nkv + kvh) * hd;     // V base

    for (uint kt = 0u; kt <= t_stop; kt += BK) {
        const uint bk = min(BK, t_stop + 1u - kt);

        // (1) QK: simdgroup sg -> S[BQ][KPS] at cols [sg*KPS,+KPS). Device-load
        //     K transposed (kf[hd][key]); contract hd 8 at a time. No staging.
        simdgroup_float8x8 Cs[FL_BQ / 8u][FL_BK / 64u];  // [TM][TK]
        for (uint i = 0u; i < TM; ++i) {
            for (uint j = 0u; j < TK; ++j) {
                Cs[i][j] = simdgroup_float8x8(0.f);
            }
        }
        for (uint kk = 0u; kk < hd; kk += 8u) {
            simdgroup_half8x8 qf[FL_BQ / 8u];
            for (uint i = 0u; i < TM; ++i) {
                simdgroup_load(qf[i], &Qs[(8u * i) * qpad_hd + kk], qpad_hd);
            }
            for (uint j = 0u; j < TK; ++j) {
                const uint key0 = kt + sgid * KPS + 8u * j;   // >= t_total safe (pad)
                simdgroup_half8x8 kf;
                simdgroup_load(kf, kb + (ulong)key0 * bstride + kk, bstride,
                               ulong2(0, 0), /*transpose=*/true);
                for (uint i = 0u; i < TM; ++i) {
                    simdgroup_multiply_accumulate(Cs[i][j], qf[i], kf, Cs[i][j]);
                }
            }
        }
        for (uint i = 0u; i < TM; ++i) {
            for (uint j = 0u; j < TK; ++j) {
                simdgroup_store(Cs[i][j], &Sf[(8u * i) * BK + sgid * KPS + 8u * j], BK);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // (2) online softmax (one thread per row).
        for (uint r = tid; r < BQ; r += TPG) {
            const uint grow = m0 + r;
            float mb = -INFINITY;
            for (uint c = 0u; c < bk; ++c) {
                if (!d.causal || kt + c <= d.kv_len + grow) {
                    if (grow < d.m) {
                        mb = max(mb, Sf[r * BK + c]);
                    }
                }
            }
            bmax[r] = mb;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = tid; r < BQ; r += TPG) {
            const uint grow = m0 + r;
            const float mold = rm[r];
            const float mnew = max(mold, bmax[r]);
            float lsum = 0.f;
            for (uint c = 0u; c < BK; ++c) {
                float p = 0.f;
                if (c < bk && (!d.causal || kt + c <= d.kv_len + grow)
                    && grow < d.m && isfinite(mnew)) {
                    p = exp(Sf[r * BK + c] - mnew);
                    lsum += p;
                }
                Ph[r * BK + c] = half(p);
            }
            const float corr = isfinite(mold) ? exp(mold - mnew) : 0.f;
            rl[r] = rl[r] * corr + lsum;
            rm[r] = mnew;
            bmax[r] = corr;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // (3) rescale O by per-row corr.
        for (uint i = 0u; i < TM; ++i) {
            for (uint j = 0u; j < TN; ++j) {
                const float corr = bmax[8u * i + fm];
                thread float2 &ec = reinterpret_cast<thread float2 &>(O[i][j].thread_elements());
                ec[0] *= corr;
                ec[1] *= corr;
            }
        }

        // (4) PV: O[:, sg*ND..] += P . V. Device-load V untransposed (vf[key][hd])
        //     in 8-key tiles; A = P from Ph (tgmem). No staging, no barriers.
        for (uint ks = 0u; ks < bk; ks += 8u) {
            simdgroup_half8x8 pf[FL_BQ / 8u];
            for (uint i = 0u; i < TM; ++i) {
                simdgroup_load(pf[i], &Ph[(8u * i) * BK + ks], BK);
            }
            for (uint j = 0u; j < TN; ++j) {
                simdgroup_half8x8 vf;
                simdgroup_load(vf, vb + (ulong)(kt + ks) * bstride + sgid * ND + 8u * j,
                               bstride);
                for (uint i = 0u; i < TM; ++i) {
                    simdgroup_multiply_accumulate(O[i][j], pf[i], vf, O[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- store O / l ----
    for (uint i = 0u; i < TM; ++i) {
        const uint row = 8u * i + fm;
        if (m0 + row >= d.m) {
            continue;
        }
        const float l = rl[row];
        const float inv = (l > 0.f) ? 1.f / l : 0.f;
        for (uint j = 0u; j < TN; ++j) {
            const uint col = sgid * ND + 8u * j + fn;
            const thread float2 &ec = reinterpret_cast<thread float2 &>(O[i][j].thread_elements());
            arena_store(out, (ulong)(m0 + row) * d.out_row_stride + (ulong)qh * hd + col,
                        ec[0] * inv);
            arena_store(out, (ulong)(m0 + row) * d.out_row_stride + (ulong)qh * hd + col + 1u,
                        ec[1] * inv);
        }
    }
}
