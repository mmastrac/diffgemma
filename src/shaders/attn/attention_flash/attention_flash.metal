#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#include "fc_axes.metal"
#include "common.metal"
#include "arena.metal"

// E18: FUSED FLASH attention for FULL-layer PREFILL.
//
// E17 (attention_gemm) decomposes attention into S=Q.K^T -> softmax -> P.V with
// S and P MATERIALIZED to device memory (~4 canvas x T plane round-trips/head).
// Flash keeps the O accumulator RESIDENT and streams the keys in BK-wide blocks,
// combining each block into the running (row-max m, denom l, O) with the online
// softmax rescale -- S/P never touch device. This is the steel-SDPA dataflow;
// the numerics match E17's cpu_causal exactly (validated: cpu_flash_blocked).
//
// hd=512 makes this hard: the O accumulator is BQ x hd, and staging Q/K/V + S/P
// must fit in threadgroup memory. Layout (v1, correctness-first, tunable BQ/BK):
//   - 256 threads = NSG=8 simdgroups; BQ query rows/threadgroup (default 16).
//   - O split across simdgroups by head-dim: simdgroup sg owns O[:, sg*ND ..],
//     ND = hd/NSG (64 for hd=512), held in C[TM][TN] fragments (TM=BQ/8, TN=ND/8).
//   - Qs[BQ][hd] staged once in tgmem, reused every key block.
//   - Per block: (1) S[BQ][BK] = Q.K^T (keys split across simdgroups, hd-chunked
//     contraction) into tgmem; (2) cooperative masked row-max/exp/sum -> P[BQ][BK]
//     in tgmem + block (m,l) update; (3) rescale O fragments by exp(m_old-m_new)
//     per row; (4) O += P.V (each simdgroup its ND slice, V staged transposed).
//   - At the end O is divided by l and stored to the attno arena.
// Grid: (ceil(M/BQ) row tiles, 1, HC heads); head_base like E17a.

#ifndef FL_BQ
#define FL_BQ 16
#endif
#ifndef FL_BK
#define FL_BK 64
#endif

constant uint BQ = FL_BQ;          // query rows per threadgroup
constant uint BK = FL_BK;          // keys streamed per block
constant uint NSG = 8u;            // simdgroups (256 threads)
constant uint TPG = NSG * 32u;
constant uint DC = 32u;            // hd contraction chunk for QK
constant uint QPAD = 8u;           // bank pad on the hd axis of Qs

// f32-side ring (FC30) vs f16 main cache -- same axis E17 uses.
constant bool KV_F32_SIDE_FC [[function_constant(30)]];
constant bool KV_F32_SIDE =
    is_function_constant_defined(KV_F32_SIDE_FC) && KV_F32_SIDE_FC;

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

// Per-simdgroup 8x8-fragment NT MMA: C[i][j] += sum_k As[sr+8i+.. ][k]*Bs[8j+..][k].
// One simdgroup owns the whole C[TM][TN] tile (no 2x2 split; flash slices the
// output across simdgroups by head-dim instead). half staging.
template <uint TM, uint TN>
static inline void flash_mma_h(
    threadgroup const half *As, uint a_stride,   // As[row][k], row-major
    threadgroup const half *Bs, uint b_stride,   // Bs[n][k], row-major (B transposed)
    thread simdgroup_float8x8 (&C)[TM][TN],
    uint kdim, uint fm, uint fn
) {
    for (uint k0 = 0u; k0 < kdim; k0 += 8u) {
        simdgroup_barrier(mem_flags::mem_none);
        simdgroup_half8x8 A[TM];
        for (uint i = 0u; i < TM; ++i) {
            thread half2 &ea = reinterpret_cast<thread half2 &>(A[i].thread_elements());
            ea[0] = As[(8u * i + fm) * a_stride + k0 + fn];
            ea[1] = As[(8u * i + fm) * a_stride + k0 + fn + 1u];
        }
        simdgroup_barrier(mem_flags::mem_none);
        simdgroup_half8x8 B[TN];
        for (uint j = 0u; j < TN; ++j) {
            thread half2 &eb = reinterpret_cast<thread half2 &>(B[j].thread_elements());
            eb[0] = Bs[(8u * j + fn) * b_stride + k0 + fm];
            eb[1] = Bs[(8u * j + fn + 1u) * b_stride + k0 + fm];
        }
        simdgroup_barrier(mem_flags::mem_none);
        for (uint i = 0u; i < TM; ++i) {
            for (uint j = 0u; j < TN; ++j) {
                simdgroup_multiply_accumulate(C[i][j], A[i], B[j], C[i][j]);
            }
        }
    }
}

kernel void attn_flash(
    device const ushort *q [[buffer(0)]],     // Q plane [canvas][n_q_heads][hd]
    device const half *kcache [[buffer(1)]],  // kv16 base (K then V regions)
    device ushort *out [[buffer(2)]],         // attno arena [canvas][n_q_heads][hd]
    constant FlashDims &d [[buffer(3)]],
    device const float *kv_f32_side [[buffer(9), function_constant(KV_F32_SIDE_FC)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint hd = d.hd;
    const uint ND = hd / NSG;              // O cols this simdgroup owns
    const uint TM = BQ / 8u;               // O fragment rows
    const uint TN = ND / 8u;               // O fragment cols
    const uint m0 = tgid.x * BQ;           // first query row
    const uint qh = d.head_base + tgid.z;  // global data head
    const uint kvh = qh / d.group;
    const uint tid = lid.x;

    // ---- tgmem ----
    // Qs[BQ][hd(+pad)] half, staged once. Union K/V staging with S/P is not done
    // in v1 (kept separate for clarity; watch the 32 KiB budget when hd=512).
    threadgroup half Qs[FL_BQ * (512u + 8u)];   // sized for max hd=512
    threadgroup float Sf[FL_BQ * FL_BK];        // scores / probs plane
    threadgroup half Ph[FL_BQ * FL_BK];         // probs (half) for PV
    threadgroup float rm[FL_BQ];                // running row max
    threadgroup float rl[FL_BQ];                // running row denom
    threadgroup float bmax[FL_BQ];             // per-block row max scratch
    // K contraction chunk [BK][DC+pad] (QK phase) and V slice transposed
    // [ND][BK] (PV phase) never coexist -> one shared staging buffer. Sized for
    // max(BK*(DC+4)=64*36=2304, ND_max*BK=64*64=4096) halfs.
    threadgroup half stage[4096];

    const uint qpad_hd = hd + QPAD;

    // Stage Q for this (row-tile, head): Qs[row][kk].
    for (uint i = tid; i < BQ * hd; i += TPG) {
        const uint row = i / hd;
        const uint kk = i % hd;
        half v = 0.h;
        if (m0 + row < d.m) {
            const ushort u = q[(ulong)(m0 + row) * d.a_row_stride
                               + (ulong)qh * hd + kk];
            v = K_ARENA_F16 ? as_type<half>(u)
                            : half(as_type<float>(uint(u) << 16u));
        }
        Qs[row * qpad_hd + kk] = v;
    }
    // Init running stats + O fragments.
    for (uint r = tid; r < BQ; r += TPG) {
        rm[r] = -INFINITY;
        rl[r] = 0.f;
    }
    simdgroup_float8x8 O[FL_BQ / 8u][(512u / 8u) / 8u];  // [TM][TN], max sizes
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

    device const half *kb = kcache + (ulong)kvh * hd;                 // K base
    device const half *vb = kcache + (ulong)(d.nkv + kvh) * hd;       // V base
    device const float *kbf = KV_F32_SIDE ? kv_f32_side + (ulong)kvh * hd : nullptr;
    device const float *vbf =
        KV_F32_SIDE ? kv_f32_side + (ulong)(d.nkv + kvh) * hd : nullptr;

    // ---- stream key blocks ----
    for (uint kt = 0u; kt <= t_stop; kt += BK) {
        const uint bk = min(BK, t_stop + 1u - kt);   // valid keys this block

        // (1) S[BQ][BK] = Q . K^T. Split keys across simdgroups: simdgroup sg
        //     computes columns [sg*KPS, sg*KPS+KPS). Contract hd in DC chunks.
        const uint KPS = BK / NSG;                    // key cols per simdgroup
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // zero S for this block (only up to bk used, but zero full for masked max)
        for (uint i = tid; i < BQ * BK; i += TPG) {
            Sf[i] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Each simdgroup: S[BQ][KPS] fragments contract over hd. KPS = BK/NSG
        // (>=8, multiple of 8): TNs = KPS/8 fragments = FL_BK/64 (NSG=8).
        simdgroup_float8x8 Cs[FL_BQ / 8u][FL_BK / 64u];
        const uint TNs = KPS / 8u;
        for (uint i = 0u; i < TM; ++i) {
            for (uint j = 0u; j < TNs; ++j) {
                Cs[i][j] = simdgroup_float8x8(0.f);
            }
        }
        for (uint dc = 0u; dc < hd; dc += DC) {
            // stage K chunk [BK][DC] -> Ks (all simdgroups cooperate)
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < BK * DC; i += TPG) {
                const uint tk = i / DC;   // key within block
                const uint cc = i % DC;   // hd col within chunk
                half v = 0.h;
                if (tk < bk) {
                    const uint gt = kt + tk;
                    if (KV_F32_SIDE) {
                        v = half(kbf[(ulong)gt * d.b_row_stride + dc + cc]);
                    } else {
                        v = kb[(ulong)gt * d.b_row_stride + dc + cc];
                    }
                }
                stage[tk * (DC + 4u) + cc] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // MMA: Cs += Qs[:, dc:dc+DC] . Ks^T  (this simdgroup's KPS keys)
            flash_mma_h<FL_BQ / 8u, FL_BK / 64u>(
                &Qs[dc], qpad_hd,
                &stage[(sgid * KPS) * (DC + 4u)], DC + 4u,
                Cs, DC, fm, fn);
        }
        // write Cs -> Sf[row][sg*KPS + col]
        for (uint i = 0u; i < TM; ++i) {
            for (uint j = 0u; j < TNs; ++j) {
                const uint row = 8u * i + fm;
                const uint col = sgid * KPS + 8u * j + fn;
                const thread float2 &ec =
                    reinterpret_cast<thread float2 &>(Cs[i][j].thread_elements());
                Sf[row * BK + col] = ec[0];
                Sf[row * BK + col + 1u] = ec[1];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // (2) online softmax: per row, block max over valid+causal cols, then
        //     m_new, corr, P=exp(S-m_new), block denom. One warp-lane group per
        //     row would be ideal; v1 uses one thread per row (BQ<=32 rows).
        for (uint r = tid; r < BQ; r += TPG) {
            const uint grow = m0 + r;
            float mb = -INFINITY;
            for (uint c = 0u; c < bk; ++c) {
                const uint gt = kt + c;
                const bool ok = !d.causal || gt <= d.kv_len + grow;
                if (ok && grow < d.m) {
                    mb = max(mb, Sf[r * BK + c]);
                }
            }
            bmax[r] = mb;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // compute per-row m_new/corr, write P (half) and accumulate denom
        for (uint r = tid; r < BQ; r += TPG) {
            const uint grow = m0 + r;
            const float mold = rm[r];
            const float mnew = max(mold, bmax[r]);
            float lsum = 0.f;
            for (uint c = 0u; c < BK; ++c) {
                float p = 0.f;
                if (c < bk) {
                    const uint gt = kt + c;
                    const bool ok = !d.causal || gt <= d.kv_len + grow;
                    if (ok && grow < d.m && isfinite(mnew)) {
                        p = exp(Sf[r * BK + c] - mnew);
                        lsum += p;
                    }
                }
                Ph[r * BK + c] = half(p);
            }
            const float corr = isfinite(mold) ? exp(mold - mnew) : 0.f;
            rl[r] = rl[r] * corr + lsum;
            rm[r] = mnew;
            bmax[r] = corr;   // stash corr for the O rescale below
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // (3) rescale O fragments by per-row corr (bmax[row]).
        for (uint i = 0u; i < TM; ++i) {
            for (uint j = 0u; j < TN; ++j) {
                const uint row = 8u * i + fm;
                const float corr = bmax[row];
                thread float2 &ec =
                    reinterpret_cast<thread float2 &>(O[i][j].thread_elements());
                ec[0] *= corr;
                ec[1] *= corr;
            }
        }

        // (4) O[:, sg slice] += P . V[:, sg slice]. Each simdgroup owns a
        //     DIFFERENT head-dim slice of V, so a per-simdgroup staged slice
        //     would race on shared tgmem. Instead stream V in 8-key sub-blocks
        //     staging the FULL transposed Vt[hd][8] (all simdgroups read their
        //     own hd-slice from the shared buffer -- race-free). Masked P cols
        //     are 0 so contracting only the valid keys is unnecessary; sub-block
        //     keys past bk are staged 0.
        for (uint ks = 0u; ks < bk; ks += 8u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < hd * 8u; i += TPG) {
                const uint gd = i / 8u;   // head-dim col (full hd)
                const uint sk = i % 8u;   // key within sub-block
                half v = 0.h;
                const uint tk = ks + sk;
                if (tk < bk) {
                    const uint gt = kt + tk;
                    if (KV_F32_SIDE) {
                        v = half(vbf[(ulong)gt * d.b_row_stride + gd]);
                    } else {
                        v = vb[(ulong)gt * d.b_row_stride + gd];
                    }
                }
                stage[gd * 8u + sk] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // O[i][j] += Ph[:, ks..ks+8] . Vt[sg*ND + 8j+..][.] (kdim=8).
            flash_mma_h<FL_BQ / 8u, (512u / 8u) / 8u>(
                &Ph[ks], BK, &stage[(sgid * ND) * 8u], 8u, O, 8u, fm, fn);
        }
    }

    // ---- store O / l ----
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0u; i < TM; ++i) {
        const uint row = 8u * i + fm;
        if (m0 + row >= d.m) {
            continue;
        }
        const float l = rl[row];
        const float inv = (l > 0.f) ? 1.f / l : 0.f;
        for (uint j = 0u; j < TN; ++j) {
            const uint col = sgid * ND + 8u * j + fn;
            const thread float2 &ec =
                reinterpret_cast<thread float2 &>(O[i][j].thread_elements());
            arena_store(out, (ulong)(m0 + row) * d.out_row_stride
                             + (ulong)qh * hd + col, ec[0] * inv);
            arena_store(out, (ulong)(m0 + row) * d.out_row_stride
                             + (ulong)qh * hd + col + 1u, ec[1] * inv);
        }
    }
}
