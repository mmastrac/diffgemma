#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#include "fc_axes.metal"
#include "common.metal"
#include "arena.metal"

// E17: GEMM-attention for FULL-layer PREFILL.
//
// attention_mma_full is pinned at ~1.25 TF/s by the hd=512 / 8x8-fragment
// occupancy shape (a complete axis sweep — MT/NKT/fragment-O/QG/occupancy — is
// Negative Knowledge in prefill-kernel-levers). During prefill the score matrix
// S = Q.K^T fits in memory per 256-row sub-chunk, so full attention decomposes
// into two big GEMMs at the tunable-GEMM rate (2.3-4 TF/s) plus a rowwise
// softmax:
//   1) S[i,t]  = <Q_i, K_t>                 (attn_gemm_qk,     NT-GEMM)
//   2) P[i,t]  = exp(S - rowmax)  (masked)   (attn_gemm_softmax, rowwise)
//   3) O[i,d]  = sum_t P[i,t] V[t,d] / L_i   (attn_gemm_pv,     NN via Vt-stage)
// No 1/sqrt(d): the scale is folded into the QK-norm upstream (same as
// attention_mma_full). P is left UNNORMALIZED (exp only) and attn_gemm_pv
// divides O by the row denom L_i at store time — mirroring mma_full's final
// divide, so the two paths share numerics (f16 K/V, f32 accumulate).
//
// Both matmuls reduce to the SAME NT fragment tiler as gemm_tunable
// (C[m][n] = sum_k Xs[m][k] * Ws[n][k], 64x64 output tile, 4 simdgroups in a
// 2x2 grid, 8x8 fragments, BK=32) — only the tile LOADERS differ:
//   QK: A = Q[i][d] (arena, row stride n_q_heads*hd);  B = K[t][d] native.
//   PV: A = P[i][t] (contiguous, row stride T);        B = V staged TRANSPOSED
//       to Ws[d][t] from native V[t][d] (so <P_i, V[:,d]> = O[i][d]).
// This is prefill-only; denoise keeps attention_mma_full. Not bit-identical.

// Tile geometry is tunable: prepend `#define AG_BM/AG_BN/AG_SM_TPG` (via
// attention_gemm::tuned_source) to sweep. Defaults reproduce the shipped 64x64
// / 256-thread kernel exactly, so the production (un-prepended) compile — and
// golden — is unchanged. BN must divide 128 (loader thread split) and be a
// multiple of 16; BM a multiple of 16 (register budget caps ~64).
#ifndef AG_BM
#define AG_BM 64
#endif
#ifndef AG_BN
#define AG_BN 64
#endif
#ifndef AG_SM_TPG
#define AG_SM_TPG 256
#endif

constant uint BM = AG_BM;
constant uint BN = AG_BN;
constant uint BK = 32u;
constant uint PAD = 40u;         // BK + 8 halfs: bank-conflict pad
constant uint TM = BM / 16u;     // fragments per simdgroup along M
constant uint TN = BN / 16u;     // fragments per simdgroup along N

// Shared NT fragment tiler (frag_mma_ktile + frag_lane_coords), the same MMA
// core gemm_tunable uses — only our tile loaders (strided arena Q, f16 KV cache
// native/transposed) differ.
#define FRAG_BM AG_BM
#define FRAG_BN AG_BN
#include "gemm_frag_tile.metal"

// E17b: read Q/K/V from the f32 side ring (buffer 9) and run the MMA all-float,
// matching attention_mma_full_side's precision (FC30, same axis the sliding/full
// mma kernels use). Unset = the f16 main-cache path. The dead branch's f32/f16
// threadgroup staging is stripped at compile time.
constant bool KV_F32_SIDE_FC [[function_constant(30)]];
constant bool KV_F32_SIDE =
    is_function_constant_defined(KV_F32_SIDE_FC) && KV_F32_SIDE_FC;

// Passed from the host per (layer, sub-chunk) dispatch. The Q head is the grid
// z index (tgid.z); this struct carries shapes, row strides, the per-head base
// strides, and the softmax mask parameters. All 16 Q heads run in one dispatch
// per kernel (grid.z = n_q_heads) with per-head S/P/lrow slices, so only two
// barriers separate the three kernels (QK -> softmax -> PV).
struct AttnGemmDims {
    uint m;             // query rows this dispatch (active_canvas, <= 256)
    uint n;             // valid N: QK -> t_total (T); PV -> head_dim
    uint k;             // contraction: QK -> head_dim; PV -> t_total (T)
    uint a_row_stride;  // elems between A rows (QK: n_q_heads*hd; PV: n_pad)
    uint b_row_stride;  // elems between native B rows (nkv*hd*2, both)
    uint s_row_stride;  // elems between S/P rows (n_pad = round_up(T, 64))
    uint out_row_stride;// elems between O rows in the attno arena (n_q_heads*hd)
    uint causal;        // softmax: 1 = causal prefill
    uint kv_len;        // softmax: causal cutoff base (valid t <= kv_len + row)
    uint hd;            // head_dim (512): per-head base stride for Q/K/V/O
    uint group;         // GQA group (n_q_heads / nkv): kvh = qh / group
    uint nkv;           // KV heads: V base = (nkv + kvh) * hd
    uint s_head_stride; // elems between heads in the S/P planes (m * s_row_stride)
    uint head_base;     // E17a head-chunk: global Q head = head_base + tgid.z;
                        // the S/P/lrow scratch is indexed by the LOCAL tgid.z,
                        // so it holds only this batch's HC heads.
};

// ---- QK: S[i,t] = <Q_i, K_t>. A = Q (arena, strided), B = K native. --------
kernel void attn_gemm_qk(
    device const ushort *q [[buffer(0)]],     // Q plane [canvas][n_q_heads][hd]
    device const half *kcache [[buffer(1)]],  // kv16 base (K region), f16 path
    device float *s [[buffer(2)]],            // [n_q_heads][m][s_row_stride] f32
    constant AttnGemmDims &d [[buffer(3)]],
    device const float *kv_f32_side [[buffer(9), function_constant(KV_F32_SIDE_FC)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint M = d.m, N = d.n, K = d.k;   // N = T (keys), K = head_dim
    // f16 path stages half; f32-side path stages float. The unused pair is
    // dead-stripped (KV_F32_SIDE is compile-time).
    threadgroup half Xs[BM][PAD];
    threadgroup half Ws[BN][PAD];
    threadgroup float Xsf[BM][PAD];
    threadgroup float Wsf[BN][PAD];

    const uint qh = d.head_base + tgid.z;     // global data head
    const uint kvh = qh / d.group;
    q += (ulong)qh * d.hd;                    // this head's Q base
    s += (ulong)tgid.z * d.s_head_stride;     // this batch-local S slice
    device const half *kb = kcache + (ulong)kvh * d.hd;         // f16 K base
    device const float *kbf =
        KV_F32_SIDE ? (kv_f32_side + (ulong)kvh * d.hd) : (device const float *)nullptr;

    const uint m0 = tgid.y * BM;
    const uint n0 = tgid.x * BN;
    const uint ltid = lid.x;

    uint fm, fn;
    frag_lane_coords(lane, fm, fn);

    simdgroup_float8x8 C[TM][TN];
    for (uint i = 0u; i < TM; ++i) {
        for (uint j = 0u; j < TN; ++j) {
            C[i][j] = simdgroup_float8x8(0.f);
        }
    }
    const uint sr = (BM / 2u) * (sgid >> 1u);
    const uint sc = (BN / 2u) * (sgid & 1u);

    // W (=K) loader geometry: 128 threads over BN rows (t), BK cols (d).
    const uint w_tpr = 128u / BN;         // 2 threads per K row
    const uint w_cols = BK / w_tpr;       // 16 cols per thread
    const uint w_r = ltid / w_tpr;        // 0..63 (t within tile)
    const uint w_q = (ltid % w_tpr) * w_cols;

    for (uint k0 = 0u; k0 < K; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // A = Q[i][d]: strided arena rows (bf16/f16 per K_ARENA_F16). K is the
        // head_dim (multiple of 32) so the BK tile is always full.
        for (uint i = ltid * 4u; i < BM * BK; i += 128u * 4u) {
            const uint mm = i / BK;
            const uint kk = i % BK;
            const bool valid = (m0 + mm < M);
            const device ushort4 *qp = (device const ushort4 *)
                (q + (ulong)(m0 + mm) * d.a_row_stride + k0 + kk);
            if (KV_F32_SIDE) {
                float4 h = float4(0);
                if (valid) {
                    const ushort4 u = *qp;
                    h = K_ARENA_F16 ? float4(as_type<half4>(u))
                                    : as_type<float4>(uint4(u) << 16u);
                }
                *(threadgroup float4 *)(&Xsf[mm][kk]) = h;
            } else {
                half4 h = half4(0);
                if (valid) {
                    const ushort4 u = *qp;
                    h = K_ARENA_F16 ? as_type<half4>(u)
                                    : half4(as_type<float4>(uint4(u) << 16u));
                }
                *(threadgroup half4 *)(&Xs[mm][kk]) = h;
            }
        }
        // B = K[t][d] native: Ws[t_local][d_local] (f16 cache or f32 side ring).
        {
            const uint gt = n0 + w_r;
            if (KV_F32_SIDE) {
                if (gt < N) {
                    device const float *row = kbf + (ulong)gt * d.b_row_stride + k0;
                    for (uint j = 0u; j < w_cols; j += 4u) {
                        *(threadgroup float4 *)(&Wsf[w_r][w_q + j]) =
                            *(device const float4 *)(row + w_q + j);
                    }
                } else {
                    for (uint j = 0u; j < w_cols; j += 4u) {
                        *(threadgroup float4 *)(&Wsf[w_r][w_q + j]) = float4(0);
                    }
                }
            } else {
                if (gt < N) {
                    device const half *row = kb + (ulong)gt * d.b_row_stride + k0;
                    for (uint j = 0u; j < w_cols; j += 4u) {
                        *(threadgroup half4 *)(&Ws[w_r][w_q + j]) =
                            *(device const half4 *)(row + w_q + j);
                    }
                } else {
                    for (uint j = 0u; j < w_cols; j += 4u) {
                        *(threadgroup half4 *)(&Ws[w_r][w_q + j]) = half4(0);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (KV_F32_SIDE) {
            frag_mma_ktile_f32(Xsf, Wsf, C, sr, sc, fm, fn);
        } else {
            frag_mma_ktile(Xs, Ws, C, sr, sc, fm, fn);
        }
    }

    for (uint i = 0u; i < TM; ++i) {
        const uint row = m0 + sr + 8u * i + fm;
        if (row >= M) {
            continue;
        }
        for (uint j = 0u; j < TN; ++j) {
            const uint col = n0 + sc + 8u * j + fn;
            const thread float2 &ec =
                reinterpret_cast<thread float2 &>(C[i][j].thread_elements());
            if (col < N) {
                s[(ulong)row * d.s_row_stride + col] = ec[0];
            }
            if (col + 1u < N) {
                s[(ulong)row * d.s_row_stride + col + 1u] = ec[1];
            }
        }
    }
}

// ---- Softmax: P = exp(S - rowmax), masked; row denom L to lrow[row]. -------
// One threadgroup per query row. P is left UNNORMALIZED (attn_gemm_pv divides
// by L at store). Masked columns (t >= t_total or, causal, t > kv_len+row) are
// written 0 so the PV contraction skips them; S for those columns is never
// read (so the QK pad past T need not be initialized).
constant uint SM_TPG = AG_SM_TPG;
kernel void attn_gemm_softmax(
    device const float *s [[buffer(0)]],  // [n_q_heads][m][s_row_stride] scores
    device half *p [[buffer(1)]],         // [n_q_heads][m][s_row_stride] probs
    device float *lrow [[buffer(2)]],     // [n_q_heads][m] row denoms
    constant AttnGemmDims &d [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    const uint tid = lid.x;
    const uint row = tgid.x;
    const uint qh = tgid.y;   // grid.y = head
    if (row >= d.m) {
        return;
    }
    s += (ulong)qh * d.s_head_stride;
    lrow += (ulong)qh * d.m;
    const uint t_total = d.n;             // T
    const uint n_pad = d.s_row_stride;    // round_up(T, 64)
    const uint qpos = d.kv_len + row;     // causal cutoff for this row
    const bool causal = d.causal != 0u;
    const ulong base = (ulong)row * n_pad;
    // P slice base (element index). f32-side path stores f32 probs (matching the
    // f32 PV MMA); the p buffer is reinterpreted + sized accordingly host-side.
    const ulong p_base = (ulong)qh * d.s_head_stride + base;
    device float *pf = (device float *)p;

    threadgroup float red[SM_TPG];
    threadgroup float mshare;
    threadgroup float lshare;

    (void)pf;
    // Pass 1: row max over valid columns.
    float m = -INFINITY;
    for (uint t = tid; t < t_total; t += SM_TPG) {
        if (!causal || t <= qpos) {
            m = max(m, s[base + t]);
        }
    }
    red[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = SM_TPG / 2u; w > 0u; w >>= 1u) {
        if (tid < w) {
            red[tid] = max(red[tid], red[tid + w]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        mshare = red[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mmax = mshare;

    // Pass 2: p = exp(s - mmax) for valid columns (0 elsewhere); sum -> L.
    // f32-side path writes f32 probs (the PV MMA runs in f32); else f16.
    float l = 0.f;
    for (uint t = tid; t < n_pad; t += SM_TPG) {
        float pv = 0.f;
        if (t < t_total && (!causal || t <= qpos)) {
            pv = exp(s[base + t] - mmax);
            l += pv;
        }
        if (KV_F32_SIDE) {
            pf[p_base + t] = pv;
        } else {
            p[p_base + t] = half(pv);
        }
    }
    red[tid] = l;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = SM_TPG / 2u; w > 0u; w >>= 1u) {
        if (tid < w) {
            red[tid] += red[tid + w];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        lshare = red[0];
        lrow[row] = red[0];
    }
    (void)lshare;
}

// ---- PV: O[i,d] = (sum_t P[i,t] V[t,d]) / L_i. A = P, B = V staged Vt. ------
kernel void attn_gemm_pv(
    device const half *p [[buffer(0)]],       // [n_q_heads][m][a_row_stride] P (f16 path)
    device const half *vcache [[buffer(1)]],  // kv16 base (V region), f16 path
    device ushort *out [[buffer(2)]],         // attno plane [canvas][n_q_heads][hd]
    device const float *lrow [[buffer(3)]],   // [n_q_heads][m] row denoms
    constant AttnGemmDims &d [[buffer(4)]],
    device const float *kv_f32_side [[buffer(9), function_constant(KV_F32_SIDE_FC)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint M = d.m, N = d.n, K = d.k;   // N = head_dim, K = T (keys)
    threadgroup half Xs[BM][PAD];
    threadgroup half Ws[BN][PAD];
    threadgroup float Xsf[BM][PAD];
    threadgroup float Wsf[BN][PAD];

    const uint qh = d.head_base + tgid.z;           // global data head
    const uint kvh = qh / d.group;
    // P slice base (elements); f32-side reinterprets p as f32 (matching softmax).
    const ulong p_base = (ulong)tgid.z * d.s_head_stride;
    device const float *pf = (device const float *)p;
    device const half *vb = vcache + (ulong)(d.nkv + kvh) * d.hd;    // f16 V base
    device const float *vbf = KV_F32_SIDE
        ? (kv_f32_side + (ulong)(d.nkv + kvh) * d.hd) : (device const float *)nullptr;
    out += (ulong)qh * d.hd;                        // this head's O base
    lrow += (ulong)tgid.z * d.m;                    // this batch-local lrow slice

    const uint m0 = tgid.y * BM;
    const uint n0 = tgid.x * BN;
    const uint ltid = lid.x;

    uint fm, fn;
    frag_lane_coords(lane, fm, fn);

    simdgroup_float8x8 C[TM][TN];
    for (uint i = 0u; i < TM; ++i) {
        for (uint j = 0u; j < TN; ++j) {
            C[i][j] = simdgroup_float8x8(0.f);
        }
    }
    const uint sr = (BM / 2u) * (sgid >> 1u);
    const uint sc = (BN / 2u) * (sgid & 1u);

    for (uint k0 = 0u; k0 < K; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // A = P[i][t]: contiguous rows (stride a_row_stride = n_pad, zeroed past
        // T so the BK tile is always full and in-bounds).
        for (uint i = ltid * 4u; i < BM * BK; i += 128u * 4u) {
            const uint mm = i / BK;
            const uint kk = i % BK;
            const bool valid = (m0 + mm < M);
            if (KV_F32_SIDE) {
                float4 h = float4(0);
                if (valid) {
                    h = *(device const float4 *)
                        (pf + p_base + (ulong)(m0 + mm) * d.a_row_stride + k0 + kk);
                }
                *(threadgroup float4 *)(&Xsf[mm][kk]) = h;
            } else {
                half4 h = half4(0);
                if (valid) {
                    h = *(device const half4 *)
                        (p + p_base + (ulong)(m0 + mm) * d.a_row_stride + k0 + kk);
                }
                *(threadgroup half4 *)(&Xs[mm][kk]) = h;
            }
        }
        // B = V staged TRANSPOSED: Ws[d_local][t_local] = V[k0+t_local][n0+d_local].
        // Coalesced device reads over d (contiguous in V), scattered tgmem write.
        for (uint idx = ltid; idx < BN * BK; idx += 128u) {
            const uint t_local = idx / BN;
            const uint d_local = idx % BN;
            const uint gd = n0 + d_local;   // head-dim column (N)
            const uint gt = k0 + t_local;   // key (K)
            const bool valid = (gd < N && gt < K);
            if (KV_F32_SIDE) {
                Wsf[d_local][t_local] =
                    valid ? vbf[(ulong)gt * d.b_row_stride + gd] : 0.f;
            } else {
                Ws[d_local][t_local] =
                    valid ? vb[(ulong)gt * d.b_row_stride + gd] : 0.h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (KV_F32_SIDE) {
            frag_mma_ktile_f32(Xsf, Wsf, C, sr, sc, fm, fn);
        } else {
            frag_mma_ktile(Xs, Ws, C, sr, sc, fm, fn);
        }
    }

    for (uint i = 0u; i < TM; ++i) {
        const uint row = m0 + sr + 8u * i + fm;
        if (row >= M) {
            continue;
        }
        const float l = lrow[row];
        const float inv = (l > 0.f) ? 1.f / l : 0.f;
        for (uint j = 0u; j < TN; ++j) {
            const uint col = n0 + sc + 8u * j + fn;   // head-dim d
            const thread float2 &ec =
                reinterpret_cast<thread float2 &>(C[i][j].thread_elements());
            if (col < N) {
                arena_store(out, (ulong)row * d.out_row_stride + col, ec[0] * inv);
            }
            if (col + 1u < N) {
                arena_store(out, (ulong)row * d.out_row_stride + col + 1u, ec[1] * inv);
            }
        }
    }
}
