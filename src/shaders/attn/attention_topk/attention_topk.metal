// E20: top-k sparse attention for full-layer PREFILL (sibling of E17).
//
// Uses E17's `attn_gemm_qk` compiled with FC32 (EMIT_PAT16): alongside the S
// plane (S[HC][canvas][n_pad] f32) it emits a u16 PATTERN plane — the top 16
// bits of each score's float-bit-monotonic key. The selection passes compare
// ONLY those 16 bits, so they read the u16 plane (half the bytes of S; the
// stage is DRAM-bound) and touch f32 S only for the <=k winners:
//   2. attn_topk_softmax  — per (row, head): 4-level radix histogram finds the
//      k-th-largest FULL 32-bit key (levels 1-2 on the u16 plane; levels 3-4
//      refine among u16-ties, reading f32 S only for those candidates — so
//      selection is EXACT top-k by f32 score); a count + exclusive prefix-sum
//      pass then emits EXACTLY k (score, idx) pairs in fixed thread-major
//      order (residual ties = bit-identical scores, broken by thread-major
//      rank — DETERMINISTIC, no atomic compaction; the cd66294 shape).
//      Softmax over the emitted set.
//      Output: P[HC][canvas][K_PAD] (f32), Idx[HC][canvas][K_PAD] (u32),
//      lrow[HC][canvas] (f32).
//   3. attn_topk_pv       — O[i,d] = sum_{j=0..K_PAD} P[i,j] * V[idx[i,j], d] / L_i.
//      Gathered-V: each query row has its own key indices, so V is gathered
//      per-row. Prototype uses a scalar per-row accumulator (lane owns 2
//      d-cols, loops over K_PAD keys) — correct, issue-bound (matches the
//      attention_mma shape that pinned E17's motivation). Sums P in list
//      order — deterministic because the list is.
//
// NOT bit-identical to dense attention (only top-k keys are kept). Quality-gated.
// Selection = exact top-k by f32 score; only bit-identical scores are
// interchangeable to the selector (tie-break = thread-major rank, fixed).
// Causal mask: invalid keys (t > kv_len + row, or t >= t_total) are treated as
// -inf and never selected. The QK kernel leaves those S entries as whatever
// the partial GEMM produced (pad cols past T are never written; causal-past
// entries are computed but the softmax masks them) — so this kernel MUST
// re-mask here, reading d.causal / d.kv_len to clamp the candidate set.
//
// Tile geometry is tunable: prepend `#define AG_BM/AG_BN/AG_SM_TPG/AG_K_PAD`
// (via attention_topk::tuned_source) to sweep. Defaults reproduce the shipped
// 64x64/256/K_PAD=64 kernel.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#include "fc_axes.metal"
#include "common.metal"
#include "arena.metal"

#ifndef AG_BM
#define AG_BM 64
#endif
#ifndef AG_BN
#define AG_BN 64
#endif
#ifndef AG_SM_TPG
#define AG_SM_TPG 256
#endif
#ifndef AG_K_PAD
#define AG_K_PAD 64
#endif

constant uint BM = AG_BM;
constant uint BN = AG_BN;
constant uint BK = 32u;
constant uint PAD = 40u;
constant uint TM = BM / 16u;
constant uint TN = BN / 16u;
constant uint SM_TPG = AG_SM_TPG;
constant uint K_PAD = AG_K_PAD;

#define FRAG_BM AG_BM
#define FRAG_BN AG_BN
#include "gemm_frag_tile.metal"

// E17b: f32 side-KV variant (FC30) — same axis as attention_gemm.
constant bool KV_F32_SIDE_FC [[function_constant(30)]];
constant bool KV_F32_SIDE =
    is_function_constant_defined(KV_F32_SIDE_FC) && KV_F32_SIDE_FC;

// The same dims struct as E17 (re-declared here so this shader is self-contained
// for compilation; the host passes the same AttnGemmDims bytes).
struct AttnGemmDims {
    uint m;
    uint n;
    uint k;
    uint a_row_stride;
    uint b_row_stride;
    uint s_row_stride;
    uint out_row_stride;
    uint causal;
    uint kv_len;
    uint hd;
    uint group;
    uint nkv;
    uint s_head_stride;
    uint head_base;
};

// Convert float to a monotonic uint (larger uint = larger float) for binary
// search. -inf -> 0, +inf -> 0xFFFFFFFE, NaN -> 0xFFFFFFFF (never selected).
inline uint topk_float_to_key(float f) {
    uint u = as_type<uint>(f);
    return (u & 0x80000000u) ? (~u) : (u ^ 0x80000000u);
}

// Per-(row, head) top-k selection + softmax.
// One threadgroup per (row, head); grid = (canvas, n_heads_batch, 1).
// Threadgroup size = SM_TPG.
//
// Algorithm: 2-level radix histogram selection.
//   Pass 1: build a 256-bin histogram of the top 8 bits of each key's
//           float-bit-monotonic pattern. Prefix-sum to find the bin
//           containing the k-th largest key.
//   Pass 2: within that bin, build another 256-bin histogram of the next 8
//           bits. Find the exact threshold pattern.
//   Pass 3: emit all keys with pattern >= threshold (atomic-append to P/Idx).
//   Phase 3: softmax over the emitted scores.
// Total: 3 passes over S (vs 33 for the binary-search prototype). The
// histograms are 256 × 4 = 1KB each in tgmem.
kernel void attn_topk_softmax(
    device const float *s     [[buffer(0)]],  // S plane [HC][m][n_pad] f32
    device float  *p         [[buffer(1)]],  // P plane [HC][m][K_PAD] f32
    device uint   *idx       [[buffer(2)]],  // Idx plane [HC][m][K_PAD] u32
    device float  *lrow      [[buffer(3)]],  // [HC][m] f32 row denoms
    constant AttnGemmDims &d [[buffer(4)]],
    constant uint &k_param   [[buffer(5)]],  // k (max keys to keep)
    device const ushort *pat [[buffer(6)]],  // u16 key plane [HC][m][n_pad]
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid  [[thread_position_in_threadgroup]]
) {
    const uint tid = lid.x;
    const uint row = tgid.x;
    const uint qh_local = tgid.y;
    if (row >= d.m) {
        return;
    }
    const uint n_pad = d.s_row_stride;
    const uint t_total = d.n;
    const uint kv_len = d.kv_len;
    const bool causal = d.causal != 0u;
    const uint qpos = kv_len + row;

    s   += (ulong)qh_local * d.s_head_stride + (ulong)row * n_pad;
    pat += (ulong)qh_local * d.s_head_stride + (ulong)row * n_pad;
    p   += (ulong)qh_local * d.m * K_PAD + (ulong)row * K_PAD;
    idx += (ulong)qh_local * d.m * K_PAD + (ulong)row * K_PAD;
    lrow += (ulong)qh_local * d.m + row;

    // Shared histogram (256 bins). Reused across the 2 radix levels.
    threadgroup atomic_uint hist[256];

    // ---- Count valid keys + find global max pattern (for threshold boundary). ----
    threadgroup atomic_uint vshare;
    if (tid == 0u) atomic_store_explicit(&vshare, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n_valid_local = 0u;
    for (uint t = tid; t < t_total; t += SM_TPG) {
        if (!causal || t <= qpos) n_valid_local++;
    }
    atomic_fetch_add_explicit(&vshare, n_valid_local, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n_valid = atomic_load_explicit(&vshare, memory_order_relaxed);
    const uint k = min(k_param, n_valid);
    if (k == 0u) {
        if (tid == 0u) lrow[0] = 0.f;
        for (uint j = tid; j < K_PAD; j += SM_TPG) { p[j] = 0.f; idx[j] = 0u; }
        return;
    }

    // ---- Pass 1: top-8-bit histogram. ----
    // Initialize the histogram (only the first 256 threads, or loop).
    for (uint b = tid; b < 256u; b += SM_TPG) {
        atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = tid; t < t_total; t += SM_TPG) {
        if (!causal || t <= qpos) {
            uint bin = uint(pat[t]) >> 8u;  // top 8 bits of the u16 key
            atomic_fetch_add_explicit(&hist[bin], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Prefix-sum from the top (largest pattern) to find the bin containing
    // the k-th largest. `cumulative_above` counts keys with top-byte > bin.
    // We want the largest `bin1` such that cumulative_above[bin1] < k.
    threadgroup uint bin1_share;
    uint cum = 0u;
    uint found_bin1 = 0u;
    bool found = false;
    if (tid == 0u) {
        for (int b = 255; b >= 0; --b) {
            uint cnt = atomic_load_explicit(&hist[b], memory_order_relaxed);
            if (cum + cnt >= k) {
                // The k-th largest is in bin b. Threshold top-byte = b.
                found_bin1 = uint(b);
                found = true;
                break;
            }
            cum += cnt;
        }
        if (!found) found_bin1 = 0u;  // all keys valid, k covers everything
        bin1_share = found_bin1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint bin1 = bin1_share;
    // `cum` = count of keys with top-byte > bin1 (i.e., strictly above this bin).
    // We need to keep all keys with top-byte > bin1, plus enough keys in bin1
    // to reach k. The threshold within bin1 will be found by Pass 2.
    // `need_in_bin1` = k - cum.
    threadgroup uint need_share;
    if (tid == 0u) {
        uint c = 0u;
        for (int b = 255; b > int(bin1); --b) {
            c += atomic_load_explicit(&hist[b], memory_order_relaxed);
        }
        need_share = k - c;  // how many keys to keep from bin1
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint need_in_bin1 = need_share;

    // ---- Pass 2: next-8-bit histogram within bin1. ----
    for (uint b = tid; b < 256u; b += SM_TPG) {
        atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = tid; t < t_total; t += SM_TPG) {
        if (!causal || t <= qpos) {
            uint p16 = uint(pat[t]);
            if ((p16 >> 8u) == bin1) {
                atomic_fetch_add_explicit(&hist[p16 & 0xFFu], 1u,
                                          memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Find the threshold second-byte: the largest b2 such that cumulative
    // count above it < need_in_bin1.
    threadgroup uint bin2_share;
    if (tid == 0u) {
        uint c = 0u;
        uint found_b2 = 0u;
        for (int b = 255; b >= 0; --b) {
            uint cnt = atomic_load_explicit(&hist[b], memory_order_relaxed);
            if (c + cnt >= need_in_bin1) {
                found_b2 = uint(b);
                break;
            }
            c += cnt;
        }
        bin2_share = found_b2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint bin2 = bin2_share;
    // The u16-level threshold is (bin1 << 8) | bin2. The tie bucket at this
    // level spans 2^16 float patterns and typically holds MANY keys (scores
    // share an exponent), so selecting among u16-ties arbitrarily would miss
    // the true top keys (k=1 must pick the argmax). Refine to full f32-key
    // precision with two more radix levels over the tie candidates only —
    // f32 S is read ONLY where the u16 key equals the threshold.
    const uint thresh_top16 = (bin1 << 8u) | bin2;

    // ---- Pass 2b: bits 15-8 of the full key, among u16-level ties. ----
    threadgroup atomic_uint above16_share;
    if (tid == 0u) atomic_store_explicit(&above16_share, 0u, memory_order_relaxed);
    for (uint b = tid; b < 256u; b += SM_TPG) {
        atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint above16_local = 0u;
    for (uint t = tid; t < t_total; t += SM_TPG) {
        if (!causal || t <= qpos) {
            uint p16 = uint(pat[t]);
            if (p16 > thresh_top16) {
                above16_local++;
            } else if (p16 == thresh_top16) {
                uint full = topk_float_to_key(s[t]);
                atomic_fetch_add_explicit(&hist[(full >> 8u) & 0xFFu], 1u,
                                          memory_order_relaxed);
            }
        }
    }
    atomic_fetch_add_explicit(&above16_share, above16_local, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint above16 = atomic_load_explicit(&above16_share, memory_order_relaxed);
    const uint need_low = k - min(above16, k);   // to take among u16-ties
    threadgroup uint bin3_share;
    threadgroup uint need2c_share;
    if (tid == 0u) {
        uint c = 0u;
        uint found = 0u;
        for (int b = 255; b >= 0; --b) {
            uint cnt = atomic_load_explicit(&hist[b], memory_order_relaxed);
            if (c + cnt >= need_low) {
                found = uint(b);
                break;
            }
            c += cnt;
        }
        bin3_share = found;
        need2c_share = need_low - c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint bin3 = bin3_share;
    const uint need_2c = need2c_share;

    // ---- Pass 2c: bits 7-0, among ties that also match bin3. ----
    for (uint b = tid; b < 256u; b += SM_TPG) {
        atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = tid; t < t_total; t += SM_TPG) {
        if (!causal || t <= qpos) {
            if (uint(pat[t]) == thresh_top16) {
                uint full = topk_float_to_key(s[t]);
                if (((full >> 8u) & 0xFFu) == bin3) {
                    atomic_fetch_add_explicit(&hist[full & 0xFFu], 1u,
                                              memory_order_relaxed);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup uint bin4_share;
    if (tid == 0u) {
        uint c = 0u;
        uint found = 0u;
        for (int b = 255; b >= 0; --b) {
            uint cnt = atomic_load_explicit(&hist[b], memory_order_relaxed);
            if (c + cnt >= need_2c) {
                found = uint(b);
                break;
            }
            c += cnt;
        }
        bin4_share = found;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Full 32-bit threshold: selection is now exact top-k by the f32 key;
    // ties below need exact bit-equality of scores (identical f32 values).
    const uint thresh32 = (thresh_top16 << 16u) | (bin3 << 8u) | bin4_share;

    // ---- Pass 3a: per-thread above/tie counts + exclusive prefix-sums. ----
    // DETERMINISTIC emission (the cd66294 shape: count -> prefix-sum ->
    // thread-major emit) replacing the old atomic-slot append, whose ORDER was
    // race-determined and — because ties at the u16 threshold overflow K_PAD on
    // the normal path — whose surviving SET was race-determined too. Ties are
    // now resolved by fixed thread-major rank (t % SM_TPG, then t), so the set
    // and the order are pure functions of the scores. attn_topk_pv sums P in
    // list order; a fixed list makes the whole path run-to-run reproducible.
    uint cnt_above = 0u;
    uint cnt_tie = 0u;
    for (uint t = tid; t < t_total; t += SM_TPG) {
        if (!causal || t <= qpos) {
            uint p16 = uint(pat[t]);
            if (p16 > thresh_top16) {
                cnt_above++;               // top16 greater => full key greater
            } else if (p16 == thresh_top16) {
                uint full = topk_float_to_key(s[t]);
                if (full > thresh32) {
                    cnt_above++;
                } else if (full == thresh32) {
                    cnt_tie++;
                }
            }
        }
    }
    threadgroup uint scan_a[SM_TPG];
    threadgroup uint scan_t[SM_TPG];
    scan_a[tid] = cnt_above;
    scan_t[tid] = cnt_tie;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = 1u; off < SM_TPG; off <<= 1u) {
        uint va = 0u, vt = 0u;
        if (tid >= off) {
            va = scan_a[tid - off];
            vt = scan_t[tid - off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scan_a[tid] += va;
        scan_t[tid] += vt;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint above_base = scan_a[tid] - cnt_above;   // exclusive
    const uint tie_base = scan_t[tid] - cnt_tie;
    const uint above_total = scan_a[SM_TPG - 1u];
    // Threshold construction guarantees above_total < k <= above_total + ties;
    // the min() is belt-and-braces against a degenerate histogram.
    const uint need_ties = k - min(above_total, k);

    // ---- Pass 3b: emit. Above keys fill [0, above_total) in thread-major
    // order; the first `need_ties` ties (thread-major rank) fill the rest.
    // k_emit == k exactly (k <= K_PAD via the host clamp + n_valid min).
    for (uint t = tid, wa = 0u, wt = 0u; t < t_total; t += SM_TPG) {
        if (!causal || t <= qpos) {
            uint p16 = uint(pat[t]);
            if (p16 < thresh_top16) {
                continue;
            }
            uint full = (p16 > thresh_top16) ? ~0u : topk_float_to_key(s[t]);
            if (p16 > thresh_top16 || full > thresh32) {
                uint slot = above_base + wa;
                p[slot] = s[t];
                idx[slot] = t;
                wa++;
            } else if (full == thresh32) {
                uint r = tie_base + wt;
                if (r < need_ties) {
                    uint slot = above_total + r;
                    p[slot] = s[t];
                    idx[slot] = t;
                }
                wt++;
            }
        }
    }
    // P/Idx are device buffers cross-read by other threads in the softmax
    // phase below — fence device memory, not just tgmem.
    threadgroup_barrier(mem_flags::mem_device);
    const uint k_emit = above_total + need_ties;
    if (k_emit == 0u) {
        if (tid == 0u) lrow[0] = 0.f;
        for (uint j = tid; j < K_PAD; j += SM_TPG) { p[j] = 0.f; idx[j] = 0u; }
        return;
    }

    // ---- Phase 3: softmax over the k_emit emitted scores. ----
    threadgroup float red[SM_TPG];
    threadgroup float mshare;
    float m = -INFINITY;
    for (uint j = tid; j < k_emit; j += SM_TPG) m = max(m, p[j]);
    red[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = SM_TPG / 2u; w > 0u; w >>= 1u) {
        if (tid < w) red[tid] = max(red[tid], red[tid + w]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) mshare = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mmax = mshare;

    float l = 0.f;
    for (uint j = tid; j < k_emit; j += SM_TPG) {
        float pv = exp(p[j] - mmax);
        p[j] = pv;
        l += pv;
    }
    red[tid] = l;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = SM_TPG / 2u; w > 0u; w >>= 1u) {
        if (tid < w) red[tid] += red[tid + w];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) lrow[0] = red[0];
    for (uint j = tid + k_emit; j < K_PAD; j += SM_TPG) {
        p[j] = 0.f;
        idx[j] = 0u;
    }
}

// ---- PV: O[i,d] = (sum_j P[i,j] V[idx[i,j], d]) / L_i. ----
// Scalar per-row accumulator: one threadgroup per (row, head, d-tile of
// PV_BN cols); one simdgroup (32 threads), each lane owns 2 output d-cols and
// loops over K_PAD keys. Grid = (hd/PV_BN, canvas, n_heads_batch).
constant uint PV_BN = 64u;
kernel void attn_topk_pv(
    device const float *p    [[buffer(0)]],   // P plane [HC][m][K_PAD] f32
    device const uint  *idx  [[buffer(1)]],   // Idx plane [HC][m][K_PAD] u32
    device const half  *vcache [[buffer(2)]],  // kv16 base (V region), f16 path
    device ushort      *out [[buffer(3)]],    // attno plane [canvas][n_q_heads][hd]
    device const float  *lrow [[buffer(4)]],  // [HC][m] f32 row denoms
    constant AttnGemmDims &d [[buffer(5)]],
    device const float *kv_f32_side [[buffer(9), function_constant(KV_F32_SIDE_FC)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid  [[thread_position_in_threadgroup]]
) {
    const uint row = tgid.y;
    const uint qh_local = tgid.z;
    const uint d0 = tgid.x * PV_BN;
    if (row >= d.m) return;
    const uint qh = d.head_base + qh_local;
    const uint kvh = qh / d.group;
    p   += (ulong)qh_local * d.m * K_PAD + (ulong)row * K_PAD;
    idx += (ulong)qh_local * d.m * K_PAD + (ulong)row * K_PAD;
    lrow += (ulong)qh_local * d.m + row;
    out += (ulong)qh * d.hd + (ulong)row * d.out_row_stride;
    device const half *vb = vcache + (ulong)(d.nkv + kvh) * d.hd;
    device const float *vbf = KV_F32_SIDE
        ? (kv_f32_side + (ulong)(d.nkv + kvh) * d.hd)
        : (device const float *)nullptr;

    const float inv = (lrow[0] > 0.f) ? 1.f / lrow[0] : 0.f;
    const uint lane = lid.x;  // 0..31
    const uint d_base = d0 + lane * 2u;

    float acc0 = 0.f, acc1 = 0.f;
    for (uint j = 0u; j < K_PAD; ++j) {
        float pj = p[j];
        if (pj == 0.f) continue;
        uint t = idx[j];
        if (KV_F32_SIDE) {
            acc0 += pj * vbf[(ulong)t * d.b_row_stride + d_base];
            acc1 += pj * vbf[(ulong)t * d.b_row_stride + d_base + 1u];
        } else {
            acc0 += pj * (float)vb[(ulong)t * d.b_row_stride + d_base];
            acc1 += pj * (float)vb[(ulong)t * d.b_row_stride + d_base + 1u];
        }
    }
    acc0 *= inv;
    acc1 *= inv;
    if (d_base < d.hd) {
        arena_store(out, d_base, acc0);
    }
    if (d_base + 1u < d.hd) {
        arena_store(out, d_base + 1u, acc1);
    }
}
