// int8-accumulated block-sparse MoE expert GEMM (E18 redirect prototype).
//
// Standalone bench kernel — NOT wired to production. Settles whether skipping
// the q4->half dequant + half-simdgroup-MMA path (the production
// `gemm_tunable_sparse`) and accumulating `int += int8*int8` instead wins on
// M3, where there is no dedicated tensor hardware and the synthetic microbench
// is unreliable in absolute terms (see PLAN.md E18 redirect).
//
// Dispatch contract mirrors `gemm_tunable_sparse` for direct comparability:
//   - grid = (N/BN, num_blocks)
//   - BM = block height (32 in production; sweepable here since int8 has no
//     MMA register-pressure trap)
//   - BN, BK tile constants
//   - 4 simdgroups in 2x2; per-lane 8x8 fragment ownership
//
// Buffer layout (self-contained — int8 is not a registered .dgq format; weights
// come from a bench-side re-quant side-channel):
//   buffer 0: device const char  *a_int8     packed int8 activations [slots, K]
//   buffer 1: device const char  *w_int8     packed int8 expert weights blob
//   buffer 2: device       float *c          f32 slot output [slots, N]
//   buffer 3: device const GroupedJob *jobs  per-expert w_byte_off
//   buffer 4: device const uint   *row_starts
//   buffer 5: constant     uint  &num_jobs
//   buffer 6: device const RouteScratch *route
//   buffer 7: device const float *scale_a   per-row f32 absmax scales [slots]
//   buffer 8: device const float *scale_w   per-output-col f32 absmax scales [N]
//                         (one scale per output column, shared across experts —
//                          the bench re-quants all expert rows with the same
//                          per-column scale envelope to keep the side-channel
//                          simple. Production int8 would use per-(expert,col)
//                          scales; deferred until the perf signal is known.)

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

#include "gemm_fc.metal"
#include "arena.metal"
#include "qgemm_grouped.metal"
#include "moe_router_device.metal"

#ifndef INT_BM
#define INT_BM 32
#endif
#ifndef INT_BN
#define INT_BN 128
#endif
#ifndef INT_BK
#define INT_BK 32
#endif

constant uint BM = INT_BM;
constant uint BN = INT_BN;
constant uint BK = INT_BK;
// bank-conflict pad: BK + 4 bytes (int8 storage, 1 byte/elem).
constant uint PAD = BK + 4u;
// 8x8 fragments per simdgroup along M / N. BM/16 and BN/16.
constant uint TM = BM / 16u;
constant uint TN = BN / 16u;

// Lane -> element coordinates within an 8x8 fragment for the int8 dot path.
// Each lane owns (fm, fn) and (fm, fn+1) — 2 of the 64 elements. 32 lanes
// cover the whole 8x8 (8 rows x 4 col-pairs x 2 = 64). Simpler than steel's
// frag_lane_coords (which is dictated by simdgroup_matrix register layout);
// valid here because we do NOT use simdgroup_matrix (float/half/bf16 only per
// PLAN.md E18 note) — the int8 dot product is a per-lane scalar chain.
inline void int_frag_lane_coords(uint lane, thread uint &fm, thread uint &fn) {
    fm = lane / 4u;
    fn = (lane % 4u) * 2u;
}

kernel void gemm_int_sparse(
    device const char  *a_int8   [[buffer(0)]],
    device const char  *w_int8   [[buffer(1)]],
    device       float *c        [[buffer(2)]],
    device const GroupedJob *jobs [[buffer(3)]],
    device const uint  *row_starts [[buffer(4)]],
    constant uint &num_jobs      [[buffer(5)]],
    device const RouteScratch *route [[buffer(6)]],
    device const float *scale_a  [[buffer(7)]],
    device const float *scale_w  [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint N = GEMM_N, K = GEMM_K;
    threadgroup char Xs[BM][PAD];
    threadgroup char Ws[BN][PAD];

    const uint b = tgid.y;
    if (b >= route->num_blocks) {
        return;
    }
    const uint e = route->block_expert[b];
    if (e >= num_jobs) {
        return;
    }
    const uint row0 = route->block_row0[b];
    const uint m1 = row_starts[e + 1u];
    const uint M_tile = min(BM, m1 - row0);
    if (M_tile == 0u) {
        return;
    }
    const uint n0 = tgid.x * BN;
    if (n0 >= N) {
        return;
    }
    const ulong w_off = jobs[e].w_byte_off;
    const uint ltid = lid.x;

    uint fm, fn;
    int_frag_lane_coords(lane, fm, fn);

    // Per-lane int accumulators: TM x TN 8x8 fragments, each lane owns 2
    // elements (fn, fn+1) per fragment.
    int C[TM][TN][2];
    for (uint i = 0u; i < TM; ++i) {
        for (uint j = 0u; j < TN; ++j) {
            C[i][j][0] = 0;
            C[i][j][1] = 0;
        }
    }

    // Sub-tile base for this simdgroup (4 simdgroups in 2x2).
    const uint sr = (BM / 2u) * (sgid >> 1u);
    const uint sc = (BN / 2u) * (sgid & 1u);

    // W tile loader split: 128 threads over BN rows.
    const uint w_tpr = 128u / BN;
    const uint w_cols = BK / w_tpr;
    const uint w_r = ltid / w_tpr;
    const uint w_q = (ltid % w_tpr) * w_cols;

    for (uint k0 = 0u; k0 < K; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // A tile: int8 packed activations, rows >= M_tile zero-filled.
        for (uint i = ltid * 4u; i < BM * BK; i += 128u * 4u) {
            const uint mm = i / BK;
            const uint kk = i % BK;
            char4 h;
            if (mm < M_tile) {
                const uint slot = row0 + mm;
                h = *(device const char4 *)(a_int8 + (ulong)slot * K + k0 + kk);
            } else {
                h = char4(0);
            }
            *(threadgroup char4 *)(&Xs[mm][kk]) = h;
        }
        // W tile: int8 packed weights, straight from the int8 blob (no dequant).
        {
            const uint gn = n0 + w_r;
            if (gn < N) {
                device const char *wr = w_int8 + w_off + (ulong)gn * K;
                for (uint j = 0u; j < w_cols; j += 4u) {
                    const char4 v = *(device const char4 *)(wr + k0 + w_q + j);
                    *(threadgroup char4 *)(&Ws[w_r][w_q + j]) = v;
                }
            } else {
                for (uint j = 0u; j < w_cols; j += 4u) {
                    *(threadgroup char4 *)(&Ws[w_r][w_q + j]) = char4(0);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Per-lane int8 dot-product accumulation over the 8x8 fragments this
        // simdgroup owns. For each fragment (i,j), the lane computes the dot
        // product of A-row (sr+8i+fm) with W-cols (sc+8j+fn) and (sc+8j+fn+1)
        // over BK=32 elements, using 4-wide char4 dots (8 dots per element).
        for (uint i = 0u; i < TM; ++i) {
            const uint ar = sr + 8u * i + fm;
            for (uint j = 0u; j < TN; ++j) {
                const uint wc0 = sc + 8u * j + fn;
                int acc0 = C[i][j][0];
                int acc1 = C[i][j][1];
                for (uint kk = 0u; kk < BK; kk += 4u) {
                    const char4 a4 = *(threadgroup char4 *)(&Xs[ar][kk]);
                    const char4 w0 = *(threadgroup char4 *)(&Ws[wc0][kk]);
                    const char4 w1 = *(threadgroup char4 *)(&Ws[wc0 + 1u][kk]);
                    acc0 += int(a4.x) * int(w0.x) + int(a4.y) * int(w0.y)
                          + int(a4.z) * int(w0.z) + int(a4.w) * int(w0.w);
                    acc1 += int(a4.x) * int(w1.x) + int(a4.y) * int(w1.y)
                          + int(a4.z) * int(w1.z) + int(a4.w) * int(w1.w);
                }
                C[i][j][0] = acc0;
                C[i][j][1] = acc1;
            }
        }
    }

    // Per-lane direct store with absmax rescale: f32 out = int_acc *
    // scale_a[row] * scale_w[col]. Same f32 slot layout as gemm_tunable_sparse.
    for (uint i = 0u; i < TM; ++i) {
        const uint mm = sr + 8u * i + fm;
        if (mm >= M_tile) {
            continue;
        }
        const uint slot = row0 + mm;
        const float sa = scale_a[slot];
        for (uint j = 0u; j < TN; ++j) {
            const uint col0 = n0 + sc + 8u * j + fn;
            if (col0 < N) {
                const float sw = scale_w[col0];
                c[(ulong)slot * N + col0] = float(C[i][j][0]) * sa * sw;
            }
            if (col0 + 1u < N) {
                const float sw = scale_w[col0 + 1u];
                c[(ulong)slot * N + col0 + 1u] = float(C[i][j][1]) * sa * sw;
            }
        }
    }
}
