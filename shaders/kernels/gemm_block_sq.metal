#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#include "gemm_fc.metal"
#include "dequant.metal"
#include "arena.metal"

/// Occupancy+AI GEMM PROTOTYPE (bench-only, not production-wired).
///
/// Research probe for the MPS/MLX GEMM gap (2026-07-02). Findings so far:
/// 32x32x32 tiles at ~9KB tgmem tie gemm_block (~2.3 TF/s) — occupancy alone
/// is not the delta; neither are bf16 A-conversion, W-dequant thread
/// utilization, or transposed B loads (all measured null). This variant tests
/// the remaining macro axis: MLX steel's LARGE tile (64x64x32, each of 4
/// simdgroups owning a 32x32 quadrant = 16 accumulators) at ~10KB tgmem
/// (2-3 resident TGs/core) — double the MMA work per loaded byte vs 32x128
/// production AND multi-TG residency, which production's 20KB forfeits.
///
/// The f32 store tile is aliased over the (dead) load buffers and staged in
/// two 32-row passes to stay within the 10KB budget.
kernel void gemm_block_sq(
    device const ushort *x [[buffer(0)]],
    device ushort *y [[buffer(1)]],
    device const uchar *blob [[buffer(2)]],
    constant ulong &w_off [[buffer(3)]],
    constant uint &M [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]
) {
    const uint N = GEMM_N, K = GEMM_K;
    constexpr uint BM = 64u;
    constexpr uint BN = 64u;
    constexpr uint BK = 32u;
    constexpr uint PAD = 40u; // BK + 8: bank-conflict pad
    // S[0] = A tile [BM][PAD], S[1] = W tile [BN][PAD] (row-major [col][kk]).
    threadgroup half S[2][BM][PAD];

    const uint m0 = tgid.y * BM;
    const uint n0 = tgid.x * BN;
    const uint ltid = lid.x;

    const bool is_raw = (K_QUANT_FORMAT == QUANT_RAW);
    const ulong rowB = is_raw ? (ulong)K * 2ul : q4_row_bytes(K);

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0u; i < 4u; ++i) {
        for (uint j = 0u; j < 4u; ++j) {
            acc[i][j] = simdgroup_float8x8(0.f);
        }
    }
    const uint sr = 32u * (sgid >> 1u); // simdgroup row quadrant base
    const uint sc = 32u * (sgid & 1u);  // simdgroup col quadrant base

    for (uint k0 = 0u; k0 < K; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // A tile BMx32: 16 halfs per thread.
        for (uint i = ltid; i < BM * BK; i += 128u) {
            const uint mm = i / BK;
            const uint kk = i % BK;
            S[0][mm][kk] = (m0 + mm < M)
                ? arena_act_half(x, (ulong)(m0 + mm) * K + k0 + kk)
                : half(0);
        }
        // W tile BNx32: 2 threads per row, 16 values each.
        {
            const uint r = ltid >> 1u;
            const uint q = (ltid & 1u) * 16u;
            const uint gn = n0 + r;
            if (gn < N) {
                if (is_raw) {
                    device const ushort *wr =
                        (device const ushort *)(blob + w_off + (ulong)gn * rowB);
                    for (uint j = 0u; j < 16u; ++j) {
                        S[1][r][q + j] = half(bf16_to_f32(wr[k0 + q + j]));
                    }
                } else {
                    dequant_q4_group_half_tg_range(
                        blob + w_off + (ulong)gn * rowB + (ulong)(k0 / 32u) * 20ul,
                        &S[1][r][q], q, 16u);
                }
            } else {
                for (uint j = 0u; j < 16u; ++j) {
                    S[1][r][q + j] = half(0);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < BK; kk += 8u) {
            simdgroup_half8x8 a[4];
            simdgroup_half8x8 b[4];
            simdgroup_load(a[0], &S[0][sr + 0u][kk], PAD);
            simdgroup_load(a[1], &S[0][sr + 8u][kk], PAD);
            simdgroup_load(a[2], &S[0][sr + 16u][kk], PAD);
            simdgroup_load(a[3], &S[0][sr + 24u][kk], PAD);
            simdgroup_load(b[0], &S[1][sc + 0u][kk], PAD, ulong2(0, 0), true);
            simdgroup_load(b[1], &S[1][sc + 8u][kk], PAD, ulong2(0, 0), true);
            simdgroup_load(b[2], &S[1][sc + 16u][kk], PAD, ulong2(0, 0), true);
            simdgroup_load(b[3], &S[1][sc + 24u][kk], PAD, ulong2(0, 0), true);
            for (uint i = 0u; i < 4u; ++i) {
                for (uint j = 0u; j < 4u; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
    }

    // Store in two 32-row passes through an f32 tile aliased over S (dead).
    threadgroup float (*ty)[BN] = (threadgroup float (*)[BN])&S[0][0][0];
    for (uint pass = 0u; pass < 2u; ++pass) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((sgid >> 1u) == pass) {
            for (uint i = 0u; i < 4u; ++i) {
                for (uint j = 0u; j < 4u; ++j) {
                    simdgroup_store(acc[i][j], &ty[8u * i][sc + 8u * j], BN);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint row_base = pass * 32u;
        for (uint i = ltid; i < 32u * BN; i += 128u) {
            const uint mm = row_base + i / BN;
            const uint nn = i % BN;
            if (m0 + mm < M && n0 + nn < N) {
                arena_store_bf16(y, (ulong)(m0 + mm) * N + n0 + nn, ty[i / BN][nn]);
            }
        }
    }
}
