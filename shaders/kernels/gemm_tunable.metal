#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#include "gemm_fc.metal"
#include "dequant.metal"
#include "arena.metal"

/// TUNABLE GEMM (task #19): steel-class fragment-level block GEMM.
///
/// The bench prototype (gemm_block_sq) proved macro tiling alone cannot reach
/// the MPS/MLX 3.4-4.4 TF/s at our shapes; the delta is fragment-level
/// codegen. This kernel replicates MLX steel's machinery in our framework:
/// - per-lane thread_elements() fragment loads with compile-time strides
///   (never simdgroup_load from threadgroup memory),
/// - simdgroup_barrier(mem_none) scheduling hints between load/mma phases,
/// - per-lane direct C store to device (no threadgroup f32 store tile),
/// - tile geometry via TUNE_BM/TUNE_BN #defines prepended at compile
///   ("tunable"): 4 simdgroups in 2x2, per-sg sub-tile (BM/2)x(BN/2),
///   8x8 fragments, BK=32 (q4 group-aligned).
///
/// Math contract: identical per-output K-accumulation chain to gemm_block
/// (ascending 32-wide K-tiles, ascending 8-wide kk chunks, same q4 dequant,
/// same bf16 I/O rounding) -> outputs must be BIT-EXACT vs gemm_block.
#ifndef TUNE_BM
#define TUNE_BM 64
#endif
#ifndef TUNE_BN
#define TUNE_BN 64
#endif

constant uint BM = TUNE_BM;
constant uint BN = TUNE_BN;
constant uint BK = 32u;
constant uint PAD = 40u; // BK + 8 halfs: bank-conflict pad
constant uint TM = BM / 16u; // 8x8 fragments per simdgroup along M
constant uint TN = BN / 16u; // 8x8 fragments per simdgroup along N

kernel void gemm_tunable(
    device const ushort *x [[buffer(0)]],
    device ushort *y [[buffer(1)]],
    device const uchar *blob [[buffer(2)]],
    constant ulong &w_off [[buffer(3)]],
    constant uint &M [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint N = GEMM_N, K = GEMM_K;
    threadgroup half Xs[BM][PAD];
    threadgroup half Ws[BN][PAD]; // row-major [n][k]

    const uint m0 = tgid.y * BM;
    const uint n0 = tgid.x * BN;
    const uint ltid = lid.x;

    const bool is_raw = (K_QUANT_FORMAT == QUANT_RAW);
    const ulong rowB = is_raw ? (ulong)K * 2ul : q4_row_bytes(K);

    // Steel lane->element coordinates within an 8x8 fragment: this lane owns
    // elements (fm, fn) and (fm, fn+1).
    const uint qid = lane / 4u;
    const uint fm = (qid & 4u) + ((lane / 2u) % 4u);
    const uint fn = (qid & 2u) * 2u + (lane % 2u) * 2u;

    simdgroup_float8x8 C[TM][TN];
    for (uint i = 0u; i < TM; ++i) {
        for (uint j = 0u; j < TN; ++j) {
            C[i][j] = simdgroup_float8x8(0.f);
        }
    }
    const uint sr = (BM / 2u) * (sgid >> 1u); // sg sub-tile row base
    const uint sc = (BN / 2u) * (sgid & 1u);  // sg sub-tile col base

    // W loader split: 128 threads over BN rows.
    const uint w_tpr = 128u / BN;            // threads per W row
    const uint w_cols = BK / w_tpr;          // cols per thread
    const uint w_r = ltid / w_tpr;
    const uint w_q = (ltid % w_tpr) * w_cols;

    for (uint k0 = 0u; k0 < K; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // A tile BMxBK: vectorized 4-wide loads + converts (K and kk are
        // 4-aligned; Metal buffers are 256B-aligned), per-4 row guard only.
        for (uint i = ltid * 4u; i < BM * BK; i += 128u * 4u) {
            const uint mm = i / BK;
            const uint kk = i % BK;
            half4 h;
            if (m0 + mm < M) {
                const ushort4 u = *(device const ushort4 *)(x + (ulong)(m0 + mm) * K + k0 + kk);
                if (K_ACT_F16) {
                    h = as_type<half4>(u);
                } else {
                    h = half4(as_type<float4>(uint4(u) << 16u));
                }
            } else {
                h = half4(0);
            }
            *(threadgroup half4 *)(&Xs[mm][kk]) = h;
        }
        // W tile BNxBK.
        {
            const uint gn = n0 + w_r;
            if (gn < N) {
                if (is_raw) {
                    device const ushort *wr =
                        (device const ushort *)(blob + w_off + (ulong)gn * rowB);
                    for (uint j = 0u; j < w_cols; j += 4u) {
                        const ushort4 u = *(device const ushort4 *)(wr + k0 + w_q + j);
                        *(threadgroup half4 *)(&Ws[w_r][w_q + j]) =
                            half4(as_type<float4>(uint4(u) << 16u));
                    }
                } else {
                    // Vectorized q4 decode: 8 nibbles per assembled u32; same
                    // per-element s*q+mn half math as the scalar helper.
                    device const uchar *g =
                        blob + w_off + (ulong)gn * rowB + (ulong)(k0 / 32u) * 20ul;
                    const half s = half(bf16_bytes(g));
                    const half mn = half(bf16_bytes(g + 2));
                    for (uint j8 = 0u; j8 < w_cols; j8 += 8u) {
                        const uint jj = w_q + j8;
                        device const uchar *p = g + 4u + jj / 2u;
                        const uint v = uint(p[0]) | (uint(p[1]) << 8u) |
                            (uint(p[2]) << 16u) | (uint(p[3]) << 24u);
                        const half4 q0 = half4(
                            half(v & 0xFu), half((v >> 4u) & 0xFu),
                            half((v >> 8u) & 0xFu), half((v >> 12u) & 0xFu));
                        const half4 q1 = half4(
                            half((v >> 16u) & 0xFu), half((v >> 20u) & 0xFu),
                            half((v >> 24u) & 0xFu), half(v >> 28u));
                        *(threadgroup half4 *)(&Ws[w_r][jj]) = s * q0 + half4(mn);
                        *(threadgroup half4 *)(&Ws[w_r][jj + 4u]) = s * q1 + half4(mn);
                    }
                }
            } else {
                for (uint j = 0u; j < w_cols; j += 4u) {
                    *(threadgroup half4 *)(&Ws[w_r][w_q + j]) = half4(0);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0u; kk < BK; kk += 8u) {
            simdgroup_barrier(mem_flags::mem_none);
            // A fragments: element (fm, fn) of frag i = Xs[sr + 8i + fm][kk + fn].
            simdgroup_half8x8 A[TM];
            for (uint i = 0u; i < TM; ++i) {
                thread half2 &ea = reinterpret_cast<thread half2 &>(A[i].thread_elements());
                ea[0] = Xs[sr + 8u * i + fm][kk + fn];
                ea[1] = Xs[sr + 8u * i + fm][kk + fn + 1u];
            }
            simdgroup_barrier(mem_flags::mem_none);
            // B fragments: element (k=fm, n=fn) of frag j = Ws[sc + 8j + fn][kk + fm].
            simdgroup_half8x8 B[TN];
            for (uint j = 0u; j < TN; ++j) {
                thread half2 &eb = reinterpret_cast<thread half2 &>(B[j].thread_elements());
                eb[0] = Ws[sc + 8u * j + fn][kk + fm];
                eb[1] = Ws[sc + 8u * j + fn + 1u][kk + fm];
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0u; i < TM; ++i) {
                for (uint j = 0u; j < TN; ++j) {
                    simdgroup_multiply_accumulate(C[i][j], A[i], B[j], C[i][j]);
                }
            }
        }
    }

    // Per-lane direct store: lane owns C elements (fm, fn) and (fm, fn+1).
    for (uint i = 0u; i < TM; ++i) {
        const uint row = m0 + sr + 8u * i + fm;
        if (row >= M) {
            continue;
        }
        for (uint j = 0u; j < TN; ++j) {
            const uint col = n0 + sc + 8u * j + fn;
            const thread float2 &ec = reinterpret_cast<thread float2 &>(C[i][j].thread_elements());
            // Match gemm_block's store semantics: logits pipelines force bf16
            // (K_OUT_BF16); others follow the toggleable arena dtype.
            if (col < N) {
                if (K_OUT_BF16) {
                    arena_store_bf16(y, (ulong)row * N + col, ec[0]);
                } else {
                    arena_store(y, (ulong)row * N + col, ec[0]);
                }
            }
            if (col + 1u < N) {
                if (K_OUT_BF16) {
                    arena_store_bf16(y, (ulong)row * N + col + 1u, ec[1]);
                } else {
                    arena_store(y, (ulong)row * N + col + 1u, ec[1]);
                }
            }
        }
    }
}
