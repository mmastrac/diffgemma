#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#include "gemm_fc.metal"
#include "dequant.metal"
#include "arena.metal"
#include "gemm_stacked_fc.metal"

/// TUNABLE GEMM: fragment-level block GEMM.
///
/// The bench prototypes proved macro tiling alone cannot reach
/// the MPS/MLX 3.4-4.4 TF/s at our shapes; the delta is fragment-level
/// codegen.
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

// Shared NT fragment tiler (frag_mma_ktile + frag_lane_coords) — the same MMA
// core attention_gemm reuses; only the tile loaders differ per op.
#define FRAG_BM TUNE_BM
#define FRAG_BN TUNE_BN
#include "gemm_frag_tile.metal"

/// A-tile loader: vectorized 4-wide bf16(or f16)->half loads, per-4 row guard.
inline void tunable_load_a(
    device const ushort *x,
    threadgroup half Xs[TUNE_BM][40],
    uint m0,
    uint M,
    uint K,
    uint k0,
    uint ltid
) {
    for (uint i = ltid * 4u; i < BM * BK; i += 128u * 4u) {
        const uint mm = i / BK;
        const uint kk = i % BK;
        half4 h;
        if (m0 + mm < M) {
            const ushort4 u = *(device const ushort4 *)(x + (ulong)(m0 + mm) * K + k0 + kk);
            h = K_ARENA_F16 ? as_type<half4>(u) : half4(as_type<float4>(uint4(u) << 16u));
        } else {
            h = half4(0);
        }
        *(threadgroup half4 *)(&Xs[mm][kk]) = h;
    }
}

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
    const bool is_q8 = (K_QUANT_FORMAT == QUANT_Q8);
    const bool is_nvfp4 = (K_QUANT_FORMAT == QUANT_NVFP4);
    const ulong rowB = is_raw
        ? (ulong)K * 2ul
        : (is_q8 ? q8_row_bytes(K) : (is_nvfp4 ? nvfp4_row_bytes(K) : q4_row_bytes(K)));
    // nvfp4: 4-byte f32 global scale header at w_off, body at w_off+4.
    const ulong body = is_nvfp4 ? w_off + 4ul : w_off;
    const float nvfp4_gscale =
        is_nvfp4 ? as_type<float>(*(device const uint *)(blob + w_off)) : 0.0f;

    // Lane->element coordinates within an 8x8 fragment: this lane owns
    // elements (fm, fn) and (fm, fn+1).
    uint fm, fn;
    frag_lane_coords(lane, fm, fn);

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
        tunable_load_a(x, Xs, m0, M, K, k0, ltid);
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
                } else if (is_q8) {
                    // Same per-element math as gemm_block's q8_at path:
                    // half(float(int8) * f32_scale).
                    device const uchar *rb = blob + w_off + (ulong)gn * rowB;
                    const float sq8 = bf16_bytes(rb);
                    for (uint j = 0u; j < w_cols; ++j) {
                        const char qv =
                            *((device const char *)(rb + 2u + k0 + w_q + j));
                        Ws[w_r][w_q + j] = half(float(qv) * sq8);
                    }
                } else if (is_nvfp4) {
                    // 16-wide nvfp4 groups; range helper decodes by ABSOLUTE
                    // column (k0 + w_q + i). Bit-identical half(e2m1*scale) to
                    // gemm_block's dense nvfp4 path.
                    dequant_nvfp4_tile_half_fused_tg_range(
                        blob + body + (ulong)gn * rowB, K, k0,
                        &Ws[w_r][w_q], nvfp4_gscale, k0 + w_q, w_cols);
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

        frag_mma_ktile(Xs, Ws, C, sr, sc, fm, fn);
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

// ---------------------------------------------------------------------------
// Double-buffered variant: overlaps the K-tile device->tgmem load of tile N+1
// with the MMA of tile N. Same K-accumulation chain + dequant + store
// rounding as gemm_tunable -> BIT-EXACT expected.
//
// Metal has no async-copy primitive; the overlap comes from issuing the load
// instructions (which execute on the load/store units) before the MMA
// instructions (which execute on the matrix units), with a single barrier per
// K-tile instead of two. tgmem footprint doubles (Xs[2][BM][PAD] + Ws[2][BN]
// [PAD]) — fits M3's 32 KiB threadgroup limit at production tiles (64x64:
// 2*64*40*2 + 2*64*40*2 = 20480 bytes; 32x128: 2*32*40*2 + 2*128*40*2 = 25600
// bytes). Prototype only — wired to bench-gemm, not production.
// ---------------------------------------------------------------------------

/// A-tile loader, double-buffered slot variant.
inline void tunable_load_a_db(
    device const ushort *x,
    threadgroup half Xs[2][TUNE_BM][40],
    uint buf,
    uint m0,
    uint M,
    uint K,
    uint k0,
    uint ltid
) {
    for (uint i = ltid * 4u; i < BM * BK; i += 128u * 4u) {
        const uint mm = i / BK;
        const uint kk = i % BK;
        half4 h;
        if (m0 + mm < M) {
            const ushort4 u = *(device const ushort4 *)(x + (ulong)(m0 + mm) * K + k0 + kk);
            h = K_ARENA_F16 ? as_type<half4>(u) : half4(as_type<float4>(uint4(u) << 16u));
        } else {
            h = half4(0);
        }
        *(threadgroup half4 *)(&Xs[buf][mm][kk]) = h;
    }
}

/// Double-buffered dense GEMM. Same buffer layout + dispatch contract as
/// `gemm_tunable`; differs only in the K-loop scheduling.
kernel void gemm_tunable_db(
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
    threadgroup half Xs[2][BM][PAD];
    threadgroup half Ws[2][BN][PAD]; // row-major [n][k]

    const uint m0 = tgid.y * BM;
    const uint n0 = tgid.x * BN;
    const uint ltid = lid.x;

    const bool is_raw = (K_QUANT_FORMAT == QUANT_RAW);
    const bool is_q8 = (K_QUANT_FORMAT == QUANT_Q8);
    const bool is_nvfp4 = (K_QUANT_FORMAT == QUANT_NVFP4);
    const ulong rowB = is_raw
        ? (ulong)K * 2ul
        : (is_q8 ? q8_row_bytes(K) : (is_nvfp4 ? nvfp4_row_bytes(K) : q4_row_bytes(K)));
    const ulong body = is_nvfp4 ? w_off + 4ul : w_off;
    const float nvfp4_gscale =
        is_nvfp4 ? as_type<float>(*(device const uint *)(blob + w_off)) : 0.0f;

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

    const uint w_tpr = 128u / BN;
    const uint w_cols = BK / w_tpr;
    const uint w_r = ltid / w_tpr;
    const uint w_q = (ltid % w_tpr) * w_cols;

    // Prologue: load tile 0 into slot 0.
    tunable_load_a_db(x, Xs, 0u, m0, M, K, 0u, ltid);
    {
        const uint gn = n0 + w_r;
        if (gn < N) {
            if (is_raw) {
                device const ushort *wr =
                    (device const ushort *)(blob + w_off + (ulong)gn * rowB);
                for (uint j = 0u; j < w_cols; j += 4u) {
                    const ushort4 u = *(device const ushort4 *)(wr + 0u + w_q + j);
                    *(threadgroup half4 *)(&Ws[0][w_r][w_q + j]) =
                        half4(as_type<float4>(uint4(u) << 16u));
                }
            } else if (is_q8) {
                device const uchar *rb = blob + w_off + (ulong)gn * rowB;
                const float sq8 = bf16_bytes(rb);
                for (uint j = 0u; j < w_cols; ++j) {
                    const char qv = *((device const char *)(rb + 2u + 0u + w_q + j));
                    Ws[0][w_r][w_q + j] = half(float(qv) * sq8);
                }
            } else if (is_nvfp4) {
                dequant_nvfp4_tile_half_fused_tg_range(
                    blob + body + (ulong)gn * rowB, K, 0u,
                    &Ws[0][w_r][w_q], nvfp4_gscale, 0u + w_q, w_cols);
            } else {
                device const uchar *g =
                    blob + w_off + (ulong)gn * rowB + (ulong)(0u / 32u) * 20ul;
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
                    *(threadgroup half4 *)(&Ws[0][w_r][jj]) = s * q0 + half4(mn);
                    *(threadgroup half4 *)(&Ws[0][w_r][jj + 4u]) = s * q1 + half4(mn);
                }
            }
        } else {
            for (uint j = 0u; j < w_cols; j += 4u) {
                *(threadgroup half4 *)(&Ws[0][w_r][w_q + j]) = half4(0);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Main loop: overlap load of tile N+1 (slot `next`) with MMA of tile N
    // (slot `cur`). One barrier per K-tile (vs two in the single-buffered
    // kernel) — it waits for both the prev-tile MMA to finish consuming `cur`
    // AND the next-tile load to finish filling `next`, then swaps.
    uint cur = 0u;
    for (uint k0 = 0u; k0 < K; k0 += BK) {
        const uint k0_next = k0 + BK;
        const uint next = cur ^ 1u;
        if (k0_next < K) {
            tunable_load_a_db(x, Xs, next, m0, M, K, k0_next, ltid);
            {
                const uint gn = n0 + w_r;
                if (gn < N) {
                    if (is_raw) {
                        device const ushort *wr =
                            (device const ushort *)(blob + w_off + (ulong)gn * rowB);
                        for (uint j = 0u; j < w_cols; j += 4u) {
                            const ushort4 u = *(device const ushort4 *)(wr + k0_next + w_q + j);
                            *(threadgroup half4 *)(&Ws[next][w_r][w_q + j]) =
                                half4(as_type<float4>(uint4(u) << 16u));
                        }
                    } else if (is_q8) {
                        device const uchar *rb = blob + w_off + (ulong)gn * rowB;
                        const float sq8 = bf16_bytes(rb);
                        for (uint j = 0u; j < w_cols; ++j) {
                            const char qv = *((device const char *)(rb + 2u + k0_next + w_q + j));
                            Ws[next][w_r][w_q + j] = half(float(qv) * sq8);
                        }
                    } else if (is_nvfp4) {
                        dequant_nvfp4_tile_half_fused_tg_range(
                            blob + body + (ulong)gn * rowB, K, k0_next,
                            &Ws[next][w_r][w_q], nvfp4_gscale, k0_next + w_q, w_cols);
                    } else {
                        device const uchar *g =
                            blob + w_off + (ulong)gn * rowB + (ulong)(k0_next / 32u) * 20ul;
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
                            *(threadgroup half4 *)(&Ws[next][w_r][jj]) = s * q0 + half4(mn);
                            *(threadgroup half4 *)(&Ws[next][w_r][jj + 4u]) = s * q1 + half4(mn);
                        }
                    }
                } else {
                    for (uint j = 0u; j < w_cols; j += 4u) {
                        *(threadgroup half4 *)(&Ws[next][w_r][w_q + j]) = half4(0);
                    }
                }
            }
        }
        // MMA on the current tile (overlapped with the next-tile load above).
        frag_mma_ktile(Xs[cur], Ws[cur], C, sr, sc, fm, fn);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        cur = next;
    }

    // Per-lane direct store: identical to gemm_tunable.
    for (uint i = 0u; i < TM; ++i) {
        const uint row = m0 + sr + 8u * i + fm;
        if (row >= M) {
            continue;
        }
        for (uint j = 0u; j < TN; ++j) {
            const uint col = n0 + sc + 8u * j + fn;
            const thread float2 &ec = reinterpret_cast<thread float2 &>(C[i][j].thread_elements());
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

/// Stacked tunable GEMM: segment table via FC12-27 (same contract as
/// gemm_block_stacked) — qkv and dense gate/up fused dispatches. Raw + q4
/// weight formats (production stacked paths); per-element segment resolve on
/// W load and store. Same accumulation chain as gemm_block_stacked ->
/// BIT-EXACT expected.
kernel void gemm_tunable_stacked(
    device const ushort *x [[buffer(0)]],
    device ushort *y_arena [[buffer(1)]],
    device const uchar *blob [[buffer(2)]],
    constant uint &M [[buffer(3)]],
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
    const bool is_nvfp4 = (K_QUANT_FORMAT == QUANT_NVFP4);
    const ulong rowB = is_raw
        ? (ulong)K * 2ul
        : (is_nvfp4 ? nvfp4_row_bytes(K) : q4_row_bytes(K));

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

    const uint w_tpr = 128u / BN;
    const uint w_cols = BK / w_tpr;
    const uint w_r = ltid / w_tpr;
    const uint w_q = (ltid % w_tpr) * w_cols;
    // Segment resolve for this thread's W row (row-constant across K).
    const uint gn = n0 + w_r;
    const uint w_seg = stacked_fc_seg_index(gn);
    const uint w_local_n = stacked_fc_local_n(gn, w_seg);
    const ulong w_base = stacked_fc_w_off(w_seg);
    // nvfp4: each segment matrix carries a 4-byte f32 global scale header at
    // its w_base; the row body starts at w_base+4.
    const float w_gscale =
        is_nvfp4 ? as_type<float>(*(device const uint *)(blob + w_base)) : 0.0f;
    const ulong w_body = is_nvfp4 ? w_base + 4ul : w_base;

    for (uint k0 = 0u; k0 < K; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tunable_load_a(x, Xs, m0, M, K, k0, ltid);
        {
            if (gn < N) {
                if (is_raw) {
                    device const ushort *wr =
                        (device const ushort *)(blob + w_base + (ulong)w_local_n * rowB);
                    for (uint j = 0u; j < w_cols; j += 4u) {
                        const ushort4 u = *(device const ushort4 *)(wr + k0 + w_q + j);
                        *(threadgroup half4 *)(&Ws[w_r][w_q + j]) =
                            half4(as_type<float4>(uint4(u) << 16u));
                    }
                } else if (is_nvfp4) {
                    dequant_nvfp4_tile_half_fused_tg_range(
                        blob + w_body + (ulong)w_local_n * rowB, K, k0,
                        &Ws[w_r][w_q], w_gscale, k0 + w_q, w_cols);
                } else {
                    device const uchar *g =
                        blob + w_base + (ulong)w_local_n * rowB + (ulong)(k0 / 32u) * 20ul;
                    const half s = half(bf16_bytes(g));
                    const half mn = half(bf16_bytes(g + 2));
                    for (uint j8 = 0u; j8 < w_cols; j8 += 8u) {
                        const uint jj = w_q + j8;
                        device const uchar *pp = g + 4u + jj / 2u;
                        const uint v = uint(pp[0]) | (uint(pp[1]) << 8u) |
                            (uint(pp[2]) << 16u) | (uint(pp[3]) << 24u);
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
        frag_mma_ktile(Xs, Ws, C, sr, sc, fm, fn);
    }

    // Per-lane store with per-element segment resolve (cols may straddle a
    // segment boundary; y offsets/strides are per segment).
    for (uint i = 0u; i < TM; ++i) {
        const uint row = m0 + sr + 8u * i + fm;
        if (row >= M) {
            continue;
        }
        for (uint j = 0u; j < TN; ++j) {
            const thread float2 &ec = reinterpret_cast<thread float2 &>(C[i][j].thread_elements());
            for (uint e = 0u; e < 2u; ++e) {
                const uint col = n0 + sc + 8u * j + fn + e;
                if (col >= N) {
                    continue;
                }
                const uint seg = stacked_fc_seg_index(col);
                const uint local_n = stacked_fc_local_n(col, seg);
                const ulong out_idx = (ulong)row * (ulong)stacked_fc_y_row_cols(seg)
                    + (ulong)stacked_fc_y_col0(seg) + (ulong)local_n;
                device ushort *dst =
                    (device ushort *)((device uchar *)y_arena + stacked_fc_y_byte_off(seg));
                arena_store(dst, out_idx, e == 0u ? ec[0] : ec[1]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Block-sparse MoE variant.
// ---------------------------------------------------------------------------
#include "qgemm_grouped.metal"
#include "moe_router_device.metal"

// Fused gather (FC28, same contract as gemm_block_sparse): A rows come from
// bf16 `moein` via route->token_list instead of a pre-gathered f32 buffer.
constant bool TUNE_GATHER_A_DEF [[function_constant(28)]];
constant bool TUNE_GATHER_A =
    is_function_constant_defined(TUNE_GATHER_A_DEF) ? TUNE_GATHER_A_DEF : false;

/// Block-sparse tunable GEMM: one <=32-row expert block x one BN-wide N-tile
/// per threadgroup (grid = (N/BN, num_blocks) via indirect slots 4/5).
/// Compile with TUNE_BM=32 (block height is fixed by moe_bucket_fill).
/// Fragment-native adaptive-M: simdgroups/fragments covering rows beyond the
/// block's actual row count skip their loads and MMAs entirely (bit-exact —
/// dead fragments are never stored; barriers remain TG-uniform).
/// Same per-output accumulation chain + dequant + f32_round_bf16 store as
/// gemm_block_sparse -> BIT-EXACT expected.
kernel void gemm_tunable_sparse(
    device const float *a [[buffer(0)]],
    device const uchar *blob [[buffer(1)]],
    device float *c [[buffer(2)]],
    device const GroupedJob *jobs [[buffer(3)]],
    device const uint *row_starts [[buffer(4)]],
    constant uint &num_jobs [[buffer(5)]],
    device const RouteScratch *route [[buffer(6)]],
    device const ushort *moein [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint N = GEMM_N, K = GEMM_K;
    threadgroup half Xs[BM][PAD];
    threadgroup half Ws[BN][PAD];

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
    const bool is_nvfp4 = (K_QUANT_FORMAT == QUANT_NVFP4);
    const bool is_q6 = (K_QUANT_FORMAT == QUANT_Q6);
    const ulong w_off = jobs[e].w_byte_off;
    // nvfp4 rows carry a 4-byte f32 global scale header at w_off; the row body
    // (packed codes + per-group fp8 scales) starts at w_off+4. q4/q6 have no
    // header (body == w_off). Same layout resolve as gemm_block_sparse.
    const ulong body = is_nvfp4 ? w_off + 4ul : w_off;
    const float nvfp4_gscale =
        is_nvfp4 ? as_type<float>(*(device const uint *)(blob + w_off)) : 0.0f;
    const ulong rowB =
        is_nvfp4 ? nvfp4_row_bytes(K) : (is_q6 ? q6_row_bytes(K) : q4_row_bytes(K));
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
    // NOTE: no runtime adaptive-M here — runtime-bounded fragment loops break
    // compile-time unrolling and spill the accumulator array (measured 2x
    // slower). Dead rows are zero-filled and simply never stored; the MMA
    // padding cost is immaterial (the adaptive-M wash proved it).

    const uint w_tpr = 128u / BN;
    const uint w_cols = BK / w_tpr;
    const uint w_r = ltid / w_tpr;
    const uint w_q = (ltid % w_tpr) * w_cols;

    for (uint k0 = 0u; k0 < K; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // A tile: gather bf16 moein rows via token_list, or the f32 slot
        // buffer (down GEMM). Rows >= M_tile zero-filled (their fragments are
        // skipped, but keep the tile defined).
        for (uint i = ltid * 4u; i < BM * BK; i += 128u * 4u) {
            const uint mm = i / BK;
            const uint kk = i % BK;
            half4 h;
            if (mm < M_tile) {
                if (TUNE_GATHER_A) {
                    const uint tok = route->token_list[row0 + mm];
                    const ushort4 u =
                        *(device const ushort4 *)(moein + (ulong)tok * K + k0 + kk);
                    h = K_ARENA_F16 ? as_type<half4>(u)
                                    : half4(as_type<float4>(uint4(u) << 16u));
                } else {
                    const float4 f =
                        *(device const float4 *)(a + (ulong)(row0 + mm) * K + k0 + kk);
                    h = half4(f);
                }
            } else {
                h = half4(0);
            }
            *(threadgroup half4 *)(&Xs[mm][kk]) = h;
        }
        // W tile: expert q4/q6/nvfp4 rows.
        {
            const uint gn = n0 + w_r;
            if (gn < N) {
                if (is_nvfp4) {
                    // nvfp4 groups are 16-wide (vs 32 for q4/q6); the range
                    // helper decodes by ABSOLUTE column (k0 + w_q + i), so it
                    // resolves the per-column 16-group + fp8 scale itself.
                    // Same half(e2m1*scale) math as gemm_block_sparse's
                    // full-tile decode -> bit-identical.
                    dequant_nvfp4_tile_half_fused_tg_range(
                        blob + body + (ulong)gn * rowB, K, k0,
                        &Ws[w_r][w_q], nvfp4_gscale, k0 + w_q, w_cols);
                } else if (is_q6) {
                    dequant_q6_group_half_tg_range(
                        blob + body + (ulong)gn * rowB + (ulong)(k0 / 32u) * 28ul,
                        &Ws[w_r][w_q], w_q, w_cols);
                } else {
                    device const uchar *g =
                        blob + body + (ulong)gn * rowB + (ulong)(k0 / 32u) * 20ul;
                    const half s = half(bf16_bytes(g));
                    const half mn = half(bf16_bytes(g + 2));
                    for (uint j8 = 0u; j8 < w_cols; j8 += 8u) {
                        const uint jj = w_q + j8;
                        device const uchar *pp = g + 4u + jj / 2u;
                        const uint v = uint(pp[0]) | (uint(pp[1]) << 8u) |
                            (uint(pp[2]) << 16u) | (uint(pp[3]) << 24u);
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

        frag_mma_ktile(Xs, Ws, C, sr, sc, fm, fn);
    }

    // Store: f32 slot rows, bf16-rounded values (matches gemm_block_sparse).
    for (uint i = 0u; i < TM; ++i) {
        const uint mm = sr + 8u * i + fm;
        if (mm >= M_tile) {
            continue;
        }
        for (uint j = 0u; j < TN; ++j) {
            const thread float2 &ec = reinterpret_cast<thread float2 &>(C[i][j].thread_elements());
            const uint col = n0 + sc + 8u * j + fn;
            if (col < N) {
                c[(ulong)(row0 + mm) * N + col] = arena_round_f32(ec[0]);
            }
            if (col + 1u < N) {
                c[(ulong)(row0 + mm) * N + col + 1u] = arena_round_f32(ec[1]);
            }
        }
    }
}
