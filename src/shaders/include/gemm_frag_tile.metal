// Steel-style per-lane fragment tiler — the shared MMA core for gemm_tunable
// (weight GEMM, task #19) and attention_gemm (E17 QK/PV). One BK=32-wide NT MMA
// phase: C[m][n] += sum_k A[m][k] * B[n][k], with A staged in Xs[m][k] and B
// staged TRANSPOSED in Ws[n][k]. 4 simdgroups in a 2x2 grid, 8x8 fragments,
// per-lane thread_elements() loads (never simdgroup_load from tgmem). Only the
// tile LOADERS differ between callers (packed weights vs strided arena Q / f16
// KV cache); this MMA core is identical, so it lives here once.
//
// The includer #defines FRAG_BM and FRAG_BN (the 64x64-style output tile, each
// a multiple of 16) BEFORE including; FRAG_PAD defaults to 40 (BK + 8-half bank
// pad). BK is fixed at 32 (q4 group-aligned). Include AFTER the includer's own
// tile constants are declared.
#ifndef DGQ_GEMM_FRAG_TILE_METAL
#define DGQ_GEMM_FRAG_TILE_METAL

#ifndef FRAG_BM
#error "gemm_frag_tile.metal: #define FRAG_BM before including"
#endif
#ifndef FRAG_BN
#error "gemm_frag_tile.metal: #define FRAG_BN before including"
#endif
#ifndef FRAG_PAD
#define FRAG_PAD 40
#endif
#define FRAG_BK 32u
#define FRAG_TM (FRAG_BM / 16u)
#define FRAG_TN (FRAG_BN / 16u)

// Steel lane->element coordinates: this lane owns fragment elements (fm, fn)
// and (fm, fn+1).
inline void frag_lane_coords(uint lane, thread uint &fm, thread uint &fn) {
    const uint qid = lane / 4u;
    fm = (qid & 4u) + ((lane / 2u) % 4u);
    fn = (qid & 2u) * 2u + (lane % 2u) * 2u;
}

inline void frag_mma_ktile(
    threadgroup half Xs[FRAG_BM][FRAG_PAD],
    threadgroup half Ws[FRAG_BN][FRAG_PAD],
    thread simdgroup_float8x8 (&C)[FRAG_TM][FRAG_TN],
    uint sr,
    uint sc,
    uint fm,
    uint fn
) {
    for (uint kk = 0u; kk < FRAG_BK; kk += 8u) {
        simdgroup_barrier(mem_flags::mem_none);
        // A fragments: element (fm, fn) of frag i = Xs[sr + 8i + fm][kk + fn].
        simdgroup_half8x8 A[FRAG_TM];
        for (uint i = 0u; i < FRAG_TM; ++i) {
            thread half2 &ea = reinterpret_cast<thread half2 &>(A[i].thread_elements());
            ea[0] = Xs[sr + 8u * i + fm][kk + fn];
            ea[1] = Xs[sr + 8u * i + fm][kk + fn + 1u];
        }
        simdgroup_barrier(mem_flags::mem_none);
        // B fragments: element (k=fm, n=fn) of frag j = Ws[sc + 8j + fn][kk + fm].
        simdgroup_half8x8 B[FRAG_TN];
        for (uint j = 0u; j < FRAG_TN; ++j) {
            thread half2 &eb = reinterpret_cast<thread half2 &>(B[j].thread_elements());
            eb[0] = Ws[sc + 8u * j + fn][kk + fm];
            eb[1] = Ws[sc + 8u * j + fn + 1u][kk + fm];
        }
        simdgroup_barrier(mem_flags::mem_none);
        for (uint i = 0u; i < FRAG_TM; ++i) {
            for (uint j = 0u; j < FRAG_TN; ++j) {
                simdgroup_multiply_accumulate(C[i][j], A[i], B[j], C[i][j]);
            }
        }
    }
}

#endif
