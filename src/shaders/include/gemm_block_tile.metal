// Tiled block GEMM geometry (FC11 GEMM_N_TILE: 32 or 128).
#ifndef DGQ_GEMM_BLOCK_TILE_METAL
#define DGQ_GEMM_BLOCK_TILE_METAL

#include "gemm_fc.metal"

constant uint GEMM_M_TILE = 32u;
constant uint GEMM_M_TILE_MAX = GEMM_M_TILE;
constant uint GEMM_K_TILE = 32u;
constant uint GEMM_N_TILE_MAX = 128u;

/// M16: simdgroups 0/1 take rows 0-7/8-15 over N columns 0-63; simdgroups 2/3
/// take the same rows over columns 64-127 (accs 0-7 per simdgroup).
/// M8: every simdgroup takes rows 0-7 over its own 32-column strip
/// (accs 0-3 per simdgroup). Per-output K-accumulation order is identical to
/// the M32 mapping, so results are bit-identical; only the simdgroup->work
/// assignment changes.
inline void gemm_block_mma_k32_m16(
    threadgroup half tx[GEMM_M_TILE_MAX][GEMM_K_TILE],
    threadgroup half tw[GEMM_N_TILE_MAX][GEMM_K_TILE],
    uint sgid,
    thread simdgroup_float8x8 &acc0,
    thread simdgroup_float8x8 &acc1,
    thread simdgroup_float8x8 &acc2,
    thread simdgroup_float8x8 &acc3,
    thread simdgroup_float8x8 &acc4,
    thread simdgroup_float8x8 &acc5,
    thread simdgroup_float8x8 &acc6,
    thread simdgroup_float8x8 &acc7
) {
    const uint arow = 8u * (sgid & 1u);
    const uint cbase = 64u * (sgid >> 1u);
    for (uint kk = 0u; kk < GEMM_K_TILE; kk += 8u) {
        simdgroup_half8x8 a;
        simdgroup_load(a, &tx[arow][kk], GEMM_K_TILE);
        if (GEMM_N_TILE > cbase + 0u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[cbase + 0u][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc0, a, b, acc0);
        }
        if (GEMM_N_TILE > cbase + 8u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[cbase + 8u][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc1, a, b, acc1);
        }
        if (GEMM_N_TILE > cbase + 16u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[cbase + 16u][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc2, a, b, acc2);
        }
        if (GEMM_N_TILE > cbase + 24u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[cbase + 24u][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc3, a, b, acc3);
        }
        if (GEMM_N_TILE > cbase + 32u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[cbase + 32u][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc4, a, b, acc4);
        }
        if (GEMM_N_TILE > cbase + 40u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[cbase + 40u][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc5, a, b, acc5);
        }
        if (GEMM_N_TILE > cbase + 48u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[cbase + 48u][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc6, a, b, acc6);
        }
        if (GEMM_N_TILE > cbase + 56u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[cbase + 56u][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc7, a, b, acc7);
        }
    }
}

inline void gemm_block_mma_k32_m8(
    threadgroup half tx[GEMM_M_TILE_MAX][GEMM_K_TILE],
    threadgroup half tw[GEMM_N_TILE_MAX][GEMM_K_TILE],
    uint sgid,
    thread simdgroup_float8x8 &acc0,
    thread simdgroup_float8x8 &acc1,
    thread simdgroup_float8x8 &acc2,
    thread simdgroup_float8x8 &acc3
) {
    const uint cbase = 32u * sgid;
    for (uint kk = 0u; kk < GEMM_K_TILE; kk += 8u) {
        simdgroup_half8x8 a;
        simdgroup_load(a, &tx[0u][kk], GEMM_K_TILE);
        if (GEMM_N_TILE > cbase + 0u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[cbase + 0u][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc0, a, b, acc0);
        }
        if (GEMM_N_TILE > cbase + 8u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[cbase + 8u][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc1, a, b, acc1);
        }
        if (GEMM_N_TILE > cbase + 16u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[cbase + 16u][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc2, a, b, acc2);
        }
        if (GEMM_N_TILE > cbase + 24u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[cbase + 24u][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc3, a, b, acc3);
        }
    }
}

inline void gemm_block_mma_k32(
    threadgroup half tx[GEMM_M_TILE_MAX][GEMM_K_TILE],
    threadgroup half tw[GEMM_N_TILE_MAX][GEMM_K_TILE],
    uint sgid,
    thread simdgroup_float8x8 &acc0,
    thread simdgroup_float8x8 &acc1,
    thread simdgroup_float8x8 &acc2,
    thread simdgroup_float8x8 &acc3,
    thread simdgroup_float8x8 &acc4,
    thread simdgroup_float8x8 &acc5,
    thread simdgroup_float8x8 &acc6,
    thread simdgroup_float8x8 &acc7,
    thread simdgroup_float8x8 &acc8,
    thread simdgroup_float8x8 &acc9,
    thread simdgroup_float8x8 &acc10,
    thread simdgroup_float8x8 &acc11,
    thread simdgroup_float8x8 &acc12,
    thread simdgroup_float8x8 &acc13,
    thread simdgroup_float8x8 &acc14,
    thread simdgroup_float8x8 &acc15
) {
    for (uint kk = 0u; kk < GEMM_K_TILE; kk += 8u) {
        simdgroup_half8x8 a;
        simdgroup_load(a, &tx[8u * sgid][kk], GEMM_K_TILE);
        if (GEMM_N_TILE > 0u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[0][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc0, a, b, acc0);
        }
        if (GEMM_N_TILE > 8u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[8][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc1, a, b, acc1);
        }
        if (GEMM_N_TILE > 16u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[16][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc2, a, b, acc2);
        }
        if (GEMM_N_TILE > 24u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[24][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc3, a, b, acc3);
        }
        if (GEMM_N_TILE > 32u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[32][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc4, a, b, acc4);
        }
        if (GEMM_N_TILE > 40u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[40][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc5, a, b, acc5);
        }
        if (GEMM_N_TILE > 48u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[48][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc6, a, b, acc6);
        }
        if (GEMM_N_TILE > 56u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[56][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc7, a, b, acc7);
        }
        if (GEMM_N_TILE > 64u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[64][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc8, a, b, acc8);
        }
        if (GEMM_N_TILE > 72u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[72][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc9, a, b, acc9);
        }
        if (GEMM_N_TILE > 80u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[80][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc10, a, b, acc10);
        }
        if (GEMM_N_TILE > 88u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[88][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc11, a, b, acc11);
        }
        if (GEMM_N_TILE > 96u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[96][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc12, a, b, acc12);
        }
        if (GEMM_N_TILE > 104u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[104][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc13, a, b, acc13);
        }
        if (GEMM_N_TILE > 112u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[112][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc14, a, b, acc14);
        }
        if (GEMM_N_TILE > 120u) {
            simdgroup_half8x8 b;
            simdgroup_load(b, &tw[120][kk], GEMM_K_TILE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc15, a, b, acc15);
        }
    }
}

inline void gemm_block_mma_store_m16(
    threadgroup float ty[GEMM_M_TILE_MAX][GEMM_N_TILE_MAX],
    uint sgid,
    thread const simdgroup_float8x8 &acc0,
    thread const simdgroup_float8x8 &acc1,
    thread const simdgroup_float8x8 &acc2,
    thread const simdgroup_float8x8 &acc3,
    thread const simdgroup_float8x8 &acc4,
    thread const simdgroup_float8x8 &acc5,
    thread const simdgroup_float8x8 &acc6,
    thread const simdgroup_float8x8 &acc7
) {
    const uint arow = 8u * (sgid & 1u);
    const uint cbase = 64u * (sgid >> 1u);
    if (GEMM_N_TILE > cbase + 0u) {
        simdgroup_store(acc0, &ty[arow][cbase + 0u], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > cbase + 8u) {
        simdgroup_store(acc1, &ty[arow][cbase + 8u], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > cbase + 16u) {
        simdgroup_store(acc2, &ty[arow][cbase + 16u], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > cbase + 24u) {
        simdgroup_store(acc3, &ty[arow][cbase + 24u], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > cbase + 32u) {
        simdgroup_store(acc4, &ty[arow][cbase + 32u], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > cbase + 40u) {
        simdgroup_store(acc5, &ty[arow][cbase + 40u], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > cbase + 48u) {
        simdgroup_store(acc6, &ty[arow][cbase + 48u], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > cbase + 56u) {
        simdgroup_store(acc7, &ty[arow][cbase + 56u], GEMM_N_TILE_MAX);
    }
}

inline void gemm_block_mma_store_m8(
    threadgroup float ty[GEMM_M_TILE_MAX][GEMM_N_TILE_MAX],
    uint sgid,
    thread const simdgroup_float8x8 &acc0,
    thread const simdgroup_float8x8 &acc1,
    thread const simdgroup_float8x8 &acc2,
    thread const simdgroup_float8x8 &acc3
) {
    const uint cbase = 32u * sgid;
    if (GEMM_N_TILE > cbase + 0u) {
        simdgroup_store(acc0, &ty[0u][cbase + 0u], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > cbase + 8u) {
        simdgroup_store(acc1, &ty[0u][cbase + 8u], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > cbase + 16u) {
        simdgroup_store(acc2, &ty[0u][cbase + 16u], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > cbase + 24u) {
        simdgroup_store(acc3, &ty[0u][cbase + 24u], GEMM_N_TILE_MAX);
    }
}

inline void gemm_block_mma_store(
    threadgroup float ty[GEMM_M_TILE_MAX][GEMM_N_TILE_MAX],
    uint sgid,
    thread const simdgroup_float8x8 &acc0,
    thread const simdgroup_float8x8 &acc1,
    thread const simdgroup_float8x8 &acc2,
    thread const simdgroup_float8x8 &acc3,
    thread const simdgroup_float8x8 &acc4,
    thread const simdgroup_float8x8 &acc5,
    thread const simdgroup_float8x8 &acc6,
    thread const simdgroup_float8x8 &acc7,
    thread const simdgroup_float8x8 &acc8,
    thread const simdgroup_float8x8 &acc9,
    thread const simdgroup_float8x8 &acc10,
    thread const simdgroup_float8x8 &acc11,
    thread const simdgroup_float8x8 &acc12,
    thread const simdgroup_float8x8 &acc13,
    thread const simdgroup_float8x8 &acc14,
    thread const simdgroup_float8x8 &acc15
) {
    if (GEMM_N_TILE > 0u) {
        simdgroup_store(acc0, &ty[8u * sgid][0], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > 8u) {
        simdgroup_store(acc1, &ty[8u * sgid][8], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > 16u) {
        simdgroup_store(acc2, &ty[8u * sgid][16], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > 24u) {
        simdgroup_store(acc3, &ty[8u * sgid][24], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > 32u) {
        simdgroup_store(acc4, &ty[8u * sgid][32], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > 40u) {
        simdgroup_store(acc5, &ty[8u * sgid][40], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > 48u) {
        simdgroup_store(acc6, &ty[8u * sgid][48], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > 56u) {
        simdgroup_store(acc7, &ty[8u * sgid][56], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > 64u) {
        simdgroup_store(acc8, &ty[8u * sgid][64], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > 72u) {
        simdgroup_store(acc9, &ty[8u * sgid][72], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > 80u) {
        simdgroup_store(acc10, &ty[8u * sgid][80], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > 88u) {
        simdgroup_store(acc11, &ty[8u * sgid][88], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > 96u) {
        simdgroup_store(acc12, &ty[8u * sgid][96], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > 104u) {
        simdgroup_store(acc13, &ty[8u * sgid][104], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > 112u) {
        simdgroup_store(acc14, &ty[8u * sgid][112], GEMM_N_TILE_MAX);
    }
    if (GEMM_N_TILE > 120u) {
        simdgroup_store(acc15, &ty[8u * sgid][120], GEMM_N_TILE_MAX);
    }
}

inline void gemm_block_zero_tw_row(threadgroup half tw[GEMM_N_TILE_MAX][GEMM_K_TILE], uint r) {
    for (uint i = 0u; i < GEMM_K_TILE; ++i) {
        tw[r][i] = half(0);
    }
}

/// Adaptive-M dispatch (GEMM_M_ADAPT): `tile` is the threadgroup-uniform
/// runtime M-mapping (8/16/32) chosen from the block's actual row count. The
/// branch is uniform per threadgroup, so there is no divergence; register
/// allocation is that of the 32-row path (identical to the legacy kernel).
inline void gemm_block_mma_k32_adapt(
    uint tile,
    threadgroup half tx[GEMM_M_TILE_MAX][GEMM_K_TILE],
    threadgroup half tw[GEMM_N_TILE_MAX][GEMM_K_TILE],
    uint sgid,
    thread simdgroup_float8x8 &acc0,
    thread simdgroup_float8x8 &acc1,
    thread simdgroup_float8x8 &acc2,
    thread simdgroup_float8x8 &acc3,
    thread simdgroup_float8x8 &acc4,
    thread simdgroup_float8x8 &acc5,
    thread simdgroup_float8x8 &acc6,
    thread simdgroup_float8x8 &acc7,
    thread simdgroup_float8x8 &acc8,
    thread simdgroup_float8x8 &acc9,
    thread simdgroup_float8x8 &acc10,
    thread simdgroup_float8x8 &acc11,
    thread simdgroup_float8x8 &acc12,
    thread simdgroup_float8x8 &acc13,
    thread simdgroup_float8x8 &acc14,
    thread simdgroup_float8x8 &acc15
) {
    if (tile == 8u) {
        gemm_block_mma_k32_m8(tx, tw, sgid, acc0, acc1, acc2, acc3);
    } else if (tile == 16u) {
        gemm_block_mma_k32_m16(tx, tw, sgid, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);
    } else {
        gemm_block_mma_k32(
            tx, tw, sgid, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10,
            acc11, acc12, acc13, acc14, acc15);
    }
}

inline void gemm_block_mma_store_adapt(
    uint tile,
    threadgroup float ty[GEMM_M_TILE_MAX][GEMM_N_TILE_MAX],
    uint sgid,
    thread const simdgroup_float8x8 &acc0,
    thread const simdgroup_float8x8 &acc1,
    thread const simdgroup_float8x8 &acc2,
    thread const simdgroup_float8x8 &acc3,
    thread const simdgroup_float8x8 &acc4,
    thread const simdgroup_float8x8 &acc5,
    thread const simdgroup_float8x8 &acc6,
    thread const simdgroup_float8x8 &acc7,
    thread const simdgroup_float8x8 &acc8,
    thread const simdgroup_float8x8 &acc9,
    thread const simdgroup_float8x8 &acc10,
    thread const simdgroup_float8x8 &acc11,
    thread const simdgroup_float8x8 &acc12,
    thread const simdgroup_float8x8 &acc13,
    thread const simdgroup_float8x8 &acc14,
    thread const simdgroup_float8x8 &acc15
) {
    if (tile == 8u) {
        gemm_block_mma_store_m8(ty, sgid, acc0, acc1, acc2, acc3);
    } else if (tile == 16u) {
        gemm_block_mma_store_m16(ty, sgid, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);
    } else {
        gemm_block_mma_store(
            ty, sgid, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11,
            acc12, acc13, acc14, acc15);
    }
}

#endif
