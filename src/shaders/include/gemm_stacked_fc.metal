// Stacked GEMM segment table baked as function constants (FC12+).
#ifndef DGQ_INCLUDE_GEMM_STACKED_FC_METAL
#define DGQ_INCLUDE_GEMM_STACKED_FC_METAL

#include "gemm_stacked.metal"

constant uint STACKED_N_SEGS [[function_constant(12)]];
constant uint STACKED_END0 [[function_constant(13)]];
constant uint STACKED_END1 [[function_constant(14)]];
constant uint STACKED_END2 [[function_constant(15)]];
constant ulong STACKED_W_OFF0 [[function_constant(16)]];
constant ulong STACKED_W_OFF1 [[function_constant(17)]];
constant ulong STACKED_W_OFF2 [[function_constant(18)]];
constant ulong STACKED_Y_BYTE_OFF0 [[function_constant(19)]];
constant ulong STACKED_Y_BYTE_OFF1 [[function_constant(20)]];
constant ulong STACKED_Y_BYTE_OFF2 [[function_constant(21)]];
constant uint STACKED_Y_COL0_0 [[function_constant(22)]];
constant uint STACKED_Y_COL0_1 [[function_constant(23)]];
constant uint STACKED_Y_COL0_2 [[function_constant(24)]];
constant uint STACKED_Y_ROW_COLS_0 [[function_constant(25)]];
constant uint STACKED_Y_ROW_COLS_1 [[function_constant(26)]];
constant uint STACKED_Y_ROW_COLS_2 [[function_constant(27)]];

inline ulong stacked_fc_w_off(uint seg) {
    if (seg == 0u) {
        return STACKED_W_OFF0;
    }
    if (seg == 1u) {
        return STACKED_W_OFF1;
    }
    return STACKED_W_OFF2;
}

inline ulong stacked_fc_y_byte_off(uint seg) {
    if (seg == 0u) {
        return STACKED_Y_BYTE_OFF0;
    }
    if (seg == 1u) {
        return STACKED_Y_BYTE_OFF1;
    }
    return STACKED_Y_BYTE_OFF2;
}

inline uint stacked_fc_y_col0(uint seg) {
    if (seg == 0u) {
        return STACKED_Y_COL0_0;
    }
    if (seg == 1u) {
        return STACKED_Y_COL0_1;
    }
    return STACKED_Y_COL0_2;
}

inline uint stacked_fc_y_row_cols(uint seg) {
    if (seg == 0u) {
        return STACKED_Y_ROW_COLS_0;
    }
    if (seg == 1u) {
        return STACKED_Y_ROW_COLS_1;
    }
    return STACKED_Y_ROW_COLS_2;
}

inline uint stacked_fc_seg_index(uint global_n) {
    if (STACKED_N_SEGS <= 1u) {
        return 0u;
    }
    if (global_n < STACKED_END0) {
        return 0u;
    }
    if (STACKED_N_SEGS == 2u) {
        return 1u;
    }
    if (global_n < STACKED_END1) {
        return 1u;
    }
    return 2u;
}

inline uint stacked_fc_local_n(uint global_n, uint seg) {
    if (seg == 0u) {
        return global_n;
    }
    if (seg == 1u) {
        return global_n - STACKED_END0;
    }
    return global_n - STACKED_END1;
}

#endif
