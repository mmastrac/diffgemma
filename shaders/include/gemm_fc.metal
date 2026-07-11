// GEMM compile-time shape axes (FC 4–6). FC 1–3 remain global (fc_axes.metal).
#ifndef DGQ_INCLUDE_GEMM_FC_METAL
#define DGQ_INCLUDE_GEMM_FC_METAL

#include "fc_axes.metal"

constant bool IS_FULL_LAYER [[function_constant(4)]];
constant uint GEMM_N [[function_constant(5)]];
constant uint GEMM_K [[function_constant(6)]];
/// Weight tile width along N (32 legacy, 128 full dequant occupancy). FC11.
constant uint GEMM_N_TILE [[function_constant(11)]];

/// FC29: force bf16 output. Set only
/// for the lm_head logits pipeline (f16 input, bf16-output logits).
constant bool K_OUT_BF16_DEF [[function_constant(29)]];
constant bool K_OUT_BF16 = is_function_constant_defined(K_OUT_BF16_DEF) ? K_OUT_BF16_DEF : false;

#endif
