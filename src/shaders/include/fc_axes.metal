// Global function-constant axes 1–3 (shared numbering convention; see build/manifest.toml).
//
//   1  K_SHAPE_ASSERT   bool   tier-2 bounds checks (legacy; implies fast asserts)
//   2  K_DUMP_STAGE     uint   0 = off; N = which intermediate to dump
//   3  K_QUANT_FORMAT   uint   0=q4_affine 1=q8 2=mxfp4 3=nvfp4
//   7  K_DEBUG_FAST     bool   shape/dim/index bounds (cheap; P3.7)
//   8  K_DEBUG_DEEP     bool   NaN/Inf scans (expensive; P3.7)
//   9  K_ARENA_F16      bool   activation-arena slots are fp16 instead of bf16
//                              (E11 fp16 prefill stream; unset/false = bf16)
//
// Per-kernel semantic axes (4+ except 7-9): declared in that kernel's entry file only.

#ifndef DGQ_FC_AXES_METAL
#define DGQ_FC_AXES_METAL

constant bool K_SHAPE_ASSERT [[function_constant(1)]];
constant uint K_DUMP_STAGE [[function_constant(2)]];
constant uint K_QUANT_FORMAT [[function_constant(3)]];
constant bool K_DEBUG_FAST [[function_constant(7)]];
constant bool K_DEBUG_DEEP [[function_constant(8)]];
#include "arena_fc.metal"

constant uint QUANT_Q4_AFFINE = 0u;
constant uint QUANT_Q8 = 1u;
constant uint QUANT_MXFP4 = 2u;
constant uint QUANT_NVFP4 = 3u;
constant uint QUANT_RAW = 4u;  // bf16 weights, no dequant (Raw/.dgq bf16 path)
constant uint QUANT_Q6 = 5u;   // affine int6 groups of 32 (q6_block, experts)

/// Element dtype for K_ELEM_DTYPE / K_IO_DTYPE axes (FC4+): 0=f32, 1=half.
constant uint ELEM_F32 = 0u;
constant uint ELEM_HALF = 1u;

/// Non-GEMM kernels: FC3 must stay q4_affine (inert). Fail loud under tier-2.
#define K_ELEMENTWISE_GUARD() \
    do { if (K_SHAPE_ASSERT && K_QUANT_FORMAT != QUANT_Q4_AFFINE) return; } while (0)

inline bool k_elem_dtype_valid(uint dtype) {
    return dtype <= ELEM_HALF;
}

#endif
