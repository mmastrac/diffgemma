// Shared *index numbering* for function constants 1–3. Each kernel translation
// unit re-declares these via #include (a numbering convention, not shared state).
//
//   1  K_SHAPE_ASSERT   bool   tier-2 bounds checks
//   2  K_DUMP_STAGE     uint   0 = off; N = which intermediate to dump
//   3  K_QUANT_FORMAT   uint   0=q4_affine 1=q8 2=mxfp4 3=nvfp4
//
// Per-kernel semantic axes (4+): declared in that kernel's .metal entry file only.
// See build/kernel_manifest.toml for the full index map and valid tuples.

#ifndef DGQ_KERNEL_COMMON_METAL
#define DGQ_KERNEL_COMMON_METAL

constant bool K_SHAPE_ASSERT [[function_constant(1)]];
constant uint K_DUMP_STAGE [[function_constant(2)]];
constant uint K_QUANT_FORMAT [[function_constant(3)]];

constant uint QUANT_Q4_AFFINE = 0u;
constant uint QUANT_Q8 = 1u;
constant uint QUANT_MXFP4 = 2u;
constant uint QUANT_NVFP4 = 3u;

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
