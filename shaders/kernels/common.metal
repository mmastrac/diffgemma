// Shared compile-time variant axes for isolated subkernels (STRATEGY.md §4).
// Indices 1–3 are uniform across all subkernel bodies; unused axes compile out.
//
//   1  K_SHAPE_ASSERT  bool   extra bounds checks (tier-2 debug)
//   2  K_DUMP_STAGE    uint   0 = off; >0 writes intermediates to buffer(5)
//   3  K_USE_FP4       bool   quant-format path (GEMM kernels; inert elsewhere)

#ifndef DGQ_KERNEL_COMMON_METAL
#define DGQ_KERNEL_COMMON_METAL

constant bool K_SHAPE_ASSERT [[function_constant(1)]];
constant uint K_DUMP_STAGE [[function_constant(2)]];
constant bool K_USE_FP4 [[function_constant(3)]];

#endif
