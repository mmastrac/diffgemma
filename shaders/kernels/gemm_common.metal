// Compile-time GEMM specialization (monolith indices 1–3).
#ifndef DGQ_KERNEL_GEMM_COMMON_METAL
#define DGQ_KERNEL_GEMM_COMMON_METAL

constant bool IS_FULL_LAYER [[function_constant(1)]];
constant uint GEMM_N [[function_constant(2)]];
constant uint GEMM_K [[function_constant(3)]];

#endif
