// FC30/31: f32 hidden/stream plane access (DGQ_HIDDEN_F32). The residual
// accumulator planes (hidden, stream) switch from bf16 to f32 so the 30-layer
// residual accumulation keeps 23-bit mantissas; every other plane stays bf16.
// Defined-default false so pipelines compiled without these constants keep the
// exact existing (bf16) behavior — flag off is bit-identical.
//  K_HIDDEN_A_F32: the kernel's hidden/stream-plane INPUT operand is f32.
//  K_HIDDEN_Y_F32: the kernel's hidden/stream-plane OUTPUT operand is f32.
#ifndef DGQ_HIDDEN_FC_METAL
#define DGQ_HIDDEN_FC_METAL

constant bool K_HIDDEN_A_F32_DEF [[function_constant(30)]];
constant bool K_HIDDEN_A_F32 = is_function_constant_defined(K_HIDDEN_A_F32_DEF) ? K_HIDDEN_A_F32_DEF : false;
constant bool K_HIDDEN_Y_F32_DEF [[function_constant(31)]];
constant bool K_HIDDEN_Y_F32 = is_function_constant_defined(K_HIDDEN_Y_F32_DEF) ? K_HIDDEN_Y_F32_DEF : false;

#endif
