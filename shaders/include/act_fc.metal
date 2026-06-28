// FC9: activation arena precision. Standalone (not via fc_axes) so arena.metal /
// dequant.metal can include it without pulling fc_axes' FC1-3, which collide with
// the monolithic step shader's own FC1-3 numbering. Defaults false (bf16) when
// unset so compile_kernel (no function constants) stays bf16. True → f16
// (10-mantissa). Logits/large-range buffers use the explicit *_bf16 path.
#ifndef DGQ_ACT_FC_METAL
#define DGQ_ACT_FC_METAL

constant bool K_ACT_F16_DEF [[function_constant(9)]];
constant bool K_ACT_F16 = is_function_constant_defined(K_ACT_F16_DEF) ? K_ACT_F16_DEF : false;

#endif
