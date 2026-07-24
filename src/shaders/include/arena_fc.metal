// FC9 = K_ARENA_F16 (activation-arena dtype: fp16 instead of bf16).
// Lives in its own header so arena.metal/dequant.metal can declare it without
// pulling in fc_axes.metal — the monolithic step shader declares its OWN
// constants at indices 1-3 and must not see fc_axes' globals.
#ifndef DGQ_ARENA_FC_METAL
#define DGQ_ARENA_FC_METAL

constant bool K_ARENA_F16_DEF [[function_constant(9)]];
constant bool K_ARENA_F16 =
    is_function_constant_defined(K_ARENA_F16_DEF) && K_ARENA_F16_DEF;

#endif
