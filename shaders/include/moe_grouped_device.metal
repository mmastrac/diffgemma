#ifndef DGQ_INCLUDE_MOE_GROUPED_METAL
#define DGQ_INCLUDE_MOE_GROUPED_METAL

constant uint MOE_MAX_FF = 704u;
constant uint MOE_MAX_HIDDEN = 2816u;

struct MoeGroupedDims {
    uint canvas;
    uint hidden;
    uint moe_ff;
    uint n_experts;
};

// Legacy: float atomic_add via CAS was unreliable on MPS for MoE scatter (removed).
// MoE scatter now uses per-(tok,d) threadgroup reduction in moe_scatter_weighted.metal.

#endif
