#ifndef DGQ_INCLUDE_MOE_ROUTER_METAL
#define DGQ_INCLUDE_MOE_ROUTER_METAL

constant uint MOE_MAX_CANVAS = 256u;
constant uint MOE_MAX_TOP_K = 8u;
constant uint MOE_MAX_EXPERTS = 128u;
constant float MOE_ROUTER_RMS_EPS = 1e-6f;

struct RouteScratch {
    half weight[256][8];
    uint expert[256][8];
    uint count[128];
    /// Per-expert slot starts; `row_start[128]` == `num_slots` after bucketing.
    uint row_start[129];
    uint num_slots;
    uint num_active_experts;
    uint active_expert[128];
    uint token_list[2048];
    uint slot_list[2048];
    /// Inverse map: `token_slot[tok][kk]` → flat slot index in `token_list`.
    uint token_slot[256][8];
    /// Block-sparse MoE GEMM: one entry per <=32-row tile (built in bucket_fill
    /// phase 1). block_expert[b] = expert; block_row0[b] = global row start.
    uint block_expert[256];
    uint block_row0[256];
    uint num_blocks;
};

struct RouterDims {
    uint canvas;
    uint hidden;
    uint n_experts;
    uint top_k;
    float router_hscale;
};

#endif
