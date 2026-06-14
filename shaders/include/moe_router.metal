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
    uint offset[128];
    uint num_slots;
    uint _pad_route;
    uint token_list[2048];
    uint slot_list[2048];
};

struct RouterDims {
    uint canvas;
    uint hidden;
    uint n_experts;
    uint top_k;
    float router_hscale;
};

#endif
