#ifndef DGQ_INCLUDE_MOE_GROUPED_DISPATCH_METAL
#define DGQ_INCLUDE_MOE_GROUPED_DISPATCH_METAL

#include <metal_stdlib>
using namespace metal;

/// `dispatchThreadgroupsWithIndirectBuffer` payload (3×uint32 per grid dimension).
struct MoeGroupedIndirectGrid {
    uint threadgroups_per_grid[3];
};

struct MoeGroupedGridInfo {
    uint gate_n;
    uint hid;
    uint n_tile;
    uint tpg;
};

#endif
