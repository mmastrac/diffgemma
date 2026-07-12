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
    /// N-tile width of the tunable sparse pipelines (indirect slots 4/5).
    uint tunable_n_tile;
    /// N-tile width of the wide-block (block_m != 32) tunable sparse
    /// pipelines (they pin BN=64; the default-BM pipelines use BN=128).
    uint tunable_wide_n_tile;
};

#endif
