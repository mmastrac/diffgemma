#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "debug_status.metal"
#include "moe_router_device.metal"
#include "moe_grouped_dispatch.metal"

/// Bucketing phases 0/1/2 (monolith k_bucket_fill).
kernel void moe_bucket_fill(
    device RouteScratch *R [[buffer(0)]],
    constant uint &phase [[buffer(1)]],
    constant RouterDims &dims [[buffer(2)]],
    device uint *layer_expert_unique [[buffer(3)]],
    constant uint &layer_idx [[buffer(4)]],
    device DebugStatus *dbg [[buffer(5)]],
    device MoeGroupedIndirectGrid *indirect [[buffer(6)]],
    constant MoeGroupedGridInfo &grid_info [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
    if (K_SHAPE_ASSERT && (dims.canvas == 0u || dims.top_k == 0u || dims.n_experts == 0u)) {
        return;
    }
    K_ELEMENTWISE_GUARD();

    if (phase == 0u) {
        uint slots = dims.canvas * dims.top_k;
        if (i < slots) {
            uint tok = i / dims.top_k;
            uint kk = i % dims.top_k;
            uint e = R->expert[tok][kk];
            dgq_assert_index(dbg, DbgKernelMoeBucketFill, e, dims.n_experts);
            atomic_fetch_add_explicit(
                (device atomic_uint *)&R->count[R->expert[tok][kk]],
                1u,
                memory_order_relaxed);
        }
    } else if (phase == 1u) {
        if (i == 0u) {
            uint s = 0u;
            uint used = 0u;
            uint ai = 0u;
            uint blk = 0u;
            // Block-sparse tile height: 32 in steady state; the wide
            // weight-stationary height during batched prefill (must match
            // the consuming GEMM's TUNE_BM).
            const uint bm = (dims.block_m != 0u) ? dims.block_m : 32u;
            for (uint e = 0u; e < dims.n_experts; ++e) {
                const uint c = R->count[e];
                if (c > 0u) {
                    R->active_expert[ai] = e;
                    ai++;
                    used++;
                    // Block-sparse: one <=bm-row tile per chunk of this expert.
                    const uint nb = (c + bm - 1u) / bm;
                    for (uint b = 0u; b < nb; ++b) {
                        R->block_expert[blk] = e;
                        R->block_row0[blk] = s + b * bm;
                        blk++;
                    }
                }
                R->row_start[e] = s;
                s += c;
                R->count[e] = 0u;
            }
            R->row_start[dims.n_experts] = s;
            R->num_slots = s;
            R->num_active_experts = used;
            R->num_blocks = blk;
            layer_expert_unique[layer_idx] = used;
            if (indirect != nullptr) {
                const uint n_tile = max(grid_info.n_tile, 1u);
                const uint gate_w = (grid_info.gate_n + n_tile - 1u) / n_tile;
                const uint down_w = (grid_info.hid + n_tile - 1u) / n_tile;
                indirect[0].threadgroups_per_grid[0] = gate_w;
                indirect[0].threadgroups_per_grid[1] = used;
                indirect[0].threadgroups_per_grid[2] = 1u;
                indirect[1].threadgroups_per_grid[0] = down_w;
                indirect[1].threadgroups_per_grid[1] = used;
                indirect[1].threadgroups_per_grid[2] = 1u;
                // Slots 2/3: block-sparse variants (height = num_blocks).
                indirect[2].threadgroups_per_grid[0] = gate_w;
                indirect[2].threadgroups_per_grid[1] = blk;
                indirect[2].threadgroups_per_grid[2] = 1u;
                indirect[3].threadgroups_per_grid[0] = down_w;
                indirect[3].threadgroups_per_grid[1] = blk;
                indirect[3].threadgroups_per_grid[2] = 1u;
                // Slots 4/5: tunable block-sparse (BN-wide N-tiles). The
                // wide-block (E1) pipelines are compiled at a different BN
                // than the default 32-row ones — the width must match the
                // pipeline the block list's height selects.
                const uint tn_sel =
                    (bm == 32u) ? grid_info.tunable_n_tile : grid_info.tunable_wide_n_tile;
                const uint tn = max(tn_sel, 1u);
                indirect[4].threadgroups_per_grid[0] = (grid_info.gate_n + tn - 1u) / tn;
                indirect[4].threadgroups_per_grid[1] = blk;
                indirect[4].threadgroups_per_grid[2] = 1u;
                indirect[5].threadgroups_per_grid[0] = (grid_info.hid + tn - 1u) / tn;
                indirect[5].threadgroups_per_grid[1] = blk;
                indirect[5].threadgroups_per_grid[2] = 1u;
            }
        }
    } else {
        uint slots = dims.canvas * dims.top_k;
        if (i < slots) {
            uint tok = i / dims.top_k;
            uint kk = i % dims.top_k;
            uint e = R->expert[tok][kk];
            dgq_assert_index(dbg, DbgKernelMoeBucketFill, e, dims.n_experts);
            uint slot = R->row_start[e]
                + atomic_fetch_add_explicit(
                      (device atomic_uint *)&R->count[e], 1u, memory_order_relaxed);
            R->token_list[slot] = tok;
            R->slot_list[slot] = kk;
            R->token_slot[tok][kk] = slot;
        }
    }
}
