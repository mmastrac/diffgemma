#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_KERNEL_COMMON_METAL
#include "common.metal"
#endif
#ifndef DGQ_INCLUDE_MOE_ROUTER_METAL
#include "moe_router.metal"
#endif

/// Zero expert slot counts before bucketing.
kernel void moe_bucket_count(
    device RouteScratch *R [[buffer(0)]],
    constant uint &n_experts [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    if (K_SHAPE_ASSERT && n_experts == 0u) {
        return;
    }
    K_ELEMENTWISE_GUARD();
    if (i < n_experts && n_experts <= MOE_MAX_EXPERTS) {
        R->count[i] = 0u;
    }
}
