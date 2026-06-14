#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_KERNEL_COMMON_METAL
#include "common.metal"
#endif
#ifndef DGQ_KERNEL_BF16_METAL
#include "bf16.metal"
#endif

constant float RMS_EPS = 1e-6f;

kernel void rmsnorm_half(
    device const half *x [[buffer(0)]],
    device half *y [[buffer(1)]],
    device const uchar *blob [[buffer(2)]],
    constant ulong &w_off [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    device float *dump [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    threadgroup float red[8];
    device const half *xr = x + (ulong)row * dim;
    float acc = 0.f;
    for (uint i = lid; i < dim; i += tpg) {
        float v = float(xr[i]);
        acc += v * v;
    }
    acc = simd_sum(acc);
    uint sg = lid / 32, nsg = (tpg + 31) / 32;
    if ((lid & 31) == 0) red[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        float t = 0.f;
        for (uint i = 0; i < nsg; ++i) t += red[i];
        red[0] = rsqrt(t / float(dim) + RMS_EPS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = red[0];
    if (K_DUMP_STAGE >= 1u && lid == 0) dump[row] = inv;
    (void)K_USE_FP4;
    for (uint i = lid; i < dim; i += tpg) {
        float v = float(xr[i]) * inv;
        if (w_off != 0) v *= bf16_bytes(blob + w_off + 2ul * i);
        y[(ulong)row * dim + i] = half(v);
    }
}
