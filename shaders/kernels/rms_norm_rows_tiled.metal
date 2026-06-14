#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_FC_AXES_METAL
#include "fc_axes.metal"
#endif
#ifndef DGQ_INCLUDE_COMMON_METAL
#error "rms_norm_rows_tiled: bundle must include shaders/include/common.metal"
#endif

constant uint K_IN_DTYPE [[function_constant(4)]];

constant float RMS_EPS = 1e-6f;

/// Threadgroup RMSNorm (monolith): f32 or half in, half out, optional bf16 weight blob.
kernel void rms_norm_rows_tiled(
    device const float *x_base [[buffer(0)]],
    device half *y [[buffer(1)]],
    device const uchar *blob [[buffer(2)]],
    constant ulong &w_off [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    device float *dump [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    K_ELEMENTWISE_GUARD();
    if (K_SHAPE_ASSERT && !k_elem_dtype_valid(K_IN_DTYPE)) {
        return;
    }

    threadgroup float red[8];
    float acc = 0.f;
    if (K_IN_DTYPE == ELEM_F32) {
        device const float *xr = x_base + (ulong)row * dim;
        for (uint i = lid; i < dim; i += tpg) {
            float v = xr[i];
            acc += v * v;
        }
    } else {
        device const half *xh = (device const half *)x_base;
        device const half *xr = xh + (ulong)row * dim;
        for (uint i = lid; i < dim; i += tpg) {
            float v = float(xr[i]);
            acc += v * v;
        }
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
    for (uint i = lid; i < dim; i += tpg) {
        float v;
        if (K_IN_DTYPE == ELEM_F32) {
            device const float *xr = x_base + (ulong)row * dim;
            v = xr[i] * inv;
        } else {
            device const half *xh = (device const half *)x_base;
            device const half *xr = xh + (ulong)row * dim;
            v = float(xr[i]) * inv;
        }
        if (w_off != 0) v *= bf16_bytes(blob + w_off + 2ul * i);
        y[(ulong)row * dim + i] = half(v);
    }
}
