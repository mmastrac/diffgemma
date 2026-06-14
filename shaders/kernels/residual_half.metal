#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "common.metal"

kernel void residual_half(
    device const half *a [[buffer(0)]],
    device const half *b [[buffer(1)]],
    device half *y [[buffer(2)]],
    device const uchar *blob [[buffer(3)]],
    constant ulong &scal_off [[buffer(4)]],
    device float *dump [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
    float s = scal_off ? bf16_bytes(blob + scal_off) : 1.0f;
    float v = (float(a[i]) + float(b[i])) * s;
    if (K_DUMP_STAGE >= 1u) dump[i] = v;
    K_ELEMENTWISE_GUARD();
    y[i] = half(v);
}
