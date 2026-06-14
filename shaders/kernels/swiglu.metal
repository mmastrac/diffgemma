#include <metal_stdlib>
using namespace metal;

#ifndef DGQ_FC_AXES_METAL
#include "fc_axes.metal"
#endif
#ifndef DGQ_INCLUDE_ACTIVATIONS_METAL
#error "swiglu: bundle must include shaders/include/activations.metal"
#endif

constant uint K_IO_DTYPE [[function_constant(4)]];
constant bool K_GELU_GATE [[function_constant(5)]];
constant bool K_IN_PLACE [[function_constant(6)]];

/// Split-layout SwiGLU: separate gate/up buffers (decoder + monolith dense MLP).
/// buffer(0)=gate, buffer(1)=up, buffer(2)=out (ignored when in-place).
/// dims.x = len, dims.y = 0.
kernel void swiglu(
    device const float *gate_base [[buffer(0)]],
    device float *up_buf [[buffer(1)]],
    device float *out_buf [[buffer(2)]],
    constant uint2 &dims [[buffer(3)]],
    device float *dump [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    K_ELEMENTWISE_GUARD();
    if (K_SHAPE_ASSERT && !k_elem_dtype_valid(K_IO_DTYPE)) {
        return;
    }

    uint len = dims.x;
    if (K_SHAPE_ASSERT && len == 0u) {
        return;
    }
    if (gid >= len) {
        return;
    }

    if (K_IO_DTYPE == ELEM_HALF) {
        device const half *gate = (device const half *)gate_base;
        device const half *up = (device const half *)up_buf;
        device half *out = (device half *)out_buf;
        float g = float(gate[gid]);
        float u = float(up[gid]);
        if (K_GELU_GATE) {
            g = gelu_tanh(g);
        }
        float v = g * u;
        if (K_DUMP_STAGE >= 1u) {
            dump[gid] = v;
        }
        out[gid] = half(v);
        return;
    }

    device float *gate = (device float *)gate_base;
    float g = gate[gid];
    float u = up_buf[gid];
    if (K_GELU_GATE) {
        g = gelu_tanh(g);
    }
    if (K_DUMP_STAGE >= 1u) {
        dump[gid] = u;
    }
    float v = g * u;
    if (K_IN_PLACE) {
        gate[gid] = v;
    } else {
        out_buf[gid] = v;
    }
}
