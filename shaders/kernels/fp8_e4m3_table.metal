#include <metal_stdlib>
using namespace metal;

#include "dequant.metal"

/// Decode FP8 E4M3 codes to f32 (tier-1 table parity vs CPU oracle).
kernel void fp8_e4m3_table(
    device const uchar *codes [[buffer(0)]],
    constant uint &n [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= n) {
        return;
    }
    out[i] = fp8_e4m3_to_f32(codes[i]);
}
