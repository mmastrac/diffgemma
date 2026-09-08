#include <metal_stdlib>
using namespace metal;

/// out[t, d] = decode(table[ids[t]], d) * embed_scale.
/// Table row is raw bf16 (\`raw\` = 1) or a q8 row (bf16 scale + int8 codes).
kernel void embed_gather(
    device const uint *blob [[buffer(0)]],
    device const uint *ids [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint2 &dims [[buffer(3)]],
    constant uint &w_off_lo [[buffer(4)]],
    constant uint &w_off_hi [[buffer(5)]],
    constant float &embed_scale [[buffer(6)]],
    constant uint &vocab [[buffer(7)]],
    constant uint &raw [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint hidden = dims.x;
    const uint num_tokens = dims.y;
    const ulong w_off = ((ulong)w_off_hi << 32) | (ulong)w_off_lo;
    const uint tok = gid / hidden;
    const uint d = gid % hidden;
    if (tok >= num_tokens) {
        return;
    }
    const uint id = ids[tok];
    if (id >= vocab) {
        return;
    }
    float v;
    if (raw != 0u) {
        device const ushort *row =
            (device const ushort *)((device const char *)blob + w_off) + (ulong)id * hidden;
        v = as_type<float>((uint)row[d] << 16);
    } else {
        device const uchar *row = (device const uchar *)blob + w_off + (ulong)id * (hidden + 2u);
        float scale = as_type<float>((uint)(row[0] | ((uint)row[1] << 8)) << 16);
        v = scale * (float)((int)(signed char)row[2 + d]);
    }
    out[(ulong)tok * hidden + d] = v * embed_scale;
}
