#include <metal_stdlib>
using namespace metal;

#include "fc_axes.metal"
#include "common.metal"

// SC sparse softembed, pass 2: per row, soft[row][h] = sum_j prob[j]*embed[idx[j]][h]
// over the row's survivors (pass 1). f32 accumulate (matches the chunked path's
// precision); embed reads are coalesced across h. The prob carries the
// SC_PROB_GEMM_SCALE that f32_to_half_scale divides back out. One threadgroup
// (256 threads) per row; each thread owns hid columns h = tid, tid+256, ...
kernel void sc_sparse_gather(
    device const uint *idx [[buffer(0)]],     // [rows, maxk]
    device const half *prob [[buffer(1)]],    // [rows, maxk]
    device const uint *cnt [[buffer(2)]],     // [rows]
    device const uchar *blob [[buffer(3)]],
    constant ulong &embed_off [[buffer(4)]],
    device float *out [[buffer(5)]],          // [rows, hid] f32
    constant uint4 &params [[buffer(6)]],     // rows, hid, maxk, _
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    const uint row = tgid.x;
    const uint rows = params.x, hid = params.y, maxk = params.z;
    if (row >= rows) {
        return;
    }
    device const ushort *embed = (device const ushort *)(blob + embed_off);
    const uint n = min(cnt[row], maxk);

    for (uint h = lid.x; h < hid; h += 256u) {
        float acc = 0.f;
        for (uint j = 0u; j < n; ++j) {
            const uint v = idx[(ulong)row * maxk + j];
            const float p = float(prob[(ulong)row * maxk + j]);
            acc += p * bf16_to_f32(embed[(ulong)v * hid + h]);
        }
        out[(ulong)row * hid + h] = acc;
    }
}
