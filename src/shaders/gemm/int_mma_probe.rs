//! Raw-throughput microbench: half tensor-MMA vs int8 4-wide dot-product vs f32
//! scalar FMA on M3 (does Apple have a fast integer MAC path, or is the matrix
//! unit strictly better?). Settles empirically whether an int8/int4 dot-product
//! MoE expert GEMM could beat the current dequant-to-half + simdgroup-MMA path,
//! rather than deferring to MLX's dequant-to-half choice (evidence, not proof).
//!
//! Each kernel runs a long compute-bound register loop (no memory traffic in the
//! loop; a data-dependent tail write defeats dead-code elimination). GMAC/s =
//! macs_per_iter * iters * lanes / seconds.

#[cfg(target_os = "macos")]
pub const SRC: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// 8-way ILP: 8 INDEPENDENT accumulator chains hide instruction latency so the
// loop measures peak THROUGHPUT, not the (serial) latency of one dependency
// chain. MAC counts below multiply by ILP=8.
#define ILP 8

// Half tensor-MMA: 8x8x8 = 512 MAC / instruction / simdgroup.
kernel void bench_mma(device float *out [[buffer(0)]],
                      constant uint &iters [[buffer(1)]],
                      uint gid [[thread_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    simdgroup_half8x8 A[ILP];
    simdgroup_half8x8 B = simdgroup_half8x8(1.0009765625h);
    simdgroup_float8x8 C[ILP];
    for (uint j = 0u; j < ILP; ++j) {
        A[j] = simdgroup_half8x8(half(lane + j) * 1e-3h + 1.0h);
        C[j] = simdgroup_float8x8(0.0f);
    }
    for (uint i = 0u; i < iters; ++i) {
        for (uint j = 0u; j < ILP; ++j) {
            simdgroup_multiply_accumulate(C[j], A[j], B, C[j]);
        }
    }
    float s = 0.0f;
    for (uint j = 0u; j < ILP; ++j) {
        thread float2 &e = reinterpret_cast<thread float2 &>(C[j].thread_elements());
        s += e[0] + e[1];
    }
    if (gid == 0xffffffffu) out[0] = s;  // never taken
}

// int8 4-wide dot: int32 += char4 . char4 = 4 MAC / instruction / lane.
kernel void bench_int8(device int *out [[buffer(0)]],
                       constant uint &iters [[buffer(1)]],
                       uint gid [[thread_position_in_grid]]) {
    char4 a[ILP];
    char4 b = char4(5, 6, 7, 8);
    int acc[ILP];
    for (uint j = 0u; j < ILP; ++j) { a[j] = char4(1 + j, 2, 3, 4); acc[j] = 0; }
    for (uint i = 0u; i < iters; ++i) {
        for (uint j = 0u; j < ILP; ++j) {
            acc[j] += int(a[j].x) * int(b.x) + int(a[j].y) * int(b.y)
                    + int(a[j].z) * int(b.z) + int(a[j].w) * int(b.w);
            a[j] += char4(char(acc[j] & 1));   // defeat strength-reduction
        }
    }
    int s = 0;
    for (uint j = 0u; j < ILP; ++j) s += acc[j];
    if (gid == 0xffffffffu) out[0] = s;
}

// f32 scalar FMA reference: 1 MAC / instruction / lane (non-tensor baseline).
kernel void bench_fma(device float *out [[buffer(0)]],
                      constant uint &iters [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    float a[ILP], acc[ILP];
    float b = 0.9999f;
    for (uint j = 0u; j < ILP; ++j) { a[j] = 1.0001f + float(j) * 1e-4f; acc[j] = 0.0f; }
    for (uint i = 0u; i < iters; ++i) {
        for (uint j = 0u; j < ILP; ++j) {
            acc[j] = fma(a[j], b, acc[j]);
            a[j] = a[j] * 1.0000001f + 1e-30f;
        }
    }
    float s = 0.0f;
    for (uint j = 0u; j < ILP; ++j) s += acc[j];
    if (gid == 0xffffffffu) out[0] = s;
}
"#;

/// Run the three throughput probes; returns (mma_gmacs, int8_gmacs, fma_gmacs).
#[cfg(target_os = "macos")]
pub fn run() -> Result<(f64, f64, f64), crate::Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use crate::shaders::variant::KernelVariant;
    use objc2_metal::{
        MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
    };
    use std::time::Instant;

    let ctx = MetalContext::new()?;
    let mut pool = BufferPool::new();
    let out = pool
        .allocate(&ctx.device, 64)
        .ok_or(crate::Error::Gpu("alloc out"))?;

    // Full-occupancy grid: many threadgroups of 256 threads.
    let tg = 256u64;
    let tgs = 512u64; // threadgroups (full occupancy, bounded dispatch time)
    let threads = tg * tgs;
    let simdgroups = threads / 32;
    let iters: u32 = 20_000;

    let bench = |entry: &str, macs_per_iter_per_unit: f64, units: f64| -> Result<f64, crate::Error> {
        let pipe = ctx.compile_subkernel(SRC, entry, KernelVariant::PRODUCTION)?;
        let grid = MTLSize { width: threads as usize, height: 1, depth: 1 };
        let tgd = MTLSize { width: tg as usize, height: 1, depth: 1 };
        let mut best = f64::INFINITY;
        for round in 0..5 {
            let cmd = ctx.queue.commandBuffer().ok_or(crate::Error::Gpu("cmd"))?;
            let enc = cmd.computeCommandEncoder().ok_or(crate::Error::Gpu("enc"))?;
            enc.setComputePipelineState(&pipe.pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&out), 0, 0);
            }
            crate::shaders::gpu_common::set_bytes(&enc, &iters, 1);
            let t = Instant::now();
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tgd);
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            if round > 0 {
                best = best.min(t.elapsed().as_secs_f64());
            }
        }
        // total MACs = macs_per_iter_per_unit * iters * units
        let macs = macs_per_iter_per_unit * iters as f64 * units;
        Ok(macs / best / 1e9) // GMAC/s
    };

    // ILP=8 independent chains. MMA: 512 MAC/iter/chain per SIMDGROUP; int8/fma:
    // 4 / 1 MAC/iter/chain per LANE(thread).
    let ilp = 8.0;
    let mma = bench("bench_mma", 512.0 * ilp, simdgroups as f64)?;
    let int8 = bench("bench_int8", 4.0 * ilp, threads as f64)?;
    let fma = bench("bench_fma", 1.0 * ilp, threads as f64)?;
    Ok((mma, int8, fma))
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    /// Run: `cargo test --release int_mma_throughput -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn int_mma_throughput() {
        let (mma, int8, fma) = super::run().expect("probe");
        println!("\n=== M3 raw compute throughput (GMAC/s) ===");
        println!("  half tensor-MMA : {mma:10.1}  (simdgroup_multiply_accumulate)");
        println!("  int8 4-wide dot : {int8:10.1}  (int32 += char4.char4)");
        println!("  f32 scalar FMA  : {fma:10.1}  (reference, non-tensor)");
        println!("  --> int8/MMA = {:.2}x ; int8/FMA = {:.2}x", int8 / mma, int8 / fma);
        println!(
            "  VERDICT: int8 dot-product {} the tensor-MMA for GEMM-shaped work.",
            if int8 >= mma { "MATCHES/BEATS" } else { "LOSES TO" }
        );
    }
}
