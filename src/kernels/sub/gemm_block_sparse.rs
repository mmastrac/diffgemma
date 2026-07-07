//! Block-sparse (megablocks-style) grouped MoE GEMM: one <=32-row tile per
//! threadgroup, driven by `route->block_expert/block_row0` (built in
//! `moe_bucket_fill` phase 1). Same math as `gemm_block_grouped`, different
//! work decomposition; production-wired behind `DGQ_MOE_BLOCK_SPARSE`.

use super::QuantFormat;
use crate::safetensors::Error;

pub const ENTRY: &str = "gemm_block_sparse";

const SHADER: &str = shader_include::include_metal!("kernels/gemm_block_sparse.metal");

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
    format: QuantFormat,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_gemm_subkernel(SHADER, ENTRY, n, k, false, format as u32, false)
}

/// Fused-gather variant (GATHER_A): A-load gathers bf16 `moein` rows via
/// `route->token_list` (buffer 7) instead of a pre-gathered f32 buffer. Used for
/// MoE gate_up to skip the separate gather pass.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_for_gather(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
    format: QuantFormat,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_gemm_subkernel_gather(SHADER, ENTRY, n, k, format as u32)
}

/// Adaptive-M variant (FC32, DGQ_MOE_TILE_ADAPT): each block runs the smallest
/// simdgroup M-mapping (8/16/32 rows) covering its actual row count.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_for_adaptive(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
    format: QuantFormat,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_gemm_subkernel_adaptive(SHADER, ENTRY, n, k, format as u32)
}

/// Adaptive-M + fused-gather (GATHER_A) variant for MoE gate_up.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_for_gather_adaptive(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
    format: QuantFormat,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_gemm_subkernel_gather_adaptive(SHADER, ENTRY, n, k, format as u32)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};

/// Run the block-sparse kernel on a grouped fixture with CPU-built blocks
/// (bucket_fill's legacy emission). `adaptive` selects the GEMM_M_ADAPT
/// pipeline (runtime per-block M-mapping); same single dispatch either way.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu_sparse(
    f: &crate::kernels::sub::gemm_linear_grouped::Fixture,
    adaptive: bool,
) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use crate::metal::{BlockGroupedJob, RouteScratch};

    let ctx = MetalContext::new()?;
    let mut pool = BufferPool::new();
    let jobs = f.jobs();
    let w_blob = f.w_blob();
    let out_len = f.out_len();
    let num_jobs = f.num_jobs();

    let mut route = {
        let mut r: RouteScratch = unsafe { std::mem::zeroed() };
        for (i, &rs) in f.row_starts.iter().enumerate().take(num_jobs + 1) {
            r.row_start[i] = rs;
        }
        r.num_active_experts = num_jobs as u32;
        for e in 0..num_jobs {
            r.active_expert[e] = e as u32;
        }
        r
    };
    // Mirror bucket_fill phase 1 block emission (one <=32-row tile per chunk).
    let counts: Vec<u32> = (0..num_jobs)
        .map(|e| f.row_starts[e + 1] - f.row_starts[e])
        .collect();
    {
        let mut blk = 0usize;
        for (e, &c) in counts.iter().enumerate() {
            let base = f.row_starts[e];
            for b in 0..c.div_ceil(32) {
                route.block_expert[blk] = e as u32;
                route.block_row0[blk] = base + b * 32;
                blk += 1;
            }
        }
        route.num_blocks = blk as u32;
    }

    let buf_a = pool
        .allocate(&ctx.device, f.a.len() * 4)
        .ok_or(Error::Format("alloc"))?;
    let buf_w = pool
        .allocate(&ctx.device, w_blob.len())
        .ok_or(Error::Format("alloc"))?;
    let buf_c = pool
        .allocate(&ctx.device, out_len * 4)
        .ok_or(Error::Format("alloc"))?;
    let buf_jobs = pool
        .allocate(
            &ctx.device,
            jobs.len() * std::mem::size_of::<BlockGroupedJob>(),
        )
        .ok_or(Error::Format("alloc"))?;
    let buf_rs = pool
        .allocate(&ctx.device, f.row_starts.len() * 4)
        .ok_or(Error::Format("alloc"))?;
    let buf_route = pool
        .allocate(&ctx.device, std::mem::size_of::<RouteScratch>())
        .ok_or(Error::Format("alloc"))?;
    BufferPool::write_f32(&buf_a, &f.a);
    BufferPool::write_bytes(&buf_w, &w_blob);
    BufferPool::write_bytes(&buf_jobs, unsafe {
        std::slice::from_raw_parts(
            jobs.as_ptr().cast::<u8>(),
            jobs.len() * std::mem::size_of::<BlockGroupedJob>(),
        )
    });
    BufferPool::write_bytes(&buf_rs, unsafe {
        std::slice::from_raw_parts(f.row_starts.as_ptr().cast::<u8>(), f.row_starts.len() * 4)
    });
    BufferPool::write_bytes(&buf_route, unsafe {
        std::slice::from_raw_parts(
            &route as *const RouteScratch as *const u8,
            std::mem::size_of::<RouteScratch>(),
        )
    });
    BufferPool::write_f32(&buf_c, &vec![0.0f32; out_len]);

    let n_tiles =
        crate::kernels::sub::gemm_common::div_up(f.n, crate::kernels::sub::gemm_common::n_tile());
    let tg = MTLSize {
        width: crate::kernels::sub::gemm_common::THREADS_PER_TG,
        height: 1,
        depth: 1,
    };
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    let pipeline = if adaptive {
        pipeline_for_adaptive(&ctx, f.n as u32, f.k as u32, f.format)?
    } else {
        pipeline_for(&ctx, f.n as u32, f.k as u32, f.format)?
    };
    enc.setComputePipelineState(&pipeline.pipeline);
    crate::kernels::sub::gemm_block_grouped::bind_gpu_buffers(
        &enc,
        &buf_a,
        &buf_w,
        &buf_c,
        &buf_jobs,
        &buf_rs,
        &buf_route,
        num_jobs as u32,
    );
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n_tiles,
            height: route.num_blocks as usize,
            depth: 1,
        },
        tg,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut out = vec![0.0f32; out_len];
    BufferPool::read_f32(&buf_c, &mut out);
    Ok(out)
}

#[cfg(test)]
#[cfg(all(feature = "metal", target_os = "macos"))]
mod tests {
    use super::*;
    use crate::kernels::sub::gemm_linear_grouped::grouped_fixture;
    use crate::kernels::sub::test_util::assert_oracle;

    /// Ragged per-expert counts covering every M-tile class boundary: C8 tails
    /// (1, 6, 8), C16 tails (9, 16), promoted C32 tails (17, 31), full chunks
    /// with tiny tails (33, 40), multi-chunk (100), and an inactive expert (0).
    const COUNTS: &[usize] = &[1, 6, 8, 9, 16, 17, 31, 33, 40, 0, 100];

    /// Adaptive-M (8/16/32 runtime mapping) output must be BIT-identical to
    /// the legacy fixed-32 kernel: same per-output K-accumulation chain, only
    /// the simdgroup->work mapping differs. Any mismatch is a bug.
    #[test]
    fn adaptive_bitwise_matches_legacy_q4() {
        let f = grouped_fixture(QuantFormat::Q4Affine, 64, 192, COUNTS);
        let legacy = gpu_sparse(&f, false).expect("legacy sparse");
        let adaptive = gpu_sparse(&f, true).expect("adaptive sparse");
        assert_eq!(legacy.len(), adaptive.len());
        for (i, (a, b)) in legacy.iter().zip(adaptive.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "output {} differs: legacy {} adaptive {}",
                i,
                a,
                b
            );
        }
        // Sanity vs the grouped CPU oracle (same tolerances as grouped tests).
        let oracle = crate::kernels::sub::gemm_block_grouped::cpu(&f);
        assert_oracle(&legacy, &oracle, 0.05, 0.999);
    }

    #[test]
    fn adaptive_bitwise_matches_legacy_nvfp4() {
        let f = grouped_fixture(QuantFormat::NvFp4, 64, 192, COUNTS);
        let legacy = gpu_sparse(&f, false).expect("legacy sparse");
        let adaptive = gpu_sparse(&f, true).expect("adaptive sparse");
        for (i, (a, b)) in legacy.iter().zip(adaptive.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "output {} differs", i);
        }
    }
}
