//! Tunable GEMM (task #19): fragment-level block GEMM with
//! per-lane thread_elements() loads and vectorized loaders; tile geometry via
//! TUNE_BM/TUNE_BN #define prepend. BIT-EXACT vs gemm_block (same ascending-K
//! accumulation chain, dequant math, and store rounding) — verified per
//! element in bench-gemm across all production shapes (q4 + raw).
//! Production wiring behind `DGQ_GEMM_TUNABLE` starts with the Raw (bf16
//! weight) plain path; 64x64 won every production shape in the sweep.

use super::QuantFormat;
use crate::safetensors::Error;

pub const ENTRY: &str = "gemm_tunable";

pub const SHADER: &str = shader_include::include_metal!("kernels/gemm_tunable.metal");

/// Production tile config: 64x64 won every swept production shape.
pub const TUNE_BM: usize = 64;
pub const TUNE_BN: usize = 64;

pub fn tuned_source(bm: usize, bn: usize) -> String {
    format!("#define TUNE_BM {bm}\n#define TUNE_BN {bn}\n{SHADER}")
}

#[cfg(target_os = "macos")]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
    format: QuantFormat,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_gemm_subkernel(
        &tuned_source(TUNE_BM, TUNE_BN),
        ENTRY,
        n,
        k,
        false,
        format as u32,
        false,
    )
}

/// Logits variant: forces bf16 output (FC29) — lm_head.
#[cfg(target_os = "macos")]
pub fn pipeline_for_logits(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
    format: QuantFormat,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_gemm_subkernel_out_bf16(&tuned_source(TUNE_BM, TUNE_BN), ENTRY, n, k, format as u32)
}

pub const ENTRY_STACKED: &str = "gemm_tunable_stacked";

/// Stacked variant (segment table FC12-27, same contract as
/// gemm_block_stacked): qkv + dense gate/up fused dispatches. Cached per
/// (n, k, format, segment table) like the legacy stacked pipeline.
#[cfg(target_os = "macos")]
pub fn stacked_pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
    format: QuantFormat,
    segs: &[crate::kernels::sub::gemm_block_stacked::GemmStackedSeg],
) -> Result<std::sync::Arc<crate::metal::device::ComputePipeline>, Error> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    type Key = crate::kernels::sub::gemm_block_stacked::StackedPipelineKey;

    static CACHE: OnceLock<
        Mutex<HashMap<Key, std::sync::Arc<crate::metal::device::ComputePipeline>>>,
    > = OnceLock::new();
    let stacked = crate::kernels::sub::gemm_block_stacked::StackedSegFc::from_segments(segs)?;
    let key = Key::new(n, k, format, stacked);
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .map_err(|_| Error::Format("tunable stacked pipeline cache poisoned"))?;
    if let Some(pipe) = guard.get(&key) {
        return Ok(std::sync::Arc::clone(pipe));
    }
    let pipe = std::sync::Arc::new(ctx.compile_gemm_stacked_subkernel(
        &tuned_source(TUNE_BM, TUNE_BN),
        ENTRY_STACKED,
        n,
        k,
        format as u32,
        &stacked,
    )?);
    guard.insert(key, std::sync::Arc::clone(&pipe));
    Ok(pipe)
}

/// Sparse (block-sparse MoE) tile config: BM fixed at the 32-row block height
/// baked into moe_bucket_fill. BN=128 since 2026-07-07 (`bench-gemm --shapes
/// sparse`, within-run A/B at the prefill distribution r64: gate_up 3.33 →
/// 3.52 TF/s, down 3.36 → 3.63; denoise r16 neutral-to-+2%). Both MoE N dims
/// (1408, 2816) divide 128 exactly; same 32-row block list; bit-identical
/// (N-tile width never touches the per-output K-chain).
pub const SPARSE_BM: usize = 32;
pub const SPARSE_BN: usize = 128;

pub const ENTRY_SPARSE: &str = "gemm_tunable_sparse";

/// Block-sparse variant (q4/q6 experts); `gather` sets GATHER_A (FC28) for
/// the fused-gather gate_up A-load.
#[cfg(target_os = "macos")]
pub fn pipeline_for_sparse(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
    gather: bool,
    format: QuantFormat,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    pipeline_for_sparse_bm(ctx, n, k, gather, format, SPARSE_BM)
}

/// Tile-parameterized sparse variant. `bm` != 32 is the weight-stationary
/// prefill experiment (DGQ_MOE_PREFILL_BM — requires the block list to be
/// built at the same height via RouterDims.block_m; disproven as perf, kept
/// opt-in). `bn` != 64 changes only the N-tile width (same 32-row block
/// list; dispatch grid width must use the same bn).
#[cfg(target_os = "macos")]
pub fn pipeline_for_sparse_tile(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
    gather: bool,
    format: QuantFormat,
    bm: usize,
    bn: usize,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    let src = tuned_source(bm, bn);
    if gather {
        ctx.compile_gemm_subkernel_gather(&src, ENTRY_SPARSE, n, k, format as u32)
    } else {
        ctx.compile_gemm_subkernel(&src, ENTRY_SPARSE, n, k, false, format as u32, false)
    }
}

#[cfg(target_os = "macos")]
pub fn pipeline_for_sparse_bm(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
    gather: bool,
    format: QuantFormat,
    bm: usize,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    // The wide-block (E1) variants pin BN=64: bm=64 x BN=128 would be 32
    // accumulator fragments (64 f32/lane) — the known register-spill regime.
    let bn = if bm == 32 { SPARSE_BN } else { 64 };
    pipeline_for_sparse_tile(ctx, n, k, gather, format, bm, bn)
}

/// Run `gemm_tunable_sparse` on a grouped MoE fixture with CPU-built 32-row
/// blocks (bucket_fill phase-1 mirror), non-gather (A from the f32 buffer).
/// Self-contained (no dependency on the legacy gemm_block_sparse harness) so it
/// survives that kernel's retirement. Used by the nvfp4 port oracle.
#[cfg(all(test, target_os = "macos"))]
pub(crate) fn gpu_sparse_tunable(
    f: &crate::kernels::sub::gemm_linear_grouped::Fixture,
) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use crate::metal::{BlockGroupedJob, RouteScratch};
    use objc2_metal::{
        MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
    };

    let ctx = MetalContext::new()?;
    let mut pool = BufferPool::new();
    let jobs = f.jobs();
    let w_blob = f.w_blob();
    let out_len = f.out_len();
    let num_jobs = f.num_jobs();

    let mut route: Box<RouteScratch> = unsafe { Box::new(std::mem::zeroed()) };
    for (i, &rs) in f.row_starts.iter().enumerate().take(num_jobs + 1) {
        route.row_start[i] = rs;
    }
    route.num_active_experts = num_jobs as u32;
    for e in 0..num_jobs {
        route.active_expert[e] = e as u32;
    }
    // 32-row block emission (SPARSE_BM), identical to bucket_fill phase 1.
    let mut blk = 0usize;
    for e in 0..num_jobs {
        let base = f.row_starts[e];
        let count = f.row_starts[e + 1] - f.row_starts[e];
        for b in 0..(count as usize).div_ceil(SPARSE_BM) {
            route.block_expert[blk] = e as u32;
            route.block_row0[blk] = base + (b * SPARSE_BM) as u32;
            blk += 1;
        }
    }
    route.num_blocks = blk as u32;

    let buf_a = pool
        .allocate(&ctx.device, f.a.len() * 4)
        .ok_or(Error::Format("alloc a"))?;
    let buf_w = pool
        .allocate(&ctx.device, w_blob.len())
        .ok_or(Error::Format("alloc w"))?;
    let buf_c = pool
        .allocate(&ctx.device, out_len * 4)
        .ok_or(Error::Format("alloc c"))?;
    let buf_jobs = pool
        .allocate(
            &ctx.device,
            jobs.len() * std::mem::size_of::<BlockGroupedJob>(),
        )
        .ok_or(Error::Format("alloc jobs"))?;
    let buf_rs = pool
        .allocate(&ctx.device, f.row_starts.len() * 4)
        .ok_or(Error::Format("alloc rs"))?;
    let buf_route = pool
        .allocate(&ctx.device, std::mem::size_of::<RouteScratch>())
        .ok_or(Error::Format("alloc route"))?;
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
            route.as_ref() as *const RouteScratch as *const u8,
            std::mem::size_of::<RouteScratch>(),
        )
    });
    BufferPool::write_f32(&buf_c, &vec![0.0f32; out_len]);

    let pipe = pipeline_for_sparse_tile(
        &ctx,
        f.n as u32,
        f.k as u32,
        false,
        f.format,
        SPARSE_BM,
        SPARSE_BN,
    )?;
    let grid = MTLSize {
        width: f.n.div_ceil(SPARSE_BN),
        height: route.num_blocks as usize,
        depth: 1,
    };
    let tg = MTLSize {
        width: crate::kernels::sub::gemm_common::THREADS_PER_TG,
        height: 1,
        depth: 1,
    };
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipe.pipeline);
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
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut out = vec![0.0f32; out_len];
    BufferPool::read_f32(&buf_c, &mut out);
    Ok(out)
}

#[cfg(all(test, target_os = "macos"))]
mod sparse_nvfp4_tests {
    use super::*;
    use crate::kernels::sub::gemm_linear_grouped::grouped_fixture;
    use crate::kernels::sub::test_util::assert_oracle;

    /// Same ragged M-tile boundary coverage as the block_sparse suite.
    const COUNTS: &[usize] = &[1, 6, 8, 9, 16, 17, 31, 33, 40, 0, 100];

    /// Port oracle: `gemm_tunable_sparse` at nvfp4 must be BIT-EXACT vs the
    /// production-proven `gemm_block_sparse` (same per-output K-accumulation
    /// chain + `half(e2m1*scale)` dequant + `arena_round_f32` store). Guards the
    /// nvfp4 branch added when the block_sparse family was retired. NOTE: this
    /// cross-check depends on gemm_block_sparse; when that kernel is deleted the
    /// durable oracle below (vs CPU) remains.
    #[test]
    fn sparse_nvfp4_bitexact_vs_block_sparse() {
        let f = grouped_fixture(QuantFormat::NvFp4, 64, 192, COUNTS);
        let legacy = crate::kernels::sub::gemm_block_sparse::gpu_sparse(&f, false)
            .expect("legacy block_sparse");
        let tuned = gpu_sparse_tunable(&f).expect("tunable sparse");
        assert_eq!(legacy.len(), tuned.len());
        for (i, (a, b)) in legacy.iter().zip(tuned.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "nvfp4 output {i} differs: block_sparse {a} tunable {b}"
            );
        }
    }

    /// Durable independent oracle (survives block_sparse retirement): tunable
    /// nvfp4 output tracks the grouped CPU reference within quant tolerance.
    #[test]
    fn sparse_nvfp4_matches_cpu_oracle() {
        let f = grouped_fixture(QuantFormat::NvFp4, 64, 192, COUNTS);
        let tuned = gpu_sparse_tunable(&f).expect("tunable sparse");
        let oracle = crate::kernels::sub::gemm_block_grouped::cpu(&f);
        assert_oracle(&tuned, &oracle, 0.05, 0.999);
    }
}
