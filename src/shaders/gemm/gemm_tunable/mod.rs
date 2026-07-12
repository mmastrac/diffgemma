//! Tunable GEMM (task #19): fragment-level block GEMM with
//! per-lane thread_elements() loads and vectorized loaders; tile geometry via
//! TUNE_BM/TUNE_BN #define prepend. BIT-EXACT vs gemm_block (same ascending-K
//! accumulation chain, dequant math, and store rounding) — verified per
//! element in bench-gemm across all production shapes (q4 + raw).
//! Production wiring behind `DGQ_GEMM_TUNABLE` starts with the Raw (bf16
//! weight) plain path; 64x64 won every production shape in the sweep.

use crate::safetensors::Error;
use crate::shaders::QuantFormat;

pub const ENTRY: &str = "gemm_tunable";

pub const SHADER: &str = include_str!("gemm_tunable.metal");

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
    segs: &[crate::shaders::gemm_block_stacked::GemmStackedSeg],
) -> Result<std::sync::Arc<crate::metal::device::ComputePipeline>, Error> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    type Key = crate::shaders::gemm_block_stacked::StackedPipelineKey;

    static CACHE: OnceLock<
        Mutex<HashMap<Key, std::sync::Arc<crate::metal::device::ComputePipeline>>>,
    > = OnceLock::new();
    let stacked = crate::shaders::gemm_block_stacked::StackedSegFc::from_segments(segs)?;
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
    f: &crate::shaders::gemm_linear_grouped::Fixture,
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
        &ctx, f.n as u32, f.k as u32, false, f.format, SPARSE_BM, SPARSE_BN,
    )?;
    let grid = MTLSize {
        width: f.n.div_ceil(SPARSE_BN),
        height: route.num_blocks as usize,
        depth: 1,
    };
    let tg = MTLSize {
        width: crate::shaders::gemm_common::THREADS_PER_TG,
        height: 1,
        depth: 1,
    };
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipe.pipeline);
    crate::shaders::gemm_block_grouped::bind_gpu_buffers(
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

/// Run the dense `gemm_tunable` kernel on a legacy nvfp4 dense fixture
/// (x bf16, single weight matrix at w_off=0). Self-contained; used by the
/// dense nvfp4 port oracle. Returns bf16-decoded outputs.
#[cfg(all(test, target_os = "macos"))]
pub(crate) fn gpu_dense_tunable_nvfp4(
    f: &crate::shaders::gemm_nvfp4::Fixture,
) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use crate::shaders::bf16;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLSize,
    };

    let ctx = MetalContext::new()?;
    let mut pool = BufferPool::new();
    let w_nvfp4 = crate::shaders::gemm_nvfp4::w_nvfp4(f);
    let buf_x = pool
        .allocate(&ctx.device, f.m * f.k * 2)
        .ok_or(Error::Format("alloc x"))?;
    let buf_y = pool
        .allocate(&ctx.device, f.m * f.n * 2)
        .ok_or(Error::Format("alloc y"))?;
    let buf_w = pool
        .allocate(&ctx.device, w_nvfp4.len())
        .ok_or(Error::Format("alloc w"))?;
    BufferPool::write_bf16(&buf_x, &bf16::f32_slice_to_bf16_bits(&f.x));
    BufferPool::write_bytes(&buf_w, &w_nvfp4);
    let pipe = pipeline_for(&ctx, f.n as u32, f.k as u32, QuantFormat::NvFp4)?;
    let grid = MTLSize {
        width: f.n.div_ceil(TUNE_BN),
        height: f.m.div_ceil(TUNE_BM),
        depth: 1,
    };
    let tg = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipe.pipeline);
    crate::shaders::gemm_nvfp4::bind_gpu_buffers(&enc, &buf_x, &buf_y, &buf_w, 0, f.m as u32);
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let ptr = buf_y.contents().as_ptr() as *const u16;
    Ok((0..f.out_len())
        .map(|i| bf16::bf16_bits_to_f32(unsafe { *ptr.add(i) }))
        .collect())
}

#[cfg(all(test, target_os = "macos"))]
mod stacked_nvfp4_tests {
    use super::*;
    use crate::dgq::layout::nvfp4_matrix_bytes;
    use crate::dgq::nvfp4::quantize_f32_matrix_nvfp4;
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use crate::shaders::bf16;
    use crate::shaders::gemm_block_stacked::{self, GemmStackedSeg};
    use crate::shaders::test_util::ElemFormat;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLSize,
    };

    /// Re-quantize a stacked fixture's segments as nvfp4 (4-byte f32 gscale
    /// header + body per segment matrix); returns (blob, gpu segments).
    fn nvfp4_stacked(f: &gemm_block_stacked::StackedFixture) -> (Vec<u8>, Vec<GemmStackedSeg>) {
        let mut blob = Vec::new();
        let mut segs = Vec::new();
        let mut w_off = 0u64;
        for s in &f.segments {
            let mut dst = vec![0u8; nvfp4_matrix_bytes(s.n, f.k)];
            quantize_f32_matrix_nvfp4(&s.w_f32, s.n, f.k, &mut dst);
            segs.push(GemmStackedSeg {
                n_cols: s.n as u32,
                y_col0: s.y_col0 as u32,
                y_row_cols: s.y_row_cols as u32,
                _pad: 0,
                w_off,
                y_byte_off: s.y_byte_off as u64,
            });
            w_off += nvfp4_matrix_bytes(s.n, f.k) as u64;
            blob.extend_from_slice(&dst);
        }
        (blob, segs)
    }

    /// Port oracle: `gemm_tunable_stacked` at nvfp4 must be BIT-EXACT vs the
    /// legacy `gemm_block_stacked` nvfp4 path (per-segment gscale header +
    /// range decode). Covers the qkv (3-seg) and gate/up (2-seg) layouts.
    #[test]
    fn stacked_nvfp4_bitexact_vs_gemm_block_stacked() {
        for fixture_fn in [
            gemm_block_stacked::gate_up_tiny_fixture as fn(_) -> _,
            gemm_block_stacked::qkv_tiny_fixture,
        ] {
            let f = fixture_fn(ElemFormat::F32);
            let n_total = f.n_total() as u32;
            let (blob, segs) = nvfp4_stacked(&f);

            let ctx = MetalContext::new().expect("metal ctx");
            let mut pool = BufferPool::new();
            let buf_x = pool.allocate(&ctx.device, f.m * f.k * 2).expect("x");
            let buf_w = pool.allocate(&ctx.device, blob.len()).expect("w");
            let buf_leg = pool.allocate(&ctx.device, f.y_bytes()).expect("y_leg");
            let buf_tun = pool.allocate(&ctx.device, f.y_bytes()).expect("y_tun");
            BufferPool::write_bf16(&buf_x, &bf16::f32_slice_to_bf16_bits(&f.x));
            BufferPool::write_bytes(&buf_w, &blob);

            // Legacy gemm_block_stacked (n_tile grid).
            {
                let pipe = gemm_block_stacked::pipeline_for(
                    &ctx,
                    n_total,
                    f.k as u32,
                    QuantFormat::NvFp4,
                    &segs,
                )
                .expect("legacy stacked pipeline");
                let (grid, tg) = crate::shaders::gemm_common::dispatch_shape(f.m, f.n_total());
                let cmd = ctx.queue.commandBuffer().expect("cmd");
                let enc = cmd.computeCommandEncoder().expect("enc");
                enc.setComputePipelineState(&pipe.pipeline);
                gemm_block_stacked::bind_gpu_buffers(&enc, &buf_x, &buf_leg, &buf_w, f.m as u32);
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
            }
            // Tunable gemm_tunable_stacked (TUNE_BM/BN grid, same buffers).
            {
                let pipe =
                    stacked_pipeline_for(&ctx, n_total, f.k as u32, QuantFormat::NvFp4, &segs)
                        .expect("tunable stacked pipeline");
                let grid = MTLSize {
                    width: (n_total as usize).div_ceil(TUNE_BN),
                    height: f.m.div_ceil(TUNE_BM),
                    depth: 1,
                };
                let tg = MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                };
                let cmd = ctx.queue.commandBuffer().expect("cmd");
                let enc = cmd.computeCommandEncoder().expect("enc");
                enc.setComputePipelineState(&pipe.pipeline);
                gemm_block_stacked::bind_gpu_buffers(&enc, &buf_x, &buf_tun, &buf_w, f.m as u32);
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
            }

            let lp = buf_leg.contents().as_ptr() as *const u16;
            let tp = buf_tun.contents().as_ptr() as *const u16;
            for i in 0..(f.y_bytes() / 2) {
                let a = unsafe { *lp.add(i) };
                let b = unsafe { *tp.add(i) };
                assert_eq!(
                    a, b,
                    "nvfp4 stacked output {i} differs: legacy {a:#06x} tunable {b:#06x}"
                );
            }
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod dense_nvfp4_tests {
    use super::*;
    use crate::shaders::gemm_nvfp4;
    use crate::shaders::test_util::{ElemFormat, assert_oracle};
    use crate::shaders::variant::KernelVariant;

    /// Port oracle: dense `gemm_tunable` at nvfp4 must be BIT-EXACT vs the
    /// legacy `gemm_block` nvfp4 dense kernel. Guards the is_nvfp4 branch added
    /// to the dense W-load when gemm_block was retired.
    #[test]
    fn dense_nvfp4_bitexact_vs_gemm_block() {
        for fixture_fn in [
            gemm_nvfp4::tile_fixture as fn(_) -> _,
            gemm_nvfp4::gscale_fixture,
        ] {
            let f = fixture_fn(ElemFormat::F32);
            let legacy = gemm_nvfp4::gpu(&f, KernelVariant::PRODUCTION).expect("legacy gemm_block");
            let tuned = gpu_dense_tunable_nvfp4(&f).expect("tunable dense");
            assert_eq!(legacy.len(), tuned.len());
            for (i, (a, b)) in legacy.iter().zip(tuned.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "nvfp4 dense output {i} differs: gemm_block {a} tunable {b}"
                );
            }
        }
    }

    /// Durable oracle (survives gemm_block retirement): dense tunable nvfp4
    /// tracks the CPU reference within quant tolerance.
    #[test]
    fn dense_nvfp4_matches_cpu_oracle() {
        let f = gemm_nvfp4::tile_fixture(ElemFormat::F32);
        let tuned = gpu_dense_tunable_nvfp4(&f).expect("tunable dense");
        let oracle = gemm_nvfp4::cpu(&f);
        assert_oracle(&tuned, &oracle, 0.08, 0.999);
    }
}

#[cfg(all(test, target_os = "macos"))]
mod sparse_nvfp4_tests {
    use super::*;
    use crate::shaders::gemm_linear_grouped::grouped_fixture;
    use crate::shaders::test_util::assert_oracle;

    /// Same ragged M-tile boundary coverage as the block_sparse suite.
    const COUNTS: &[usize] = &[1, 6, 8, 9, 16, 17, 31, 33, 40, 0, 100];

    /// Independent oracle: tunable nvfp4 block-sparse output tracks the grouped
    /// CPU reference within quant tolerance. (The block_sparse bit-exact
    /// cross-check retired with that kernel.)
    #[test]
    fn sparse_nvfp4_matches_cpu_oracle() {
        let f = grouped_fixture(QuantFormat::NvFp4, 64, 192, COUNTS);
        let tuned = gpu_sparse_tunable(&f).expect("tunable sparse");
        let oracle = crate::shaders::gemm_block_grouped::cpu(&f);
        assert_oracle(&tuned, &oracle, 0.05, 0.999);
    }
}
