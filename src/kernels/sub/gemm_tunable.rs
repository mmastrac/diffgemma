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
