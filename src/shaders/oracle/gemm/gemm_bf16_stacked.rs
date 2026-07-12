//! ORACLE (not production): validation twin of `gemm_tunable` stacked raw.
//! Kernel source lives in `shaders/oracle/gemm_block_stacked.metal`.
//!
//! bf16 stacked GEMM (fused QKV / gate+up on the bf16 default path). Now a thin
//! delegate over `gemm_block_stacked` with `QuantFormat::Raw` (bf16 [N,K] rows
//! read with no dequant) — the bf16 analog, mirroring how `gemm_bf16` folded
//! into `gemm_block`. The shared kernel is `kernels/gemm_block_stacked.metal`.

use super::gemm_block_stacked::GemmStackedSeg;
use crate::safetensors::Error;

#[cfg(target_os = "macos")]
pub fn stacked_pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
    segs: &[GemmStackedSeg],
) -> Result<std::sync::Arc<crate::metal::device::ComputePipeline>, Error> {
    super::gemm_block_stacked::stacked_pipeline_for(
        ctx,
        n,
        k,
        crate::shaders::QuantFormat::Raw,
        segs,
    )
}
