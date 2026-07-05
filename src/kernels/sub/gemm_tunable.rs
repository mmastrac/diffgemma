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

#[cfg(all(feature = "metal", target_os = "macos"))]
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
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_for_logits(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
    format: QuantFormat,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_gemm_subkernel_out_bf16(&tuned_source(TUNE_BM, TUNE_BN), ENTRY, n, k, format as u32)
}
