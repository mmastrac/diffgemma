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
