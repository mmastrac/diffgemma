//! Tiled bf16-weight GEMM: `y[M,N] = x[M,K] @ W[N,K]^T` with bf16 weights (no
//! quantization). Same tiling as `gemm_q8` but reads the weight tile straight
//! from bf16 rows — lossless into the half tiles since the weights are bf16.
//! Used for the mixed-precision `.dgq` attention/dense-FFN path.

use crate::safetensors::Error;

pub const ENTRY: &str = "gemm_bf16";

const SHADER: &str = shader_include::include_metal!("kernels/gemm_bf16.metal");

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    // quant_format / x_fp16 are unused by the bf16 kernel; pass inert values.
    ctx.compile_gemm_subkernel(SHADER, ENTRY, n, k, false, super::QuantFormat::Q8 as u32, false)
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn pipeline_for(
    _ctx: &crate::metal::device::MetalContext,
    _n: u32,
    _k: u32,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    Err(Error::Format("Metal unavailable"))
}
