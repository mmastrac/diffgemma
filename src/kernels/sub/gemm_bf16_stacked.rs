//! bf16 stacked GEMM (fused QKV / gate+up on the bf16 default path). Same
//! FC segment machinery as `gemm_block_stacked` (q4/nvfp4) but reads bf16 [N,K]
//! weight rows with no dequant — the bf16 analog, mirroring how `gemm_bf16`
//! derives from `gemm_block`.

use super::gemm_block_stacked::{GemmStackedSeg, StackedSegFc};
use crate::safetensors::Error;

pub const ENTRY: &str = "gemm_bf16_stacked";

const SHADER: &str = shader_include::include_metal!("kernels/gemm_bf16_stacked.metal");

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn stacked_pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
    segs: &[GemmStackedSeg],
) -> Result<std::sync::Arc<crate::metal::device::ComputePipeline>, Error> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static CACHE: OnceLock<Mutex<HashMap<(u32, u32, StackedSegFc), std::sync::Arc<crate::metal::device::ComputePipeline>>>> =
        OnceLock::new();
    let stacked = StackedSegFc::from_segments(segs)?;
    let key = (n, k, stacked);
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .map_err(|_| Error::Format("bf16 stacked pipeline cache poisoned"))?;
    if let Some(pipe) = guard.get(&key) {
        return Ok(std::sync::Arc::clone(pipe));
    }
    // Format FC is unused by the bf16 kernel (no dequant); pass Q4Affine as filler.
    let pipe = std::sync::Arc::new(ctx.compile_gemm_stacked_subkernel(
        SHADER,
        ENTRY,
        n,
        k,
        super::QuantFormat::Q4Affine as u32,
        &stacked,
    )?);
    guard.insert(key, std::sync::Arc::clone(&pipe));
    Ok(pipe)
}
