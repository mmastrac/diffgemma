//! In-place fp16 scale (logits buffer — not bf16 arena).

use super::f16;
use super::gpu_common;
use super::variant::KernelVariant;
use crate::safetensors::Error;

pub const ENTRY: &str = "half_scale_fp16";

const SHADER: &str = shader_include::include_metal!("kernels/half_scale_fp16.metal");

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER, ENTRY, variant)
}
