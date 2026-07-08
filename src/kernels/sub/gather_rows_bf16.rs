//! Gather bf16 arena rows by index (`ushort` slots, MLX-like bf16 bits).

use super::variant::KernelVariant;
use crate::safetensors::Error;

pub const ENTRY: &str = "gather_rows_bf16";

const SHADER: &str = shader_include::include_metal!("kernels/gather_rows_bf16.metal");

#[cfg(target_os = "macos")]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER, ENTRY, variant)
}
