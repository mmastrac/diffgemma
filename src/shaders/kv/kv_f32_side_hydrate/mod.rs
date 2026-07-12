//! E14: (re)hydrate a sliding layer's f32 side KV ring from the monolithic
//! cache (f16 or q8 source) — used when a fast prefill resumes at an offset
//! the side ring is not valid for.

use crate::safetensors::Error;
use crate::shaders::variant::KernelVariant;

pub const ENTRY: &str = "kv_f32_side_hydrate";

pub const SHADER: &str = include_str!("kv_f32_side_hydrate.metal");

#[cfg(target_os = "macos")]
pub fn pipeline_for_kv(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
    fmt: crate::shaders::kv_quant::KvFormat,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    let uints = [crate::shaders::variant::FcUInt {
        index: 4,
        value: fmt.code(),
    }];
    ctx.compile_subkernel_ex(SHADER, ENTRY, variant, fmt.label(), &[], &uints)
}
