//! E14: (re)hydrate a sliding layer's f32 side KV ring from the monolithic
//! cache (f16 or q8 source) — used when a fast prefill resumes at an offset
//! the side ring is not valid for.

use super::variant::KernelVariant;
use crate::safetensors::Error;

pub const ENTRY: &str = "kv_f32_side_hydrate";

const SHADER: &str = shader_include::include_metal!("kernels/kv_f32_side_hydrate.metal");

#[cfg(target_os = "macos")]
pub fn pipeline_for_kv(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
    fmt: crate::kernels::sub::kv_quant::KvFormat,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    let uints = [crate::kernels::sub::variant::FcUInt {
        index: 4,
        value: fmt.code(),
    }];
    ctx.compile_subkernel_ex(SHADER, ENTRY, variant, fmt.label(), &[], &uints)
}
