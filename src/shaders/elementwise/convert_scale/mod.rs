//! Generic elementwise convert + scale kernel:
//! `dst[dst_base+gid] = to_dst(scale * from_src(src[src_base+gid]))`, generic
//! over src/dst element dtype (f32 vs activation-arena bf16/fp16) via FC4/FC5.
//! Subsumes half_to_f32, half_scale (in-place), copy_f32, f32_to_half_scale.

use crate::Error;
use crate::shaders::variant::KernelVariant;

crate::shader_kernel! {
    name = "convert_scale",
    metal = "convert_scale.metal",
    spec = {
            quant_formats: &[QuantFormat::Q4Affine],
            fc: &[(4, "K_SRC_F32"), (5, "K_DST_F32")],
            variants: KernelVariants::Elementwise,
    },
    tests = {},
}

/// Compile the convert/scale kernel specialized for (src_f32, dst_f32).
#[cfg(target_os = "macos")]
pub fn pipeline_for_fmt(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
    src_f32: bool,
    dst_f32: bool,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    use crate::shaders::variant::FcBool;
    let bools = [
        FcBool {
            index: 4,
            value: src_f32,
        },
        FcBool {
            index: 5,
            value: dst_f32,
        },
    ];
    let label = match (src_f32, dst_f32) {
        (false, true) => "arena2f32",
        (false, false) => "arena2arena",
        (true, true) => "f322f32",
        (true, false) => "f322arena",
    };
    ctx.compile_subkernel_ex(SHADER, ENTRY, variant, label, &bools, &[])
}

#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};

/// Bind the 7-arg convert/scale signature. `src`/`dst` may be the same buffer
/// (in-place, e.g. half_scale). `dump` is only written when K_DUMP_STAGE >= 1.
#[cfg(target_os = "macos")]
pub fn bind_gpu_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    src: &ProtocolObject<dyn MTLBuffer>,
    src_off: usize,
    dst: &ProtocolObject<dyn MTLBuffer>,
    dst_off: usize,
    src_base: u32,
    dst_base: u32,
    len: u32,
    scale: f32,
    dump: &ProtocolObject<dyn MTLBuffer>,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(src), src_off, 0);
        enc.setBuffer_offset_atIndex(Some(dst), dst_off, 1);
        enc.setBuffer_offset_atIndex(Some(dump), 0, 6);
    }
    crate::shaders::gpu_common::set_bytes(enc, &src_base, 2);
    crate::shaders::gpu_common::set_bytes(enc, &dst_base, 3);
    crate::shaders::gpu_common::set_bytes(enc, &len, 4);
    crate::shaders::gpu_common::set_bytes(enc, &scale, 5);
}
