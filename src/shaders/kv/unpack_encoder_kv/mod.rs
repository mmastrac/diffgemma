//! Hydrate engine GpuKvCache f32 K/V from monolithic b4 layout (inverse of
//! pack_encoder_kv; ring-aware source, linear engine destination).

use crate::Error;
use crate::shaders::variant::KernelVariant;

pub const ENTRY: &str = "unpack_encoder_kv";

pub const SHADER: &str = include_str!("unpack_encoder_kv.metal");

#[cfg(target_os = "macos")]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER, ENTRY, variant)
}

/// Quantized-KV variant (uint function constant 4 = KvFormat code).
#[cfg(target_os = "macos")]
pub fn pipeline_fmt_for(
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

#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};

#[cfg(target_os = "macos")]
pub fn bind_gpu_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    src: &ProtocolObject<dyn MTLBuffer>,
    keys: &ProtocolObject<dyn MTLBuffer>,
    values: &ProtocolObject<dyn MTLBuffer>,
    token_count: u32,
    dst_pos: u32,
    nkv: u32,
    hd: u32,
    kv_region_bytes: u64,
    src_pos: u32,
    kv_ring_mask: u32,
) {
    let shape = [token_count, dst_pos, nkv, hd];
    unsafe {
        enc.setBuffer_offset_atIndex(Some(src), 0, 0);
        enc.setBuffer_offset_atIndex(Some(keys), 0, 1);
        enc.setBuffer_offset_atIndex(Some(values), 0, 2);
    }
    crate::shaders::gpu_common::set_bytes(enc, &shape, 3);
    crate::shaders::gpu_common::set_bytes(enc, &kv_region_bytes, 4);
    crate::shaders::gpu_common::set_bytes(enc, &src_pos, 5);
    crate::shaders::gpu_common::set_bytes(enc, &kv_ring_mask, 6);
}

/// Grid: (token_count, nkv, head_dim) — z is elements for both formats.
#[cfg(target_os = "macos")]
pub fn dispatch_shape(
    token_count: usize,
    nkv: usize,
    hd: usize,
) -> (objc2_metal::MTLSize, objc2_metal::MTLSize) {
    use objc2_metal::MTLSize;
    (
        MTLSize {
            width: token_count,
            height: nkv,
            depth: hd,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    )
}
