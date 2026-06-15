//! Pack engine GpuKvCache f32 K/V into monolithic b4 layout.

use super::variant::KernelVariant;
use crate::safetensors::Error;

pub const ENTRY: &str = "pack_encoder_kv";

const SHADER: &str = shader_include::include_metal!("kernels/pack_encoder_kv.metal");

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER, ENTRY, variant)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2::runtime::ProtocolObject;
#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn bind_gpu_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    keys: &ProtocolObject<dyn MTLBuffer>,
    values: &ProtocolObject<dyn MTLBuffer>,
    dst: &ProtocolObject<dyn MTLBuffer>,
    token_count: u32,
    dst_pos: u32,
    nkv: u32,
    hd: u32,
    kv_region_bytes: u64,
    src_pos: u32,
) {
    let shape = [token_count, dst_pos, nkv, hd];
    unsafe {
        enc.setBuffer_offset_atIndex(Some(keys), 0, 0);
        enc.setBuffer_offset_atIndex(Some(values), 0, 1);
        enc.setBuffer_offset_atIndex(Some(dst), 0, 2);
    }
    super::gpu_common::set_bytes(enc, &shape, 3);
    super::gpu_common::set_bytes(enc, &kv_region_bytes, 4);
    super::gpu_common::set_bytes(enc, &src_pos, 5);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
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
