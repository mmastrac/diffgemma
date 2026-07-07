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

/// Quantized-KV variant (uint function constant 4 = KvFormat code): grid
/// depth = head_dim/32 groups. `fmt` must be a quantized format (q8/q4).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_fmt_for(
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
    kv_ring_mask: u32,
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
    super::gpu_common::set_bytes(enc, &kv_ring_mask, 6);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn dispatch_shape(
    token_count: usize,
    nkv: usize,
    hd: usize,
    fmt: crate::kernels::sub::kv_quant::KvFormat,
) -> (objc2_metal::MTLSize, objc2_metal::MTLSize) {
    use crate::kernels::sub::kv_quant::KvFormat;
    use objc2_metal::MTLSize;
    (
        MTLSize {
            width: token_count,
            height: nkv,
            // quantized: one thread per 32-group (quantizes K + V); f16: per element.
            depth: if fmt == KvFormat::F16 { hd } else { hd / 32 },
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    )
}
