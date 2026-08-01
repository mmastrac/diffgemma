//! Generic elementwise convert + scale kernel:
//! `dst[dst_base+gid] = to_dst(scale * from_src(src[src_base+gid]))`, generic
//! over src/dst element dtype (f32 vs activation-arena bf16/fp16) via FC4/FC5.
//! Subsumes half_to_f32, half_scale (in-place), copy_f32, f32_to_half_scale.

use crate::Error;
use crate::shaders::variant::KernelVariant;

pub const ENTRY: &str = "convert_scale";

pub const SHADER: &str = include_str!("convert_scale.metal");

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

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use crate::shaders::bf16;
    use crate::shaders::variant::KernelVariant;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLSize,
    };

    /// Each (src_f32, dst_f32) corner converts+scales bit-for-bit like a CPU
    /// mirror at the arena's bf16 precision (bf16-exact inputs).
    #[test]
    fn all_four_corners_match_cpu() {
        // bf16-exact input values so arena round-trips are exact.
        let src_f32_vals: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.5).collect();
        let scale = 0.25f32;
        let n = src_f32_vals.len();
        let ctx = MetalContext::new().expect("ctx");

        for &(src_f32, dst_f32) in &[(false, true), (false, false), (true, true), (true, false)] {
            let mut pool = BufferPool::new();
            // src buffer: f32 or arena bf16 bits.
            let buf_src = if src_f32 {
                let b = pool.allocate(&ctx.device, n * 4).unwrap();
                BufferPool::write_f32(&b, &src_f32_vals);
                b
            } else {
                let b = pool.allocate(&ctx.device, n * 2).unwrap();
                BufferPool::write_bf16(&b, &bf16::f32_slice_to_bf16_bits(&src_f32_vals));
                b
            };
            let dst_elem = if dst_f32 { 4 } else { 2 };
            let buf_dst = pool.allocate(&ctx.device, n * dst_elem).unwrap();
            let dump = pool.allocate(&ctx.device, n * 4).unwrap();

            let pipe =
                pipeline_for_fmt(&ctx, KernelVariant::PRODUCTION, src_f32, dst_f32).expect("pipe");
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pipe.pipeline);
            bind_gpu_buffers(&enc, &buf_src, 0, &buf_dst, 0, 0, 0, n as u32, scale, &dump);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: n.div_ceil(64),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 64,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();

            for (i, &sv) in src_f32_vals.iter().enumerate().take(n) {
                // CPU mirror: src already bf16-exact, so from_src == value.
                let want = bf16::store_bf16_round_half(sv * scale);
                let want = if dst_f32 {
                    // dst f32 stores the (bf16-rounded on read? no) exact product.
                    sv * scale
                } else {
                    want
                };
                let got = if dst_f32 {
                    let p = buf_dst.contents().as_ptr() as *const f32;
                    unsafe { *p.add(i) }
                } else {
                    let p = buf_dst.contents().as_ptr() as *const u16;
                    bf16::bf16_bits_to_f32(unsafe { *p.add(i) })
                };
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "corner (src_f32={src_f32}, dst_f32={dst_f32}) elem {i}: got {got} want {want}"
                );
            }
        }
    }
}
