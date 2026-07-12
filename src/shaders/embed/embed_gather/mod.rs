//! Q8 embed-table gather: dequant row by token id and apply embed scale.

use crate::dgq::block::{dequant_row_q8, quantize_row_q8};
use crate::dgq::embed_row::EMBED_SCALE;
use crate::dgq::layout::q8_row_bytes;
use crate::safetensors::Error;
use crate::shaders::bf16;
use crate::shaders::gpu_common;
use crate::shaders::test_util::ElemFormat;
use crate::shaders::variant::KernelVariant;

pub const ENTRY: &str = "embed_gather";

const SHADER: &str = shader_include::include_metal!("embed/embed_gather/embed_gather.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub embed_f32: Vec<f32>,
    pub ids: Vec<u32>,
    pub num_tokens: usize,
    pub hidden: usize,
    pub embed_scale: f32,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.num_tokens * self.hidden
    }

    pub fn embed_q8(&self) -> Vec<u8> {
        let vocab = self.embed_f32.len() / self.hidden;
        let row_bytes = q8_row_bytes(self.hidden);
        let mut dst = vec![0u8; vocab * row_bytes];
        for row in 0..vocab {
            let off = row * self.hidden;
            quantize_row_q8(
                &self.embed_f32[off..off + self.hidden],
                self.hidden,
                &mut dst[row * row_bytes..],
            );
        }
        dst
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    let hidden = 8usize;
    let vocab = 4usize;
    let embed_f32: Vec<f32> = (0..vocab * hidden)
        .map(|i| ((i as f32) * 0.07).sin() * 0.05)
        .collect();
    Fixture {
        embed_f32,
        ids: vec![2, 0, 1],
        num_tokens: 3,
        hidden,
        embed_scale: EMBED_SCALE,
    }
}

pub fn tile_fixture(_: ElemFormat) -> Fixture {
    let hidden = 64usize;
    let vocab = 32usize;
    let embed_f32: Vec<f32> = (0..vocab * hidden)
        .map(|i| ((i as f32) * 0.011).cos() * 0.03)
        .collect();
    Fixture {
        embed_f32,
        ids: (0..8).map(|i| (i * 3) as u32).collect(),
        num_tokens: 8,
        hidden,
        embed_scale: EMBED_SCALE,
    }
}

pub fn embed_gather_cpu(
    embed_q8: &[u8],
    ids: &[u32],
    num_tokens: usize,
    hidden: usize,
    embed_scale: f32,
    out: &mut [f32],
) {
    assert_eq!(out.len(), num_tokens * hidden);
    let row_bytes = q8_row_bytes(hidden);
    let mut raw = vec![0.0f32; hidden];
    for tok in 0..num_tokens {
        let row = ids[tok] as usize;
        let row_off = row * row_bytes;
        dequant_row_q8(&embed_q8[row_off..row_off + row_bytes], hidden, &mut raw);
        let dst = tok * hidden;
        for d in 0..hidden {
            out[dst + d] = bf16::store_bf16_round_half(raw[d] * embed_scale);
        }
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let embed_q8 = f.embed_q8();
    let mut out = vec![0.0f32; f.out_len()];
    embed_gather_cpu(
        &embed_q8,
        &f.ids,
        f.num_tokens,
        f.hidden,
        f.embed_scale,
        &mut out,
    );
    out
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(target_os = "macos")]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    pipeline_for_fmt(ctx, variant, false)
}

/// Format-generic embed gather: `bf16` picks the Raw bf16 embed table
/// (K_EMBED_BF16 / FC4); false = q8 per-row dequant (default). FC3 stays inert
/// so K_ELEMENTWISE_GUARD passes for both.
#[cfg(target_os = "macos")]
pub fn pipeline_for_fmt(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
    bf16: bool,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    use crate::shaders::variant::FcBool;
    let bools = [FcBool {
        index: 4,
        value: bf16,
    }];
    let label = if bf16 { "bf16" } else { "q8" };
    ctx.compile_subkernel_ex(SHADER, ENTRY, variant, label, &bools, &[])
}

#[cfg(target_os = "macos")]
pub fn dispatch_shape(
    hidden: usize,
    num_tokens: usize,
) -> (objc2_metal::MTLSize, objc2_metal::MTLSize) {
    use objc2_metal::MTLSize;
    (
        MTLSize {
            width: hidden,
            height: num_tokens,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    )
}

#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
};

#[cfg(target_os = "macos")]
pub fn bind_gpu_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    blob: &ProtocolObject<dyn MTLBuffer>,
    ids: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    w_off: u64,
    hidden: u32,
    num_tokens: u32,
    embed_scale: f32,
    vocab: u32,
    debug_status: Option<&ProtocolObject<dyn MTLBuffer>>,
) {
    let dims = [hidden, num_tokens];
    unsafe {
        enc.setBuffer_offset_atIndex(Some(blob), 0, 0);
        enc.setBuffer_offset_atIndex(Some(ids), 0, 1);
        enc.setBuffer_offset_atIndex(Some(out), 0, 2);
    }
    gpu_common::set_bytes(enc, &w_off, 3);
    gpu_common::set_bytes(enc, &dims, 4);
    gpu_common::set_bytes(enc, &embed_scale, 5);
    gpu_common::set_bytes(enc, &vocab, 6);
    if let Some(dbg) = debug_status {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(dbg), 0, 7);
        }
    }
}

#[cfg(target_os = "macos")]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let embed_q8 = f.embed_q8();
    let buf_blob = pool
        .allocate(&ctx.device, embed_q8.len())
        .ok_or(Error::Format("alloc"))?;
    let buf_ids = pool
        .allocate(&ctx.device, f.ids.len() * 4)
        .ok_or(Error::Format("alloc"))?;
    let buf_out = pool
        .allocate(&ctx.device, f.out_len() * 2)
        .ok_or(Error::Format("alloc"))?;
    let buf_dbg = pool
        .allocate(&ctx.device, crate::metal::debug_status::DEBUG_STATUS_BYTES)
        .ok_or(Error::Format("alloc"))?;
    BufferPool::write_bytes(&buf_blob, &embed_q8);
    let idx_bytes =
        unsafe { std::slice::from_raw_parts(f.ids.as_ptr().cast::<u8>(), f.ids.len() * 4) };
    BufferPool::write_bytes(&buf_ids, idx_bytes);
    let (grid, tg) = dispatch_shape(f.hidden, f.num_tokens);
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    bind_gpu_buffers(
        &enc,
        &buf_blob,
        &buf_ids,
        &buf_out,
        0,
        f.hidden as u32,
        f.num_tokens as u32,
        f.embed_scale,
        (f.embed_f32.len() / f.hidden) as u32,
        if variant.debug_fast || variant.shape_assert || variant.debug_deep {
            Some(&buf_dbg)
        } else {
            None
        },
    );
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let ptr = buf_out.contents().as_ptr() as *const u16;
    Ok((0..f.out_len())
        .map(|i| bf16::bf16_bits_to_f32(unsafe { *ptr.add(i) }))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::shaders::embed_gather::cpu,
        cpu_oracle = crate::shaders::embed_gather::cpu_oracle,
        gpu = crate::shaders::embed_gather::gpu,
        fixture = crate::shaders::embed_gather::tiny_fixture,
        out_len = crate::shaders::embed_gather::fixture_len,
        formats: [F32],
        max_tol = 0.002,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod tile,
        cpu = crate::shaders::embed_gather::cpu,
        cpu_oracle = crate::shaders::embed_gather::cpu_oracle,
        gpu = crate::shaders::embed_gather::gpu,
        fixture = crate::shaders::embed_gather::tile_fixture,
        out_len = crate::shaders::embed_gather::fixture_len,
        formats: [F32],
        max_tol = 1e-2,
        min_cos = 0.9999,
    }
}
