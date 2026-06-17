//! Gather bf16 arena rows by index into f32 (MoE batched path; fuses half_to_f32 + gather).

use super::bf16;
use super::gpu_common;
use super::gather_rows;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::safetensors::Error;

pub const ENTRY: &str = "gather_rows_bf16_to_f32";

const SHADER: &str = shader_include::include_metal!("kernels/gather_rows_bf16_to_f32.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub src_bf16: Vec<u16>,
    pub indices: Vec<u32>,
    pub hidden: usize,
    pub num_tokens: usize,
}

impl Fixture {
    pub fn batch_size(&self) -> usize {
        self.indices.len()
    }

    pub fn out_len(&self) -> usize {
        self.batch_size() * self.hidden
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

fn from_gather_fixture(g: &gather_rows::Fixture) -> Fixture {
    Fixture {
        src_bf16: g
            .src
            .iter()
            .map(|&v| bf16::f32_to_bf16_bits(bf16::store_bf16_round_half(v)))
            .collect(),
        indices: g.indices.clone(),
        hidden: g.hidden,
        num_tokens: g.num_tokens,
    }
}

pub fn tiny_fixture(fmt: ElemFormat) -> Fixture {
    from_gather_fixture(&gather_rows::tiny_fixture(fmt))
}

pub fn moe_fixture(fmt: ElemFormat) -> Fixture {
    from_gather_fixture(&gather_rows::moe_fixture(fmt))
}

pub fn moe_routing_fixture(fmt: ElemFormat) -> Fixture {
    from_gather_fixture(&gather_rows::moe_routing_fixture(fmt))
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let mut out = vec![0.0f32; f.out_len()];
    for (bi, &tok) in f.indices.iter().enumerate() {
        let src_off = tok as usize * f.hidden;
        let dst_off = bi * f.hidden;
        for h in 0..f.hidden {
            out[dst_off + h] = bf16::bf16_bits_to_f32(f.src_bf16[src_off + h]);
        }
    }
    out
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

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
    src: &ProtocolObject<dyn MTLBuffer>,
    indices: &ProtocolObject<dyn MTLBuffer>,
    dst: &ProtocolObject<dyn MTLBuffer>,
    dump: &ProtocolObject<dyn MTLBuffer>,
    hidden: u32,
    batch_size: u32,
    elem_base: u32,
) {
    let dims = [0u32, hidden];
    unsafe {
        enc.setBuffer_offset_atIndex(Some(src), 0, 0);
        enc.setBuffer_offset_atIndex(Some(indices), 0, 1);
        enc.setBuffer_offset_atIndex(Some(dst), 0, 2);
        enc.setBuffer_offset_atIndex(Some(dump), 0, 5);
    }
    gpu_common::set_bytes(enc, &dims, 3);
    gpu_common::set_bytes(enc, &batch_size, 4);
    gpu_common::set_bytes(enc, &elem_base, 6);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let out_len = f.out_len();
    let grid = out_len;
    let buf_src = pool
        .allocate(&ctx.device, f.src_bf16.len() * 2)
        .ok_or(Error::Format("alloc"))?;
    let buf_idx = pool
        .allocate(&ctx.device, f.indices.len() * 4)
        .ok_or(Error::Format("alloc"))?;
    let buf_dst = pool.allocate(&ctx.device, out_len * 4).ok_or(Error::Format("alloc"))?;
    let dump_bytes = if variant.dump_stage > 0 { out_len * 4 } else { 4 };
    let buf_d = pool.allocate(&ctx.device, dump_bytes).ok_or(Error::Format("alloc"))?;
    let src_bytes = unsafe {
        std::slice::from_raw_parts(f.src_bf16.as_ptr().cast::<u8>(), f.src_bf16.len() * 2)
    };
    BufferPool::write_bytes(&buf_src, src_bytes);
    let idx_bytes = unsafe {
        std::slice::from_raw_parts(f.indices.as_ptr().cast::<u8>(), f.indices.len() * 4)
    };
    BufferPool::write_bytes(&buf_idx, idx_bytes);
    gpu_common::dispatch_1d(&ctx.queue, &pipeline.pipeline, grid, |enc| {
        bind_gpu_buffers(
            enc,
            &buf_src,
            &buf_idx,
            &buf_dst,
            &buf_d,
            f.hidden as u32,
            f.batch_size() as u32,
            0,
        );
    })?;
    let mut out = vec![0.0f32; out_len];
    BufferPool::read_f32(&buf_dst, &mut out);
    Ok(out)
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn gpu(_: &Fixture, _: KernelVariant) -> Result<Vec<f32>, Error> {
    Err(Error::Format("Metal unavailable"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::kernels::sub::gather_rows_bf16_to_f32::cpu,
        cpu_oracle = crate::kernels::sub::gather_rows_bf16_to_f32::cpu_oracle,
        gpu = crate::kernels::sub::gather_rows_bf16_to_f32::gpu,
        fixture = crate::kernels::sub::gather_rows_bf16_to_f32::tiny_fixture,
        out_len = crate::kernels::sub::gather_rows_bf16_to_f32::fixture_len,
        formats: [F32],
        max_tol = 1e-5,
        min_cos = 0.99999,
    }

    kernel_oracle_matrix! {
        mod moe,
        cpu = crate::kernels::sub::gather_rows_bf16_to_f32::cpu,
        cpu_oracle = crate::kernels::sub::gather_rows_bf16_to_f32::cpu_oracle,
        gpu = crate::kernels::sub::gather_rows_bf16_to_f32::gpu,
        fixture = crate::kernels::sub::gather_rows_bf16_to_f32::moe_fixture,
        out_len = crate::kernels::sub::gather_rows_bf16_to_f32::fixture_len,
        formats: [F32],
        max_tol = 1e-5,
        min_cos = 0.99999,
    }

    kernel_oracle_matrix! {
        mod moe_routing,
        cpu = crate::kernels::sub::gather_rows_bf16_to_f32::cpu,
        cpu_oracle = crate::kernels::sub::gather_rows_bf16_to_f32::cpu_oracle,
        gpu = crate::kernels::sub::gather_rows_bf16_to_f32::gpu,
        fixture = crate::kernels::sub::gather_rows_bf16_to_f32::moe_routing_fixture,
        out_len = crate::kernels::sub::gather_rows_bf16_to_f32::fixture_len,
        formats: [F32],
        max_tol = 1e-5,
        min_cos = 0.99999,
    }
}
