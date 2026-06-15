//! Gather rows by index from a row-major `[tokens, hidden]` source.

use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::safetensors::Error;

pub const ENTRY: &str = "gather_rows";

const SHADER: &str = shader_include::include_metal!("kernels/gather_rows.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub src: Vec<f32>,
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

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        src: vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0,
        ],
        indices: vec![2, 0],
        hidden: 4,
        num_tokens: 3,
    }
}

pub fn moe_fixture(_: ElemFormat) -> Fixture {
    let hidden = 64;
    let num_tokens = 16;
    let src: Vec<f32> = (0..num_tokens * hidden)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let indices: Vec<u32> = (0..8).map(|i| (i * 2) as u32).collect();
    Fixture {
        src,
        indices,
        hidden,
        num_tokens,
    }
}

/// 32-slot gather order mimicking batched MoE bucket fill (non-sequential token indices).
pub fn moe_routing_fixture(_: ElemFormat) -> Fixture {
    let hidden = 64;
    let num_tokens = 256;
    let src: Vec<f32> = (0..num_tokens * hidden)
        .map(|i| ((i as f32) * 0.0023).sin() * 0.5 + 0.25)
        .collect();
    let indices: Vec<u32> = vec![
        103, 87, 44, 12, 201, 155, 3, 98, //
        17, 240, 66, 129, 8, 190, 51, 222, //
        74, 11, 183, 145, 28, 99, 167, 4, //
        131, 58, 210, 37, 172, 95, 21, 248,
    ];
    Fixture {
        src,
        indices,
        hidden,
        num_tokens,
    }
}

/// 64-slot gather order from Calgary L0 batched MoE capture (seed 42).
pub fn moe_batched_pin_l0_fixture(_: ElemFormat) -> Fixture {
    let hidden = 64;
    let num_tokens = crate::metal::CANVAS;
    let token_list = crate::kernels::sub::moe_batched_pin::calgary_l0_token_list();
    let src: Vec<f32> = (0..num_tokens * hidden)
        .map(|i| ((i as f32) * 0.0023).sin() * 0.5 + 0.25)
        .collect();
    Fixture {
        src,
        indices: token_list.to_vec(),
        hidden,
        num_tokens,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let mut out = vec![0.0f32; f.out_len()];
    for (bi, &tok) in f.indices.iter().enumerate() {
        let src_off = tok as usize * f.hidden;
        let dst_off = bi * f.hidden;
        out[dst_off..dst_off + f.hidden]
            .copy_from_slice(&f.src[src_off..src_off + f.hidden]);
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
    let buf_src = pool.allocate(&ctx.device, f.src.len() * 4).ok_or(Error::Format("alloc"))?;
    let buf_idx = pool
        .allocate(&ctx.device, f.indices.len() * 4)
        .ok_or(Error::Format("alloc"))?;
    let buf_dst = pool.allocate(&ctx.device, out_len * 4).ok_or(Error::Format("alloc"))?;
    let dump_bytes = if variant.dump_stage > 0 { out_len * 4 } else { 4 };
    let buf_d = pool.allocate(&ctx.device, dump_bytes).ok_or(Error::Format("alloc"))?;
    BufferPool::write_f32(&buf_src, &f.src);
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
        cpu = crate::kernels::sub::gather_rows::cpu,
        cpu_oracle = crate::kernels::sub::gather_rows::cpu_oracle,
        gpu = crate::kernels::sub::gather_rows::gpu,
        fixture = crate::kernels::sub::gather_rows::tiny_fixture,
        out_len = crate::kernels::sub::gather_rows::fixture_len,
        formats: [F32],
        max_tol = 1e-6,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod moe,
        cpu = crate::kernels::sub::gather_rows::cpu,
        cpu_oracle = crate::kernels::sub::gather_rows::cpu_oracle,
        gpu = crate::kernels::sub::gather_rows::gpu,
        fixture = crate::kernels::sub::gather_rows::moe_fixture,
        out_len = crate::kernels::sub::gather_rows::fixture_len,
        formats: [F32],
        max_tol = 1e-6,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod moe_routing,
        cpu = crate::kernels::sub::gather_rows::cpu,
        cpu_oracle = crate::kernels::sub::gather_rows::cpu_oracle,
        gpu = crate::kernels::sub::gather_rows::gpu,
        fixture = crate::kernels::sub::gather_rows::moe_routing_fixture,
        out_len = crate::kernels::sub::gather_rows::fixture_len,
        formats: [F32],
        max_tol = 1e-6,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod moe_batched_pin_l0,
        cpu = crate::kernels::sub::gather_rows::cpu,
        cpu_oracle = crate::kernels::sub::gather_rows::cpu_oracle,
        gpu = crate::kernels::sub::gather_rows::gpu,
        fixture = crate::kernels::sub::gather_rows::moe_batched_pin_l0_fixture,
        out_len = crate::kernels::sub::gather_rows::fixture_len,
        formats: [F32],
        max_tol = 1e-6,
        min_cos = 0.9999,
    }
}
