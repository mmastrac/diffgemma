//! Per-row scale for MoE router input: `x[s,:] *= scale[:] * root`.

use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::safetensors::Error;

pub const ENTRY: &str = "router_scale_rows";

const SHADER: &str = concat!(
    include_str!("../../../shaders/kernels/common.metal"),
    include_str!("../../../shaders/kernels/router_scale_rows.metal"),
);

#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
    pub scale: Vec<f32>,
    pub seq_len: usize,
    pub hidden: usize,
    pub root: f32,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.seq_len * self.hidden
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        x: vec![1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.5, 0.25],
        scale: vec![1.0, 0.5, 2.0, 1.5],
        seq_len: 2,
        hidden: 4,
        root: 0.1,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let mut out = f.x.clone();
    for s in 0..f.seq_len {
        let off = s * f.hidden;
        for i in 0..f.hidden {
            out[off + i] *= f.scale[i] * f.root;
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
    x: &ProtocolObject<dyn MTLBuffer>,
    scale: &ProtocolObject<dyn MTLBuffer>,
    dump: &ProtocolObject<dyn MTLBuffer>,
    dims: &[u32; 2],
    root: f32,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(x), 0, 0);
        enc.setBuffer_offset_atIndex(Some(scale), 0, 1);
        enc.setBuffer_offset_atIndex(Some(dump), 0, 4);
    }
    gpu_common::set_bytes(enc, dims, 2);
    gpu_common::set_bytes(enc, &root, 3);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let len = f.len();
    let buf_x = pool.allocate(&ctx.device, len * 4).ok_or(Error::Format("alloc"))?;
    let buf_s = pool
        .allocate(&ctx.device, f.hidden * 4)
        .ok_or(Error::Format("alloc"))?;
    let dump_bytes = if variant.dump_stage > 0 { f.seq_len * 4 } else { 4 };
    let buf_d = pool.allocate(&ctx.device, dump_bytes).ok_or(Error::Format("alloc"))?;
    BufferPool::write_f32(&buf_x, &f.x);
    BufferPool::write_f32(&buf_s, &f.scale);
    gpu_common::dispatch_1d(&ctx.queue, &pipeline.pipeline, f.seq_len, |enc| {
        let dims = [f.seq_len as u32, f.hidden as u32];
        bind_gpu_buffers(enc, &buf_x, &buf_s, &buf_d, &dims, f.root);
    })?;
    let mut out = vec![0.0f32; len];
    BufferPool::read_f32(&buf_x, &mut out);
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
        cpu = crate::kernels::sub::router_scale_rows::cpu,
        cpu_oracle = crate::kernels::sub::router_scale_rows::cpu_oracle,
        gpu = crate::kernels::sub::router_scale_rows::gpu,
        fixture = crate::kernels::sub::router_scale_rows::tiny_fixture,
        out_len = crate::kernels::sub::router_scale_rows::fixture_len,
        formats: [F32],
        max_tol = 1e-6,
        min_cos = 0.9999,
    }
}
