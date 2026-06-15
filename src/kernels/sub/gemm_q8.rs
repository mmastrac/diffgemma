//! Tiled Q8 GEMM: `y[M,N] = x[M,K] @ Wq8[N,K]^T` (monolith `k_gemm_q8` body).

use super::f16;
use super::gemm_common;
use super::test_util::ElemFormat;
use crate::dgq::block::{q8_gemm_cpu, quantize_row_q8};
use crate::dgq::layout::{q8_matrix_bytes, q8_row_bytes};
use crate::safetensors::Error;

pub const ENTRY: &str = "gemm_q8";

const SHADER: &str = shader_include::include_metal!("kernels/gemm_q8.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
    pub w_f32: Vec<f32>,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.m * self.n
    }

    pub fn w_q8(&self) -> Vec<u8> {
        quantize_f32_matrix_q8(&self.w_f32, self.n, self.k)
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

fn quantize_f32_matrix_q8(rows: &[f32], out_dim: usize, in_dim: usize) -> Vec<u8> {
    let mut dst = vec![0u8; q8_matrix_bytes(out_dim, in_dim)];
    let row_bytes = q8_row_bytes(in_dim);
    for row in 0..out_dim {
        let off = row * in_dim;
        quantize_row_q8(&rows[off..off + in_dim], in_dim, &mut dst[row * row_bytes..]);
    }
    dst
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    let m = 4usize;
    let n = 64usize;
    let k = 64usize;
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.013).sin() * 0.25)
        .collect();
    let w_f32: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.007).cos() * 0.02)
        .collect();
    Fixture { x, w_f32, m, n, k }
}

pub fn tile_fixture(_: ElemFormat) -> Fixture {
    let m = 8usize;
    let n = 128usize;
    let k = 128usize;
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.011).sin() * 0.1)
        .collect();
    let w_f32: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.005).cos() * 0.03)
        .collect();
    Fixture { x, w_f32, m, n, k }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let w_q8 = f.w_q8();
    let mut out = vec![0.0f32; f.out_len()];
    q8_gemm_cpu(&f.x, f.m, f.k, &w_q8, f.n, &mut out);
    out.iter().map(|&v| f16::round_half(v)).collect()
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_gemm_subkernel(SHADER, ENTRY, n, k, false, super::QuantFormat::Q8 as u32)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2::runtime::ProtocolObject;
#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn bind_gpu_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    x: &ProtocolObject<dyn MTLBuffer>,
    y: &ProtocolObject<dyn MTLBuffer>,
    blob: &ProtocolObject<dyn MTLBuffer>,
    w_off: u64,
    m: u32,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(x), 0, 0);
        enc.setBuffer_offset_atIndex(Some(y), 0, 1);
        enc.setBuffer_offset_atIndex(Some(blob), 0, 2);
    }
    super::gpu_common::set_bytes(enc, &w_off, 3);
    super::gpu_common::set_bytes(enc, &m, 4);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture, _variant: super::KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, f.n as u32, f.k as u32)?;
    let mut pool = BufferPool::new();
    let w_q8 = f.w_q8();
    let buf_x = pool
        .allocate(&ctx.device, f.m * f.k * 2)
        .ok_or(Error::Format("alloc"))?;
    let buf_y = pool
        .allocate(&ctx.device, f.m * f.n * 2)
        .ok_or(Error::Format("alloc"))?;
    let buf_w = pool
        .allocate(&ctx.device, w_q8.len())
        .ok_or(Error::Format("alloc"))?;
    BufferPool::write_bf16(&buf_x, &f16::f32_slice_to_f16(&f.x));
    BufferPool::write_bytes(&buf_w, &w_q8);
    let (grid, tg) = gemm_common::dispatch_shape(f.m, f.n);
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    bind_gpu_buffers(&enc, &buf_x, &buf_y, &buf_w, 0, f.m as u32);
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let ptr = buf_y.contents().as_ptr() as *const u16;
    Ok((0..f.out_len())
        .map(|i| f16::f16_bits_to_f32(unsafe { *ptr.add(i) }))
        .collect())
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn gpu(_: &Fixture, _: super::KernelVariant) -> Result<Vec<f32>, Error> {
    Err(Error::Format("Metal unavailable"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::kernels::sub::gemm_q8::cpu,
        cpu_oracle = crate::kernels::sub::gemm_q8::cpu_oracle,
        gpu = crate::kernels::sub::gemm_q8::gpu,
        fixture = crate::kernels::sub::gemm_q8::tiny_fixture,
        out_len = crate::kernels::sub::gemm_q8::fixture_len,
        formats: [F32],
        max_tol = 0.05,
        min_cos = 0.999,
    }

    kernel_oracle_matrix! {
        mod tile,
        cpu = crate::kernels::sub::gemm_q8::cpu,
        cpu_oracle = crate::kernels::sub::gemm_q8::cpu_oracle,
        gpu = crate::kernels::sub::gemm_q8::gpu,
        fixture = crate::kernels::sub::gemm_q8::tile_fixture,
        out_len = crate::kernels::sub::gemm_q8::fixture_len,
        formats: [F32],
        max_tol = 0.05,
        min_cos = 0.999,
    }
}
