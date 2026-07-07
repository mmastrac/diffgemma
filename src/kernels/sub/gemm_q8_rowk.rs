//! Tiled Q8 GEMM: `y[M,N] = x[M,K] @ W[K,N]` with K-indexed rows (monolith `k_gemm_q8_rowk`).

use super::bf16;
use super::f16;
use super::gemm_common;
use super::test_util::ElemFormat;
use crate::dgq::block::{q8_gemm_rowk_cpu, quantize_row_q8};
use crate::dgq::layout::q8_row_bytes;
use crate::safetensors::Error;

pub const ENTRY: &str = "gemm_q8_rowk";

const SHADER: &str = shader_include::include_metal!("kernels/gemm_q8_rowk.metal");

#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
    /// Row-major `[K, N]` (vocab × hidden).
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
        quantize_f32_matrix_q8_rowk(&self.w_f32, self.k, self.n)
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

fn quantize_f32_matrix_q8_rowk(rows: &[f32], k_dim: usize, n_dim: usize) -> Vec<u8> {
    let row_bytes = q8_row_bytes(n_dim);
    let mut dst = vec![0u8; k_dim * row_bytes];
    for row in 0..k_dim {
        let off = row * n_dim;
        quantize_row_q8(&rows[off..off + n_dim], n_dim, &mut dst[row * row_bytes..]);
    }
    dst
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    let m = 2usize;
    let k = 64usize;
    let n = 32usize;
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.017).sin() * 0.2)
        .collect();
    let w_f32: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.009).cos() * 0.015)
        .collect();
    Fixture { x, w_f32, m, n, k }
}

pub fn tile_fixture(_: ElemFormat) -> Fixture {
    let m = 4usize;
    let k = 128usize;
    let n = 64usize;
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.012).sin() * 0.12)
        .collect();
    let w_f32: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.006).cos() * 0.025)
        .collect();
    Fixture { x, w_f32, m, n, k }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let w_q8 = f.w_q8();
    let mut out = vec![0.0f32; f.out_len()];
    q8_gemm_rowk_cpu(&f.x, f.m, f.k, &w_q8, f.n, &mut out);
    out.iter()
        .map(|&v| bf16::store_bf16_round_half(v))
        .collect()
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
    ctx.compile_gemm_subkernel(
        SHADER,
        ENTRY,
        n,
        k,
        false,
        super::QuantFormat::Q8 as u32,
        false,
    )
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_for_fp16_input(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_gemm_subkernel(
        SHADER,
        ENTRY,
        n,
        k,
        false,
        super::QuantFormat::Q8 as u32,
        true,
    )
}

#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2::runtime::ProtocolObject;
#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
};

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
    BufferPool::write_bf16(&buf_x, &bf16::f32_slice_to_bf16_bits(&f.x));
    BufferPool::write_bytes(&buf_w, &w_q8);
    // gemm_q8_rowk is hardcoded to 32x32 tiles (tgid.{x,y} * 32) — production
    // dispatches div_up(n,32) x div_up(m,32). gemm_common::dispatch_shape's
    // 128-wide n-tile left columns 32..127 of each block unwritten (the
    // long-standing cos~0.34 harness failure).
    let grid = objc2_metal::MTLSize {
        width: f.n.div_ceil(32),
        height: f.m.div_ceil(32),
        depth: 1,
    };
    let tg = objc2_metal::MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
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
        .map(|i| bf16::bf16_bits_to_f32(unsafe { *ptr.add(i) }))
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
        cpu = crate::kernels::sub::gemm_q8_rowk::cpu,
        cpu_oracle = crate::kernels::sub::gemm_q8_rowk::cpu_oracle,
        gpu = crate::kernels::sub::gemm_q8_rowk::gpu,
        fixture = crate::kernels::sub::gemm_q8_rowk::tiny_fixture,
        out_len = crate::kernels::sub::gemm_q8_rowk::fixture_len,
        formats: [F32],
        max_tol = 0.05,
        min_cos = 0.999,
    }

    kernel_oracle_matrix! {
        mod tile,
        cpu = crate::kernels::sub::gemm_q8_rowk::cpu,
        cpu_oracle = crate::kernels::sub::gemm_q8_rowk::cpu_oracle,
        gpu = crate::kernels::sub::gemm_q8_rowk::gpu,
        fixture = crate::kernels::sub::gemm_q8_rowk::tile_fixture,
        out_len = crate::kernels::sub::gemm_q8_rowk::fixture_len,
        formats: [F32],
        max_tol = 0.05,
        min_cos = 0.999,
    }
}
