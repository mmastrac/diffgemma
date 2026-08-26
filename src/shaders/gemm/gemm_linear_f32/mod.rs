//! Scalar f32 block GEMM: `C[M,N] = A[M,K] @ W[N,K]^T` (engine reference path).

use crate::Error;
use crate::dgq::block::quantize_row_q4;
use crate::dgq::layout::{nvfp4_matrix_bytes, q4_matrix_bytes, q4_row_bytes};
use crate::dgq::nvfp4::quantize_f32_matrix_nvfp4_with_scale;
use crate::shaders::QuantFormat;
use crate::shaders::cpu::gemm_linear_f32::gemm_linear_f32_cpu;
use crate::shaders::gpu_common;
use crate::shaders::test_util::ElemFormat;
use crate::shaders::variant::KernelVariant;

pub const ENTRY: &str = "gemm_linear_f32";

pub const SHADER: &str = include_str!("gemm_linear_f32.metal");

const THREADGROUP: usize = 16;

#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
    pub w_f32: Vec<f32>,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub format: QuantFormat,
    pub global_scale: f32,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.m * self.n
    }

    pub fn w_blob(&self) -> Vec<u8> {
        match self.format {
            QuantFormat::NvFp4 => {
                let mut dst = vec![0u8; nvfp4_matrix_bytes(self.n, self.k)];
                quantize_f32_matrix_nvfp4_with_scale(
                    &self.w_f32,
                    self.n,
                    self.k,
                    &mut dst,
                    self.global_scale,
                );
                dst
            }
            _ => {
                let mut dst = vec![0u8; q4_matrix_bytes(self.n, self.k)];
                let row_bytes = q4_row_bytes(self.k);
                for row in 0..self.n {
                    let off = row * self.k;
                    quantize_row_q4(
                        &self.w_f32[off..off + self.k],
                        self.k,
                        &mut dst[row * row_bytes..],
                    );
                }
                dst
            }
        }
    }

    pub fn groups_per_row(&self) -> u32 {
        (self.k as u32).div_ceil(32)
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

pub fn tiny_fixture_q4(_: ElemFormat) -> Fixture {
    matrix_fixture(QuantFormat::Q4Affine, 4, 32, 64)
}

pub fn tiny_fixture_nvfp4(_: ElemFormat) -> Fixture {
    matrix_fixture(QuantFormat::NvFp4, 4, 32, 64)
}

fn matrix_fixture(format: QuantFormat, m: usize, n: usize, k: usize) -> Fixture {
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.013).sin() * 0.25)
        .collect();
    let w_f32: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.007).cos() * 0.02)
        .collect();
    Fixture {
        x,
        w_f32,
        m,
        n,
        k,
        format,
        global_scale: 1.0,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    gemm_linear_f32_cpu(&f.x, f.m, f.k, &f.w_blob(), f.n, f.format)
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(target_os = "macos")]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    format: QuantFormat,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    let v = KernelVariant {
        quant_format: format,
        ..variant
    };
    ctx.compile_subkernel(SHADER, ENTRY, v)
}

#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLSize,
};

#[cfg(target_os = "macos")]
pub fn bind_gpu_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    a: &ProtocolObject<dyn MTLBuffer>,
    w: &ProtocolObject<dyn MTLBuffer>,
    c: &ProtocolObject<dyn MTLBuffer>,
    m: u32,
    n: u32,
    k: u32,
    groups_per_row: u32,
) {
    let dims = [m, n, k, groups_per_row];
    unsafe {
        enc.setBuffer_offset_atIndex(Some(a), 0, 0);
        enc.setBuffer_offset_atIndex(Some(w), 0, 1);
        enc.setBuffer_offset_atIndex(Some(c), 0, 2);
    }
    gpu_common::set_bytes(enc, &dims, 3);
}

#[cfg(target_os = "macos")]
pub fn dispatch_shape(m: usize, n: usize) -> (MTLSize, MTLSize) {
    let tg = MTLSize {
        width: THREADGROUP,
        height: THREADGROUP,
        depth: 1,
    };
    let grid = MTLSize {
        width: gpu_common::div_up(n, THREADGROUP),
        height: gpu_common::div_up(m, THREADGROUP),
        depth: 1,
    };
    (grid, tg)
}

#[cfg(target_os = "macos")]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, f.format, variant)?;
    let mut pool = BufferPool::new();
    let w_blob = f.w_blob();
    let buf_a = pool
        .allocate(&ctx.device, f.x.len() * 4)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_w = pool
        .allocate(&ctx.device, w_blob.len())
        .ok_or(Error::Gpu("alloc"))?;
    let buf_c = pool
        .allocate(&ctx.device, f.out_len() * 4)
        .ok_or(Error::Gpu("alloc"))?;
    BufferPool::write_f32(&buf_a, &f.x);
    BufferPool::write_bytes(&buf_w, &w_blob);
    BufferPool::write_f32(&buf_c, &vec![0.0f32; f.out_len()]);

    let (grid, tg) = dispatch_shape(f.m, f.n);
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    bind_gpu_buffers(
        &enc,
        &buf_a,
        &buf_w,
        &buf_c,
        f.m as u32,
        f.n as u32,
        f.k as u32,
        f.groups_per_row(),
    );
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut out = vec![0.0f32; f.out_len()];
    BufferPool::read_f32(&buf_c, &mut out);
    Ok(out)
}

#[cfg(target_os = "macos")]
pub fn gpu_q4(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    gpu(
        f,
        KernelVariant {
            quant_format: QuantFormat::Q4Affine,
            ..variant
        },
    )
}

#[cfg(target_os = "macos")]
pub fn gpu_nvfp4(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    gpu(
        f,
        KernelVariant {
            quant_format: QuantFormat::NvFp4,
            ..variant
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny_q4,
        cpu = crate::shaders::gemm_linear_f32::cpu,
        cpu_oracle = crate::shaders::gemm_linear_f32::cpu_oracle,
        gpu = crate::shaders::gemm_linear_f32::gpu_q4,
        fixture = crate::shaders::gemm_linear_f32::tiny_fixture_q4,
        out_len = crate::shaders::gemm_linear_f32::fixture_len,
        formats: [F32],
        max_tol = 1e-4,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod tiny_nvfp4,
        cpu = crate::shaders::gemm_linear_f32::cpu,
        cpu_oracle = crate::shaders::gemm_linear_f32::cpu_oracle,
        gpu = crate::shaders::gemm_linear_f32::gpu_nvfp4,
        fixture = crate::shaders::gemm_linear_f32::tiny_fixture_nvfp4,
        out_len = crate::shaders::gemm_linear_f32::fixture_len,
        formats: [F32],
        max_tol = 1e-4,
        min_cos = 0.9999,
    }
}

pub mod cpu;

crate::kernel_spec! {
    pub const SPEC {
        name: "gemm_linear_f32",
        entry: "gemm_linear_f32",
        source: SHADER,
        quant_formats: &[
            QuantFormat::Q4Affine,
            QuantFormat::NvFp4,
        ],
        fc: &[],
        variants: KernelVariants::Elementwise,
    }
}
