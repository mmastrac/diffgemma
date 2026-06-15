//! Block matrix dequant for MPS dense scratch path.

use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use super::QuantFormat;
use crate::dgq::block::quantize_row_q4;
use crate::dgq::layout::{nvfp4_matrix_bytes, q4_matrix_bytes, q4_row_bytes};
use crate::dgq::nvfp4::quantize_f32_matrix_nvfp4;
use crate::kernels::cpu::dequant_block_matrix::dequant_block_matrix_cpu;
use crate::safetensors::Error;

pub const ENTRY: &str = "dequant_block_matrix";

const SHADER: &str = shader_include::include_metal!("kernels/dequant_block_matrix.metal");

const THREADGROUP: usize = 16;

#[derive(Debug, Clone)]
pub struct Fixture {
    pub w_f32: Vec<f32>,
    pub n: usize,
    pub k: usize,
    pub format: QuantFormat,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.n * self.k
    }

    pub fn w_blob(&self) -> Vec<u8> {
        match self.format {
            QuantFormat::NvFp4 => {
                let mut dst = vec![0u8; nvfp4_matrix_bytes(self.n, self.k)];
                quantize_f32_matrix_nvfp4(&self.w_f32, self.n, self.k, &mut dst);
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
        match self.format {
            QuantFormat::NvFp4 => 0,
            _ => (self.k as u32).div_ceil(32),
        }
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

pub fn tiny_fixture_q4(_: ElemFormat) -> Fixture {
    matrix_fixture(QuantFormat::Q4Affine, 8, 64)
}

pub fn tiny_fixture_nvfp4(_: ElemFormat) -> Fixture {
    matrix_fixture(QuantFormat::NvFp4, 8, 64)
}

pub fn wide_fixture_nvfp4(_: ElemFormat) -> Fixture {
    matrix_fixture(QuantFormat::NvFp4, 16, 128)
}

fn matrix_fixture(format: QuantFormat, n: usize, k: usize) -> Fixture {
    let w_f32: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.011).sin() * 0.15)
        .collect();
    Fixture {
        w_f32,
        n,
        k,
        format,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    dequant_block_matrix_cpu(&f.w_blob(), f.n, f.k, f.format)
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
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

#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2::runtime::ProtocolObject;
#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLSize,
};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn bind_gpu_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    w: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
    k: u32,
    groups_per_row: u32,
) {
    let dims = [n, k, groups_per_row];
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w), 0, 0);
        enc.setBuffer_offset_atIndex(Some(out), 0, 1);
    }
    gpu_common::set_bytes(enc, &dims, 2);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn dispatch_shape(n: usize, k: usize) -> (MTLSize, MTLSize) {
    let tg = MTLSize {
        width: THREADGROUP,
        height: THREADGROUP,
        depth: 1,
    };
    let grid = MTLSize {
        width: (k + THREADGROUP - 1) / THREADGROUP,
        height: (n + THREADGROUP - 1) / THREADGROUP,
        depth: 1,
    };
    (grid, tg)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, f.format, variant)?;
    let mut pool = BufferPool::new();
    let w_blob = f.w_blob();
    let out_len = f.out_len();
    let buf_w = pool
        .allocate(&ctx.device, w_blob.len())
        .ok_or(Error::Format("alloc"))?;
    let buf_out = pool
        .allocate(&ctx.device, out_len * 4)
        .ok_or(Error::Format("alloc"))?;
    BufferPool::write_bytes(&buf_w, &w_blob);
    BufferPool::write_f32(&buf_out, &vec![0.0f32; out_len]);

    let (grid, tg) = dispatch_shape(f.n, f.k);
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    bind_gpu_buffers(
        &enc,
        &buf_w,
        &buf_out,
        f.n as u32,
        f.k as u32,
        f.groups_per_row(),
    );
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut out = vec![0.0f32; out_len];
    BufferPool::read_f32(&buf_out, &mut out);
    Ok(out)
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn gpu(_: &Fixture, _: KernelVariant) -> Result<Vec<f32>, Error> {
    Err(Error::Format("Metal unavailable"))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu_q4(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    gpu(
        f,
        KernelVariant {
            quant_format: QuantFormat::Q4Affine,
            ..variant
        },
    )
}

#[cfg(all(feature = "metal", target_os = "macos"))]
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
        cpu = crate::kernels::sub::dequant_block_matrix::cpu,
        cpu_oracle = crate::kernels::sub::dequant_block_matrix::cpu_oracle,
        gpu = crate::kernels::sub::dequant_block_matrix::gpu_q4,
        fixture = crate::kernels::sub::dequant_block_matrix::tiny_fixture_q4,
        out_len = crate::kernels::sub::dequant_block_matrix::fixture_len,
        formats: [F32],
        max_tol = 1e-4,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod tiny_nvfp4,
        cpu = crate::kernels::sub::dequant_block_matrix::cpu,
        cpu_oracle = crate::kernels::sub::dequant_block_matrix::cpu_oracle,
        gpu = crate::kernels::sub::dequant_block_matrix::gpu_nvfp4,
        fixture = crate::kernels::sub::dequant_block_matrix::tiny_fixture_nvfp4,
        out_len = crate::kernels::sub::dequant_block_matrix::fixture_len,
        formats: [F32],
        max_tol = 1e-4,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod wide_nvfp4,
        cpu = crate::kernels::sub::dequant_block_matrix::cpu,
        cpu_oracle = crate::kernels::sub::dequant_block_matrix::cpu_oracle,
        gpu = crate::kernels::sub::dequant_block_matrix::gpu_nvfp4,
        fixture = crate::kernels::sub::dequant_block_matrix::wide_fixture_nvfp4,
        out_len = crate::kernels::sub::dequant_block_matrix::fixture_len,
        formats: [F32],
        max_tol = 1e-4,
        min_cos = 0.9999,
    }
}
