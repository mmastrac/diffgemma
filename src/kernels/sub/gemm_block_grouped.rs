//! Tiled grouped MoE GEMM: expert segments × 32×32 output tiles (simdgroup matmul).

use super::gemm_common;
use super::gpu_common;
use super::test_util::ElemFormat;
use super::QuantFormat;
use crate::kernels::cpu::gemm_linear_grouped::gemm_linear_grouped_cpu;
use crate::kernels::sub::gemm_linear_grouped::{grouped_fixture, Fixture};
use crate::metal::BlockGroupedJob;
use crate::safetensors::Error;

pub const ENTRY: &str = "gemm_block_grouped";

const SHADER: &str = shader_include::include_metal!("kernels/gemm_block_grouped.metal");

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

/// One expert with M=100 (>32 M-tile) plus smaller segments for multi-expert routing.
pub fn tiny_fixture_q4(_: ElemFormat) -> Fixture {
    grouped_fixture(QuantFormat::Q4Affine, 64, 32, &[100, 4])
}

pub fn tiny_fixture_nvfp4(_: ElemFormat) -> Fixture {
    grouped_fixture(QuantFormat::NvFp4, 64, 32, &[100, 4])
}

/// M=100 with N/K tiling (128×128).
pub fn tile_fixture_q4(_: ElemFormat) -> Fixture {
    grouped_fixture(QuantFormat::Q4Affine, 128, 128, &[100, 48, 4])
}

pub fn tile_fixture_nvfp4(_: ElemFormat) -> Fixture {
    grouped_fixture(QuantFormat::NvFp4, 128, 128, &[100, 48, 4])
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    gemm_linear_grouped_cpu(
        &f.a,
        f.total_m(),
        f.k,
        f.n,
        &f.w_blob(),
        &f.jobs(),
        &f.row_starts,
        f.format,
    )
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
    format: QuantFormat,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_gemm_subkernel(
        SHADER,
        ENTRY,
        n,
        k,
        false,
        format as u32,
    )
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
    a: &ProtocolObject<dyn MTLBuffer>,
    w_blob: &ProtocolObject<dyn MTLBuffer>,
    c: &ProtocolObject<dyn MTLBuffer>,
    jobs: &ProtocolObject<dyn MTLBuffer>,
    row_starts: &ProtocolObject<dyn MTLBuffer>,
    num_jobs: u32,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(a), 0, 0);
        enc.setBuffer_offset_atIndex(Some(w_blob), 0, 1);
        enc.setBuffer_offset_atIndex(Some(c), 0, 2);
        enc.setBuffer_offset_atIndex(Some(jobs), 0, 3);
        enc.setBuffer_offset_atIndex(Some(row_starts), 0, 4);
    }
    gpu_common::set_bytes(enc, &num_jobs, 5);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn dispatch_shape(n: usize, num_jobs: usize) -> (MTLSize, MTLSize) {
    (
        MTLSize {
            width: gemm_common::div_up(n, 32),
            height: num_jobs,
            depth: 1,
        },
        MTLSize {
            width: gemm_common::THREADS_PER_TG,
            height: 1,
            depth: 1,
        },
    )
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture, _variant: super::KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, f.n as u32, f.k as u32, f.format)?;
    let mut pool = BufferPool::new();
    let jobs = f.jobs();
    let w_blob = f.w_blob();
    let out_len = f.out_len();
    let num_jobs = f.num_jobs();

    let buf_a = pool
        .allocate(&ctx.device, f.a.len() * 4)
        .ok_or(Error::Format("alloc"))?;
    let buf_w = pool
        .allocate(&ctx.device, w_blob.len())
        .ok_or(Error::Format("alloc"))?;
    let buf_c = pool
        .allocate(&ctx.device, out_len * 4)
        .ok_or(Error::Format("alloc"))?;
    let buf_jobs = pool
        .allocate(&ctx.device, jobs.len() * std::mem::size_of::<BlockGroupedJob>())
        .ok_or(Error::Format("alloc"))?;
    let buf_rs = pool
        .allocate(&ctx.device, f.row_starts.len() * 4)
        .ok_or(Error::Format("alloc"))?;

    BufferPool::write_f32(&buf_a, &f.a);
    BufferPool::write_bytes(&buf_w, &w_blob);
    BufferPool::write_bytes(
        &buf_jobs,
        unsafe {
            std::slice::from_raw_parts(
                jobs.as_ptr().cast::<u8>(),
                jobs.len() * std::mem::size_of::<BlockGroupedJob>(),
            )
        },
    );
    BufferPool::write_bytes(
        &buf_rs,
        unsafe {
            std::slice::from_raw_parts(
                f.row_starts.as_ptr().cast::<u8>(),
                f.row_starts.len() * 4,
            )
        },
    );
    BufferPool::write_f32(&buf_c, &vec![0.0f32; out_len]);

    let (grid, tg) = dispatch_shape(f.n, num_jobs);
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    bind_gpu_buffers(
        &enc,
        &buf_a,
        &buf_w,
        &buf_c,
        &buf_jobs,
        &buf_rs,
        num_jobs as u32,
    );
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut out = vec![0.0f32; out_len];
    BufferPool::read_f32(&buf_c, &mut out);
    Ok(out)
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn gpu(_: &Fixture, _: super::KernelVariant) -> Result<Vec<f32>, Error> {
    Err(Error::Format("Metal unavailable"))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu_q4(f: &Fixture, variant: super::KernelVariant) -> Result<Vec<f32>, Error> {
    gpu(
        f,
        super::KernelVariant {
            quant_format: QuantFormat::Q4Affine,
            ..variant
        },
    )
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu_nvfp4(f: &Fixture, variant: super::KernelVariant) -> Result<Vec<f32>, Error> {
    gpu(
        f,
        super::KernelVariant {
            quant_format: QuantFormat::NvFp4,
            ..variant
        },
    )
}

/// Grouped tiled GEMM against a shared `.dgq` blob (absolute `w_byte_off` in jobs).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub struct BlobGroupedParams<'a> {
    pub blob: &'a ProtocolObject<dyn MTLBuffer>,
    pub a: &'a [f32],
    pub jobs: &'a [BlockGroupedJob],
    pub row_starts: &'a [u32],
    pub k: usize,
    pub n: usize,
    pub format: QuantFormat,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl BlobGroupedParams<'_> {
    pub fn total_m(&self) -> usize {
        *self.row_starts.last().unwrap_or(&0) as usize
    }

    pub fn out_len(&self) -> usize {
        self.total_m() * self.n
    }

    pub fn num_jobs(&self) -> usize {
        self.row_starts.len().saturating_sub(1)
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu_on_blob(
    p: &BlobGroupedParams<'_>,
    _variant: super::KernelVariant,
) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, p.n as u32, p.k as u32, p.format)?;
    let mut pool = BufferPool::new();
    let out_len = p.out_len();
    let num_jobs = p.num_jobs();

    let buf_a = pool
        .allocate(&ctx.device, p.a.len() * 4)
        .ok_or(Error::Format("alloc"))?;
    let buf_c = pool
        .allocate(&ctx.device, out_len * 4)
        .ok_or(Error::Format("alloc"))?;
    let buf_jobs = pool
        .allocate(
            &ctx.device,
            p.jobs.len() * std::mem::size_of::<BlockGroupedJob>(),
        )
        .ok_or(Error::Format("alloc"))?;
    let buf_rs = pool
        .allocate(&ctx.device, p.row_starts.len() * 4)
        .ok_or(Error::Format("alloc"))?;

    BufferPool::write_f32(&buf_a, p.a);
    BufferPool::write_f32(&buf_c, &vec![0.0f32; out_len]);
    BufferPool::write_bytes(
        &buf_jobs,
        unsafe {
            std::slice::from_raw_parts(
                p.jobs.as_ptr().cast::<u8>(),
                p.jobs.len() * std::mem::size_of::<BlockGroupedJob>(),
            )
        },
    );
    BufferPool::write_bytes(
        &buf_rs,
        unsafe {
            std::slice::from_raw_parts(
                p.row_starts.as_ptr().cast::<u8>(),
                p.row_starts.len() * 4,
            )
        },
    );

    let (grid, tg) = dispatch_shape(p.n, num_jobs);
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    bind_gpu_buffers(
        &enc,
        &buf_a,
        p.blob,
        &buf_c,
        &buf_jobs,
        &buf_rs,
        num_jobs as u32,
    );
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut out = vec![0.0f32; out_len];
    BufferPool::read_f32(&buf_c, &mut out);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny_q4,
        cpu = crate::kernels::sub::gemm_block_grouped::cpu,
        cpu_oracle = crate::kernels::sub::gemm_block_grouped::cpu_oracle,
        gpu = crate::kernels::sub::gemm_block_grouped::gpu_q4,
        fixture = crate::kernels::sub::gemm_block_grouped::tiny_fixture_q4,
        out_len = crate::kernels::sub::gemm_block_grouped::fixture_len,
        formats: [F32],
        max_tol = 0.05,
        min_cos = 0.999,
    }

    kernel_oracle_matrix! {
        mod tiny_nvfp4,
        cpu = crate::kernels::sub::gemm_block_grouped::cpu,
        cpu_oracle = crate::kernels::sub::gemm_block_grouped::cpu_oracle,
        gpu = crate::kernels::sub::gemm_block_grouped::gpu_nvfp4,
        fixture = crate::kernels::sub::gemm_block_grouped::tiny_fixture_nvfp4,
        out_len = crate::kernels::sub::gemm_block_grouped::fixture_len,
        formats: [F32],
        max_tol = 0.05,
        min_cos = 0.999,
    }

    kernel_oracle_matrix! {
        mod tile_q4,
        cpu = crate::kernels::sub::gemm_block_grouped::cpu,
        cpu_oracle = crate::kernels::sub::gemm_block_grouped::cpu_oracle,
        gpu = crate::kernels::sub::gemm_block_grouped::gpu_q4,
        fixture = crate::kernels::sub::gemm_block_grouped::tile_fixture_q4,
        out_len = crate::kernels::sub::gemm_block_grouped::fixture_len,
        formats: [F32],
        max_tol = 0.05,
        min_cos = 0.999,
    }

    kernel_oracle_matrix! {
        mod tile_nvfp4,
        cpu = crate::kernels::sub::gemm_block_grouped::cpu,
        cpu_oracle = crate::kernels::sub::gemm_block_grouped::cpu_oracle,
        gpu = crate::kernels::sub::gemm_block_grouped::gpu_nvfp4,
        fixture = crate::kernels::sub::gemm_block_grouped::tile_fixture_nvfp4,
        out_len = crate::kernels::sub::gemm_block_grouped::fixture_len,
        formats: [F32],
        max_tol = 0.05,
        min_cos = 0.999,
    }
}
