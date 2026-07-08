//! Scalar f32 Q8 GEMM: `C[M,N] = A[M,K] @ W[K,N]` (K-indexed rows / softembed).

use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::dgq::block::{q8_gemm_rowk_cpu, quantize_row_q8};
use crate::dgq::layout::q8_row_bytes;
use crate::safetensors::Error;

pub const ENTRY: &str = "gemm_q8_linear_kxn_f32";

const SHADER: &str = shader_include::include_metal!("kernels/gemm_q8_linear_kxn_f32.metal");

const THREADGROUP: usize = 16;

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

    /// `[K,N]` row-major q8 rows (each row length N).
    pub fn w_q8(&self) -> Vec<u8> {
        let row_bytes = q8_row_bytes(self.n);
        let mut dst = vec![0u8; self.k * row_bytes];
        for row in 0..self.k {
            let off = row * self.n;
            quantize_row_q8(
                &self.w_f32[off..off + self.n],
                self.n,
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
    let m = 2usize;
    let n = 16usize;
    let k = 8usize;
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.11).sin() * 0.3)
        .collect();
    let w_f32: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.05).cos() * 0.04)
        .collect();
    Fixture { x, w_f32, m, n, k }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let w = f.w_q8();
    let mut out = vec![0.0f32; f.out_len()];
    q8_gemm_rowk_cpu(&f.x, f.m, f.k, &w, f.n, &mut out);
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
    ctx.compile_subkernel(SHADER, ENTRY, variant)
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
) {
    let dims = [m, n, k];
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
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let w_q8 = f.w_q8();
    let buf_a = pool
        .allocate(&ctx.device, f.x.len() * 4)
        .ok_or(Error::Format("alloc"))?;
    let buf_w = pool
        .allocate(&ctx.device, w_q8.len())
        .ok_or(Error::Format("alloc"))?;
    let buf_c = pool
        .allocate(&ctx.device, f.out_len() * 4)
        .ok_or(Error::Format("alloc"))?;
    BufferPool::write_f32(&buf_a, &f.x);
    BufferPool::write_bytes(&buf_w, &w_q8);
    BufferPool::write_f32(&buf_c, &vec![0.0f32; f.out_len()]);

    let (grid, tg) = dispatch_shape(f.m, f.n);
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    bind_gpu_buffers(
        &enc, &buf_a, &buf_w, &buf_c, f.m as u32, f.n as u32, f.k as u32,
    );
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut out = vec![0.0f32; f.out_len()];
    BufferPool::read_f32(&buf_c, &mut out);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::kernels::sub::gemm_q8_linear_kxn_f32::cpu,
        cpu_oracle = crate::kernels::sub::gemm_q8_linear_kxn_f32::cpu_oracle,
        gpu = crate::kernels::sub::gemm_q8_linear_kxn_f32::gpu,
        fixture = crate::kernels::sub::gemm_q8_linear_kxn_f32::tiny_fixture,
        out_len = crate::kernels::sub::gemm_q8_linear_kxn_f32::fixture_len,
        formats: [F32],
        max_tol = 1e-4,
        min_cos = 0.9999,
    }
}
