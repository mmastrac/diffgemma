//! Scalar f32 Q8 GEMM: `C[M,N] = A[M,K] @ W[N,K]^T`.

use crate::dgq::block::{q8_gemm_cpu, quantize_row_q8};
use crate::dgq::layout::{q8_matrix_bytes, q8_row_bytes};
use crate::safetensors::Error;
use crate::shaders::gpu_common;
use crate::shaders::test_util::ElemFormat;
use crate::shaders::variant::KernelVariant;

pub const ENTRY: &str = "gemm_q8_linear_f32";

pub const SHADER: &str = include_str!("gemm_q8_linear_f32.metal");

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

    pub fn w_q8(&self) -> Vec<u8> {
        let mut dst = vec![0u8; q8_matrix_bytes(self.n, self.k)];
        let row_bytes = q8_row_bytes(self.k);
        for row in 0..self.n {
            let off = row * self.k;
            quantize_row_q8(
                &self.w_f32[off..off + self.k],
                self.k,
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
    let m = 4usize;
    let n = 32usize;
    let k = 64usize;
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.013).sin() * 0.25)
        .collect();
    let w_f32: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.007).cos() * 0.02)
        .collect();
    Fixture { x, w_f32, m, n, k }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let w = f.w_q8();
    let mut out = vec![0.0f32; f.out_len()];
    q8_gemm_cpu(&f.x, f.m, f.k, &w, f.n, &mut out);
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
    pipeline_for_kxn(ctx, variant, false)
}

/// Weight K-order axis (K_W_KXN / FC4): false = W[N,K]^T, true = W[K,N].
/// One kernel; the kxn variant serves the embed / SC-softembed layout.
#[cfg(target_os = "macos")]
pub fn pipeline_for_kxn(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
    kxn: bool,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    use crate::shaders::variant::FcBool;
    let bools = [FcBool {
        index: 4,
        value: kxn,
    }];
    let label = if kxn { "kxn" } else { "nxk" };
    ctx.compile_subkernel_ex(SHADER, ENTRY, variant, label, &bools, &[])
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
        .ok_or(Error::Gpu("alloc"))?;
    let buf_w = pool
        .allocate(&ctx.device, w_q8.len())
        .ok_or(Error::Gpu("alloc"))?;
    let buf_c = pool
        .allocate(&ctx.device, f.out_len() * 4)
        .ok_or(Error::Gpu("alloc"))?;
    BufferPool::write_f32(&buf_a, &f.x);
    BufferPool::write_bytes(&buf_w, &w_q8);
    BufferPool::write_f32(&buf_c, &vec![0.0f32; f.out_len()]);

    let (grid, tg) = dispatch_shape(f.m, f.n);
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
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
        cpu = crate::shaders::gemm_q8_linear_f32::cpu,
        cpu_oracle = crate::shaders::gemm_q8_linear_f32::cpu_oracle,
        gpu = crate::shaders::gemm_q8_linear_f32::gpu,
        fixture = crate::shaders::gemm_q8_linear_f32::tiny_fixture,
        out_len = crate::shaders::gemm_q8_linear_f32::fixture_len,
        formats: [F32],
        max_tol = 1e-4,
        min_cos = 0.9999,
    }
}

pub mod kxn;

/// Manifest registration; collected in common/manifest.rs::MANIFEST.
pub const SPEC: crate::shaders::manifest::KernelSpec = crate::shaders::manifest::KernelSpec {
    name: "gemm_q8_linear_f32",
    entry: "gemm_q8_linear_f32",
    quant_formats: &[crate::shaders::variant::QuantFormat::Q4Affine],
    fc: &[],
    variants: crate::shaders::manifest::KernelVariants::Elementwise,
};
