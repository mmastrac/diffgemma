//! Tiled bf16-weight GEMM: `y[M,N] = x[M,K] @ W[N,K]^T` with bf16 weights (no
//! quantization). Same tiling as `gemm_q8` but reads the weight tile straight
//! from bf16 rows — lossless into the half tiles since the weights are bf16.
//! Used for the mixed-precision `.dgq` attention/dense-FFN path.

use crate::safetensors::Error;

pub const ENTRY: &str = "gemm_bf16";

const SHADER: &str = shader_include::include_metal!("kernels/gemm_bf16.metal");

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    // quant_format / x_fp16 are unused by the bf16 kernel; pass inert values.
    ctx.compile_gemm_subkernel(SHADER, ENTRY, n, k, false, super::QuantFormat::Q8 as u32, false)
}

/// lm_head logits pipeline: forces bf16 output (FC29) so logits keep bf16's range
/// even when K_ACT_F16 (f16 activations) is on for the input.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_for_logits(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_gemm_subkernel_out_bf16(SHADER, ENTRY, n, k, super::QuantFormat::Q8 as u32)
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn pipeline_for(
    _ctx: &crate::metal::device::MetalContext,
    _n: u32,
    _k: u32,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    Err(Error::Format("Metal unavailable"))
}

#[cfg(test)]
use super::bf16;
#[cfg(test)]
use super::test_util::ElemFormat;

#[cfg(test)]
#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
    pub w_f32: Vec<f32>,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

#[cfg(test)]
impl Fixture {
    pub fn out_len(&self) -> usize {
        self.m * self.n
    }
}

#[cfg(test)]
pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

#[cfg(test)]
pub fn tile_fixture(_: ElemFormat) -> Fixture {
    // Non-uniform, multi-tile shape (N=160 > 128 N-tile, K=160 > 32 K-tile,
    // M=40 > 32 M-tile) so the test exercises grid tiling + tail handling.
    let m = 40usize;
    let n = 160usize;
    let k = 160usize;
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.017).sin() * 0.3 - 0.05)
        .collect();
    let w_f32: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.011).cos() * 0.04 + 0.01)
        .collect();
    Fixture { x, w_f32, m, n, k }
}

/// `y[m,n] = sum_k bf16(x) * bf16(w)`, f32 accumulate, bf16-rounded result —
/// mirrors the kernel (bf16 inputs into half tiles, float8x8 accumulate, arena store).
#[cfg(test)]
pub fn cpu(f: &Fixture) -> Vec<f32> {
    let xb: Vec<f32> = f.x.iter().map(|&v| bf16::round_bf16_f32(v)).collect();
    let wb: Vec<f32> = f.w_f32.iter().map(|&v| bf16::round_bf16_f32(v)).collect();
    let mut out = vec![0.0f32; f.out_len()];
    for mi in 0..f.m {
        for ni in 0..f.n {
            let mut acc = 0.0f32;
            for ki in 0..f.k {
                acc += xb[mi * f.k + ki] * wb[ni * f.k + ki];
            }
            out[mi * f.n + ni] = bf16::store_bf16_round_half(acc);
        }
    }
    out
}

#[cfg(test)]
pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(all(test, feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture, _variant: super::KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::kernels::sub::gemm_common;
    use crate::kernels::sub::gemm_q8;
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder};

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, f.n as u32, f.k as u32)?;
    let mut pool = BufferPool::new();
    let buf_x = pool
        .allocate(&ctx.device, f.m * f.k * 2)
        .ok_or(Error::Format("alloc"))?;
    let buf_y = pool
        .allocate(&ctx.device, f.m * f.n * 2)
        .ok_or(Error::Format("alloc"))?;
    let buf_w = pool
        .allocate(&ctx.device, f.n * f.k * 2)
        .ok_or(Error::Format("alloc"))?;
    BufferPool::write_bf16(&buf_x, &bf16::f32_slice_to_bf16_bits(&f.x));
    BufferPool::write_bf16(&buf_w, &bf16::f32_slice_to_bf16_bits(&f.w_f32));
    let (grid, tg) = gemm_common::dispatch_shape(f.m, f.n);
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    gemm_q8::bind_gpu_buffers(&enc, &buf_x, &buf_y, &buf_w, 0, f.m as u32);
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let ptr = buf_y.contents().as_ptr() as *const u16;
    Ok((0..f.out_len())
        .map(|i| bf16::bf16_bits_to_f32(unsafe { *ptr.add(i) }))
        .collect())
}

#[cfg(all(test, not(all(feature = "metal", target_os = "macos"))))]
pub fn gpu(_: &Fixture, _: super::KernelVariant) -> Result<Vec<f32>, Error> {
    Err(Error::Format("Metal unavailable"))
}

#[cfg(test)]
mod tests {
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tile,
        cpu = crate::kernels::sub::gemm_bf16::cpu,
        cpu_oracle = crate::kernels::sub::gemm_bf16::cpu_oracle,
        gpu = crate::kernels::sub::gemm_bf16::gpu,
        fixture = crate::kernels::sub::gemm_bf16::tile_fixture,
        out_len = crate::kernels::sub::gemm_bf16::fixture_len,
        formats: [F32],
        max_tol = 0.02,
        min_cos = 0.999,
    }
}
