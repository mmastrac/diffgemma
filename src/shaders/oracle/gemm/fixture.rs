//! Shared fixtures + GPU runner for the dense `gemm_block` oracle family
//! (bf16 / q8 / q4 / nvfp4). Each format module keeps only its CPU reference,
//! weight-quant, and `QuantFormat`; the fixture shape and the GPU dispatch dance
//! live here. Validation-only (bit-exact twins of `gemm_tunable`).

use crate::Error;

pub const ENTRY: &str = "gemm_block";
pub const SHADER: &str = include_str!("gemm_block/gemm_block.metal");

/// Dense GEMM oracle fixture: `y[m,n] = x[m,k] @ w[n,k]^T`. `global_scale` is
/// only meaningful for nvfp4 (1.0 for the other formats).
#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
    pub w_f32: Vec<f32>,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub global_scale: f32,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.m * self.n
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

/// Deterministic fixture: `x[i] = sin(i*xf)*xa + xb`, `w[i] = cos(i*wf)*wa + wb`.
pub fn make_fixture(
    m: usize,
    n: usize,
    k: usize,
    (xf, xa, xb): (f32, f32, f32),
    (wf, wa, wb): (f32, f32, f32),
    global_scale: f32,
) -> Fixture {
    let x = (0..m * k)
        .map(|i| ((i as f32) * xf).sin() * xa + xb)
        .collect();
    let w_f32 = (0..n * k)
        .map(|i| ((i as f32) * wf).cos() * wa + wb)
        .collect();
    Fixture {
        x,
        w_f32,
        m,
        n,
        k,
        global_scale,
    }
}

#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
};

/// Bind the dense `gemm_block` buffers: x@0, y@1, weight blob@2, w_off@3, m@4.
#[cfg(target_os = "macos")]
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
    crate::shaders::gpu_common::set_bytes(enc, &w_off, 3);
    crate::shaders::gpu_common::set_bytes(enc, &m, 4);
}

/// Compile the dense `gemm_block` pipeline for `format`.
#[cfg(target_os = "macos")]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    n: u32,
    k: u32,
    format: crate::shaders::QuantFormat,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_gemm_subkernel(SHADER, ENTRY, n, k, false, format as u32, false)
}

/// Run the dense `gemm_block` kernel: `x` written as bf16, weights as raw
/// `w_bytes` (bf16 bits for `Raw`, the quantized blob otherwise), bf16-decoded
/// output.
#[cfg(target_os = "macos")]
pub fn run_gpu(
    f: &Fixture,
    format: crate::shaders::QuantFormat,
    w_bytes: &[u8],
) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use crate::shaders::{bf16, gemm_common};

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, f.n as u32, f.k as u32, format)?;
    let mut pool = BufferPool::new();
    let buf_x = pool
        .allocate(&ctx.device, f.m * f.k * 2)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_y = pool
        .allocate(&ctx.device, f.m * f.n * 2)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_w = pool
        .allocate(&ctx.device, w_bytes.len())
        .ok_or(Error::Gpu("alloc"))?;
    BufferPool::write_bf16(&buf_x, &bf16::f32_slice_to_bf16_bits(&f.x));
    BufferPool::write_bytes(&buf_w, w_bytes);
    let (grid, tg) = gemm_common::dispatch_shape(f.m, f.n);
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
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
