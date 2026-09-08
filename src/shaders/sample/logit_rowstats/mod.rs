//! Per-row max + sumexp over half logits (SC row stats at t=1).

use crate::Error;
use crate::shaders::bf16;
use crate::shaders::gpu_common;
use crate::shaders::test_util::ElemFormat;
use crate::shaders::variant::KernelVariant;

crate::shader_kernel! {
    name = "logit_rowstats",
    metal = "logit_rowstats.metal",
    pipeline = |ctx, variant| ctx.compile_subkernel(SHADER, ENTRY, variant),
    spec = {
            quant_formats: &[QuantFormat::Q4Affine],
            fc: &[],
            variants: KernelVariants::Elementwise,
    },
    tests = [
        tiny => tiny_fixture => (1e-3, 0.9999),
        wide => wide_fixture => (1e-2, 0.9999),
    ],
}
pub const THREADGROUP_WIDTH: usize = 256;

#[derive(Debug, Clone)]
pub struct Fixture {
    pub logits: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.rows * 2
    }
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        logits: vec![1.0, 2.0, 3.0, 0.0, -1.0, 0.5],
        rows: 2,
        cols: 3,
    }
}

pub fn wide_fixture(_: ElemFormat) -> Fixture {
    let rows = 4usize;
    let cols = 512usize;
    let logits: Vec<f32> = (0..rows * cols).map(|i| (i as f32 % 17.0) - 8.0).collect();
    Fixture { logits, rows, cols }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let mut out = vec![0.0f32; f.out_len()];
    // GPU reads bf16-rounded logits; mirror that here for parity.
    let logits = bf16::bf16_slice_to_f32(&bf16::f32_slice_to_bf16_bits(&f.logits));
    for row in 0..f.rows {
        let lr = &logits[row * f.cols..(row + 1) * f.cols];
        let mx = lr.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = lr.iter().map(|&x| (x - mx).exp()).sum();
        out[row * 2] = mx;
        out[row * 2 + 1] = sum;
    }
    out
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(target_os = "macos")]
pub fn dispatch_shape(rows: usize) -> (objc2_metal::MTLSize, objc2_metal::MTLSize) {
    use objc2_metal::MTLSize;
    (
        MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        },
    )
}

#[cfg(target_os = "macos")]
use objc2::runtime::ProtocolObject;
#[cfg(target_os = "macos")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
};

#[cfg(target_os = "macos")]
pub fn bind_gpu_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    logits: &ProtocolObject<dyn MTLBuffer>,
    rowstat: &ProtocolObject<dyn MTLBuffer>,
    dims: &[u32; 2],
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(logits), 0, 0);
        enc.setBuffer_offset_atIndex(Some(rowstat), 0, 1);
    }
    gpu_common::set_bytes(enc, dims, 2);
}

#[cfg(target_os = "macos")]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let buf_logits = pool
        .allocate(&ctx.device, f.logits.len() * 2)
        .ok_or(Error::Gpu("alloc"))?;
    let buf_out = pool
        .allocate(&ctx.device, f.out_len() * 4)
        .ok_or(Error::Gpu("alloc"))?;
    BufferPool::write_bf16(&buf_logits, &bf16::f32_slice_to_bf16_bits(&f.logits));
    let dims = [f.rows as u32, f.cols as u32];
    let (grid, tg) = dispatch_shape(f.rows);
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Gpu("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    bind_gpu_buffers(&enc, &buf_logits, &buf_out, &dims);
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let mut out = vec![0.0f32; f.out_len()];
    BufferPool::read_f32(&buf_out, &mut out);
    Ok(out)
}
