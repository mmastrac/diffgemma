//! Threadgroup RMSNorm: half in, half out, optional bf16 weight blob.

use super::bf16;
use super::f16;
use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::kernels::cpu;
use crate::safetensors::Error;

pub const ENTRY: &str = "rmsnorm_half";
pub const RMS_EPS: f32 = 1e-6;
pub const THREADS_PER_TG: usize = 256;

const SHADER: &str = concat!(
    include_str!("../../../shaders/kernels/common.metal"),
    include_str!("../../../shaders/kernels/bf16.metal"),
    include_str!("../../../shaders/kernels/rmsnorm_half.metal"),
);

#[derive(Debug, Clone)]
pub struct Fixture {
    pub x: Vec<f32>,
    pub weight: Option<Vec<f32>>,
    pub rows: usize,
    pub dim: usize,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.rows * self.dim
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        x: vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -0.5],
        weight: Some(vec![1.0, 0.5, 2.0, 1.5]),
        rows: 2,
        dim: 4,
    }
}

pub fn no_scale_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        x: vec![1.0, -1.0, 2.0, 0.0, 0.5, -0.5, 1.5, 2.0],
        weight: None,
        rows: 2,
        dim: 4,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let mut out = vec![0.0f32; f.out_len()];
    if let Some(ref w) = f.weight {
        for r in 0..f.rows {
            let off = r * f.dim;
            let x = &f.x[off..off + f.dim];
            let mut row = vec![0.0f32; f.dim];
            cpu::rms_norm(&mut row, x, w, RMS_EPS);
            for i in 0..f.dim {
                out[off + i] = f16::round_half(row[i]);
            }
        }
    } else {
        for r in 0..f.rows {
            let off = r * f.dim;
            let x = &f.x[off..off + f.dim];
            let mut row = vec![0.0f32; f.dim];
            cpu::rms_norm_no_scale(&mut row, x, RMS_EPS);
            for i in 0..f.dim {
                out[off + i] = f16::round_half(row[i]);
            }
        }
    }
    out
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn dispatch_shape(rows: usize) -> (objc2_metal::MTLSize, objc2_metal::MTLSize) {
    use objc2_metal::MTLSize;
    (
        MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        },
    )
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER, ENTRY, variant)
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
    dump: &ProtocolObject<dyn MTLBuffer>,
    w_off: u64,
    dim: u32,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(x), 0, 0);
        enc.setBuffer_offset_atIndex(Some(y), 0, 1);
        enc.setBuffer_offset_atIndex(Some(blob), 0, 2);
        enc.setBuffer_offset_atIndex(Some(dump), 0, 5);
    }
    gpu_common::set_bytes(enc, &w_off, 3);
    gpu_common::set_bytes(enc, &dim, 4);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let mut pool = BufferPool::new();
    let len = f.out_len();
    let buf_x = pool.allocate(&ctx.device, len * 2).ok_or(Error::Format("alloc"))?;
    let buf_y = pool.allocate(&ctx.device, len * 2).ok_or(Error::Format("alloc"))?;
    let (blob, w_off) = match &f.weight {
        Some(w) => {
            let mut b = vec![0u8; 2];
            b.extend(bf16::pack_bf16_slice(w));
            (b, 2u64)
        }
        None => (vec![0u8; 2], 0u64),
    };
    let buf_blob = pool
        .allocate(&ctx.device, blob.len())
        .ok_or(Error::Format("alloc"))?;
    let dump_bytes = if variant.dump_stage > 0 {
        f.rows * 4
    } else {
        4
    };
    let buf_d = pool.allocate(&ctx.device, dump_bytes).ok_or(Error::Format("alloc"))?;
    BufferPool::write_bf16(&buf_x, &f16::f32_slice_to_f16(&f.x));
    BufferPool::write_bytes(&buf_blob, &blob);
    let (grid, tg) = dispatch_shape(f.rows);
    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    bind_gpu_buffers(
        &enc,
        &buf_x,
        &buf_y,
        &buf_blob,
        &buf_d,
        w_off,
        f.dim as u32,
    );
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let ptr = unsafe { buf_y.contents().as_ptr() as *const u16 };
    Ok((0..len)
        .map(|i| f16::f16_bits_to_f32(unsafe { *ptr.add(i) }))
        .collect())
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn gpu(_: &Fixture, _: KernelVariant) -> Result<Vec<f32>, Error> {
    Err(Error::Format("Metal unavailable"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny,
        cpu = crate::kernels::sub::rmsnorm_half::cpu,
        cpu_oracle = crate::kernels::sub::rmsnorm_half::cpu_oracle,
        gpu = crate::kernels::sub::rmsnorm_half::gpu,
        fixture = crate::kernels::sub::rmsnorm_half::tiny_fixture,
        out_len = crate::kernels::sub::rmsnorm_half::fixture_len,
        formats: [F32],
        max_tol = 1e-2,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod no_scale,
        cpu = crate::kernels::sub::rmsnorm_half::cpu,
        cpu_oracle = crate::kernels::sub::rmsnorm_half::cpu_oracle,
        gpu = crate::kernels::sub::rmsnorm_half::gpu,
        fixture = crate::kernels::sub::rmsnorm_half::no_scale_fixture,
        out_len = crate::kernels::sub::rmsnorm_half::fixture_len,
        formats: [F32],
        max_tol = 1e-3,
        min_cos = 0.9999,
    }
}
