//! Elementwise kernels encoded into a shared `GpuBatch`.

use crate::metal::batch::{set_bytes, GpuBatch};
use crate::metal::kernels::GpuKernels;
use crate::safetensors::Error;
use objc2_metal::MTLComputeCommandEncoder;

pub fn rms_norm_rows(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuKernels,
    out: &mut [f32],
    x: &[f32],
    weight: &[f32],
    seq_len: usize,
    hidden: usize,
    eps: f32,
) -> Result<(), Error> {
    let len = seq_len * hidden;
    if out.len() != len || x.len() != len || weight.len() != hidden {
        return Err(Error::Format("rms_norm_rows shape mismatch"));
    }
    let buf_x = batch.alloc_f32(x)?;
    let buf_w = batch.alloc_f32(weight)?;
    let buf_o = batch.alloc_f32_out(len)?;
    let dims = [seq_len as u32, hidden as u32];
    batch.dispatch_1d(&kernels.rms_norm.pipeline, seq_len, |enc| {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&buf_x), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&buf_w), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&buf_o), 0, 2);
        }
        set_bytes(enc, &dims, 3);
        set_bytes(enc, &eps, 4);
    });
    batch.register_read(buf_o, out);
    Ok(())
}

pub fn vec_add_inplace(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuKernels,
    out: &mut [f32],
    addend: &[f32],
) -> Result<(), Error> {
    if out.len() != addend.len() {
        return Err(Error::Format("vec_add shape mismatch"));
    }
    let buf_o = batch.alloc_f32(out)?;
    let buf_a = batch.alloc_f32(addend)?;
    let len = out.len() as u32;
    batch.dispatch_1d(&kernels.vec_add.pipeline, out.len(), |enc| {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&buf_o), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&buf_a), 0, 1);
        }
        set_bytes(enc, &len, 2);
    });
    batch.register_read(buf_o, out);
    Ok(())
}

pub fn vec_scale_inplace(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuKernels,
    x: &mut [f32],
    scale: f32,
) -> Result<(), Error> {
    let buf = batch.alloc_f32(x)?;
    let len = x.len() as u32;
    batch.dispatch_1d(&kernels.vec_scale.pipeline, x.len(), |enc| {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&buf), 0, 0);
        }
        set_bytes(enc, &scale, 1);
        set_bytes(enc, &len, 2);
    });
    batch.register_read(buf, x);
    Ok(())
}

pub fn f32_bf16_gemm(
    batch: &mut GpuBatch<'_>,
    pipeline: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    c: &mut [f32],
    a: &[f32],
    b: &[u16],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), Error> {
    if a.len() != m * k || b.len() != k * n || c.len() != m * n {
        return Err(Error::Format("f32_bf16 gemm shape mismatch"));
    }
    let buf_a = batch.alloc_f32(a)?;
    let buf_b = batch.alloc_bf16(b)?;
    let buf_c = batch.alloc_f32_out(c.len())?;
    batch.dispatch_gemm(pipeline, &buf_a, &buf_b, &buf_c, m, n, k);
    batch.register_read(buf_c, c);
    Ok(())
}
