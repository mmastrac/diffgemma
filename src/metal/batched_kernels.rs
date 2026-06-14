//! Elementwise kernels encoded into a shared `GpuBatch`.

use crate::fast_slice::FastSlice;
use crate::kernels::sub::rms_norm_rows;
use crate::metal::batch::{set_bytes, GpuBatch};
use crate::metal::kernels::GpuKernels;
use crate::safetensors::Error;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState};

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
    let buf_dump = batch.alloc_f32_out(1)?;
    let dims = [seq_len as u32, hidden as u32];
    batch.dispatch_1d(&kernels.rms_norm.pipeline, seq_len, |enc| {
        rms_norm_rows::bind_gpu_buffers(&enc, &buf_x, &buf_w, &buf_o, &buf_dump, &dims, eps);
    });
    batch.register_read(buf_o, out);
    Ok(())
}

/// RMSNorm rows; output stays on GPU until the batch ends.
pub fn rms_norm_rows_gpu(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuKernels,
    x: &[f32],
    weight: &[f32],
    seq_len: usize,
    hidden: usize,
    eps: f32,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    let len = seq_len * hidden;
    if x.len() != len || weight.len() != hidden {
        return Err(Error::Format("rms_norm_rows shape mismatch"));
    }
    let buf_x = batch.alloc_f32(x)?;
    let buf_w = batch.alloc_f32(weight)?;
    let buf_o = batch.alloc_f32_out(len)?;
    let buf_dump = batch.alloc_f32_out(1)?;
    let dims = [seq_len as u32, hidden as u32];
    batch.dispatch_1d(&kernels.rms_norm.pipeline, seq_len, |enc| {
        rms_norm_rows::bind_gpu_buffers(&enc, &buf_x, &buf_w, &buf_o, &buf_dump, &dims, eps);
    });
    Ok(buf_o)
}

/// RMSNorm rows from a GPU buffer input.
pub fn rms_norm_rows_gpu_buf(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuKernels,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    weight: &[f32],
    seq_len: usize,
    hidden: usize,
    eps: f32,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    if weight.len() != hidden {
        return Err(Error::Format("rms_norm_rows shape mismatch"));
    }
    let len = seq_len * hidden;
    let buf_w = batch.alloc_f32(weight)?;
    let buf_o = batch.alloc_f32_out(len)?;
    let buf_dump = batch.alloc_f32_out(1)?;
    let dims = [seq_len as u32, hidden as u32];
    batch.dispatch_1d(&kernels.rms_norm.pipeline, seq_len, |enc| {
        rms_norm_rows::bind_gpu_buffers(&enc, x_buf, &buf_w, &buf_o, &buf_dump, &dims, eps);
    });
    Ok(buf_o)
}

/// RMSNorm without affine weight; output stays on GPU until the batch ends.
pub fn rms_norm_rows_no_scale_gpu_buf(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuKernels,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    seq_len: usize,
    hidden: usize,
    eps: f32,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    let len = seq_len * hidden;
    let buf_o = batch.alloc_f32_out(len)?;
    let dims = [seq_len as u32, hidden as u32];
    batch.dispatch_1d(&kernels.rms_norm_no_scale.pipeline, seq_len, |enc| {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&buf_o), 0, 1);
        }
        set_bytes(enc, &dims, 2);
        set_bytes(enc, &eps, 3);
    });
    Ok(buf_o)
}

pub fn gelu_pytorch_tanh_gpu_buf(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuKernels,
    buf: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
) -> Result<(), Error> {
    let len_u = len as u32;
    batch.dispatch_1d(&kernels.gelu.pipeline, len, |enc| {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf), 0, 0);
        }
        set_bytes(enc, &len_u, 1);
    });
    Ok(())
}

pub fn swiglu_mul_gpu_bufs(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuKernels,
    gate_buf: &ProtocolObject<dyn MTLBuffer>,
    up_buf: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
) -> Result<(), Error> {
    let len_u = len as u32;
    batch.dispatch_1d(&kernels.swiglu_mul.pipeline, len, |enc| {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(gate_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(up_buf), 0, 1);
        }
        set_bytes(enc, &len_u, 2);
    });
    Ok(())
}

pub fn gelu_swiglu_gate_up_gpu(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuKernels,
    gate_up_buf: &ProtocolObject<dyn MTLBuffer>,
    out_len: usize,
    batch_size: usize,
    moe_inter: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    if out_len != batch_size * moe_inter {
        return Err(Error::Format("gelu_swiglu_gate_up shape mismatch"));
    }
    let buf_o = batch.alloc_f32_out(out_len)?;
    let dims = [batch_size as u32, moe_inter as u32];
    batch.dispatch_1d(&kernels.gelu_swiglu_gate_up.pipeline, out_len, |enc| {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(gate_up_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&buf_o), 0, 1);
        }
        set_bytes(enc, &dims, 2);
    });
    Ok(buf_o)
}

pub fn f32_bf16_gemm_gpu_out(
    batch: &mut GpuBatch<'_>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a: &[f32],
    b: FastSlice<'_, u16>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    if a.len() != m * k || b.len() != k * n {
        return Err(Error::Format("f32_bf16 gemm shape mismatch"));
    }
    let buf_a = batch.alloc_f32(a)?;
    f32_bf16_gemm_gpu_in_out(batch, pipeline, &buf_a, b, m, k, n)
}

pub fn f32_bf16_gemm_gpu_in_out(
    batch: &mut GpuBatch<'_>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a_buf: &ProtocolObject<dyn MTLBuffer>,
    b: FastSlice<'_, u16>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    if b.len() != k * n {
        return Err(Error::Format("f32_bf16 gemm shape mismatch"));
    }
    let buf_b = batch.alloc_bf16_fast(b)?;
    let buf_c = batch.alloc_f32_out(m * n)?;
    batch.dispatch_gemm(pipeline, a_buf, &buf_b, &buf_c, m, n, k);
    Ok(buf_c)
}

/// `C = A @ W^T` with f32 weights already on GPU (router proj).
pub fn f32_f32_linear_gpu_bufs(
    batch: &mut GpuBatch<'_>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a_buf: &ProtocolObject<dyn MTLBuffer>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    let buf_c = batch.alloc_f32_out(m * n)?;
    batch.dispatch_linear(pipeline, a_buf, w_buf, &buf_c, m, n, k);
    Ok(buf_c)
}

pub fn router_scale_rows_gpu_buf(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuKernels,
    buf: &ProtocolObject<dyn MTLBuffer>,
    scale: &[f32],
    seq_len: usize,
    hidden: usize,
    root: f32,
) -> Result<(), Error> {
    if scale.len() != hidden {
        return Err(Error::Format("router_scale shape mismatch"));
    }
    let buf_scale = batch.alloc_f32(scale)?;
    let dims = [seq_len as u32, hidden as u32];
    batch.dispatch_1d(&kernels.router_scale.pipeline, seq_len, |enc| {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&buf_scale), 0, 1);
        }
        set_bytes(enc, &dims, 2);
        set_bytes(enc, &root, 3);
    });
    Ok(())
}

/// `C = A @ W^T` with PyTorch `W[out,in]` already on GPU.
pub fn f32_bf16_linear_gpu_bufs(
    batch: &mut GpuBatch<'_>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a_buf: &ProtocolObject<dyn MTLBuffer>,
    w_buf: &ProtocolObject<dyn MTLBuffer>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    let buf_c = batch.alloc_f32_out(m * n)?;
    batch.dispatch_linear(pipeline, a_buf, w_buf, &buf_c, m, n, k);
    Ok(buf_c)
}

/// Copy `indices[b]` rows from `src` into contiguous `dst` rows on GPU.
pub fn gather_rows_gpu(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuKernels,
    src: &ProtocolObject<dyn MTLBuffer>,
    indices: &[u32],
    hidden: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    let batch_size = indices.len();
    if batch_size == 0 {
        return Err(Error::Format("gather_rows empty batch"));
    }
    let idx_bytes = unsafe {
        std::slice::from_raw_parts(indices.as_ptr().cast::<u8>(), batch_size * 4)
    };
    let buf_idx = batch.alloc_bytes(idx_bytes)?;
    let buf_dst = batch.alloc_f32_out(batch_size * hidden)?;
    let dims = [0u32, hidden as u32];
    let batch_u = batch_size as u32;
    let grid = batch_size * hidden;
    batch.dispatch_1d(&kernels.gather_rows.pipeline, grid, |enc| {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&buf_idx), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&buf_dst), 0, 2);
        }
        set_bytes(enc, &dims, 3);
        set_bytes(enc, &batch_u, 4);
    });
    Ok(buf_dst)
}

/// `out_buf += addend_buf` on GPU (in-place on `out_buf`).
pub fn vec_add_gpu_bufs(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuKernels,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
    addend_buf: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
) -> Result<(), Error> {
    let len_u = len as u32;
    batch.dispatch_1d(&kernels.vec_add.pipeline, len, |enc| {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(out_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(addend_buf), 0, 1);
        }
        set_bytes(enc, &len_u, 2);
    });
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::cpu::gelu_pytorch_tanh;
    use crate::metal::batch::GpuBatch;
    use crate::metal::device::MetalContext;
    use crate::metal::kernels::GpuKernels;

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn gpu_gelu_large_input() {
        let mut cpu = vec![10.229641f32];
        let mut gpu = cpu.clone();
        gelu_pytorch_tanh(&mut cpu);
        let ctx = MetalContext::new().expect("metal");
        let kernels = GpuKernels::new(&ctx).expect("kernels");
        kernels
            .gelu_pytorch_tanh(&ctx.queue, &mut crate::metal::buffer::BufferPool::new(), &ctx.device, &mut gpu)
            .expect("gelu");
        assert!(gpu[0].is_finite(), "gpu={}", gpu[0]);
        assert!((cpu[0] - gpu[0]).abs() < 1e-4);
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn gpu_gelu_matches_cpu() {
        let mut cpu = vec![-2.0f32, -1.0, 0.0, 0.5, 1.5, 3.0];
        let mut gpu = cpu.clone();
        gelu_pytorch_tanh(&mut cpu);

        let ctx = MetalContext::new().expect("metal");
        let kernels = GpuKernels::new(&ctx).expect("kernels");
        let mut pool = crate::metal::buffer::BufferPool::new();
        let mut batch = GpuBatch::begin(&ctx.queue, &mut pool, &ctx.device).expect("batch");
        let buf = batch.alloc_f32(&gpu).expect("buf");
        gelu_pytorch_tanh_gpu_buf(&mut batch, &kernels, &buf, gpu.len()).expect("gelu");
        batch.register_read(buf, &mut gpu);
        batch.end().expect("end");

        for (a, b) in cpu.iter().zip(gpu.iter()) {
            assert!((a - b).abs() < 1e-5, "cpu={a} gpu={b}");
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn gpu_gelu_matches_cpu_mlp_shape() {
        let len = 16 * 2112;
        let data: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.001).sin()).collect();
        let mut cpu = data.clone();
        let mut gpu = data;
        gelu_pytorch_tanh(&mut cpu);

        let ctx = MetalContext::new().expect("metal");
        let kernels = GpuKernels::new(&ctx).expect("kernels");
        kernels
            .gelu_pytorch_tanh(&ctx.queue, &mut crate::metal::buffer::BufferPool::new(), &ctx.device, &mut gpu)
            .expect("gelu");

        let nan = gpu.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(nan, 0, "gpu gelu produced {nan} non-finite values");
        let max_diff = cpu
            .iter()
            .zip(gpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "max_diff={max_diff}");
    }
}

pub fn f32_bf16_gemm(
    batch: &mut GpuBatch<'_>,
    pipeline: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    c: &mut [f32],
    a: &[f32],
    b: FastSlice<'_, u16>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), Error> {
    if a.len() != m * k || b.len() != k * n || c.len() != m * n {
        return Err(Error::Format("f32_bf16 gemm shape mismatch"));
    }
    let buf_a = batch.alloc_f32(a)?;
    let buf_b = batch.alloc_bf16_fast(b)?;
    let buf_c = batch.alloc_f32_out(c.len())?;
    batch.dispatch_gemm(pipeline, &buf_a, &buf_b, &buf_c, m, n, k);
    batch.register_read(buf_c, c);
    Ok(())
}
