//! Elementwise kernels encoded into a shared `GpuBatch`.

use crate::Error;
use crate::metal::batch::GpuBatch;
use crate::metal::kernels::GpuKernels;
use crate::shaders::{
    gather_rows, gelu, rms_norm_rows, router_scale_rows, swiglu, vec_add_inplace,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputePipelineState};

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
        rms_norm_rows::bind_gpu_buffers(enc, &buf_x, &buf_w, &buf_o, &buf_dump, &dims, eps);
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
        rms_norm_rows::bind_gpu_buffers(enc, &buf_x, &buf_w, &buf_o, &buf_dump, &dims, eps);
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
        rms_norm_rows::bind_gpu_buffers(enc, x_buf, &buf_w, &buf_o, &buf_dump, &dims, eps);
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
    let buf_w = batch.alloc_f32_out(1)?;
    let buf_dump = batch.alloc_f32_out(1)?;
    let dims = [seq_len as u32, hidden as u32];
    batch.dispatch_1d(&kernels.rms_norm_no_scale.pipeline, seq_len, |enc| {
        rms_norm_rows::bind_gpu_buffers(enc, x_buf, &buf_w, &buf_o, &buf_dump, &dims, eps);
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
    let buf_dump = batch.alloc_f32_out(1)?;
    batch.dispatch_1d(&kernels.gelu.pipeline, len, |enc| {
        gelu::bind_gpu_in_place(enc, buf, &buf_dump, len_u);
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
    let buf_dump = batch.alloc_f32_out(1)?;
    batch.dispatch_1d(&kernels.swiglu_mul.pipeline, len, |enc| {
        swiglu::bind_split_in_place_f32(enc, gate_buf, up_buf, &buf_dump, len_u);
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
    let buf_dump = batch.alloc_f32_out(1)?;
    let dims = [batch_size as u32, moe_inter as u32];
    batch.dispatch_1d(&kernels.gelu_swiglu_gate_up.pipeline, out_len, |enc| {
        swiglu::bind_moe_gate_up(enc, gate_up_buf, &buf_o, &buf_dump, &dims);
    });
    Ok(buf_o)
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
    let buf_dump = batch.alloc_f32_out(1)?;
    let dims = [seq_len as u32, hidden as u32];
    batch.dispatch_1d(&kernels.router_scale.pipeline, seq_len, |enc| {
        router_scale_rows::bind_gpu_buffers(enc, buf, &buf_scale, &buf_dump, &dims, root);
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
        return Err(Error::Runtime("gather_rows empty batch"));
    }
    let idx_bytes =
        unsafe { std::slice::from_raw_parts(indices.as_ptr().cast::<u8>(), batch_size * 4) };
    let buf_idx = batch.alloc_bytes(idx_bytes)?;
    let buf_dst = batch.alloc_f32_out(batch_size * hidden)?;
    let buf_dump = batch.alloc_f32_out(1)?;
    let batch_u = batch_size as u32;
    let grid = batch_size * hidden;
    batch.dispatch_1d(&kernels.gather_rows.pipeline, grid, |enc| {
        gather_rows::bind_gpu_buffers(
            enc,
            src,
            &buf_idx,
            &buf_dst,
            &buf_dump,
            hidden as u32,
            batch_u,
            0,
        );
    });
    Ok(buf_dst)
}

/// Weighted scatter of grouped-MoE arena rows back to `[seq, hidden]` token rows.
/// `rows`/`weights` are per-token top_k lists in expert-JOB order (see
/// `build_scatter_lists`) so the f32 sum order matches the CPU scatter exactly.
pub fn scatter_rows_weighted_gpu(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuKernels,
    arena_buf: &ProtocolObject<dyn MTLBuffer>,
    rows: &[u32],
    weights: &[f32],
    seq_len: usize,
    hidden: usize,
    top_k: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    if rows.len() != seq_len * top_k || weights.len() != seq_len * top_k {
        return Err(Error::Format("scatter_rows_weighted shape mismatch"));
    }
    let rows_bytes =
        unsafe { std::slice::from_raw_parts(rows.as_ptr().cast::<u8>(), rows.len() * 4) };
    let buf_rows = batch.alloc_bytes(rows_bytes)?;
    let buf_w = batch.alloc_f32(weights)?;
    let buf_out = batch.alloc_f32_out(seq_len * hidden)?;
    let buf_dump = batch.alloc_f32_out(1)?;
    let dims = [seq_len as u32, hidden as u32, top_k as u32];
    batch.dispatch_1d(
        &kernels.scatter_rows_weighted.pipeline,
        seq_len * hidden,
        |enc| {
            crate::shaders::scatter_rows_weighted::bind_gpu_buffers(
                enc, arena_buf, &buf_rows, &buf_w, &buf_out, &buf_dump, &dims,
            );
        },
    );
    Ok(buf_out)
}

/// `buf *= scale` in place on GPU (engine analog of the CPU layer_scalar multiply).
pub fn vec_scale_gpu_buf(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuKernels,
    buf: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
    scale: f32,
) -> Result<(), Error> {
    let len_u = len as u32;
    let buf_dump = batch.alloc_f32_out(1)?;
    batch.dispatch_1d(&kernels.vec_scale.pipeline, len, |enc| {
        crate::shaders::vec_scale_inplace::bind_gpu_buffers(enc, buf, &buf_dump, scale, len_u);
    });
    Ok(())
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
    let buf_dump = batch.alloc_f32_out(1)?;
    batch.dispatch_1d(&kernels.vec_add.pipeline, len, |enc| {
        vec_add_inplace::bind_gpu_buffers(enc, out_buf, addend_buf, &buf_dump, len_u);
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::batch::GpuBatch;
    use crate::metal::device::MetalContext;
    use crate::metal::kernels::GpuKernels;
    use crate::shaders::cpu::gelu_pytorch_tanh;

    #[cfg(target_os = "macos")]
    #[test]
    fn gpu_gelu_large_input() {
        let mut cpu = vec![10.229641f32];
        let mut gpu = cpu.clone();
        gelu_pytorch_tanh(&mut cpu);
        let ctx = MetalContext::new().expect("metal");
        let kernels = GpuKernels::new(&ctx).expect("kernels");
        let mut pool = crate::metal::buffer::BufferPool::new();
        let mut batch = GpuBatch::begin_with_telemetry(&ctx.queue, &mut pool, &ctx.device, None)
            .expect("batch");
        let buf = batch.alloc_f32(&gpu).expect("buf");
        gelu_pytorch_tanh_gpu_buf(&mut batch, &kernels, &buf, gpu.len()).expect("gelu");
        batch.register_read(buf, &mut gpu);
        batch.end().expect("end");
        assert!(gpu[0].is_finite(), "gpu={}", gpu[0]);
        assert!((cpu[0] - gpu[0]).abs() < 1e-4);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn gpu_gelu_matches_cpu() {
        let mut cpu = vec![-2.0f32, -1.0, 0.0, 0.5, 1.5, 3.0];
        let mut gpu = cpu.clone();
        gelu_pytorch_tanh(&mut cpu);

        let ctx = MetalContext::new().expect("metal");
        let kernels = GpuKernels::new(&ctx).expect("kernels");
        let mut pool = crate::metal::buffer::BufferPool::new();
        let mut batch = GpuBatch::begin_with_telemetry(&ctx.queue, &mut pool, &ctx.device, None)
            .expect("batch");
        let buf = batch.alloc_f32(&gpu).expect("buf");
        gelu_pytorch_tanh_gpu_buf(&mut batch, &kernels, &buf, gpu.len()).expect("gelu");
        batch.register_read(buf, &mut gpu);
        batch.end().expect("end");

        for (a, b) in cpu.iter().zip(gpu.iter()) {
            assert!((a - b).abs() < 1e-5, "cpu={a} gpu={b}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn gpu_gelu_matches_cpu_mlp_shape() {
        let len = 16 * 2112;
        let data: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.001).sin()).collect();
        let mut cpu = data.clone();
        let mut gpu = data;
        gelu_pytorch_tanh(&mut cpu);

        let ctx = MetalContext::new().expect("metal");
        let kernels = GpuKernels::new(&ctx).expect("kernels");
        let mut pool = crate::metal::buffer::BufferPool::new();
        let mut batch = GpuBatch::begin_with_telemetry(&ctx.queue, &mut pool, &ctx.device, None)
            .expect("batch");
        let buf = batch.alloc_f32(&gpu).expect("buf");
        gelu_pytorch_tanh_gpu_buf(&mut batch, &kernels, &buf, len).expect("gelu");
        batch.register_read(buf, &mut gpu);
        batch.end().expect("end");

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
