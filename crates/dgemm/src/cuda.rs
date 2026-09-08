//! CUDA dispatch for the f32 GEMM.

use crate::Error;
use crate::problem::{AbiParams, Call};
use crate::{BM, BN, CUBIN, ENTRY, THREADS};
use gpukit::cuda::{BufferPool, KernelArgs, div_up, launch_grid};

pub fn gpu(call: &Call) -> Result<Vec<f32>, Error> {
    if CUBIN.is_empty() {
        return Err(Error::Gpu(
            "CUDA kernels were not built (nvcc missing at build time)",
        ));
    }
    let ctx = gpukit::cuda::cached_context()?;
    let kernel = gpukit::cuda::cached_kernel(CUBIN, ENTRY)?;
    let p: AbiParams = call.problem.into();
    let out_len = call.out_len();
    let mut pool = BufferPool::new();
    let buf_a = pool.allocate(ctx, call.a.len() * 4)?;
    let buf_b = pool.allocate(ctx, call.b.len() * 4)?;
    let buf_c = pool.allocate(ctx, out_len * 4)?;
    buf_a.write_f32(&call.a)?;
    buf_b.write_f32(&call.b)?;
    buf_c.write_f32(&call.c)?;

    let mut args = KernelArgs::new();
    args.device_ptr(buf_a.device_ptr())
        .device_ptr(buf_b.device_ptr())
        .device_ptr(buf_c.device_ptr())
        .bytes(gpukit::cuda::pod_bytes(&p));
    launch_grid(
        ctx,
        &kernel,
        div_up(p.n as usize, BN),
        div_up(p.m as usize, BM),
        THREADS,
        &mut args,
    )?;
    ctx.synchronize()?;

    let mut out = vec![0.0f32; out_len];
    buf_c.read_f32(&mut out)?;
    Ok(out)
}
