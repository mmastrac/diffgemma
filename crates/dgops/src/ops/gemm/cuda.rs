//! CUDA dispatch for gemm.

use super::{BM, BN, CUBIN, ENTRY, THREADS};
use crate::Error;
use gpukit::cuda::{BufferPool, KernelArgs, div_up, launch_grid};

pub fn gpu(fix: &super::Fixture) -> Result<Vec<f32>, Error> {
    if CUBIN.is_empty() {
        return Err(Error::Gpu(
            "CUDA kernels were not built (nvcc missing at build time)",
        ));
    }
    let ctx = crate::cuda_rt::context()?;
    let kernel = crate::cuda_rt::kernel(CUBIN, ENTRY)?;
    let p = fix.params;
    let mut pool = BufferPool::new();
    let out_len = fix.out_len();
    let buf_a = pool.allocate(ctx, fix.a.len() * 4)?;
    let buf_b = pool.allocate(ctx, fix.b.len() * 4)?;
    let buf_c = pool.allocate(ctx, out_len * 4)?;
    buf_a.write_f32(&fix.a)?;
    buf_b.write_f32(&fix.b)?;
    buf_c.write_f32(&fix.c)?;

    let mut args = KernelArgs::new();
    args.device_ptr(buf_a.device_ptr())
        .device_ptr(buf_b.device_ptr())
        .device_ptr(buf_c.device_ptr())
        .bytes(crate::cuda_rt::pod_bytes(&p));
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
