//! CUDA dispatch for the f32 GEMM.

use crate::Error;
use crate::problem::{AbiParams, Call, Problem};
use crate::{BM, BN, CUDA, ENTRY, THREADS};
use gpukit::cuda::{BufferPool, DeviceBuffer, KernelArgs, div_up, launch_grid};

/// Launch the GEMM into caller-owned device buffers — no host round-trip.
///
/// This is the entry point a multi-kernel pipeline uses: the buffers stay on
/// the device across dispatches and only the final result is read back. The
/// one-shot `gpu` below is the tier-1 convenience wrapper over it.
pub fn gpu_device(
    ctx: &gpukit::cuda::Context,
    problem: &Problem,
    a: &DeviceBuffer,
    b: &DeviceBuffer,
    c: &DeviceBuffer,
) -> Result<(), Error> {
    let kernel = gpukit::cuda::cached_source_kernel(CUDA, ENTRY)?;
    let p: AbiParams = (*problem).into();
    let mut args = KernelArgs::new();
    args.device_ptr(a.device_ptr())
        .device_ptr(b.device_ptr())
        .device_ptr(c.device_ptr())
        .bytes(gpukit::cuda::pod_bytes(&p));
    launch_grid(
        ctx,
        &kernel,
        div_up(p.n as usize, BN),
        div_up(p.m as usize, BM),
        THREADS,
        &mut args,
    )?;
    Ok(())
}

pub fn gpu(call: &Call) -> Result<Vec<f32>, Error> {
    let ctx = gpukit::cuda::cached_context()?;
    let kernel = gpukit::cuda::cached_source_kernel(CUDA, ENTRY)?;
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
