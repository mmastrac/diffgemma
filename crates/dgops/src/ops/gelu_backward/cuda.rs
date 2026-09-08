//! CUDA dispatch for gelu_backward.

use super::{CUDA, ENTRY};
use crate::Error;
use gpukit::cuda::{BufferPool, KernelArgs, launch_1d};

pub fn gpu(fix: &super::Fixture) -> Result<Vec<f32>, Error> {
    let ctx = gpukit::cuda::cached_context()?;
    let kernel = gpukit::cuda::cached_source_kernel(CUDA, ENTRY)?;
    let mut pool = BufferPool::new();
    let len = fix.len();
    let buf_g = pool.allocate(ctx, len * 4)?;
    let buf_dy = pool.allocate(ctx, len * 4)?;
    let buf_out = pool.allocate(ctx, len * 4)?;
    buf_g.write_f32(&fix.g)?;
    buf_dy.write_f32(&fix.dy)?;

    let mut args = KernelArgs::new();
    args.device_ptr(buf_g.device_ptr())
        .device_ptr(buf_dy.device_ptr())
        .device_ptr(buf_out.device_ptr())
        .u32(len as u32);
    launch_1d(ctx, &kernel, len, &mut args)?;
    ctx.synchronize()?;

    let mut out = vec![0.0f32; len];
    buf_out.read_f32(&mut out)?;
    Ok(out)
}
