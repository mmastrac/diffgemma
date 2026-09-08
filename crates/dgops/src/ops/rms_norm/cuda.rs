//! CUDA dispatch for rms_norm.

use super::{CUDA, ENTRY};
use crate::Error;
use gpukit::cuda::{BufferPool, KernelArgs, launch_rows};

pub fn gpu(fix: &super::Fixture) -> Result<Vec<f32>, Error> {
    let ctx = gpukit::cuda::cached_context()?;
    let kernel = gpukit::cuda::cached_source_kernel(CUDA, ENTRY)?;
    let mut pool = BufferPool::new();
    let len = fix.len();
    let buf_x = pool.allocate(ctx, len * 4)?;
    let buf_w = pool.allocate(ctx, fix.hidden * 4)?;
    let buf_out = pool.allocate(ctx, len * 4)?;
    buf_x.write_f32(&fix.x)?;
    buf_w.write_f32(&fix.weight)?;

    let mut args = KernelArgs::new();
    args.device_ptr(buf_x.device_ptr())
        .device_ptr(buf_w.device_ptr())
        .device_ptr(buf_out.device_ptr())
        .u32(fix.rows as u32)
        .u32(fix.hidden as u32)
        .f32(fix.eps);
    launch_rows(ctx, &kernel, fix.rows, 256, &mut args)?;
    ctx.synchronize()?;

    let mut out = vec![0.0f32; len];
    buf_out.read_f32(&mut out)?;
    Ok(out)
}
