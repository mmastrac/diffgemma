//! CUDA dispatch for vec_scale.

use super::{CUDA, ENTRY};
use crate::Error;
use gpukit::cuda::{BufferPool, KernelArgs, launch_1d};

pub fn gpu(fix: &super::Fixture) -> Result<Vec<f32>, Error> {
    let ctx = gpukit::cuda::cached_context()?;
    let kernel = gpukit::cuda::cached_source_kernel(CUDA, ENTRY)?;
    let mut pool = BufferPool::new();
    let len = fix.len();
    let buf = pool.allocate(ctx, len * 4)?;
    buf.write_f32(&fix.x)?;

    let mut args = KernelArgs::new();
    args.device_ptr(buf.device_ptr())
        .f32(fix.scale)
        .u32(len as u32);
    launch_1d(ctx, &kernel, len, &mut args)?;
    ctx.synchronize()?;

    let mut out = vec![0.0f32; len];
    buf.read_f32(&mut out)?;
    Ok(out)
}
