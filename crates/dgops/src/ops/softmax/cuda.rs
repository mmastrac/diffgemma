//! CUDA dispatch for softmax.

use super::{CUBIN, ENTRY};
use crate::Error;
use gpukit::cuda::{BufferPool, KernelArgs, launch_rows};

pub fn gpu(fix: &super::Fixture) -> Result<Vec<f32>, Error> {
    if CUBIN.is_empty() {
        return Err(Error::Gpu(
            "CUDA kernels were not built (nvcc missing at build time)",
        ));
    }
    let ctx = gpukit::cuda::cached_context()?;
    let kernel = gpukit::cuda::cached_kernel(CUBIN, ENTRY)?;
    let mut pool = BufferPool::new();
    let len = fix.len();
    let buf = pool.allocate(ctx, len * 4)?;
    buf.write_f32(&fix.logits)?;

    let mut args = KernelArgs::new();
    args.device_ptr(buf.device_ptr())
        .u32(fix.rows as u32)
        .u32(fix.cols as u32);
    launch_rows(ctx, &kernel, fix.rows, 256, &mut args)?;
    ctx.synchronize()?;

    let mut out = vec![0.0f32; len];
    buf.read_f32(&mut out)?;
    Ok(out)
}
