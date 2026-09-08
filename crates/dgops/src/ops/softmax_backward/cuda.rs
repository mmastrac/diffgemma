//! CUDA dispatch for softmax_backward.

use super::{CUBIN, ENTRY};
use crate::Error;
use gpukit::cuda::{BufferPool, KernelArgs, launch_rows};

pub fn gpu(fix: &super::Fixture) -> Result<Vec<f32>, Error> {
    if CUBIN.is_empty() {
        return Err(Error::Gpu(
            "CUDA kernels were not built (nvcc missing at build time)",
        ));
    }
    let ctx = crate::cuda_rt::context()?;
    let kernel = crate::cuda_rt::kernel(CUBIN, ENTRY)?;
    let mut pool = BufferPool::new();
    let len = fix.len();
    let buf_probs = pool.allocate(ctx, len * 4)?;
    let buf_dp = pool.allocate(ctx, len * 4)?;
    let buf_out = pool.allocate(ctx, len * 4)?;
    buf_probs.write_f32(&fix.probs)?;
    buf_dp.write_f32(&fix.dp)?;

    let mut args = KernelArgs::new();
    args.device_ptr(buf_probs.device_ptr())
        .device_ptr(buf_dp.device_ptr())
        .device_ptr(buf_out.device_ptr())
        .u32(fix.rows as u32)
        .u32(fix.cols as u32);
    launch_rows(ctx, &kernel, fix.rows, 256, &mut args)?;
    ctx.synchronize()?;

    let mut out = vec![0.0f32; len];
    buf_out.read_f32(&mut out)?;
    Ok(out)
}
