//! CUDA dispatch for vec_add.

use super::{CUBIN, ENTRY};
use crate::Error;
use gpukit::cuda::{BufferPool, KernelArgs, launch_1d};

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
    let buf = pool.allocate(ctx, len * 4)?;
    let buf_addend = pool.allocate(ctx, len * 4)?;
    buf.write_f32(&fix.x)?;
    buf_addend.write_f32(&fix.addend)?;

    let mut args = KernelArgs::new();
    args.device_ptr(buf.device_ptr())
        .device_ptr(buf_addend.device_ptr())
        .u32(len as u32);
    launch_1d(ctx, &kernel, len, &mut args)?;
    ctx.synchronize()?;

    let mut out = vec![0.0f32; len];
    buf.read_f32(&mut out)?;
    Ok(out)
}
