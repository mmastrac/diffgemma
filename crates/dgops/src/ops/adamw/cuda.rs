//! CUDA dispatch for adamw.

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
    let out_len = super::out_len(fix);
    let buf_p = pool.allocate(ctx, len * 4)?;
    let buf_g = pool.allocate(ctx, len * 4)?;
    let buf_m = pool.allocate(ctx, len * 4)?;
    let buf_v = pool.allocate(ctx, len * 4)?;
    let buf_out = pool.allocate(ctx, out_len * 4)?;
    buf_p.write_f32(&fix.p)?;
    buf_g.write_f32(&fix.g)?;
    buf_m.write_f32(&fix.m)?;
    buf_v.write_f32(&fix.v)?;

    let params = fix.params();
    let mut args = KernelArgs::new();
    args.device_ptr(buf_p.device_ptr())
        .device_ptr(buf_g.device_ptr())
        .device_ptr(buf_m.device_ptr())
        .device_ptr(buf_v.device_ptr())
        .device_ptr(buf_out.device_ptr())
        .bytes(crate::cuda_rt::pod_bytes(&params))
        .u32(len as u32);
    launch_1d(ctx, &kernel, len, &mut args)?;
    ctx.synchronize()?;

    let mut out = vec![0.0f32; out_len];
    buf_out.read_f32(&mut out)?;
    Ok(out)
}
