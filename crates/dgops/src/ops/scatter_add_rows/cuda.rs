//! CUDA dispatch for scatter_add_rows.

use super::{CUDA, ENTRY};
use crate::Error;
use gpukit::cuda::{BufferPool, KernelArgs, launch_rows};

pub fn gpu(fix: &super::Fixture) -> Result<Vec<f32>, Error> {
    let ctx = gpukit::cuda::cached_context()?;
    let kernel = gpukit::cuda::cached_source_kernel(CUDA, ENTRY)?;
    let mut pool = BufferPool::new();
    let out_len = fix.len();
    let buf_dst = pool.allocate(ctx, out_len * 4)?;
    let buf_idx = pool.allocate(ctx, fix.indices.len() * 4)?;
    let buf_src = pool.allocate(ctx, fix.src.len() * 4)?;
    buf_dst.write_f32(&fix.dst)?;
    buf_idx.write_bytes(unsafe {
        std::slice::from_raw_parts(fix.indices.as_ptr().cast::<u8>(), fix.indices.len() * 4)
    })?;
    buf_src.write_f32(&fix.src)?;

    let mut args = KernelArgs::new();
    args.device_ptr(buf_dst.device_ptr())
        .device_ptr(buf_idx.device_ptr())
        .device_ptr(buf_src.device_ptr())
        .u32(fix.indices.len() as u32)
        .u32(fix.hidden as u32);
    launch_rows(ctx, &kernel, fix.indices.len(), 256, &mut args)?;
    ctx.synchronize()?;

    let mut out = vec![0.0f32; out_len];
    buf_dst.read_f32(&mut out)?;
    Ok(out)
}
