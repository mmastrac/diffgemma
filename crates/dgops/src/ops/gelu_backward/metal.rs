//! Metal dispatch for gelu_backward.

use super::{ENTRY, METAL};
use crate::Error;
use gpukit::metal::{BufferPool, dispatch_1d, set_bytes};
use objc2_metal::MTLComputeCommandEncoder;

pub fn gpu(fix: &super::Fixture) -> Result<Vec<f32>, Error> {
    let ctx = gpukit::metal::cached_context(gpukit::metal::CacheConfig::MEMORY)?;
    let pipeline = gpukit::metal::cached_pipeline(&ctx, METAL, ENTRY)?;
    let mut pool = BufferPool::new();
    let len = fix.len();
    let buf_g = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Gpu("buffer alloc"))?;
    let buf_dy = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Gpu("buffer alloc"))?;
    let buf_out = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Gpu("buffer alloc"))?;
    BufferPool::write_f32(&buf_g, &fix.g);
    BufferPool::write_f32(&buf_dy, &fix.dy);

    dispatch_1d(&ctx.queue, &pipeline.pipeline, len, |enc| unsafe {
        enc.setBuffer_offset_atIndex(Some(&*buf_g), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&*buf_dy), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&*buf_out), 0, 2);
        set_bytes(enc, &(len as u32), 3);
    })?;

    let mut out = vec![0.0f32; len];
    BufferPool::read_f32(&buf_out, &mut out);
    pool.release(len * 4, buf_g);
    pool.release(len * 4, buf_dy);
    pool.release(len * 4, buf_out);
    Ok(out)
}
