//! Metal dispatch for vec_add.

use super::{ENTRY, METAL};
use crate::Error;
use gpukit::metal::{BufferPool, dispatch_1d, set_bytes};
use objc2_metal::MTLComputeCommandEncoder;

pub fn gpu(fix: &super::Fixture) -> Result<Vec<f32>, Error> {
    let ctx = gpukit::metal::cached_context(gpukit::metal::CacheConfig::MEMORY)?;
    let pipeline = gpukit::metal::cached_pipeline(&ctx, METAL, ENTRY)?;
    let mut pool = BufferPool::new();
    let len = fix.len();
    let buf = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Gpu("buffer alloc"))?;
    let buf_addend = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Gpu("buffer alloc"))?;
    BufferPool::write_f32(&buf, &fix.x);
    BufferPool::write_f32(&buf_addend, &fix.addend);

    dispatch_1d(&ctx.queue, &pipeline.pipeline, len, |enc| unsafe {
        enc.setBuffer_offset_atIndex(Some(&*buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&*buf_addend), 0, 1);
        set_bytes(enc, &(len as u32), 2);
    })?;

    let mut out = vec![0.0f32; len];
    BufferPool::read_f32(&buf, &mut out);
    pool.release(len * 4, buf);
    pool.release(len * 4, buf_addend);
    Ok(out)
}
