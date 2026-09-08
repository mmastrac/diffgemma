//! Metal dispatch for fill_zero.

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
    BufferPool::write_f32(&buf, &fix.x);

    let range = [fix.base as u32, fix.count as u32];
    dispatch_1d(&ctx.queue, &pipeline.pipeline, fix.count, |enc| unsafe {
        enc.setBuffer_offset_atIndex(Some(&*buf), 0, 0);
        set_bytes(enc, &range, 1);
    })?;

    let mut out = vec![0.0f32; len];
    BufferPool::read_f32(&buf, &mut out);
    pool.release(len * 4, buf);
    Ok(out)
}
