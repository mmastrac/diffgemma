//! Metal dispatch for softmax.

use super::{ENTRY, METAL};
use crate::Error;
use gpukit::metal::{BufferPool, dispatch_rows, set_bytes};
use objc2_metal::MTLComputeCommandEncoder;

pub fn gpu(fix: &super::Fixture) -> Result<Vec<f32>, Error> {
    let ctx = crate::metal_rt::context()?;
    let pipeline = ctx.compile_kernel(METAL, ENTRY)?;
    let mut pool = BufferPool::new();
    let len = fix.len();
    let buf = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Gpu("buffer alloc"))?;
    BufferPool::write_f32(&buf, &fix.logits);

    let dims = [fix.rows as u32, fix.cols as u32];
    dispatch_rows(&ctx.queue, &pipeline.pipeline, fix.rows, |enc| unsafe {
        enc.setBuffer_offset_atIndex(Some(&*buf), 0, 0);
        set_bytes(enc, &dims, 1);
    })?;

    let mut out = vec![0.0f32; len];
    BufferPool::read_f32(&buf, &mut out);
    pool.release(len * 4, buf);
    Ok(out)
}
