//! Metal dispatch for softmax_backward.

use super::{ENTRY, METAL};
use crate::Error;
use gpukit::metal::{BufferPool, dispatch_rows, set_bytes};
use objc2_metal::MTLComputeCommandEncoder;

pub fn gpu(fix: &super::Fixture) -> Result<Vec<f32>, Error> {
    let ctx = crate::metal_rt::context()?;
    let pipeline = crate::metal_rt::pipeline(&ctx, METAL, ENTRY)?;
    let mut pool = BufferPool::new();
    let len = fix.len();
    let buf_probs = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Gpu("buffer alloc"))?;
    let buf_dp = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Gpu("buffer alloc"))?;
    let buf_out = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Gpu("buffer alloc"))?;
    BufferPool::write_f32(&buf_probs, &fix.probs);
    BufferPool::write_f32(&buf_dp, &fix.dp);

    let dims = [fix.rows as u32, fix.cols as u32];
    dispatch_rows(&ctx.queue, &pipeline.pipeline, fix.rows, |enc| unsafe {
        enc.setBuffer_offset_atIndex(Some(&*buf_probs), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&*buf_dp), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&*buf_out), 0, 2);
        set_bytes(enc, &dims, 3);
    })?;

    let mut out = vec![0.0f32; len];
    BufferPool::read_f32(&buf_out, &mut out);
    pool.release(len * 4, buf_probs);
    pool.release(len * 4, buf_dp);
    pool.release(len * 4, buf_out);
    Ok(out)
}
