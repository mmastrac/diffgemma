//! Metal dispatch for rms_norm_backward.

use super::{ENTRY, METAL};
use crate::Error;
use gpukit::metal::{BufferPool, dispatch_rows, set_bytes};
use objc2_metal::MTLComputeCommandEncoder;

pub fn gpu(fix: &super::Fixture) -> Result<Vec<f32>, Error> {
    let ctx = crate::metal_rt::context()?;
    let pipeline = crate::metal_rt::pipeline(&ctx, METAL, ENTRY)?;
    let mut pool = BufferPool::new();
    let len = fix.len();
    let buf_x = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Gpu("buffer alloc"))?;
    let buf_w = pool
        .allocate(&ctx.device, fix.hidden * 4)
        .ok_or(Error::Gpu("buffer alloc"))?;
    let buf_dy = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Gpu("buffer alloc"))?;
    let buf_out = pool
        .allocate(&ctx.device, len * 4)
        .ok_or(Error::Gpu("buffer alloc"))?;
    BufferPool::write_f32(&buf_x, &fix.x);
    BufferPool::write_f32(&buf_w, &fix.weight);
    BufferPool::write_f32(&buf_dy, &fix.dy);

    let dims = [fix.rows as u32, fix.hidden as u32];
    dispatch_rows(&ctx.queue, &pipeline.pipeline, fix.rows, |enc| unsafe {
        enc.setBuffer_offset_atIndex(Some(&*buf_x), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&*buf_w), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&*buf_dy), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&*buf_out), 0, 3);
        set_bytes(enc, &dims, 4);
        set_bytes(enc, &fix.eps, 5);
    })?;

    let mut out = vec![0.0f32; len];
    BufferPool::read_f32(&buf_out, &mut out);
    pool.release(len * 4, buf_x);
    pool.release(fix.hidden * 4, buf_w);
    pool.release(len * 4, buf_dy);
    pool.release(len * 4, buf_out);
    Ok(out)
}
