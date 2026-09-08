//! Metal dispatch for scatter_add_rows.

use super::{ENTRY, METAL};
use crate::Error;
use gpukit::metal::{BufferPool, dispatch_rows, set_bytes};
use objc2_metal::MTLComputeCommandEncoder;

pub fn gpu(fix: &super::Fixture) -> Result<Vec<f32>, Error> {
    let ctx = gpukit::metal::cached_context(gpukit::metal::CacheConfig::MEMORY)?;
    let pipeline = gpukit::metal::cached_pipeline(&ctx, METAL, ENTRY)?;
    let mut pool = BufferPool::new();
    let out_len = fix.len();
    let buf_dst = pool
        .allocate(&ctx.device, out_len * 4)
        .ok_or(Error::Gpu("buffer alloc"))?;
    let buf_idx = pool
        .allocate(&ctx.device, fix.indices.len() * 4)
        .ok_or(Error::Gpu("buffer alloc"))?;
    let buf_src = pool
        .allocate(&ctx.device, fix.src.len() * 4)
        .ok_or(Error::Gpu("buffer alloc"))?;
    BufferPool::write_f32(&buf_dst, &fix.dst);
    BufferPool::write_bytes(&buf_idx, unsafe {
        std::slice::from_raw_parts(fix.indices.as_ptr().cast::<u8>(), fix.indices.len() * 4)
    });
    BufferPool::write_f32(&buf_src, &fix.src);

    let dims = [fix.indices.len() as u32, fix.hidden as u32];
    dispatch_rows(
        &ctx.queue,
        &pipeline.pipeline,
        fix.indices.len(),
        |enc| unsafe {
            enc.setBuffer_offset_atIndex(Some(&*buf_dst), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&*buf_idx), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&*buf_src), 0, 2);
            set_bytes(enc, &dims, 3);
        },
    )?;

    let mut out = vec![0.0f32; out_len];
    BufferPool::read_f32(&buf_dst, &mut out);
    pool.release(out_len * 4, buf_dst);
    pool.release(fix.indices.len() * 4, buf_idx);
    pool.release(fix.src.len() * 4, buf_src);
    Ok(out)
}
