//! Metal dispatch for gemm.

use super::{BM, BN, ENTRY, METAL, THREADS};
use crate::Error;
use gpukit::metal::{BufferPool, dispatch_grid, div_up, set_bytes};
use objc2_metal::MTLComputeCommandEncoder;

pub fn gpu(fix: &super::Fixture) -> Result<Vec<f32>, Error> {
    let ctx = crate::metal_rt::context()?;
    let pipeline = crate::metal_rt::pipeline(&ctx, METAL, ENTRY)?;
    let p = fix.params;
    let mut pool = BufferPool::new();
    let out_len = fix.out_len();
    let buf_a = pool
        .allocate(&ctx.device, fix.a.len() * 4)
        .ok_or(Error::Gpu("buffer alloc"))?;
    let buf_b = pool
        .allocate(&ctx.device, fix.b.len() * 4)
        .ok_or(Error::Gpu("buffer alloc"))?;
    let buf_c = pool
        .allocate(&ctx.device, out_len * 4)
        .ok_or(Error::Gpu("buffer alloc"))?;
    BufferPool::write_f32(&buf_a, &fix.a);
    BufferPool::write_f32(&buf_b, &fix.b);
    BufferPool::write_f32(&buf_c, &fix.c);

    let grid_w = div_up(p.n as usize, BN);
    let grid_h = div_up(p.m as usize, BM);
    dispatch_grid(
        &ctx.queue,
        &pipeline.pipeline,
        grid_w,
        grid_h,
        THREADS as usize,
        |enc| unsafe {
            enc.setBuffer_offset_atIndex(Some(&*buf_a), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&*buf_b), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&*buf_c), 0, 2);
            set_bytes(enc, &p, 3);
        },
    )?;

    let mut out = vec![0.0f32; out_len];
    BufferPool::read_f32(&buf_c, &mut out);
    pool.release(fix.a.len() * 4, buf_a);
    pool.release(fix.b.len() * 4, buf_b);
    pool.release(out_len * 4, buf_c);
    Ok(out)
}
