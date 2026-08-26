//! One-shot dispatch helpers: encode, dispatch, and wait on one kernel.

use crate::Error;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

/// Bind `value`'s bytes as kernel argument `index`. `T` must be plain
/// `#[repr(C)]` data with no padding the kernel would misread.
pub fn set_bytes<T>(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    value: &T,
    index: usize,
) {
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from_ref(value).cast(),
            std::mem::size_of_val(value),
            index,
        );
    }
}

pub fn div_up(value: usize, group: usize) -> usize {
    value.div_ceil(group)
}

/// 1D dispatch chunked to Metal's 65535-threadgroup grid limit; `encode`
/// receives each chunk's `(base, count)` to bind as kernel arguments.
pub fn dispatch_1d_ranged(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    count: usize,
    mut encode: impl FnMut(&ProtocolObject<dyn MTLComputeCommandEncoder>, u32, u32),
) -> Result<(), Error> {
    const THREADS_PER_TG: usize = 256;
    const MAX_GROUPS: usize = 65535;
    const CHUNK: usize = MAX_GROUPS * THREADS_PER_TG;
    let cmd = queue.commandBuffer().ok_or(Error::Gpu("command buffer"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("encoder"))?;
    enc.setComputePipelineState(pipeline);
    let mut base = 0usize;
    while base < count {
        let chunk = (count - base).min(CHUNK);
        encode(&enc, base as u32, chunk as u32);
        let tg_width = THREADS_PER_TG.min(chunk);
        let grid_width = div_up(chunk, tg_width);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: grid_width,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            },
        );
        base += chunk;
    }
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    Ok(())
}

/// Row kernels: one threadgroup (width 256) per row.
pub fn dispatch_rows(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    rows: usize,
    encode: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
) -> Result<(), Error> {
    const THREADGROUP_WIDTH: usize = 256;
    let cmd = queue.commandBuffer().ok_or(Error::Gpu("command buffer"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("encoder"))?;
    enc.setComputePipelineState(pipeline);
    encode(&enc);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: 1,
            height: rows,
            depth: 1,
        },
        MTLSize {
            width: THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    Ok(())
}

pub fn dispatch_1d(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    count: usize,
    encode: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
) -> Result<(), Error> {
    let cmd = queue.commandBuffer().ok_or(Error::Gpu("command buffer"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("encoder"))?;
    enc.setComputePipelineState(pipeline);
    encode(&enc);
    let tg = 256usize.min(count);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: div_up(count, tg),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    Ok(())
}

/// 2D grid with fixed threadgroup width (e.g. one threadgroup per `(x, y)` pair).
pub fn dispatch_grid(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    width: usize,
    height: usize,
    tg_width: usize,
    encode: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
) -> Result<(), Error> {
    let cmd = queue.commandBuffer().ok_or(Error::Gpu("command buffer"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("encoder"))?;
    enc.setComputePipelineState(pipeline);
    encode(&enc);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width,
            height,
            depth: 1,
        },
        MTLSize {
            width: tg_width,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    Ok(())
}
