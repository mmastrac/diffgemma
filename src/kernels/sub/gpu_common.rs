//! Shared Metal helpers for tier-1 subkernel GPU tests.

#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::safetensors::Error;
#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2::runtime::ProtocolObject;
#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn set_bytes<T>(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, value: &T, index: usize) {
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from_ref(value).cast(),
            std::mem::size_of_val(value),
            index,
        );
    }
}

pub fn div_up(value: usize, group: usize) -> usize {
    (value + group - 1) / group
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn dispatch_1d(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    count: usize,
    encode: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
) -> Result<(), Error> {
    let cmd = queue.commandBuffer().ok_or(Error::Format("command buffer"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("encoder"))?;
    enc.setComputePipelineState(pipeline);
    encode(&enc);
    let tg = 256usize.min(count);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: div_up(count, tg), height: 1, depth: 1 },
        MTLSize { width: tg, height: 1, depth: 1 },
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    Ok(())
}
