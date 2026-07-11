//! Full-matrix block dequant kernels (test / validation helpers).

#[cfg(test)]
use crate::metal::device::ComputePipeline;
#[cfg(test)]
use crate::metal::dgq_gpu::Q4LinearGpu;
#[cfg(test)]
use objc2::runtime::ProtocolObject;
#[cfg(test)]
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};

#[cfg(test)]
pub fn dispatch_dequant_block_matrix(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ComputePipeline,
    q4: &Q4LinearGpu,
    buf_out: &ProtocolObject<dyn MTLBuffer>,
) {
    let (buf_w, off) = q4.weight_buffer();
    encoder.setComputePipelineState(&pipeline.pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(
            Some(buf_w),
            crate::dgq::layout::blob_offset_for_mtl(off),
            0,
        );
        encoder.setBuffer_offset_atIndex(Some(buf_out), 0, 1);
    }
    let groups_per_row = if q4.is_nvfp4() {
        0u32
    } else {
        q4.groups_per_row()
    };
    let dims = [q4.out_dim as u32, q4.in_dim as u32, groups_per_row];
    crate::metal::batch::set_bytes(encoder, &dims, 2);
    let (grid, tg) =
        crate::kernels::sub::dequant_block_matrix::dispatch_shape(q4.out_dim, q4.in_dim);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}
