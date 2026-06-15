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
    let dims = [
        q4.out_dim as u32,
        q4.in_dim as u32,
        groups_per_row,
    ];
    crate::metal::batch::set_bytes(encoder, &dims, 2);
    let (grid, tg) =
        crate::kernels::sub::dequant_block_matrix::dispatch_shape(q4.out_dim, q4.in_dim);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod tests {
    use super::*;
    use crate::dgq::DgqStore;
    use crate::kernels::sub::variant::KernelVariant;
    use crate::kernels::sub::QuantFormat;
    use crate::metal::batch::GpuBatch;
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use crate::metal::dgq_gpu::{load_block_linear, DgqGpuBlob};
    use std::sync::Arc;

    #[test]
    fn dequant_q4_matrix_matches_cpu_row() {
        let dgq_dir = std::path::Path::new("/tmp/quantized-weights");
        if !dgq_dir.join("model.dgq.json").exists() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let store = DgqStore::open(dgq_dir).expect("open dgq");
        let ctx = MetalContext::new().expect("metal");
        let blob = DgqGpuBlob::from_store(&store, &ctx.device).expect("blob");
        let q4 = load_block_linear(
            &store,
            Arc::clone(&blob),
            "model.decoder.layers.0.self_attn.q_proj.weight",
        )
        .expect("q4 view");
        let f32_w = store
            .tensor_f32("model.decoder.layers.0.self_attn.q_proj.weight")
            .expect("dequant");

        let pipeline = crate::kernels::sub::dequant_block_matrix::pipeline_for(
            &ctx,
            QuantFormat::Q4Affine,
            KernelVariant::PRODUCTION,
        )
        .expect("pipeline");
        let mut pool = BufferPool::new();
        let mut batch = GpuBatch::begin_with_telemetry(&ctx.queue, &mut pool, &ctx.device, None).expect("batch");
        let buf_out = batch.alloc_f32_out(q4.out_dim * q4.in_dim).expect("out");
        {
            let enc = batch.encoder();
            dispatch_dequant_block_matrix(enc, &pipeline, &q4, &buf_out);
        }
        let mut gpu_w = vec![0.0f32; q4.out_dim * q4.in_dim];
        batch.register_read(buf_out, &mut gpu_w);
        batch.end().expect("end");

        let mut max_err = 0.0f32;
        for (a, b) in f32_w.iter().zip(gpu_w.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        eprintln!("dequant_q4_matrix vs cpu max_err={max_err:.6}");
        assert!(max_err < 0.05, "max_err={max_err}");
    }
}
