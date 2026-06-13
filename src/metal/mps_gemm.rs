//! MPSMatrixMultiplication wrapper for dense f32 linear layers (Q2.5).

use crate::metal::device::ComputePipeline;
use crate::metal::dgq_gpu::Q4LinearGpu;
use crate::safetensors::Error;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLComputeCommandEncoder, MTLDevice};
use objc2::AllocAnyThread;
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};
use std::collections::HashMap;

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
struct MatmulKey {
    m: usize,
    k: usize,
    n: usize,
}

pub struct MpsMatmulCache {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    kernels: HashMap<MatmulKey, Retained<MPSMatrixMultiplication>>,
}

impl MpsMatmulCache {
    pub fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self {
            device,
            kernels: HashMap::new(),
        }
    }

    fn kernel(&mut self, m: usize, k: usize, n: usize) -> &MPSMatrixMultiplication {
        let key = MatmulKey { m, k, n };
        self.kernels.entry(key).or_insert_with(|| unsafe {
            MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                MPSMatrixMultiplication::alloc(),
                &self.device,
                false,
                true,
                m,
                n,
                k,
                1.0,
                0.0,
            )
        })
    }

    /// `C[M,N] = A[M,K] @ W[N,K]^T` with f32 row-major buffers (PyTorch linear layout).
    pub fn encode_f32_linear(
        &mut self,
        cmd: &ProtocolObject<dyn MTLCommandBuffer>,
        buf_a: &ProtocolObject<dyn MTLBuffer>,
        buf_w: &ProtocolObject<dyn MTLBuffer>,
        buf_c: &ProtocolObject<dyn MTLBuffer>,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let kernel = self.kernel(m, k, n);
        let (a_desc, a_mat) = matrix_view(buf_a, m, k);
        let (w_desc, w_mat) = matrix_view(buf_w, n, k);
        let (c_desc, c_mat) = matrix_view(buf_c, m, n);
        unsafe {
            kernel.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                cmd,
                &a_mat,
                &w_mat,
                &c_mat,
            );
        }
        let _ = (a_desc, w_desc, c_desc);
    }
}

fn matrix_view(
    buf: &ProtocolObject<dyn MTLBuffer>,
    rows: usize,
    cols: usize,
) -> (Retained<MPSMatrixDescriptor>, Retained<MPSMatrix>) {
    let row_bytes = unsafe {
        MPSMatrixDescriptor::rowBytesForColumns_dataType(cols, MPSDataType::Float32)
    };
    let desc = unsafe {
        MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            rows,
            cols,
            row_bytes,
            MPSDataType::Float32,
        )
    };
    let mat =
        unsafe { MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), buf, &desc) };
    (desc, mat)
}

pub fn dispatch_dequant_q4_matrix(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ComputePipeline,
    q4: &Q4LinearGpu,
    buf_out: &ProtocolObject<dyn MTLBuffer>,
) {
    const THREADGROUP: usize = 16;
    let (buf_w, off) = q4.weight_buffer();
    encoder.setComputePipelineState(&pipeline.pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(buf_w), off as usize, 0);
        encoder.setBuffer_offset_atIndex(Some(buf_out), 0, 1);
    }
    let dims = [
        q4.out_dim as u32,
        q4.in_dim as u32,
        q4.groups_per_row(),
    ];
    crate::metal::batch::set_bytes(encoder, &dims, 2);
    let tg = objc2_metal::MTLSize {
        width: THREADGROUP,
        height: THREADGROUP,
        depth: 1,
    };
    let grid = objc2_metal::MTLSize {
        width: (q4.in_dim + THREADGROUP - 1) / THREADGROUP,
        height: (q4.out_dim + THREADGROUP - 1) / THREADGROUP,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

/// Q4 weight dequant → f32 scratch → MPS linear. Requires compute encoder pause/resume on batch.
pub fn q4_dequant_mps_linear(
    batch: &mut crate::metal::batch::GpuBatch<'_>,
    dequant_pipeline: &ComputePipeline,
    mps: &mut MpsMatmulCache,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    q4: &Q4LinearGpu,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    if q4.in_dim != k || q4.out_dim != n {
        return Err(Error::Format("q4/mps shape mismatch"));
    }
    let buf_w = batch.alloc_f32_out(n * k)?;
    {
        let enc = batch.encoder();
        dispatch_dequant_q4_matrix(enc, dequant_pipeline, q4, &buf_w);
    }
    let buf_c = batch.alloc_f32_out(m * n)?;
    batch.pause_compute_for_mps(|cmd| {
        mps.encode_f32_linear(cmd, x_buf, &buf_w, &buf_c, m, k, n);
    });
    Ok(buf_c)
}

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod tests {
    use super::*;
    use crate::dgq::DgqStore;
    use crate::metal::batch::GpuBatch;
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use crate::metal::dgq_gpu::{load_q4_linear, DgqGpuBlob};
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
        let q4 = load_q4_linear(
            &store,
            Arc::clone(&blob),
            "model.decoder.layers.0.self_attn.q_proj.weight",
        )
        .expect("q4 view");
        let f32_w = store
            .tensor_f32("model.decoder.layers.0.self_attn.q_proj.weight")
            .expect("dequant");

        let pipeline = ctx
            .compile_kernel(
                include_str!("../../shaders/qgemm.metal"),
                "dequant_q4_matrix",
            )
            .expect("pipeline");
        let mut pool = BufferPool::new();
        let mut batch = GpuBatch::begin(&ctx.queue, &mut pool, &ctx.device).expect("batch");
        let buf_out = batch.alloc_f32_out(q4.out_dim * q4.in_dim).expect("out");
        {
            let enc = batch.encoder();
            dispatch_dequant_q4_matrix(enc, &pipeline, &q4, &buf_out);
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
