use crate::buffer::Buffer;
use crate::metal::batch::GpuBatch;
use crate::metal::buffer::BufferPool;
use crate::metal::device::ComputePipeline;
use crate::safetensors::Error;
use crate::tensor::Bf16Slice;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice};
use std::cell::RefCell;

/// GPU linear weight in PyTorch `[out_dim, in_dim]` bf16 layout (no transpose).
pub struct CachedLinear {
    pub w: Buffer<u16>,
    pub in_dim: usize,
    pub out_dim: usize,
    gpu_w: RefCell<Option<Retained<ProtocolObject<dyn MTLBuffer>>>>,
}

impl CachedLinear {
    /// Copy mmap/safetensors weights as-is (`[out, in]`).
    pub fn from_bf16(w: Bf16Slice<'_>, out_dim: usize, in_dim: usize) -> Self {
        assert_eq!(w.len(), out_dim * in_dim);
        let mut buf = Buffer::new(w.len());
        let bytes = w.as_bytes();
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buf.as_slice_mut().as_mut_ptr() as *mut u8,
                bytes.len(),
            );
        }
        Self {
            w: buf,
            in_dim,
            out_dim,
            gpu_w: RefCell::new(None),
        }
    }

    pub fn gpu_weight(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        pool: &mut BufferPool,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        self.gpu_weight_tracked(device, pool, None)
    }

    pub fn gpu_weight_tracked(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        pool: &mut BufferPool,
        upload_bytes: Option<&mut u64>,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let mut slot = self.gpu_w.borrow_mut();
        if let Some(buf) = slot.as_ref() {
            return buf.clone();
        }
        let bytes = self.w.len() * 2;
        let buf = pool
            .allocate(device, bytes)
            .expect("Metal weight buffer alloc failed");
        BufferPool::write_bf16_ptr(&buf, self.w.as_slice().as_ptr(), self.w.len());
        *slot = Some(buf.clone());
        if let Some(acc) = upload_bytes {
            *acc += bytes as u64;
        }
        buf
    }

    pub fn clear_gpu(&self) {
        *self.gpu_w.borrow_mut() = None;
    }
}

/// Batched `y = x @ W^T` with pre-transposed weights; readback on `batch.end()`.
pub fn linear_cached_batched(
    batch: &mut GpuBatch<'_>,
    pipeline: &ComputePipeline,
    y: &mut [f32],
    x: &[f32],
    w: &CachedLinear,
    seq_len: usize,
) -> Result<(), Error> {
    if x.len() != seq_len * w.in_dim || y.len() != seq_len * w.out_dim {
        return Err(Error::Format("linear_cached_batched shape mismatch"));
    }
    let buf_a = batch.alloc_f32(x)?;
    let mut upload = 0u64;
    let buf_w = w.gpu_weight_tracked(batch.device, batch.pool, Some(&mut upload));
    batch.record_dense_upload(upload);
    let buf_c = batch.alloc_f32_out(y.len())?;
    batch.dispatch_linear(
        &pipeline.pipeline,
        &buf_a,
        &buf_w,
        &buf_c,
        seq_len,
        w.out_dim,
        w.in_dim,
    );
    batch.register_read(buf_c, y);
    Ok(())
}

/// `y = x_buf @ W^T`; output stays on GPU until the batch ends.
pub fn linear_cached_batched_in_buf(
    batch: &mut GpuBatch<'_>,
    pipeline: &ComputePipeline,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    w: &CachedLinear,
    seq_len: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    let out_len = seq_len * w.out_dim;
    let mut upload = 0u64;
    let buf_w = w.gpu_weight_tracked(batch.device, batch.pool, Some(&mut upload));
    batch.record_dense_upload(upload);
    let buf_c = batch.alloc_f32_out(out_len)?;
    batch.dispatch_linear(
        &pipeline.pipeline,
        x_buf,
        &buf_w,
        &buf_c,
        seq_len,
        w.out_dim,
        w.in_dim,
    );
    Ok(buf_c)
}

/// `y = x_buf @ W^T` with CPU readback on `batch.end()`.
pub fn linear_cached_batched_in_cpu_out(
    batch: &mut GpuBatch<'_>,
    pipeline: &ComputePipeline,
    y: &mut [f32],
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    w: &CachedLinear,
    seq_len: usize,
) -> Result<(), Error> {
    if y.len() != seq_len * w.out_dim {
        return Err(Error::Format("linear_cached_batched shape mismatch"));
    }
    let buf_c = linear_cached_batched_in_buf(batch, pipeline, x_buf, w, seq_len)?;
    batch.register_read(buf_c, y);
    Ok(())
}
