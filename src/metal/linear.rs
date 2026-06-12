use crate::buffer::Buffer;
use crate::metal::batch::GpuBatch;
use crate::metal::buffer::BufferPool;
use crate::metal::device::ComputePipeline;
use crate::metal::dgq_gpu::{Q4LinearGpu, Q8LinearGpu};
use crate::safetensors::Error;
use crate::tensor::Bf16Slice;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice};

/// GPU linear weight: bf16 upload path (safetensors) or q4 blob view (`.dgq`).
pub enum GpuLinearWeight {
    Bf16(CachedLinear),
    Q4(Q4LinearGpu),
}

impl GpuLinearWeight {
    pub fn bf16(w: Bf16Slice<'_>, out_dim: usize, in_dim: usize) -> Self {
        Self::Bf16(CachedLinear::from_bf16(w, out_dim, in_dim))
    }

    pub fn q4(view: Q4LinearGpu) -> Self {
        Self::Q4(view)
    }

    pub fn in_dim(&self) -> usize {
        match self {
            Self::Bf16(w) => w.in_dim,
            Self::Q4(w) => w.in_dim,
        }
    }

    pub fn out_dim(&self) -> usize {
        match self {
            Self::Bf16(w) => w.out_dim,
            Self::Q4(w) => w.out_dim,
        }
    }

    pub fn is_q4(&self) -> bool {
        matches!(self, Self::Q4(_))
    }
}

/// GPU linear weight in PyTorch `[out_dim, in_dim]` bf16 layout (no transpose).
pub struct CachedLinear {
    pub w: Buffer<u16>,
    pub in_dim: usize,
    pub out_dim: usize,
    gpu_w: std::cell::RefCell<Option<Retained<ProtocolObject<dyn MTLBuffer>>>>,
}

impl CachedLinear {
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
            gpu_w: std::cell::RefCell::new(None),
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

pub fn linear_batched_in_buf(
    batch: &mut GpuBatch<'_>,
    bf16_pipeline: &ComputePipeline,
    q4_pipeline: &ComputePipeline,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    w: &GpuLinearWeight,
    seq_len: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    let out_len = seq_len * w.out_dim();
    let buf_c = batch.alloc_f32_out(out_len)?;
    match w {
        GpuLinearWeight::Bf16(cached) => {
            let mut upload = 0u64;
            let buf_w = cached.gpu_weight_tracked(batch.device, batch.pool, Some(&mut upload));
            batch.record_dense_upload(upload);
            batch.dispatch_linear(
                &bf16_pipeline.pipeline,
                x_buf,
                &buf_w,
                &buf_c,
                seq_len,
                w.out_dim(),
                w.in_dim(),
            );
        }
        GpuLinearWeight::Q4(q4) => {
            let (buf_w, off) = q4.weight_buffer();
            batch.dispatch_q4_linear(
                &q4_pipeline.pipeline,
                x_buf,
                buf_w,
                off,
                &buf_c,
                seq_len,
                w.out_dim(),
                w.in_dim(),
                q4.groups_per_row(),
            );
        }
    }
    Ok(buf_c)
}

pub fn linear_batched_in_cpu_out(
    batch: &mut GpuBatch<'_>,
    bf16_pipeline: &ComputePipeline,
    q4_pipeline: &ComputePipeline,
    y: &mut [f32],
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    w: &GpuLinearWeight,
    seq_len: usize,
) -> Result<(), Error> {
    if y.len() != seq_len * w.out_dim() {
        return Err(Error::Format("linear_batched shape mismatch"));
    }
    let buf_c = linear_batched_in_buf(batch, bf16_pipeline, q4_pipeline, x_buf, w, seq_len)?;
    batch.register_read(buf_c, y);
    Ok(())
}

/// Batched `y = x @ W^T` with pre-transposed weights; readback on `batch.end()`.
pub fn linear_cached_batched(
    batch: &mut GpuBatch<'_>,
    bf16_pipeline: &ComputePipeline,
    q4_pipeline: &ComputePipeline,
    y: &mut [f32],
    x: &[f32],
    w: &GpuLinearWeight,
    seq_len: usize,
) -> Result<(), Error> {
    if x.len() != seq_len * w.in_dim() || y.len() != seq_len * w.out_dim() {
        return Err(Error::Format("linear_cached_batched shape mismatch"));
    }
    let buf_a = batch.alloc_f32(x)?;
    let buf_c = linear_batched_in_buf(batch, bf16_pipeline, q4_pipeline, &buf_a, w, seq_len)?;
    batch.register_read(buf_c, y);
    Ok(())
}

/// `y = x_buf @ W^T`; output stays on GPU until the batch ends.
pub fn linear_cached_batched_in_buf(
    batch: &mut GpuBatch<'_>,
    bf16_pipeline: &ComputePipeline,
    q4_pipeline: &ComputePipeline,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    w: &GpuLinearWeight,
    seq_len: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    linear_batched_in_buf(batch, bf16_pipeline, q4_pipeline, x_buf, w, seq_len)
}

/// `y = x_buf @ W^T` with CPU readback on `batch.end()`.
pub fn linear_cached_batched_in_cpu_out(
    batch: &mut GpuBatch<'_>,
    bf16_pipeline: &ComputePipeline,
    q4_pipeline: &ComputePipeline,
    y: &mut [f32],
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    w: &GpuLinearWeight,
    seq_len: usize,
) -> Result<(), Error> {
    linear_batched_in_cpu_out(
        batch,
        bf16_pipeline,
        q4_pipeline,
        y,
        x_buf,
        w,
        seq_len,
    )
}

pub fn f32_q4_linear_gpu_bufs(
    batch: &mut GpuBatch<'_>,
    q4_pipeline: &ComputePipeline,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    w: &Q4LinearGpu,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    let buf_c = batch.alloc_f32_out(m * n)?;
    let (buf_w, off) = w.weight_buffer();
    batch.dispatch_q4_linear(
        &q4_pipeline.pipeline,
        x_buf,
        buf_w,
        off,
        &buf_c,
        m,
        n,
        k,
        w.groups_per_row(),
    );
    Ok(buf_c)
}

pub fn f32_q8_linear_gpu_bufs(
    batch: &mut GpuBatch<'_>,
    q8_pipeline: &ComputePipeline,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    w: &Q8LinearGpu,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    let buf_c = batch.alloc_f32_out(m * n)?;
    let (buf_w, off) = w.weight_buffer();
    batch.dispatch_q8_linear(
        &q8_pipeline.pipeline,
        x_buf,
        buf_w,
        off,
        &buf_c,
        m,
        n,
        k,
    );
    Ok(buf_c)
}

pub fn f32_linear_gpu_bufs(
    batch: &mut GpuBatch<'_>,
    bf16_pipeline: &ComputePipeline,
    q4_pipeline: &ComputePipeline,
    x_buf: &ProtocolObject<dyn MTLBuffer>,
    w: &GpuLinearWeight,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    match w {
        GpuLinearWeight::Bf16(cached) => {
            let mut upload = 0u64;
            let buf_w = cached.gpu_weight_tracked(batch.device, batch.pool, Some(&mut upload));
            batch.record_dense_upload(upload);
            let buf_c = batch.alloc_f32_out(m * n)?;
            batch.dispatch_linear(
                &bf16_pipeline.pipeline,
                x_buf,
                &buf_w,
                &buf_c,
                m,
                n,
                k,
            );
            Ok(buf_c)
        }
        GpuLinearWeight::Q4(q4) => f32_q4_linear_gpu_bufs(batch, q4_pipeline, x_buf, q4, m, k, n),
    }
}
