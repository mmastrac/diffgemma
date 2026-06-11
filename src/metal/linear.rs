use crate::buffer::Buffer;
use crate::fast_slice::{transpose_bf16_weight_into, FastBf16Slice, FastSlice};
use crate::metal::batch::GpuBatch;
use crate::metal::device::ComputePipeline;
use crate::safetensors::Error;
use crate::tensor::Bf16Slice;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

pub struct CachedLinear {
    pub w_t: Buffer<u16>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl CachedLinear {
    pub fn from_bf16(w: Bf16Slice<'_>, out_dim: usize, in_dim: usize) -> Self {
        let mut w_t = Buffer::new(in_dim * out_dim);
        let src = FastBf16Slice::from_bf16(w);
        transpose_bf16_weight_into(src, w_t.as_fast_slice_mut(), out_dim, in_dim);
        Self {
            w_t,
            in_dim,
            out_dim,
        }
    }

    pub fn w_t_fast(&self) -> FastSlice<'_, u16> {
        self.w_t.as_fast_slice()
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
    let w_fast = w.w_t_fast();
    let buf_b = batch.alloc_bf16_fast(w_fast)?;
    let buf_c = batch.alloc_f32_out(y.len())?;
    batch.dispatch_gemm(
        &pipeline.pipeline,
        &buf_a,
        &buf_b,
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
    let w_fast = w.w_t_fast();
    let buf_b = batch.alloc_bf16_fast(w_fast)?;
    let buf_c = batch.alloc_f32_out(out_len)?;
    batch.dispatch_gemm(
        &pipeline.pipeline,
        x_buf,
        &buf_b,
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
