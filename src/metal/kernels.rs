use crate::metal::buffer::BufferPool;
use crate::metal::device::{ComputePipeline, MetalContext};
use crate::safetensors::Error;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

const DECODER_SHADER: &str = include_str!("../../shaders/decoder.metal");

/// One-byte scratch for optional dump buffer slot (compiled out when dump_stage=0).
fn dummy_dump_buf(
    pool: &mut BufferPool,
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>, Error> {
    pool.allocate(device, 4)
        .ok_or(Error::Format("Metal buffer alloc failed"))
}

pub struct GpuKernels {
    pub(crate) rms_norm: ComputePipeline,
    pub(crate) rms_norm_no_scale: ComputePipeline,
    pub(crate) vec_add: ComputePipeline,
    pub(crate) gather_rows: ComputePipeline,
    pub(crate) vec_mul: ComputePipeline,
    pub(crate) vec_scale: ComputePipeline,
    pub(crate) router_scale: ComputePipeline,
    pub(crate) gelu: ComputePipeline,
    pub(crate) swiglu_mul: ComputePipeline,
    pub(crate) gelu_swiglu_gate_up: ComputePipeline,
    pub(crate) softmax_rows: ComputePipeline,
    pub(crate) router_top_k: ComputePipeline,
    pub(crate) gather_prob_cols: ComputePipeline,
    pub(crate) vec_fill_zero: ComputePipeline,
}

impl GpuKernels {
    pub fn new(ctx: &MetalContext) -> Result<Self, Error> {
        let names = [
            "vec_add_inplace",
            "gather_rows",
            "vec_mul_inplace",
            "vec_scale_inplace",
            "router_scale_rows",
            "gelu_pytorch_tanh",
            "swiglu_mul",
        ];
        let pipelines = ctx.compile_kernels(DECODER_SHADER, &names)?;
        let rms_norm = crate::kernels::sub::rms_norm_rows::pipeline_for(
            ctx,
            crate::kernels::sub::variant::KernelVariant::PRODUCTION,
        )?;
        let rms_norm_no_scale = crate::kernels::sub::rms_norm_rows_no_scale::pipeline_for(
            ctx,
            crate::kernels::sub::variant::KernelVariant::PRODUCTION,
        )?;
        let softmax_rows = crate::kernels::sub::softmax_rows::pipeline_for(
            ctx,
            crate::kernels::sub::variant::KernelVariant::PRODUCTION,
        )?;
        let mut it = pipelines.into_iter().rev();
        Ok(Self {
            vec_fill_zero: ctx.compile_kernel(DECODER_SHADER, "vec_fill_zero")?,
            gather_prob_cols: ctx.compile_kernel(DECODER_SHADER, "gather_prob_cols")?,
            softmax_rows,
            router_top_k: ctx.compile_kernel(DECODER_SHADER, "router_top_k_rows")?,
            gelu_swiglu_gate_up: ctx.compile_kernel(DECODER_SHADER, "gelu_swiglu_gate_up")?,
            swiglu_mul: it.next().ok_or(Error::Format("pipeline missing"))?,
            gelu: it.next().ok_or(Error::Format("pipeline missing"))?,
            router_scale: it.next().ok_or(Error::Format("pipeline missing"))?,
            vec_scale: it.next().ok_or(Error::Format("pipeline missing"))?,
            vec_mul: it.next().ok_or(Error::Format("pipeline missing"))?,
            gather_rows: it.next().ok_or(Error::Format("pipeline missing"))?,
            vec_add: it.next().ok_or(Error::Format("pipeline missing"))?,
            rms_norm_no_scale,
            rms_norm,
        })
    }

    pub fn rms_norm_rows(
        &self,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        pool: &mut BufferPool,
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        out: &mut [f32],
        x: &[f32],
        weight: &[f32],
        seq_len: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<(), Error> {
        let len = seq_len * hidden;
        if out.len() != len || x.len() != len || weight.len() != hidden {
            return Err(Error::Format("rms_norm_rows shape mismatch"));
        }

        let x_bytes = len * 4;
        let w_bytes = hidden * 4;
        let o_bytes = len * 4;
        let buf_x = pool
            .allocate(device, x_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        let buf_w = pool
            .allocate(device, w_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        let buf_o = pool
            .allocate(device, o_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        BufferPool::write_f32(&buf_x, x);
        BufferPool::write_f32(&buf_w, weight);
        BufferPool::write_f32(&buf_o, out);

        let buf_dump = dummy_dump_buf(pool, device)?;
        let dims = [seq_len as u32, hidden as u32];
        run_1d(queue, &self.rms_norm.pipeline, seq_len, |enc| {
            crate::kernels::sub::rms_norm_rows::bind_gpu_buffers(
                &enc, &buf_x, &buf_w, &buf_o, &buf_dump, &dims, eps,
            );
        })?;

        BufferPool::read_f32(&buf_o, out);
        pool.release(x_bytes, buf_x);
        pool.release(w_bytes, buf_w);
        pool.release(o_bytes, buf_o);
        pool.release(4, buf_dump);
        Ok(())
    }

    pub fn rms_norm_rows_no_scale(
        &self,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        pool: &mut BufferPool,
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        out: &mut [f32],
        x: &[f32],
        seq_len: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<(), Error> {
        let len = seq_len * hidden;
        if out.len() != len || x.len() != len {
            return Err(Error::Format("rms_norm_no_scale shape mismatch"));
        }

        let x_bytes = len * 4;
        let o_bytes = len * 4;
        let buf_x = pool
            .allocate(device, x_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        let buf_o = pool
            .allocate(device, o_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        let buf_dump = dummy_dump_buf(pool, device)?;
        BufferPool::write_f32(&buf_x, x);
        BufferPool::write_f32(&buf_o, out);

        let dims = [seq_len as u32, hidden as u32];
        run_1d(queue, &self.rms_norm_no_scale.pipeline, seq_len, |enc| {
            crate::kernels::sub::rms_norm_rows_no_scale::bind_gpu_buffers(
                &enc, &buf_x, &buf_o, &buf_dump, &dims, eps,
            );
        })?;

        BufferPool::read_f32(&buf_o, out);
        pool.release(x_bytes, buf_x);
        pool.release(o_bytes, buf_o);
        pool.release(4, buf_dump);
        Ok(())
    }

    pub fn vec_add_inplace(
        &self,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        pool: &mut BufferPool,
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        out: &mut [f32],
        addend: &[f32],
    ) -> Result<(), Error> {
        if out.len() != addend.len() {
            return Err(Error::Format("vec_add shape mismatch"));
        }
        let bytes = out.len() * 4;
        let buf_o = pool
            .allocate(device, bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        let buf_a = pool
            .allocate(device, bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        BufferPool::write_f32(&buf_o, out);
        BufferPool::write_f32(&buf_a, addend);
        let len = out.len() as u32;
        run_1d(queue, &self.vec_add.pipeline, out.len(), |enc| {
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_o), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&buf_a), 0, 1);
            }
            set_bytes(enc, &len, 2);
        })?;
        BufferPool::read_f32(&buf_o, out);
        pool.release(bytes, buf_o);
        pool.release(bytes, buf_a);
        Ok(())
    }

    pub fn vec_mul_inplace(
        &self,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        pool: &mut BufferPool,
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        a: &mut [f32],
        b: &[f32],
    ) -> Result<(), Error> {
        if a.len() != b.len() {
            return Err(Error::Format("vec_mul shape mismatch"));
        }
        let bytes = a.len() * 4;
        let buf_a = pool
            .allocate(device, bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        let buf_b = pool
            .allocate(device, bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        BufferPool::write_f32(&buf_a, a);
        BufferPool::write_f32(&buf_b, b);
        let len = a.len() as u32;
        run_1d(queue, &self.vec_mul.pipeline, a.len(), |enc| {
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_a), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&buf_b), 0, 1);
            }
            set_bytes(enc, &len, 2);
        })?;
        BufferPool::read_f32(&buf_a, a);
        pool.release(bytes, buf_a);
        pool.release(bytes, buf_b);
        Ok(())
    }

    pub fn vec_scale_inplace(
        &self,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        pool: &mut BufferPool,
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        x: &mut [f32],
        scale: f32,
    ) -> Result<(), Error> {
        let bytes = x.len() * 4;
        let buf = pool
            .allocate(device, bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        BufferPool::write_f32(&buf, x);
        let len = x.len() as u32;
        run_1d(queue, &self.vec_scale.pipeline, x.len(), |enc| {
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf), 0, 0);
            }
            set_bytes(enc, &scale, 1);
            set_bytes(enc, &len, 2);
        })?;
        BufferPool::read_f32(&buf, x);
        pool.release(bytes, buf);
        Ok(())
    }

    pub fn router_scale_rows(
        &self,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        pool: &mut BufferPool,
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        x: &mut [f32],
        scale: &[f32],
        seq_len: usize,
        hidden: usize,
        root: f32,
    ) -> Result<(), Error> {
        let len = seq_len * hidden;
        if x.len() != len || scale.len() != hidden {
            return Err(Error::Format("router_scale shape mismatch"));
        }
        let x_bytes = len * 4;
        let s_bytes = hidden * 4;
        let buf_x = pool
            .allocate(device, x_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        let buf_s = pool
            .allocate(device, s_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        BufferPool::write_f32(&buf_x, x);
        BufferPool::write_f32(&buf_s, scale);
        let dims = [seq_len as u32, hidden as u32];
        run_1d(queue, &self.router_scale.pipeline, seq_len, |enc| {
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_x), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&buf_s), 0, 1);
            }
            set_bytes(enc, &dims, 2);
            set_bytes(enc, &root, 3);
        })?;
        BufferPool::read_f32(&buf_x, x);
        pool.release(x_bytes, buf_x);
        pool.release(s_bytes, buf_s);
        Ok(())
    }

    pub fn gelu_pytorch_tanh(
        &self,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        pool: &mut BufferPool,
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        x: &mut [f32],
    ) -> Result<(), Error> {
        let bytes = x.len() * 4;
        let buf = pool
            .allocate(device, bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        BufferPool::write_f32(&buf, x);
        let len = x.len() as u32;
        run_1d(queue, &self.gelu.pipeline, x.len(), |enc| {
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf), 0, 0);
            }
            set_bytes(enc, &len, 1);
        })?;
        BufferPool::read_f32(&buf, x);
        pool.release(bytes, buf);
        Ok(())
    }

    pub fn swiglu_mul(
        &self,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        pool: &mut BufferPool,
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        gate: &mut [f32],
        up: &[f32],
    ) -> Result<(), Error> {
        if gate.len() != up.len() {
            return Err(Error::Format("swiglu_mul shape mismatch"));
        }
        let bytes = gate.len() * 4;
        let buf_g = pool
            .allocate(device, bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        let buf_u = pool
            .allocate(device, bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        BufferPool::write_f32(&buf_g, gate);
        BufferPool::write_f32(&buf_u, up);
        let len = gate.len() as u32;
        run_1d(queue, &self.swiglu_mul.pipeline, gate.len(), |enc| {
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_g), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&buf_u), 0, 1);
            }
            set_bytes(enc, &len, 2);
        })?;
        BufferPool::read_f32(&buf_g, gate);
        pool.release(bytes, buf_g);
        pool.release(bytes, buf_u);
        Ok(())
    }
}

fn set_bytes<T>(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, value: &T, index: usize) {
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from_ref(value).cast(),
            std::mem::size_of_val(value),
            index,
        );
    }
}

const THREADS_PER_TG: usize = 256;

fn run_1d(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    count: usize,
    encode: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
) -> Result<(), Error> {
    let cmd = queue
        .commandBuffer()
        .ok_or(Error::Format("Metal command buffer alloc failed"))?;
    let enc = cmd
        .computeCommandEncoder()
        .ok_or(Error::Format("Metal compute encoder alloc failed"))?;
    enc.setComputePipelineState(pipeline);
    encode(&enc);
    let tg_width = THREADS_PER_TG.min(count);
    let grid_width = div_up(count, tg_width);
    let grid = MTLSize {
        width: grid_width,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    Ok(())
}

fn div_up(value: usize, group: usize) -> usize {
    (value + group - 1) / group
}
