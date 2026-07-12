use crate::metal::buffer::BufferPool;
use crate::metal::device::{ComputePipeline, MetalContext};
use crate::model::attention::{AttentionParams, GqaMask};
use crate::safetensors::Error;
use crate::shaders::engine_gqa_common::{self, GqaParams};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState,
};

const ROPE_SHADER: &str = crate::shaders::apply_rope_heads::SHADER;
const GQA_SHADER: &str = crate::shaders::gqa_attention::SHADER;
const ROPE_ENTRY: &str = crate::shaders::apply_rope_heads::ENTRY;
const GQA_ENTRY: &str = crate::shaders::gqa_attention::ENTRY;

pub struct GpuAttentionKernels {
    pub rope_pipeline: ComputePipeline,
    pub attn_pipeline: ComputePipeline,
}

impl GpuAttentionKernels {
    pub fn new(ctx: &MetalContext) -> Result<Self, Error> {
        let mut rope = ctx.compile_kernels(ROPE_SHADER, &[ROPE_ENTRY])?;
        let mut attn = ctx.compile_kernels(GQA_SHADER, &[GQA_ENTRY])?;
        let attn_pipeline = attn.pop().ok_or(Error::Format("Metal pipeline missing"))?;
        let rope_pipeline = rope.pop().ok_or(Error::Format("Metal pipeline missing"))?;
        Ok(Self {
            rope_pipeline,
            attn_pipeline,
        })
    }
}

pub struct GpuAttention {
    ctx: MetalContext,
    kernels: GpuAttentionKernels,
    pool: BufferPool,
}

impl GpuAttention {
    pub fn new() -> Result<Self, Error> {
        let ctx = MetalContext::new()?;
        let kernels = GpuAttentionKernels::new(&ctx)?;
        Ok(Self {
            ctx,
            kernels,
            pool: BufferPool::new(),
        })
    }

    pub fn apply_rope(
        &mut self,
        x: &mut [f32],
        freqs: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
    ) -> Result<(), Error> {
        if x.len() != seq_len * num_heads * head_dim || freqs.len() != seq_len * rotary_dim {
            return Err(Error::Format("rope shape mismatch"));
        }

        let x_bytes = x.len() * 4;
        let f_bytes = freqs.len() * 4;
        let buf_x = self
            .pool
            .allocate(&self.ctx.device, x_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        let buf_f = self
            .pool
            .allocate(&self.ctx.device, f_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;

        BufferPool::write_f32(&buf_x, x);
        BufferPool::write_f32(&buf_f, freqs);

        let params = GqaParams::for_rope(seq_len, num_heads, head_dim, rotary_dim, 0);

        run_kernel(
            &self.ctx.queue,
            &self.kernels.rope_pipeline.pipeline,
            |encoder| {
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&buf_x), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&buf_f), 0, 1);
                }
                engine_gqa_common::set_params(encoder, &params, 2);
                engine_gqa_common::dispatch_head_query_2d(encoder, num_heads, seq_len);
            },
        )?;

        BufferPool::read_f32(&buf_x, x);
        self.pool.release(x_bytes, buf_x);
        self.pool.release(f_bytes, buf_f);
        Ok(())
    }

    pub fn gqa_attention(
        &mut self,
        attn_out: &mut [f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        total_kv: usize,
        params: &AttentionParams,
        mask: GqaMask<'_>,
    ) -> Result<(), Error> {
        let q_dim = seq_len * params.n_heads * params.head_dim;
        let kv_dim = total_kv * params.n_kv_heads * params.head_dim;
        let out_dim = seq_len * params.n_heads * params.head_dim;

        if q.len() != q_dim || k.len() != kv_dim || v.len() != kv_dim || attn_out.len() != out_dim {
            return Err(Error::Format("gqa attention shape mismatch"));
        }

        let (mask_kind, kv_cache_len, positions, decoder_mask) = match mask {
            GqaMask::CausalSliding => (engine_gqa_common::MASK_CAUSAL_SLIDING, 0usize, None, None),
            GqaMask::EncoderExtend {
                kv_cache_len,
                positions,
            } => (
                engine_gqa_common::MASK_ENCODER_EXTEND,
                kv_cache_len,
                Some(positions),
                None,
            ),
            GqaMask::DecoderBitmap(m) => (
                engine_gqa_common::MASK_DECODER_BITMAP,
                m.kv_cache_len,
                None,
                Some(m),
            ),
        };

        let q_bytes = q.len() * 4;
        let k_bytes = k.len() * 4;
        let v_bytes = v.len() * 4;
        let o_bytes = attn_out.len() * 4;

        let buf_q = self
            .pool
            .allocate(&self.ctx.device, q_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        let buf_k = self
            .pool
            .allocate(&self.ctx.device, k_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        let buf_v = self
            .pool
            .allocate(&self.ctx.device, v_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        let buf_o = self
            .pool
            .allocate(&self.ctx.device, o_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;

        BufferPool::write_f32(&buf_q, q);
        BufferPool::write_f32(&buf_k, k);
        BufferPool::write_f32(&buf_v, v);
        BufferPool::write_f32(&buf_o, attn_out);

        let mut buf_mask = None;
        if let Some(m) = decoder_mask {
            let packed: Vec<u8> = m.attend.iter().map(|&b| u8::from(b)).collect();
            let mask_bytes = packed.len();
            let b = self
                .pool
                .allocate(&self.ctx.device, mask_bytes)
                .ok_or(Error::Format("Metal buffer alloc failed"))?;
            BufferPool::write_bytes(&b, &packed);
            buf_mask = Some((mask_bytes, b));
        }

        let mut buf_pos = None;
        if let Some(pos) = positions {
            let pos_bytes = pos.len() * 8;
            let b = self
                .pool
                .allocate(&self.ctx.device, pos_bytes)
                .ok_or(Error::Format("Metal buffer alloc failed"))?;
            BufferPool::write_i64(&b, pos);
            buf_pos = Some((pos_bytes, b));
        }

        let gpu_params =
            GqaParams::for_attention(seq_len, total_kv, params, mask_kind, kv_cache_len);

        run_kernel(
            &self.ctx.queue,
            &self.kernels.attn_pipeline.pipeline,
            |encoder| {
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&buf_q), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&buf_k), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&buf_v), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(&buf_o), 0, 3);
                }
                if let Some((_, ref mask_buf)) = buf_mask {
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(mask_buf), 0, 4);
                    }
                }
                if let Some((_, ref pos_buf)) = buf_pos {
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(pos_buf), 0, 5);
                    }
                }
                engine_gqa_common::set_params(encoder, &gpu_params, 6);
                engine_gqa_common::dispatch_head_query_2d(encoder, params.n_heads, seq_len);
            },
        )?;

        BufferPool::read_f32(&buf_o, attn_out);

        self.pool.release(q_bytes, buf_q);
        self.pool.release(k_bytes, buf_k);
        self.pool.release(v_bytes, buf_v);
        self.pool.release(o_bytes, buf_o);
        if let Some((bytes, b)) = buf_mask {
            self.pool.release(bytes, b);
        }
        if let Some((bytes, b)) = buf_pos {
            self.pool.release(bytes, b);
        }
        Ok(())
    }

    /// RoPE on Q/K then GQA attention (matches CPU `forward_to_attn_out` attention stage).
    pub fn rope_and_gqa(
        &mut self,
        attn_out: &mut [f32],
        q: &mut [f32],
        k: &mut [f32],
        v: &[f32],
        freqs: &[f32],
        seq_len: usize,
        total_kv: usize,
        params: &AttentionParams,
        mask: GqaMask<'_>,
    ) -> Result<(), Error> {
        self.apply_rope(
            q,
            freqs,
            seq_len,
            params.n_heads,
            params.head_dim,
            params.rotary_dim,
        )?;
        self.apply_rope(
            k,
            freqs,
            seq_len,
            params.n_kv_heads,
            params.head_dim,
            params.rotary_dim,
        )?;
        self.gqa_attention(attn_out, q, k, v, seq_len, total_kv, params, mask)
    }
}

fn run_kernel(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    encode: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
) -> Result<(), Error> {
    let cmd_buf = queue
        .commandBuffer()
        .ok_or(Error::Format("Metal command buffer alloc failed"))?;
    let encoder = cmd_buf
        .computeCommandEncoder()
        .ok_or(Error::Format("Metal compute encoder alloc failed"))?;
    encoder.setComputePipelineState(pipeline);
    encode(&encoder);
    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    Ok(())
}
