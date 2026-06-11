use crate::metal::buffer::BufferPool;
use crate::metal::device::{ComputePipeline, MetalContext};
use crate::model::attention::{AttentionParams, GqaMask, MASK_NEG};
use crate::safetensors::Error;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

const ATTENTION_SHADER: &str = include_str!("../../shaders/attention.metal");
const ROPE_ENTRY: &str = "apply_rope_heads";
const GQA_ENTRY: &str = "gqa_attention";

const MASK_CAUSAL_SLIDING: u32 = 0;
const MASK_ENCODER_EXTEND: u32 = 1;
const MASK_DECODER_BITMAP: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy)]
struct GqaParams {
    seq_len: u32,
    total_kv: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    n_groups: u32,
    mask_kind: u32,
    sliding_window: u32,
    kv_cache_len: u32,
    mask_neg: f32,
    rotary_dim: u32,
    num_heads_rope: u32,
}

pub struct GpuAttention {
    ctx: MetalContext,
    rope_pipeline: ComputePipeline,
    attn_pipeline: ComputePipeline,
    pool: BufferPool,
}

impl GpuAttention {
    pub fn new() -> Result<Self, Error> {
        let ctx = MetalContext::new()?;
        let mut pipelines = ctx.compile_kernels(ATTENTION_SHADER, &[ROPE_ENTRY, GQA_ENTRY])?;
        let attn_pipeline = pipelines.pop().ok_or(Error::Format("Metal pipeline missing"))?;
        let rope_pipeline = pipelines.pop().ok_or(Error::Format("Metal pipeline missing"))?;
        Ok(Self {
            ctx,
            rope_pipeline,
            attn_pipeline,
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

        let params = GqaParams {
            seq_len: seq_len as u32,
            total_kv: 0,
            n_heads: 0,
            n_kv_heads: 0,
            head_dim: head_dim as u32,
            n_groups: 0,
            mask_kind: 0,
            sliding_window: 0,
            kv_cache_len: 0,
            mask_neg: MASK_NEG,
            rotary_dim: rotary_dim as u32,
            num_heads_rope: num_heads as u32,
        };

        run_kernel(
            &self.ctx.queue,
            &self.rope_pipeline.pipeline,
            |encoder| {
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&buf_x), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&buf_f), 0, 1);
                }
                set_params(encoder, &params, 2);
                dispatch_2d(encoder, num_heads, seq_len);
            },
        )?;

        BufferPool::read_f32(&buf_x, x);
        self.pool.release(x_bytes, buf_x);
        self.pool.release(f_bytes, buf_f);
        Ok(())
    }

    /// RoPE on Q and K canvas tensors in a single GPU submit.
    pub fn apply_rope_qk(
        &mut self,
        q: &mut [f32],
        k: &mut [f32],
        freqs: &[f32],
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
    ) -> Result<(), Error> {
        let q_bytes = q.len() * 4;
        let k_bytes = k.len() * 4;
        let f_bytes = freqs.len() * 4;
        if q.len() != seq_len * n_heads * head_dim
            || k.len() != seq_len * n_kv_heads * head_dim
            || freqs.len() != seq_len * rotary_dim
        {
            return Err(Error::Format("rope qk shape mismatch"));
        }

        let buf_q = self
            .pool
            .allocate(&self.ctx.device, q_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        let buf_k = self
            .pool
            .allocate(&self.ctx.device, k_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;
        let buf_f = self
            .pool
            .allocate(&self.ctx.device, f_bytes)
            .ok_or(Error::Format("Metal buffer alloc failed"))?;

        BufferPool::write_f32(&buf_q, q);
        BufferPool::write_f32(&buf_k, k);
        BufferPool::write_f32(&buf_f, freqs);

        let rope_q = GqaParams {
            seq_len: seq_len as u32,
            total_kv: 0,
            n_heads: 0,
            n_kv_heads: 0,
            head_dim: head_dim as u32,
            n_groups: 0,
            mask_kind: 0,
            sliding_window: 0,
            kv_cache_len: 0,
            mask_neg: MASK_NEG,
            rotary_dim: rotary_dim as u32,
            num_heads_rope: n_heads as u32,
        };
        let rope_k = GqaParams {
            num_heads_rope: n_kv_heads as u32,
            ..rope_q
        };

        run_kernel(
            &self.ctx.queue,
            &self.rope_pipeline.pipeline,
            |encoder| {
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&buf_q), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&buf_f), 0, 1);
                }
                set_params(encoder, &rope_q, 2);
                dispatch_2d(encoder, n_heads, seq_len);

                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&buf_k), 0, 0);
                }
                set_params(encoder, &rope_k, 2);
                dispatch_2d(encoder, n_kv_heads, seq_len);
            },
        )?;

        BufferPool::read_f32(&buf_q, q);
        BufferPool::read_f32(&buf_k, k);
        self.pool.release(q_bytes, buf_q);
        self.pool.release(k_bytes, buf_k);
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
            GqaMask::CausalSliding => (MASK_CAUSAL_SLIDING, 0usize, None, None),
            GqaMask::EncoderExtend {
                kv_cache_len,
                positions,
            } => (MASK_ENCODER_EXTEND, kv_cache_len, Some(positions), None),
            GqaMask::DecoderBitmap(m) => (MASK_DECODER_BITMAP, m.kv_cache_len, None, Some(m)),
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

        let gpu_params = GqaParams {
            seq_len: seq_len as u32,
            total_kv: total_kv as u32,
            n_heads: params.n_heads as u32,
            n_kv_heads: params.n_kv_heads as u32,
            head_dim: params.head_dim as u32,
            n_groups: params.n_groups as u32,
            mask_kind,
            sliding_window: params.sliding_window.unwrap_or(0) as u32,
            kv_cache_len: kv_cache_len as u32,
            mask_neg: MASK_NEG,
            rotary_dim: params.rotary_dim as u32,
            num_heads_rope: 0,
        };

        run_kernel(
            &self.ctx.queue,
            &self.attn_pipeline.pipeline,
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
                set_params(encoder, &gpu_params, 6);
                dispatch_2d(encoder, params.n_heads, seq_len);
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
        self.apply_rope(q, freqs, seq_len, params.n_heads, params.head_dim, params.rotary_dim)?;
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

fn set_params(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    params: &GqaParams,
    index: usize,
) {
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from_ref(params).cast(),
            std::mem::size_of::<GqaParams>(),
            index,
        );
    }
}

fn dispatch_2d(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, width: usize, height: usize) {
    let grid = MTLSize {
        width,
        height,
        depth: 1,
    };
    let tg = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
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

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod tests {
    use super::*;
    use crate::model::attention::gqa_attention as gqa_attention_cpu;

    #[test]
    fn gpu_gqa_matches_cpu_small() {
        let seq_len = 4usize;
        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = 8usize;
        let params = AttentionParams {
            n_heads,
            n_kv_heads,
            head_dim,
            rotary_dim: head_dim,
            n_groups: n_heads / n_kv_heads,
            sliding_window: Some(2),
        };

        let q_dim = seq_len * n_heads * head_dim;
        let kv_dim = seq_len * n_kv_heads * head_dim;
        let q: Vec<f32> = (0..q_dim).map(|i| (i as f32) * 0.03 - 0.2).collect();
        let k: Vec<f32> = (0..kv_dim).map(|i| (i as f32) * 0.02 - 0.1).collect();
        let v: Vec<f32> = (0..kv_dim).map(|i| (i as f32) * 0.01).collect();
        let mut cpu_out = vec![0.0f32; q_dim];
        let mut cpu_scores = vec![0.0f32; seq_len * n_heads * seq_len];
        gqa_attention_cpu(
            &mut cpu_out,
            &mut cpu_scores,
            &q,
            &k,
            &v,
            seq_len,
            seq_len,
            &crate::model::attention::AttentionParams {
                n_heads,
                n_kv_heads,
                head_dim,
                rotary_dim: head_dim,
                n_groups: n_heads / n_kv_heads,
                sliding_window: Some(2),
            },
            GqaMask::CausalSliding,
        );

        let mut gpu = GpuAttention::new().expect("gpu");
        let mut gpu_out = vec![0.0f32; q_dim];
        gpu.gqa_attention(
            &mut gpu_out,
            &q,
            &k,
            &v,
            seq_len,
            seq_len,
            &params,
            GqaMask::CausalSliding,
        )
        .expect("gpu attention");

        let max_diff = cpu_out
            .iter()
            .zip(gpu_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "max_diff={max_diff}");
    }
}
