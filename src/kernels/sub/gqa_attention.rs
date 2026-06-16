//! Encoder-path GQA attention (f32, naive softmax, mask modes for prefill/extend).

use super::engine_gqa_common::{self, GqaParams, MASK_CAUSAL_SLIDING, MASK_DECODER_BITMAP, MASK_ENCODER_EXTEND};
use super::test_util::ElemFormat;
use crate::model::attention::{self, AttentionParams, GqaMask};
use crate::model::mask::DecoderAttnMask;
use crate::safetensors::Error;

pub const ENTRY: &str = "gqa_attention";

const SHADER: &str = shader_include::include_metal!("kernels/gqa_attention.metal");

#[derive(Debug, Clone)]
pub enum MaskMode {
    CausalSliding,
    EncoderExtend {
        kv_cache_len: usize,
        positions: Vec<i64>,
    },
    DecoderBitmap(DecoderAttnMask),
}

#[derive(Debug, Clone)]
pub struct Fixture {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub params: AttentionParams,
    pub seq_len: usize,
    pub total_kv: usize,
    pub mask: MaskMode,
}

impl Fixture {
    pub fn out_len(&self) -> usize {
        self.seq_len * self.params.n_heads * self.params.head_dim
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.out_len()
}

fn fill(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32) * seed).sin() * 0.29 + ((i as f32) * seed * 0.61).cos() * 0.17)
        .collect()
}

fn qkv(
    seq_len: usize,
    total_kv: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_seed: f32,
    kv_seed: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q_dim = seq_len * n_heads * head_dim;
    let kv_dim = total_kv * n_kv_heads * head_dim;
    (
        fill(q_dim, q_seed),
        fill(kv_dim, kv_seed),
        fill(kv_dim, kv_seed * 0.83),
    )
}

pub fn tiny_fixture(_: ElemFormat) -> Fixture {
    let seq_len = 4usize;
    let n_heads = 2usize;
    let n_kv_heads = 1usize;
    let head_dim = 8usize;
    let (q, k, v) = qkv(seq_len, seq_len, n_heads, n_kv_heads, head_dim, 0.03, 0.02);
    Fixture {
        q,
        k,
        v,
        params: AttentionParams {
            n_heads,
            n_kv_heads,
            head_dim,
            rotary_dim: head_dim,
            n_groups: n_heads / n_kv_heads,
            sliding_window: Some(2),
        },
        seq_len,
        total_kv: seq_len,
        mask: MaskMode::CausalSliding,
    }
}

/// Sliding-layer causal prefill: 128 tokens, 16×8 GQA, hd 256.
pub fn sliding_prefill_fixture(_: ElemFormat) -> Fixture {
    let seq_len = 128usize;
    let n_heads = 16usize;
    let n_kv_heads = 8usize;
    let head_dim = 256usize;
    let (q, k, v) = qkv(seq_len, seq_len, n_heads, n_kv_heads, head_dim, 0.011, 0.009);
    Fixture {
        q,
        k,
        v,
        params: AttentionParams {
            n_heads,
            n_kv_heads,
            head_dim,
            rotary_dim: head_dim,
            n_groups: n_heads / n_kv_heads,
            sliding_window: Some(1024),
        },
        seq_len,
        total_kv: seq_len,
        mask: MaskMode::CausalSliding,
    }
}

/// Encoder extend: 96 cached tokens + 32 new (total_kv=128).
pub fn encoder_extend_fixture(_: ElemFormat) -> Fixture {
    let seq_len = 32usize;
    let kv_cache_len = 96usize;
    let total_kv = kv_cache_len + seq_len;
    let n_heads = 16usize;
    let n_kv_heads = 8usize;
    let head_dim = 256usize;
    let (q, k, v) = qkv(seq_len, total_kv, n_heads, n_kv_heads, head_dim, 0.014, 0.012);
    let positions: Vec<i64> = (kv_cache_len as i64..kv_cache_len as i64 + seq_len as i64).collect();
    Fixture {
        q,
        k,
        v,
        params: AttentionParams {
            n_heads,
            n_kv_heads,
            head_dim,
            rotary_dim: head_dim,
            n_groups: n_heads / n_kv_heads,
            sliding_window: Some(1024),
        },
        seq_len,
        total_kv,
        mask: MaskMode::EncoderExtend {
            kv_cache_len,
            positions,
        },
    }
}

/// Full-attention layer: 64 tokens, 16×2 GQA, hd 512, no sliding window.
pub fn full_layer_fixture(_: ElemFormat) -> Fixture {
    let seq_len = 64usize;
    let n_heads = 16usize;
    let n_kv_heads = 2usize;
    let head_dim = 512usize;
    let (q, k, v) = qkv(seq_len, seq_len, n_heads, n_kv_heads, head_dim, 0.008, 0.007);
    Fixture {
        q,
        k,
        v,
        params: AttentionParams {
            n_heads,
            n_kv_heads,
            head_dim,
            rotary_dim: 128,
            n_groups: n_heads / n_kv_heads,
            sliding_window: None,
        },
        seq_len,
        total_kv: seq_len,
        mask: MaskMode::CausalSliding,
    }
}

/// Decoder canvas attention: 32 canvas queries × (14 prompt + 32 canvas) KV.
pub fn decoder_bitmap_fixture(_: ElemFormat) -> Fixture {
    let canvas_len = 32usize;
    let kv_cache_len = 14usize;
    let total_kv = kv_cache_len + canvas_len;
    let n_heads = 16usize;
    let n_kv_heads = 8usize;
    let head_dim = 256usize;
    let (q, k, v) = qkv(canvas_len, total_kv, n_heads, n_kv_heads, head_dim, 0.016, 0.013);
    Fixture {
        q,
        k,
        v,
        params: AttentionParams {
            n_heads,
            n_kv_heads,
            head_dim,
            rotary_dim: head_dim,
            n_groups: n_heads / n_kv_heads,
            sliding_window: None,
        },
        seq_len: canvas_len,
        total_kv,
        mask: MaskMode::DecoderBitmap(DecoderAttnMask::all_valid(canvas_len, kv_cache_len)),
    }
}

fn mask_kind_and_side_data(f: &Fixture) -> (u32, usize, Option<&[i64]>, Option<&DecoderAttnMask>) {
    match &f.mask {
        MaskMode::CausalSliding => (MASK_CAUSAL_SLIDING, 0, None, None),
        MaskMode::EncoderExtend {
            kv_cache_len,
            positions,
        } => (MASK_ENCODER_EXTEND, *kv_cache_len, Some(positions.as_slice()), None),
        MaskMode::DecoderBitmap(m) => (MASK_DECODER_BITMAP, m.kv_cache_len, None, Some(m)),
    }
}

fn gqa_mask<'a>(f: &'a Fixture) -> GqaMask<'a> {
    match &f.mask {
        MaskMode::CausalSliding => GqaMask::CausalSliding,
        MaskMode::EncoderExtend {
            kv_cache_len,
            positions,
        } => GqaMask::EncoderExtend {
            kv_cache_len: *kv_cache_len,
            positions,
        },
        MaskMode::DecoderBitmap(m) => GqaMask::DecoderBitmap(m),
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    let mut out = vec![0.0f32; f.out_len()];
    let mut scores = vec![0.0f32; f.seq_len * f.params.n_heads * f.total_kv];
    attention::gqa_attention(
        &mut out,
        &mut scores,
        &f.q,
        &f.k,
        &f.v,
        f.seq_len,
        f.total_kv,
        &f.params,
        gqa_mask(f),
    );
    out
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    let mut pipelines = ctx.compile_kernels(SHADER, &[ENTRY])?;
    pipelines
        .pop()
        .ok_or(Error::Format("Metal pipeline missing"))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use objc2_metal::{
        MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    };

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx)?;
    let mut pool = BufferPool::new();

    let q_bytes = f.q.len() * 4;
    let k_bytes = f.k.len() * 4;
    let v_bytes = f.v.len() * 4;
    let o_bytes = f.out_len() * 4;

    let buf_q = pool.allocate(&ctx.device, q_bytes).ok_or(Error::Format("alloc"))?;
    let buf_k = pool.allocate(&ctx.device, k_bytes).ok_or(Error::Format("alloc"))?;
    let buf_v = pool.allocate(&ctx.device, v_bytes).ok_or(Error::Format("alloc"))?;
    let buf_o = pool.allocate(&ctx.device, o_bytes).ok_or(Error::Format("alloc"))?;

    BufferPool::write_f32(&buf_q, &f.q);
    BufferPool::write_f32(&buf_k, &f.k);
    BufferPool::write_f32(&buf_v, &f.v);
    BufferPool::write_f32(&buf_o, &vec![0.0f32; f.out_len()]);

    let (mask_kind, kv_cache_len, positions, decoder_mask) = mask_kind_and_side_data(f);

    let mut buf_mask = None;
    if let Some(m) = decoder_mask {
        let packed: Vec<u8> = m.attend.iter().map(|&b| u8::from(b)).collect();
        let mask_bytes = packed.len();
        let b = pool.allocate(&ctx.device, mask_bytes).ok_or(Error::Format("alloc"))?;
        BufferPool::write_bytes(&b, &packed);
        buf_mask = Some((mask_bytes, b));
    }

    let mut buf_pos = None;
    if let Some(pos) = positions {
        let pos_bytes = pos.len() * 8;
        let b = pool.allocate(&ctx.device, pos_bytes).ok_or(Error::Format("alloc"))?;
        BufferPool::write_i64(&b, pos);
        buf_pos = Some((pos_bytes, b));
    }

    let params = GqaParams::for_attention(f.seq_len, f.total_kv, &f.params, mask_kind, kv_cache_len);

    let cmd = ctx.queue.commandBuffer().ok_or(Error::Format("cmd"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Format("enc"))?;
    enc.setComputePipelineState(&pipeline.pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&buf_q), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&buf_k), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&buf_v), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&buf_o), 0, 3);
    }
    if let Some((_, ref mask_buf)) = buf_mask {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(mask_buf), 0, 4);
        }
    }
    if let Some((_, ref pos_buf)) = buf_pos {
        unsafe {
            enc.setBuffer_offset_atIndex(Some(pos_buf), 0, 5);
        }
    }
    engine_gqa_common::set_params(&enc, &params, 6);
    engine_gqa_common::dispatch_head_query_2d(&enc, f.params.n_heads, f.seq_len);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut out = vec![0.0f32; f.out_len()];
    BufferPool::read_f32(&buf_o, &mut out);

    pool.release(q_bytes, buf_q);
    pool.release(k_bytes, buf_k);
    pool.release(v_bytes, buf_v);
    pool.release(o_bytes, buf_o);
    if let Some((bytes, b)) = buf_mask {
        pool.release(bytes, b);
    }
    if let Some((bytes, b)) = buf_pos {
        pool.release(bytes, b);
    }
    Ok(out)
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn gpu(_: &Fixture) -> Result<Vec<f32>, Error> {
    Err(Error::Format("Metal unavailable"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::sub::test_util::{assert_oracle, ElemFormat};

    fn run_matrix(fixture_fn: fn(ElemFormat) -> Fixture, max_tol: f32) {
        let fix = fixture_fn(ElemFormat::F32);
        let cpu = cpu(&fix);
        assert!(cpu.iter().all(|v| v.is_finite()));
        assert_eq!(cpu.len(), fixture_len(&fix));

        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            let gpu = gpu(&fix).expect("gpu");
            assert_oracle(&gpu, &cpu, max_tol, 0.9999);
        }
    }

    #[test]
    fn cpu_tiny() {
        run_matrix(tiny_fixture, 1e-4);
    }

    #[test]
    fn cpu_sliding_prefill() {
        run_matrix(sliding_prefill_fixture, 1e-4);
    }

    #[test]
    fn cpu_encoder_extend() {
        run_matrix(encoder_extend_fixture, 1e-4);
    }

    #[test]
    fn cpu_full_layer() {
        run_matrix(full_layer_fixture, 1e-4);
    }

    #[test]
    fn cpu_decoder_bitmap() {
        run_matrix(decoder_bitmap_fixture, 1e-4);
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn gpu_tiny() {
        run_matrix(tiny_fixture, 1e-4);
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn gpu_sliding_prefill() {
        run_matrix(sliding_prefill_fixture, 1e-4);
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn gpu_encoder_extend() {
        run_matrix(encoder_extend_fixture, 1e-4);
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn gpu_full_layer() {
        run_matrix(full_layer_fixture, 1e-4);
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn gpu_decoder_bitmap() {
        run_matrix(decoder_bitmap_fixture, 1e-4);
    }
}
