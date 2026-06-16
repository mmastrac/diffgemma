//! Attention kernels batched through `GpuBatch` (shared engine pool/queue).

use crate::kernels::sub::engine_gqa_common::{self, GqaParams};
use crate::metal::batch::{set_bytes, GpuBatch};
use crate::metal::device::ComputePipeline;
use crate::model::attention::{AttentionParams, GqaMask, MASK_NEG};
use crate::safetensors::Error;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState};

use super::attention::GpuAttentionKernels;

pub fn rope_qk_batched(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuAttentionKernels,
    q: &mut [f32],
    k: &mut [f32],
    freqs: &[f32],
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<(), Error> {
    if q.len() != seq_len * n_heads * head_dim
        || k.len() != seq_len * n_kv_heads * head_dim
        || freqs.len() != seq_len * rotary_dim
    {
        return Err(Error::Format("rope qk shape mismatch"));
    }

    let buf_q = batch.alloc_f32(q)?;
    let buf_k = batch.alloc_f32(k)?;
    let buf_f = batch.alloc_f32(freqs)?;

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
        elem_offset: 0,
    };
    let rope_k = GqaParams {
        num_heads_rope: n_kv_heads as u32,
        ..rope_q
    };

    encode_rope(batch, &kernels.rope_pipeline.pipeline, &buf_q, &buf_f, &rope_q, n_heads, seq_len);
    encode_rope(batch, &kernels.rope_pipeline.pipeline, &buf_k, &buf_f, &rope_k, n_kv_heads, seq_len);

    batch.register_read(buf_q, q);
    batch.register_read(buf_k, k);
    Ok(())
}

/// RoPE canvas K in-place at `k_canvas_elem_offset` within `k_buf`, RoPE Q in pool buffer, then GQA.
/// `k_buf`/`v_buf` must already hold encoder prefix; canvas V written at the same suffix offset.
pub fn decoder_gqa_gpu_kv_batched(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuAttentionKernels,
    attn_out: Option<&mut [f32]>,
    q_pre_rope: &[f32],
    k_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    v_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_canvas_elem_offset: usize,
    freqs: &[f32],
    seq_len: usize,
    total_kv: usize,
    params: &AttentionParams,
    mask: GqaMask<'_>,
) -> Result<Option<Retained<ProtocolObject<dyn MTLBuffer>>>, Error> {
    let q_dim = seq_len * params.n_heads * params.head_dim;
    let out_dim = seq_len * params.n_heads * params.head_dim;
    if q_pre_rope.len() != q_dim {
        return Err(Error::Format("decoder gpu kv gqa shape mismatch"));
    }
    if let Some(out) = &attn_out {
        if out.len() != out_dim {
            return Err(Error::Format("decoder gpu kv gqa shape mismatch"));
        }
    }

    let buf_q = batch.alloc_f32(q_pre_rope)?;
    let buf_f = batch.alloc_f32(freqs)?;

    let rope_base = GqaParams {
        seq_len: seq_len as u32,
        total_kv: 0,
        n_heads: 0,
        n_kv_heads: 0,
        head_dim: params.head_dim as u32,
        n_groups: 0,
        mask_kind: 0,
        sliding_window: 0,
        kv_cache_len: 0,
        mask_neg: MASK_NEG,
        rotary_dim: params.rotary_dim as u32,
        num_heads_rope: params.n_heads as u32,
        elem_offset: 0,
    };
    let rope_k = GqaParams {
        num_heads_rope: params.n_kv_heads as u32,
        elem_offset: k_canvas_elem_offset as u32,
        ..rope_base
    };

    encode_rope(
        batch,
        &kernels.rope_pipeline.pipeline,
        &buf_q,
        &buf_f,
        &rope_base,
        params.n_heads,
        seq_len,
    );
    encode_rope(
        batch,
        &kernels.rope_pipeline.pipeline,
        &k_buf,
        &buf_f,
        &rope_k,
        params.n_kv_heads,
        seq_len,
    );

    gqa_batched_inner(
        batch,
        kernels,
        attn_out,
        q_pre_rope,
        None,
        Some((k_buf, v_buf)),
        seq_len,
        total_kv,
        params,
        mask,
        Some(buf_q),
    )
}

/// RoPE + GQA over GPU KV; returns attention output buffer when `attn_out` is `None`.
pub fn decoder_gqa_gpu_kv_batched_chained(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuAttentionKernels,
    q_pre_rope: &[f32],
    k_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    v_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_canvas_elem_offset: usize,
    freqs: &[f32],
    seq_len: usize,
    total_kv: usize,
    params: &AttentionParams,
    mask: GqaMask<'_>,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    decoder_gqa_gpu_kv_batched(
        batch,
        kernels,
        None,
        q_pre_rope,
        k_buf,
        v_buf,
        k_canvas_elem_offset,
        freqs,
        seq_len,
        total_kv,
        params,
        mask,
    )?
    .ok_or(Error::Format("gqa gpu kv missing output buffer"))
}

/// Device-to-device f32 copy: `dst[dst_byte_off/4 .. +len] = src[0..len]`.
pub fn dispatch_copy_f32_to_buf(
    batch: &mut GpuBatch<'_>,
    copy_pipeline: &ComputePipeline,
    src: &ProtocolObject<dyn MTLBuffer>,
    dst: &ProtocolObject<dyn MTLBuffer>,
    dst_byte_off: usize,
    len: usize,
) {
    batch.dispatch_1d_ranged(&copy_pipeline.pipeline, len, |enc, base, chunk| {
        let range = [base, chunk];
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src), 0, 0);
            enc.setBuffer_offset_atIndex(Some(dst), dst_byte_off, 1);
        }
        set_bytes(enc, &range, 2);
    });
}

/// RoPE + GQA over GPU KV when Q is already in a pool buffer (pre-RoPE).
pub fn decoder_gqa_gpu_kv_batched_chained_qbuf(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuAttentionKernels,
    buf_q: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    v_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    k_canvas_elem_offset: usize,
    freqs: &[f32],
    seq_len: usize,
    total_kv: usize,
    params: &AttentionParams,
    mask: GqaMask<'_>,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    let buf_f = batch.alloc_f32(freqs)?;

    let rope_base = GqaParams {
        seq_len: seq_len as u32,
        total_kv: 0,
        n_heads: 0,
        n_kv_heads: 0,
        head_dim: params.head_dim as u32,
        n_groups: 0,
        mask_kind: 0,
        sliding_window: 0,
        kv_cache_len: 0,
        mask_neg: MASK_NEG,
        rotary_dim: params.rotary_dim as u32,
        num_heads_rope: params.n_heads as u32,
        elem_offset: 0,
    };
    let rope_k = GqaParams {
        num_heads_rope: params.n_kv_heads as u32,
        elem_offset: k_canvas_elem_offset as u32,
        ..rope_base
    };

    encode_rope(
        batch,
        &kernels.rope_pipeline.pipeline,
        &buf_q,
        &buf_f,
        &rope_base,
        params.n_heads,
        seq_len,
    );
    encode_rope(
        batch,
        &kernels.rope_pipeline.pipeline,
        &k_buf,
        &buf_f,
        &rope_k,
        params.n_kv_heads,
        seq_len,
    );

    gqa_batched_inner(
        batch,
        kernels,
        None,
        &[],
        None,
        Some((k_buf, v_buf)),
        seq_len,
        total_kv,
        params,
        mask,
        Some(buf_q),
    )?
    .ok_or(Error::Format("gqa gpu kv missing output buffer"))
}

pub fn gqa_batched_chained(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuAttentionKernels,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    total_kv: usize,
    params: &AttentionParams,
    mask: GqaMask<'_>,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, Error> {
    gqa_batched_inner(
        batch,
        kernels,
        None,
        q,
        Some((k, v)),
        None,
        seq_len,
        total_kv,
        params,
        mask,
        None,
    )?
    .ok_or(Error::Format("gqa missing output buffer"))
}

fn gqa_batched_inner(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuAttentionKernels,
    attn_out: Option<&mut [f32]>,
    q: &[f32],
    kv_cpu: Option<(&[f32], &[f32])>,
    kv_gpu: Option<(Retained<ProtocolObject<dyn MTLBuffer>>, Retained<ProtocolObject<dyn MTLBuffer>>)>,
    seq_len: usize,
    total_kv: usize,
    params: &AttentionParams,
    mask: GqaMask<'_>,
    q_gpu: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
) -> Result<Option<Retained<ProtocolObject<dyn MTLBuffer>>>, Error> {
    let q_dim = seq_len * params.n_heads * params.head_dim;
    let kv_dim_elems = total_kv * params.n_kv_heads * params.head_dim;
    let out_dim = seq_len * params.n_heads * params.head_dim;

    if q_gpu.is_none() && q.len() != q_dim {
        return Err(Error::Format("gqa attention shape mismatch"));
    }
    if let Some(out) = &attn_out {
        if out.len() != out_dim {
            return Err(Error::Format("gqa attention shape mismatch"));
        }
    }
    if let Some((k, v)) = kv_cpu {
        if k.len() != kv_dim_elems || v.len() != kv_dim_elems {
            return Err(Error::Format("gqa attention shape mismatch"));
        }
    } else if kv_gpu.is_none() {
        return Err(Error::Format("gqa gpu kv buffers missing"));
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

    let buf_q = if let Some(buf) = q_gpu {
        buf
    } else {
        batch.alloc_f32(q)?
    };
    let (buf_k, buf_v) = match kv_cpu {
        Some((k, v)) => (batch.alloc_f32(k)?, batch.alloc_f32(v)?),
        None => kv_gpu.expect("gpu kv"),
    };
    let buf_o = batch.alloc_f32_out(out_dim)?;

    let mut mask_buf = None;
    let mut pos_buf = None;

    if let Some(m) = decoder_mask {
        let packed: Vec<u8> = m.attend.iter().map(|&b| u8::from(b)).collect();
        mask_buf = Some(batch.alloc_bytes(&packed)?);
    }

    if let Some(pos) = positions {
        pos_buf = Some(batch.alloc_i64(pos)?);
    }

    let gpu_params = GqaParams::for_attention(seq_len, total_kv, params, mask_kind, kv_cache_len);

    encode_gqa(
        batch,
        &kernels.attn_pipeline.pipeline,
        &buf_q,
        &buf_k,
        &buf_v,
        &buf_o,
        mask_buf.as_deref(),
        pos_buf.as_deref(),
        &gpu_params,
        params.n_heads,
        seq_len,
    );

    if let Some(out) = attn_out {
        batch.register_read(buf_o, out);
        Ok(None)
    } else {
        Ok(Some(buf_o))
    }
}

fn encode_rope(
    batch: &GpuBatch<'_>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buf_x: &ProtocolObject<dyn MTLBuffer>,
    buf_f: &ProtocolObject<dyn MTLBuffer>,
    params: &GqaParams,
    num_heads: usize,
    seq_len: usize,
) {
    batch.encoder().setComputePipelineState(pipeline);
    unsafe {
        batch.encoder().setBuffer_offset_atIndex(Some(buf_x), 0, 0);
        batch.encoder().setBuffer_offset_atIndex(Some(buf_f), 0, 1);
    }
    engine_gqa_common::set_params(batch.encoder(), params, 2);
    engine_gqa_common::dispatch_head_query_2d(batch.encoder(), num_heads, seq_len);
}

fn encode_gqa(
    batch: &GpuBatch<'_>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buf_q: &ProtocolObject<dyn MTLBuffer>,
    buf_k: &ProtocolObject<dyn MTLBuffer>,
    buf_v: &ProtocolObject<dyn MTLBuffer>,
    buf_o: &ProtocolObject<dyn MTLBuffer>,
    mask_buf: Option<&ProtocolObject<dyn MTLBuffer>>,
    pos_buf: Option<&ProtocolObject<dyn MTLBuffer>>,
    params: &GqaParams,
    n_heads: usize,
    seq_len: usize,
) {
    batch.encoder().setComputePipelineState(pipeline);
    unsafe {
        batch.encoder().setBuffer_offset_atIndex(Some(buf_q), 0, 0);
        batch.encoder().setBuffer_offset_atIndex(Some(buf_k), 0, 1);
        batch.encoder().setBuffer_offset_atIndex(Some(buf_v), 0, 2);
        batch.encoder().setBuffer_offset_atIndex(Some(buf_o), 0, 3);
    }
    if let Some(mask) = mask_buf {
        unsafe {
            batch.encoder().setBuffer_offset_atIndex(Some(mask), 0, 4);
        }
    }
    if let Some(pos) = pos_buf {
        unsafe {
            batch.encoder().setBuffer_offset_atIndex(Some(pos), 0, 5);
        }
    }
    engine_gqa_common::set_params(batch.encoder(), params, 6);
    engine_gqa_common::dispatch_head_query_2d(batch.encoder(), n_heads, seq_len);
}

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod prefill_attn_tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::kernels::cpu::{apply_rope_tensor, compute_rope_freqs, rope_kind_for_layer};
    use crate::metal::attention::GpuAttentionKernels;
    use crate::metal::batch::GpuBatch;
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use crate::metal::kv_cache::GpuKvCache;
    use crate::model::attention::gqa_attention;

    fn dgq_fixture_dir() -> Option<std::path::PathBuf> {
        for dir in ["/tmp/nvfp4-weights", "/tmp/quantized-weights"] {
            let p = std::path::Path::new(dir);
            if p.join("model.dgq.json").exists() {
                return Some(p.to_path_buf());
            }
        }
        None
    }

    fn fill_synthetic(src: &mut [f32], seed: f32) {
        for (i, x) in src.iter_mut().enumerate() {
            let t = i as f32 * 0.0011 + seed;
            *x = t.sin() * 0.04 + (t * 0.7).cos() * 0.02;
        }
    }

    /// Encoder prefill attention: `decoder_gqa_gpu_kv_batched` at model L0 dims, `seq_len=25`.
    #[test]
    fn decoder_gqa_gpu_kv_prefill_seq_len_matches_cpu() {
        let Some(model_dir) = dgq_fixture_dir() else {
            eprintln!("skip: no .dgq weights under /tmp");
            return;
        };
        let cfg = ModelConfig::load(&model_dir).expect("cfg");
        let text = &cfg.text_config;
        let layer = 0usize;
        let params = AttentionParams::for_layer(text, layer).expect("L0 params");
        let seq_len = 25usize;
        let total_kv = seq_len;
        let positions: Vec<i64> = (0..seq_len as i64).collect();

        let q_dim = seq_len * params.n_heads * params.head_dim;
        let kv_dim = seq_len * params.n_kv_heads * params.head_dim;
        let mut q_pre = vec![0.0f32; q_dim];
        let mut k_pre = vec![0.0f32; kv_dim];
        let mut v_pre = vec![0.0f32; kv_dim];
        fill_synthetic(&mut q_pre, 0.1);
        fill_synthetic(&mut k_pre, 0.2);
        fill_synthetic(&mut v_pre, 0.3);

        let mut freqs = vec![0.0f32; seq_len * params.rotary_dim];
        let rope_kind = rope_kind_for_layer(text, layer).expect("rope kind");
        compute_rope_freqs(&mut freqs, &positions, rope_kind);

        let mut q_cpu = q_pre.clone();
        let mut k_cpu = k_pre.clone();
        apply_rope_tensor(
            &mut q_cpu,
            seq_len,
            params.n_heads,
            params.head_dim,
            &freqs,
            params.rotary_dim,
        );
        apply_rope_tensor(
            &mut k_cpu,
            seq_len,
            params.n_kv_heads,
            params.head_dim,
            &freqs,
            params.rotary_dim,
        );

        let mut cpu_out = vec![0.0f32; q_dim];
        let mut scores = vec![0.0f32; seq_len * params.n_heads * total_kv];
        gqa_attention(
            &mut cpu_out,
            &mut scores,
            &q_cpu,
            &k_cpu,
            &v_pre,
            seq_len,
            total_kv,
            &params,
            GqaMask::CausalSliding,
        );

        let ctx = MetalContext::new().expect("metal");
        let gpu_kv = GpuKvCache::new(&ctx.device, text, seq_len, 0).expect("gpu kv");
        gpu_kv
            .write_canvas_kv_pre_rope(layer, seq_len, &k_pre, &v_pre)
            .expect("kv write");
        let k_off = gpu_kv.canvas_k_elem_offset(layer).expect("k off");
        assert_eq!(k_off, 0);
        let (k_buf, v_buf) = gpu_kv.layer_buffers(layer).expect("layer bufs");

        let kernels = GpuAttentionKernels::new(&ctx).expect("attn kernels");
        let mut pool = BufferPool::new();
        let mut batch = GpuBatch::begin_with_telemetry(&ctx.queue, &mut pool, &ctx.device, None).expect("batch");
        let mut gpu_out = vec![0.0f32; q_dim];
        decoder_gqa_gpu_kv_batched(
            &mut batch,
            &kernels,
            Some(&mut gpu_out),
            &q_pre,
            k_buf,
            v_buf,
            k_off,
            &freqs,
            seq_len,
            total_kv,
            &params,
            GqaMask::CausalSliding,
        )
        .expect("decoder gqa");
        batch.end().expect("batch end");

        let mut max_err = 0.0f32;
        let mut nan = 0usize;
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (a, b) in cpu_out.iter().zip(gpu_out.iter()) {
            if !b.is_finite() {
                nan += 1;
            }
            max_err = max_err.max((a - b).abs());
            dot += *a as f64 * *b as f64;
            na += *a as f64 * *a as f64;
            nb += *b as f64 * *b as f64;
        }
        let cos = if na > 0.0 && nb > 0.0 {
            (dot / (na.sqrt() * nb.sqrt())) as f32
        } else {
            0.0
        };
        eprintln!(
            "decoder_gqa prefill L{layer} seq_len={seq_len} heads={} kv_heads={} hd={} max_err={max_err:.6} cos={cos:.6} nan={nan}",
            params.n_heads, params.n_kv_heads, params.head_dim
        );
        assert_eq!(nan, 0, "gpu attention produced {nan} non-finite values");
        assert!(cos > 0.999, "cos={cos}");
        assert!(max_err < 0.05, "max_err={max_err}");
    }
}
