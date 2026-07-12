use crate::config::TextConfig;
use crate::metal::attention_batch::{
    decoder_gqa_gpu_kv_batched_chained_qbuf, dispatch_copy_f32_to_buf, gqa_batched_chained,
    rope_qk_batched,
};
use crate::metal::batch::begin_engine_batch;
use crate::metal::batched_kernels::{self as bk};
use crate::metal::engine::GpuDecoderEngine;
use crate::metal::kv_cache::GpuKvCache;
use crate::metal::linear::{linear_cached_batched_in_buf, linear_cached_batched_in_cpu_out};
use crate::metal::weights::GpuLayerWeightCache;
use crate::model::attention::{AttentionParams, AttentionScratch, GqaMask, concat_kv_for_decoder};
use crate::model::kv_cache::LayerKvView;
use crate::model::mask::DecoderAttnMask;
use crate::safetensors::Error;
use crate::shaders::cpu::{compute_rope_freqs, rope_kind_for_layer};

fn fused_input_qkv_heads(
    engine: &mut GpuDecoderEngine,
    hidden: &[f32],
    cached: &GpuLayerWeightCache,
    seq_len: usize,
    hidden_size: usize,
    eps: f32,
    params: &AttentionParams,
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
) -> Result<(), Error> {
    let telemetry = engine.batch_telemetry();
    let mut batch = begin_engine_batch(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        telemetry,
    )?;
    let buf_normed = bk::rms_norm_rows_gpu(
        &mut batch,
        &engine.kernels,
        hidden,
        cached.input_layernorm.as_slice(),
        seq_len,
        hidden_size,
        eps,
    )?;
    let buf_q = linear_cached_batched_in_buf(
        &mut batch,
        &engine.f32_bf16_linear_pipeline,
        &engine.f32_q4_linear_pipeline,
        &engine.f32_nvfp4_linear_pipeline,
        &engine.f32_q8_linear_pipeline,
        &buf_normed,
        &cached.q_proj,
        seq_len,
    )?;
    let buf_k = linear_cached_batched_in_buf(
        &mut batch,
        &engine.f32_bf16_linear_pipeline,
        &engine.f32_q4_linear_pipeline,
        &engine.f32_nvfp4_linear_pipeline,
        &engine.f32_q8_linear_pipeline,
        &buf_normed,
        &cached.k_proj,
        seq_len,
    )?;
    let buf_v = if let Some(v_proj) = &cached.v_proj {
        linear_cached_batched_in_buf(
            &mut batch,
            &engine.f32_bf16_linear_pipeline,
            &engine.f32_q4_linear_pipeline,
            &engine.f32_nvfp4_linear_pipeline,
            &engine.f32_q8_linear_pipeline,
            &buf_normed,
            v_proj,
            seq_len,
        )?
    } else {
        buf_k.clone()
    };
    let q_rows = seq_len * params.n_heads;
    let kv_rows = seq_len * params.n_kv_heads;
    let head_dim = params.head_dim;
    let buf_q = bk::rms_norm_rows_gpu_buf(
        &mut batch,
        &engine.kernels,
        &buf_q,
        cached.q_norm.as_slice(),
        q_rows,
        head_dim,
        eps,
    )?;
    let buf_k = bk::rms_norm_rows_gpu_buf(
        &mut batch,
        &engine.kernels,
        &buf_k,
        cached.k_norm.as_slice(),
        kv_rows,
        head_dim,
        eps,
    )?;
    let buf_v = bk::rms_norm_rows_no_scale_gpu_buf(
        &mut batch,
        &engine.kernels,
        &buf_v,
        kv_rows,
        head_dim,
        eps,
    )?;
    batch.register_read(buf_q, q);
    batch.register_read(buf_k, k);
    batch.register_read(buf_v, v);
    batch.end()
}

/// Input layernorm + Q/K/V projections + per-head norms; input and Q/K/V stay
/// on GPU (no readback).
fn encode_input_qkv_gpu_bufs(
    batch: &mut crate::metal::batch::GpuBatch<'_>,
    kernels: &crate::metal::kernels::GpuKernels,
    f32_bf16: &crate::metal::device::ComputePipeline,
    f32_q4: &crate::metal::device::ComputePipeline,
    f32_nvfp4: &crate::metal::device::ComputePipeline,
    f32_q8: &crate::metal::device::ComputePipeline,
    hidden_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    cached: &GpuLayerWeightCache,
    seq_len: usize,
    hidden_size: usize,
    eps: f32,
    params: &AttentionParams,
) -> Result<
    (
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    ),
    Error,
> {
    let buf_normed = bk::rms_norm_rows_gpu_buf(
        batch,
        kernels,
        hidden_buf,
        cached.input_layernorm.as_slice(),
        seq_len,
        hidden_size,
        eps,
    )?;
    let buf_q = linear_cached_batched_in_buf(
        batch,
        f32_bf16,
        f32_q4,
        f32_nvfp4,
        f32_q8,
        &buf_normed,
        &cached.q_proj,
        seq_len,
    )?;
    let buf_k = linear_cached_batched_in_buf(
        batch,
        f32_bf16,
        f32_q4,
        f32_nvfp4,
        f32_q8,
        &buf_normed,
        &cached.k_proj,
        seq_len,
    )?;
    let buf_v = if let Some(v_proj) = &cached.v_proj {
        linear_cached_batched_in_buf(
            batch,
            f32_bf16,
            f32_q4,
            f32_nvfp4,
            f32_q8,
            &buf_normed,
            v_proj,
            seq_len,
        )?
    } else {
        buf_k.clone()
    };
    let q_rows = seq_len * params.n_heads;
    let kv_rows = seq_len * params.n_kv_heads;
    let head_dim = params.head_dim;
    let buf_q = bk::rms_norm_rows_gpu_buf(
        batch,
        kernels,
        &buf_q,
        cached.q_norm.as_slice(),
        q_rows,
        head_dim,
        eps,
    )?;
    let buf_k = bk::rms_norm_rows_gpu_buf(
        batch,
        kernels,
        &buf_k,
        cached.k_norm.as_slice(),
        kv_rows,
        head_dim,
        eps,
    )?;
    let buf_v = bk::rms_norm_rows_no_scale_gpu_buf(batch, kernels, &buf_v, kv_rows, head_dim, eps)?;
    Ok((buf_q, buf_k, buf_v))
}

/// Encode QKV → KV suffix write → RoPE/GQA → o_proj into an open batch; the
/// hidden input and the o_proj output stay on GPU. Core of both the classic
/// (CPU-out) attention and the GPU-resident prefill path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_fused_gpu_kv_attention_buf(
    batch: &mut crate::metal::batch::GpuBatch<'_>,
    kernels: &crate::metal::kernels::GpuKernels,
    attention: &crate::metal::attention::GpuAttentionKernels,
    copy_pipeline: &crate::metal::device::ComputePipeline,
    f32_bf16: &crate::metal::device::ComputePipeline,
    f32_q4: &crate::metal::device::ComputePipeline,
    f32_nvfp4: &crate::metal::device::ComputePipeline,
    f32_q8: &crate::metal::device::ComputePipeline,
    hidden_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    cached: &GpuLayerWeightCache,
    seq_len: usize,
    hidden_size: usize,
    eps: f32,
    params: &AttentionParams,
    freqs: &[f32],
    k_buf: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    v_buf: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    k_canvas_elem_offset: usize,
    kv_suffix_byte_off: usize,
    kv_suffix_elems: usize,
    total_kv: usize,
    mask: GqaMask<'_>,
    o_proj: &crate::metal::linear::GpuLinearWeight,
) -> Result<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>, Error>
{
    let (buf_q, buf_k, buf_v) = encode_input_qkv_gpu_bufs(
        batch,
        kernels,
        f32_bf16,
        f32_q4,
        f32_nvfp4,
        f32_q8,
        hidden_buf,
        cached,
        seq_len,
        hidden_size,
        eps,
        params,
    )?;
    dispatch_copy_f32_to_buf(
        batch,
        copy_pipeline,
        &buf_k,
        &k_buf,
        kv_suffix_byte_off,
        kv_suffix_elems,
    );
    dispatch_copy_f32_to_buf(
        batch,
        copy_pipeline,
        &buf_v,
        &v_buf,
        kv_suffix_byte_off,
        kv_suffix_elems,
    );
    let buf_attn = decoder_gqa_gpu_kv_batched_chained_qbuf(
        batch,
        attention,
        buf_q,
        k_buf,
        v_buf,
        k_canvas_elem_offset,
        freqs,
        seq_len,
        total_kv,
        params,
        mask,
    )?;
    linear_cached_batched_in_buf(
        batch, f32_bf16, f32_q4, f32_nvfp4, f32_q8, &buf_attn, o_proj, seq_len,
    )
}

/// One batch: QKV on GPU → KV suffix write → RoPE/GQA/o_proj; single sync, output readback only.
fn fused_gpu_kv_attention(
    engine: &mut GpuDecoderEngine,
    out: &mut [f32],
    hidden: &[f32],
    cached: &GpuLayerWeightCache,
    seq_len: usize,
    hidden_size: usize,
    eps: f32,
    params: &AttentionParams,
    freqs: &[f32],
    k_buf: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    v_buf: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    k_canvas_elem_offset: usize,
    kv_suffix_byte_off: usize,
    kv_suffix_elems: usize,
    total_kv: usize,
    mask: GqaMask<'_>,
    o_proj: &crate::metal::linear::GpuLinearWeight,
) -> Result<(), Error> {
    let telemetry = engine.batch_telemetry();
    let mut batch = begin_engine_batch(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        telemetry,
    )?;
    let buf_hidden = batch.alloc_f32(hidden)?;
    let buf_o = encode_fused_gpu_kv_attention_buf(
        &mut batch,
        &engine.kernels,
        &engine.attention,
        &engine.sampler_kernels.copy_f32,
        &engine.f32_bf16_linear_pipeline,
        &engine.f32_q4_linear_pipeline,
        &engine.f32_nvfp4_linear_pipeline,
        &engine.f32_q8_linear_pipeline,
        &buf_hidden,
        cached,
        seq_len,
        hidden_size,
        eps,
        params,
        freqs,
        k_buf,
        v_buf,
        k_canvas_elem_offset,
        kv_suffix_byte_off,
        kv_suffix_elems,
        total_kv,
        mask,
        o_proj,
    )?;
    batch.register_read(buf_o, out);
    batch.end()
}

/// Decoder self-attention: fused GPU norm/QKV/head-norms + fused RoPE/GQA/o_proj.
pub fn forward_decoder_attention(
    out: &mut [f32],
    hidden: &[f32],
    cached: &GpuLayerWeightCache,
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    positions: &[i64],
    kv: LayerKvView<'_>,
    mask: Option<&DecoderAttnMask>,
    scratch: &mut AttentionScratch,
    engine: &mut GpuDecoderEngine,
    gpu_kv: Option<&GpuKvCache>,
) -> Result<(), Error> {
    let hidden_size = cfg.hidden_size;
    let eps = cfg.rms_norm_eps as f32;
    let params = AttentionParams::for_layer(cfg, layer)?;
    let total_kv = scratch.total_kv_len;

    assert_eq!(hidden.len(), seq_len * hidden_size);
    assert_eq!(out.len(), seq_len * hidden_size);
    assert_eq!(positions.len(), seq_len);
    assert_eq!(kv.kv_len + seq_len, total_kv);

    let rope_kind = rope_kind_for_layer(cfg, layer).ok_or(Error::Format("rope kind"))?;
    compute_rope_freqs(&mut scratch.rope_freqs, positions, rope_kind);

    let use_gpu_kv = gpu_kv.map(|g| g.kv_len == kv.kv_len).unwrap_or(false);

    let default_mask;
    let gqa_mask = match mask {
        Some(m) => GqaMask::DecoderBitmap(m),
        None => {
            default_mask = DecoderAttnMask::all_valid(seq_len, kv.kv_len);
            GqaMask::DecoderBitmap(&default_mask)
        }
    };

    if use_gpu_kv {
        let gpu_kv = gpu_kv.expect("gpu kv");
        let k_canvas_off = gpu_kv.canvas_k_elem_offset(layer)?;
        let kv_suffix_elems = seq_len * params.n_kv_heads * params.head_dim;
        let kv_suffix_byte_off = k_canvas_off * std::mem::size_of::<f32>();
        let (k_buf, v_buf) = gpu_kv.layer_buffers(layer)?;
        return fused_gpu_kv_attention(
            engine,
            out,
            hidden,
            cached,
            seq_len,
            hidden_size,
            eps,
            &params,
            &scratch.rope_freqs,
            k_buf,
            v_buf,
            k_canvas_off,
            kv_suffix_byte_off,
            kv_suffix_elems,
            total_kv,
            gqa_mask,
            &cached.o_proj,
        );
    }

    fused_input_qkv_heads(
        engine,
        hidden,
        cached,
        seq_len,
        hidden_size,
        eps,
        &params,
        &mut scratch.q,
        &mut scratch.k,
        &mut scratch.v,
    )?;

    scratch.ensure_cpu_kv_buffers(&params);
    {
        let telemetry = engine.batch_telemetry();
        let mut batch = begin_engine_batch(
            &engine.ctx.queue,
            &mut engine.pool,
            &engine.ctx.device,
            telemetry,
        )?;
        rope_qk_batched(
            &mut batch,
            &engine.attention,
            &mut scratch.q,
            &mut scratch.k,
            &scratch.rope_freqs,
            seq_len,
            params.n_heads,
            params.n_kv_heads,
            params.head_dim,
            params.rotary_dim,
        )?;
        batch.end()?;
    }

    concat_kv_for_decoder(
        &mut scratch.k_full,
        &mut scratch.v_full,
        &kv,
        &scratch.k,
        &scratch.v,
        &params,
    );

    let telemetry = engine.batch_telemetry();
    let mut batch = begin_engine_batch(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        telemetry,
    )?;
    let buf_attn = gqa_batched_chained(
        &mut batch,
        &engine.attention,
        &scratch.q,
        &scratch.k_full,
        &scratch.v_full,
        seq_len,
        total_kv,
        &params,
        gqa_mask,
    )?;
    linear_cached_batched_in_cpu_out(
        &mut batch,
        &engine.f32_bf16_linear_pipeline,
        &engine.f32_q4_linear_pipeline,
        &engine.f32_nvfp4_linear_pipeline,
        &engine.f32_q8_linear_pipeline,
        out,
        &buf_attn,
        &cached.o_proj,
        seq_len,
    )?;
    batch.end()
}

/// Shared encoder attention on GPU: writes RoPE K + V for new tokens into `gpu_kv` at current suffix.
fn forward_encoder_kv_attention_gpu(
    out: &mut [f32],
    hidden: &[f32],
    cached: &GpuLayerWeightCache,
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    positions: &[i64],
    scratch: &mut AttentionScratch,
    engine: &mut GpuDecoderEngine,
    gpu_kv: &GpuKvCache,
    gqa_mask: GqaMask<'_>,
) -> Result<(), Error> {
    let hidden_size = cfg.hidden_size;
    let eps = cfg.rms_norm_eps as f32;
    let params = AttentionParams::for_layer(cfg, layer)?;
    let total_kv = scratch.total_kv_len;

    assert_eq!(hidden.len(), seq_len * hidden_size);
    assert_eq!(out.len(), seq_len * hidden_size);
    assert_eq!(positions.len(), seq_len);

    let rope_kind = rope_kind_for_layer(cfg, layer).ok_or(Error::Format("rope kind"))?;
    compute_rope_freqs(&mut scratch.rope_freqs, positions, rope_kind);

    let k_canvas_off = gpu_kv.canvas_k_elem_offset(layer)?;
    let kv_suffix_elems = seq_len * params.n_kv_heads * params.head_dim;
    let kv_suffix_byte_off = k_canvas_off * std::mem::size_of::<f32>();
    let (k_buf, v_buf) = gpu_kv.layer_buffers(layer)?;

    fused_gpu_kv_attention(
        engine,
        out,
        hidden,
        cached,
        seq_len,
        hidden_size,
        eps,
        &params,
        &scratch.rope_freqs,
        k_buf,
        v_buf,
        k_canvas_off,
        kv_suffix_byte_off,
        kv_suffix_elems,
        total_kv,
        gqa_mask,
        &cached.o_proj,
    )
}

/// Causal encoder prefill: writes post-RoPE K/V into `gpu_kv` at offset 0.
pub fn forward_encoder_prefill_attention(
    out: &mut [f32],
    hidden: &[f32],
    cached: &GpuLayerWeightCache,
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    positions: &[i64],
    scratch: &mut AttentionScratch,
    engine: &mut GpuDecoderEngine,
    gpu_kv: &GpuKvCache,
) -> Result<(), Error> {
    assert_eq!(gpu_kv.kv_len, 0);
    assert_eq!(scratch.total_kv_len, seq_len);
    forward_encoder_kv_attention_gpu(
        out,
        hidden,
        cached,
        cfg,
        layer,
        seq_len,
        positions,
        scratch,
        engine,
        gpu_kv,
        GqaMask::CausalSliding,
    )
}

/// Encoder extend attention: causal GQA over GPU prefix + new tokens; suffix K/V stay on GPU.
pub fn forward_encoder_extend_attention(
    out: &mut [f32],
    hidden: &[f32],
    cached: &GpuLayerWeightCache,
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    positions: &[i64],
    kv_cache_len: usize,
    scratch: &mut AttentionScratch,
    engine: &mut GpuDecoderEngine,
    gpu_kv: &GpuKvCache,
) -> Result<(), Error> {
    assert_eq!(kv_cache_len + seq_len, scratch.total_kv_len);
    assert_eq!(gpu_kv.kv_len, kv_cache_len);
    forward_encoder_kv_attention_gpu(
        out,
        hidden,
        cached,
        cfg,
        layer,
        seq_len,
        positions,
        scratch,
        engine,
        gpu_kv,
        GqaMask::EncoderExtend {
            kv_cache_len,
            positions,
        },
    )
}
