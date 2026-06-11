use crate::config::TextConfig;
use crate::kernels::cpu::{compute_rope_freqs, rope_kind_for_layer};
use crate::metal::attention_batch::{decoder_gqa_gpu_kv_batched, gqa_batched, rope_qk_batched};
use crate::metal::batched_kernels::{self as bk};
use crate::metal::batch::GpuBatch;
use crate::metal::engine::GpuDecoderEngine;
use crate::metal::kv_cache::GpuKvCache;
use crate::metal::linear::linear_cached_batched;
use crate::metal::weights::GpuLayerWeightCache;
use crate::model::attention::{
    concat_kv_for_decoder, normalize_qkv_heads, AttentionParams, AttentionScratch, GqaMask,
};
use crate::model::kv_cache::LayerKvView;
use crate::model::mask::DecoderAttnMask;
use crate::safetensors::Error;

fn rms_norm_batch(
    engine: &mut GpuDecoderEngine,
    out: &mut [f32],
    x: &[f32],
    weight: &[f32],
    seq_len: usize,
    hidden: usize,
    eps: f32,
) -> Result<(), Error> {
    let mut batch = GpuBatch::begin(&engine.ctx.queue, &mut engine.pool, &engine.ctx.device)?;
    bk::rms_norm_rows(
        &mut batch,
        &engine.kernels,
        out,
        x,
        weight,
        seq_len,
        hidden,
        eps,
    )?;
    batch.end()
}

/// Decoder self-attention: GPU input norm + Q/K/V + o_proj; GPU RoPE + GQA; CPU head norms.
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

    rms_norm_batch(
        engine,
        &mut scratch.normed,
        hidden,
        cached.input_layernorm.as_slice(),
        seq_len,
        hidden_size,
        eps,
    )?;

    {
        let mut batch = GpuBatch::begin(&engine.ctx.queue, &mut engine.pool, &engine.ctx.device)?;
        linear_cached_batched(
            &mut batch,
            &engine.f32_bf16_gemm_pipeline,
            &mut scratch.q,
            &scratch.normed,
            &cached.q_proj,
            seq_len,
        )?;
        linear_cached_batched(
            &mut batch,
            &engine.f32_bf16_gemm_pipeline,
            &mut scratch.k,
            &scratch.normed,
            &cached.k_proj,
            seq_len,
        )?;
        if let Some(v_proj) = &cached.v_proj {
            linear_cached_batched(
                &mut batch,
                &engine.f32_bf16_gemm_pipeline,
                &mut scratch.v,
                &scratch.normed,
                v_proj,
                seq_len,
            )?;
        }
        batch.end()?;
    }

    if cached.v_proj.is_none() {
        scratch.v.copy_from_slice(&scratch.k);
    }

    normalize_qkv_heads(
        &mut scratch.q,
        &mut scratch.k,
        &mut scratch.v,
        seq_len,
        &params,
        eps,
        &mut scratch.head_buf,
        cached.q_norm.as_slice(),
        cached.k_norm.as_slice(),
    );

    let rope_kind = rope_kind_for_layer(cfg, layer).ok_or(Error::Format("rope kind"))?;
    compute_rope_freqs(&mut scratch.rope_freqs, positions, rope_kind);

    let use_gpu_kv = gpu_kv
        .map(|g| g.kv_len == kv.kv_len)
        .unwrap_or(false);

    if !use_gpu_kv {
        scratch.ensure_cpu_kv_buffers(&params);
        {
            let mut batch = GpuBatch::begin(&engine.ctx.queue, &mut engine.pool, &engine.ctx.device)?;
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
    }

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
        gpu_kv.write_canvas_kv_pre_rope(layer, seq_len, &scratch.k, &scratch.v)?;
        let k_canvas_off = gpu_kv.canvas_k_elem_offset(layer)?;
        let (k_buf, v_buf) = gpu_kv.layer_buffers(layer)?;
        let mut batch = GpuBatch::begin(&engine.ctx.queue, &mut engine.pool, &engine.ctx.device)?;
        decoder_gqa_gpu_kv_batched(
            &mut batch,
            &engine.attention,
            &mut scratch.attn_out,
            &scratch.q,
            k_buf,
            v_buf,
            k_canvas_off,
            &scratch.rope_freqs,
            seq_len,
            total_kv,
            &params,
            gqa_mask,
        )?;
        batch.end()?;
    } else {
        let mut batch = GpuBatch::begin(&engine.ctx.queue, &mut engine.pool, &engine.ctx.device)?;
        gqa_batched(
            &mut batch,
            &engine.attention,
            &mut scratch.attn_out,
            &scratch.q,
            &scratch.k_full,
            &scratch.v_full,
            seq_len,
            total_kv,
            &params,
            gqa_mask,
        )?;
        batch.end()?;
    }

    {
        let mut batch = GpuBatch::begin(&engine.ctx.queue, &mut engine.pool, &engine.ctx.device)?;
        linear_cached_batched(
            &mut batch,
            &engine.f32_bf16_gemm_pipeline,
            out,
            &scratch.attn_out,
            &cached.o_proj,
            seq_len,
        )?;
        batch.end()?;
    }

    Ok(())
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
    let hidden_size = cfg.hidden_size;
    let eps = cfg.rms_norm_eps as f32;
    let params = AttentionParams::for_layer(cfg, layer)?;
    let total_kv = scratch.total_kv_len;

    assert_eq!(hidden.len(), seq_len * hidden_size);
    assert_eq!(out.len(), seq_len * hidden_size);
    assert_eq!(positions.len(), seq_len);
    assert_eq!(kv_cache_len + seq_len, total_kv);
    assert_eq!(gpu_kv.kv_len, kv_cache_len);

    rms_norm_batch(
        engine,
        &mut scratch.normed,
        hidden,
        cached.input_layernorm.as_slice(),
        seq_len,
        hidden_size,
        eps,
    )?;

    {
        let mut batch = GpuBatch::begin(&engine.ctx.queue, &mut engine.pool, &engine.ctx.device)?;
        linear_cached_batched(
            &mut batch,
            &engine.f32_bf16_gemm_pipeline,
            &mut scratch.q,
            &scratch.normed,
            &cached.q_proj,
            seq_len,
        )?;
        linear_cached_batched(
            &mut batch,
            &engine.f32_bf16_gemm_pipeline,
            &mut scratch.k,
            &scratch.normed,
            &cached.k_proj,
            seq_len,
        )?;
        if let Some(v_proj) = &cached.v_proj {
            linear_cached_batched(
                &mut batch,
                &engine.f32_bf16_gemm_pipeline,
                &mut scratch.v,
                &scratch.normed,
                v_proj,
                seq_len,
            )?;
        }
        batch.end()?;
    }

    if cached.v_proj.is_none() {
        scratch.v.copy_from_slice(&scratch.k);
    }

    normalize_qkv_heads(
        &mut scratch.q,
        &mut scratch.k,
        &mut scratch.v,
        seq_len,
        &params,
        eps,
        &mut scratch.head_buf,
        cached.q_norm.as_slice(),
        cached.k_norm.as_slice(),
    );

    let rope_kind = rope_kind_for_layer(cfg, layer).ok_or(Error::Format("rope kind"))?;
    compute_rope_freqs(&mut scratch.rope_freqs, positions, rope_kind);

    gpu_kv.write_canvas_kv_pre_rope(layer, seq_len, &scratch.k, &scratch.v)?;
    let k_suffix_off = gpu_kv.canvas_k_elem_offset(layer)?;
    let (k_buf, v_buf) = gpu_kv.layer_buffers(layer)?;
    let gqa_mask = GqaMask::EncoderExtend {
        kv_cache_len,
        positions,
    };

    {
        let mut batch = GpuBatch::begin(&engine.ctx.queue, &mut engine.pool, &engine.ctx.device)?;
        decoder_gqa_gpu_kv_batched(
            &mut batch,
            &engine.attention,
            &mut scratch.attn_out,
            &scratch.q,
            k_buf,
            v_buf,
            k_suffix_off,
            &scratch.rope_freqs,
            seq_len,
            total_kv,
            &params,
            gqa_mask,
        )?;
        batch.end()?;
    }

    {
        let mut batch = GpuBatch::begin(&engine.ctx.queue, &mut engine.pool, &engine.ctx.device)?;
        linear_cached_batched(
            &mut batch,
            &engine.f32_bf16_gemm_pipeline,
            out,
            &scratch.attn_out,
            &cached.o_proj,
            seq_len,
        )?;
        batch.end()?;
    }

    Ok(())
}
