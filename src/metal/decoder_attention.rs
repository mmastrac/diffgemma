use crate::config::TextConfig;
use crate::kernels::cpu::{compute_rope_freqs, rope_kind_for_layer};
use crate::metal::attention_batch::{
    decoder_gqa_gpu_kv_batched_chained, gqa_batched_chained, rope_qk_batched,
};
use crate::metal::batched_kernels::{self as bk};
use crate::metal::batch::begin_engine_batch;
use crate::metal::engine::GpuDecoderEngine;
use crate::metal::kv_cache::GpuKvCache;
use crate::metal::linear::{linear_cached_batched_in_buf, linear_cached_batched_in_cpu_out};
use crate::metal::weights::GpuLayerWeightCache;
use crate::model::attention::{
    concat_kv_for_decoder, AttentionParams, AttentionScratch, GqaMask,
};
use crate::model::kv_cache::LayerKvView;
use crate::model::mask::DecoderAttnMask;
use crate::safetensors::Error;

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
    use_q4_mps: bool,
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
        if use_q4_mps {
            Some((&mut engine.mps_matmul, &engine.dequant_q4_matrix_pipeline))
        } else {
            None
        },
        &buf_normed,
        &cached.q_proj,
        seq_len,
    )?;
    let buf_k = linear_cached_batched_in_buf(
        &mut batch,
        &engine.f32_bf16_linear_pipeline,
        &engine.f32_q4_linear_pipeline,
        if use_q4_mps {
            Some((&mut engine.mps_matmul, &engine.dequant_q4_matrix_pipeline))
        } else {
            None
        },
        &buf_normed,
        &cached.k_proj,
        seq_len,
    )?;
    let buf_v = if let Some(v_proj) = &cached.v_proj {
        linear_cached_batched_in_buf(
            &mut batch,
            &engine.f32_bf16_linear_pipeline,
            &engine.f32_q4_linear_pipeline,
            if use_q4_mps {
                Some((&mut engine.mps_matmul, &engine.dequant_q4_matrix_pipeline))
            } else {
                None
            },
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

fn fused_gqa_o_proj_gpu_kv(
    engine: &mut GpuDecoderEngine,
    out: &mut [f32],
    q: &[f32],
    k_buf: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    v_buf: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    k_canvas_off: usize,
    freqs: &[f32],
    seq_len: usize,
    total_kv: usize,
    params: &AttentionParams,
    mask: GqaMask<'_>,
    o_proj: &crate::metal::linear::GpuLinearWeight,
    use_q4_mps: bool,
) -> Result<(), Error> {
    let telemetry = engine.batch_telemetry();
    let mut batch = begin_engine_batch(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        telemetry,
    )?;
    let q4_mps = if use_q4_mps {
        Some((&mut engine.mps_matmul, &engine.dequant_q4_matrix_pipeline))
    } else {
        None
    };
    let buf_attn = decoder_gqa_gpu_kv_batched_chained(
        &mut batch,
        &engine.attention,
        q,
        k_buf,
        v_buf,
        k_canvas_off,
        freqs,
        seq_len,
        total_kv,
        params,
        mask,
    )?;
    linear_cached_batched_in_cpu_out(
        &mut batch,
        &engine.f32_bf16_linear_pipeline,
        &engine.f32_q4_linear_pipeline,
        q4_mps,
        out,
        &buf_attn,
        o_proj,
        seq_len,
    )?;
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
    use_q4_mps: bool,
) -> Result<(), Error> {
    let hidden_size = cfg.hidden_size;
    let eps = cfg.rms_norm_eps as f32;
    let params = AttentionParams::for_layer(cfg, layer)?;
    let total_kv = scratch.total_kv_len;

    assert_eq!(hidden.len(), seq_len * hidden_size);
    assert_eq!(out.len(), seq_len * hidden_size);
    assert_eq!(positions.len(), seq_len);
    assert_eq!(kv.kv_len + seq_len, total_kv);

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
        use_q4_mps,
    )?;

    let rope_kind = rope_kind_for_layer(cfg, layer).ok_or(Error::Format("rope kind"))?;
    compute_rope_freqs(&mut scratch.rope_freqs, positions, rope_kind);

    let use_gpu_kv = gpu_kv
        .map(|g| g.kv_len == kv.kv_len)
        .unwrap_or(false);

    if !use_gpu_kv {
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
        fused_gqa_o_proj_gpu_kv(
            engine,
            out,
            &scratch.q,
            k_buf,
            v_buf,
            k_canvas_off,
            &scratch.rope_freqs,
            seq_len,
            total_kv,
            &params,
            gqa_mask,
            &cached.o_proj,
            use_q4_mps,
        )?;
    } else {
        let telemetry = engine.batch_telemetry();
    let mut batch = begin_engine_batch(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        telemetry,
    )?;
        let q4_mps = if use_q4_mps {
            Some((&mut engine.mps_matmul, &engine.dequant_q4_matrix_pipeline))
        } else {
            None
        };
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
            q4_mps,
            out,
            &buf_attn,
            &cached.o_proj,
            seq_len,
        )?;
        batch.end()?;
    }

    Ok(())
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
    use_q4_mps: bool,
) -> Result<(), Error> {
    let hidden_size = cfg.hidden_size;
    let eps = cfg.rms_norm_eps as f32;
    let params = AttentionParams::for_layer(cfg, layer)?;
    let total_kv = scratch.total_kv_len;

    assert_eq!(hidden.len(), seq_len * hidden_size);
    assert_eq!(out.len(), seq_len * hidden_size);
    assert_eq!(positions.len(), seq_len);

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
        use_q4_mps,
    )?;

    let rope_kind = rope_kind_for_layer(cfg, layer).ok_or(Error::Format("rope kind"))?;
    compute_rope_freqs(&mut scratch.rope_freqs, positions, rope_kind);

    gpu_kv.write_canvas_kv_pre_rope(layer, seq_len, &scratch.k, &scratch.v)?;
    let k_suffix_off = gpu_kv.canvas_k_elem_offset(layer)?;
    let (k_buf, v_buf) = gpu_kv.layer_buffers(layer)?;

    fused_gqa_o_proj_gpu_kv(
        engine,
        out,
        &scratch.q,
        k_buf,
        v_buf,
        k_suffix_off,
        &scratch.rope_freqs,
        seq_len,
        total_kv,
        &params,
        gqa_mask,
        &cached.o_proj,
        use_q4_mps,
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
    use_q4_mps: bool,
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
        use_q4_mps,
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
    use_q4_mps: bool,
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
        use_q4_mps,
    )
}
