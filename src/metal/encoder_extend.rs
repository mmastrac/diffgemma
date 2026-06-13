//! GPU encoder prefill and extend (KV stays on Metal buffers).

use crate::config::ModelConfig;
use crate::metal::decoder_layer::{forward_encoder_extend, forward_encoder_prefill};
use crate::model::encoder::EncoderPrefillInput;
use crate::metal::engine::GpuDecoderEngine;
use crate::metal::weights::GpuDecoderWeightCache;
use crate::model::embed::embed_tokens_from_store;
use crate::model::encoder::EncoderScratch;
use crate::model::kv_cache::KvCache;
use crate::model::layer_weights::DecoderLayerWeights;
use crate::safetensors::Error;
use crate::weights::WeightStore;

fn progress_enabled() -> bool {
    match std::env::var("DGQ_QUIET") {
        Ok(v) => v != "1" && !v.eq_ignore_ascii_case("true"),
        Err(_) => true,
    }
}

pub fn prefill_gpu(
    store: &WeightStore,
    cfg: &ModelConfig,
    input: &EncoderPrefillInput<'_>,
    enc_scratch: &mut EncoderScratch,
    dec_scratch: &mut crate::metal::GpuDecoderScratch,
    weights: &mut GpuDecoderWeightCache,
    engine: &mut GpuDecoderEngine,
    max_encoder_kv: usize,
    max_canvas: usize,
    max_layers: Option<usize>,
) -> Result<KvCache, Error> {
    let text = &cfg.text_config;
    let seq_len = input.token_ids.len();
    if seq_len == 0 {
        return Err(Error::Format("prefill requires at least one token"));
    }
    let hidden = text.hidden_size;
    let embed_scale = (hidden as f32).sqrt();

    dec_scratch.ensure_gpu_kv(&engine.ctx.device, text, max_encoder_kv, max_canvas)?;
    let mut gpu_kv = dec_scratch
        .gpu_kv
        .take()
        .ok_or(Error::Format("gpu kv cache missing"))?;
    gpu_kv.reset_len();

    embed_tokens_from_store(
        store,
        &mut enc_scratch.embed_buf[..seq_len * hidden],
        input.token_ids,
        hidden,
        embed_scale,
    )?;

    let positions: Vec<i64> = (0..seq_len as i64)
        .map(|i| input.position_offset + i)
        .collect();

    enc_scratch.hidden_a[..seq_len * hidden]
        .copy_from_slice(&enc_scratch.embed_buf[..seq_len * hidden]);

    let n_layers = max_layers
        .unwrap_or(text.num_hidden_layers)
        .min(text.num_hidden_layers);
    let prefill_started = std::time::Instant::now();
    if progress_enabled() {
        eprintln!(
            "encoder: prefill starting ({n_layers} layers, {seq_len} tokens)..."
        );
    }
    let mut use_a_input = true;
    let mut bf16_layer_weights = None;
    for layer in 0..n_layers {
        let layer_started = std::time::Instant::now();
        let lw = if weights.is_dgq() {
            None
        } else {
            bf16_layer_weights = Some(DecoderLayerWeights::load(store, layer, text)?);
            bf16_layer_weights.as_ref()
        };
        weights.ensure_layer(store, text, layer, &engine.ctx.device, &mut engine.pool)?;
        {
            let layer_cache = weights.layer_ref(layer);
            let layer_scratch = dec_scratch.ensure_layer_scratch(cfg, seq_len, 0, layer)?;
            if use_a_input {
                forward_encoder_prefill(
                    &mut enc_scratch.hidden_b[..seq_len * hidden],
                    &enc_scratch.hidden_a[..seq_len * hidden],
                    lw,
                    &layer_cache,
                    weights,
                    text,
                    layer,
                    seq_len,
                    &positions,
                    layer_scratch,
                    engine,
                    &gpu_kv,
                )?;
            } else {
                forward_encoder_prefill(
                    &mut enc_scratch.hidden_a[..seq_len * hidden],
                    &enc_scratch.hidden_b[..seq_len * hidden],
                    lw,
                    &layer_cache,
                    weights,
                    text,
                    layer,
                    seq_len,
                    &positions,
                    layer_scratch,
                    engine,
                    &gpu_kv,
                )?;
            }
        }
        if !weights.is_dgq() {
            weights.release_layer();
        }
        if progress_enabled()
            && (layer == 0 || layer + 1 == n_layers || (layer + 1) % 5 == 0)
        {
            eprintln!(
                "encoder: prefill layer {}/{} ({layer_elapsed:.2?}, cumulative {cum:.2?})",
                layer + 1,
                n_layers,
                layer_elapsed = layer_started.elapsed(),
                cum = prefill_started.elapsed(),
            );
        }
        use_a_input = !use_a_input;
    }

    if progress_enabled() {
        eprintln!(
            "encoder: prefill done ({:.2?}, kv_len={seq_len})",
            prefill_started.elapsed()
        );
    }

    gpu_kv.advance_kv_len(seq_len)?;
    dec_scratch.gpu_kv = Some(gpu_kv);

    let mut kv_cache = KvCache::empty(text)?;
    kv_cache.kv_len = seq_len;
    Ok(kv_cache)
}

pub fn extend_prefill_gpu(
    store: &WeightStore,
    cfg: &ModelConfig,
    kv_cache: &mut KvCache,
    token_ids: &[u32],
    enc_scratch: &mut EncoderScratch,
    dec_scratch: &mut crate::metal::GpuDecoderScratch,
    weights: &mut GpuDecoderWeightCache,
    engine: &mut GpuDecoderEngine,
    max_layers: Option<usize>,
) -> Result<(), Error> {
    let mut gpu_kv = dec_scratch
        .gpu_kv
        .take()
        .ok_or(Error::Format("gpu kv cache missing"))?;
    let text = &cfg.text_config;
    let seq_len = token_ids.len();
    if seq_len == 0 {
        return Ok(());
    }
    let hidden = text.hidden_size;
    let embed_scale = (hidden as f32).sqrt();
    let kv_len_before = gpu_kv.kv_len;
    if kv_len_before != kv_cache.kv_len {
        return Err(Error::Format("gpu/cpu kv_len mismatch"));
    }

    embed_tokens_from_store(
        store,
        &mut enc_scratch.embed_buf[..seq_len * hidden],
        token_ids,
        hidden,
        embed_scale,
    )?;

    let positions: Vec<i64> = (0..seq_len as i64)
        .map(|i| kv_len_before as i64 + i)
        .collect();

    enc_scratch.hidden_a[..seq_len * hidden]
        .copy_from_slice(&enc_scratch.embed_buf[..seq_len * hidden]);

    let n_layers = max_layers
        .unwrap_or(text.num_hidden_layers)
        .min(text.num_hidden_layers);
    let mut use_a_input = true;
    let mut bf16_layer_weights = None;
    for layer in 0..n_layers {
        let lw = if weights.is_dgq() {
            None
        } else {
            bf16_layer_weights = Some(DecoderLayerWeights::load(store, layer, text)?);
            bf16_layer_weights.as_ref()
        };
        weights.ensure_layer(store, text, layer, &engine.ctx.device, &mut engine.pool)?;
        {
            let layer_cache = weights.layer_ref(layer);
            let layer_scratch =
                dec_scratch.ensure_layer_scratch(cfg, seq_len, kv_len_before, layer)?;
            if use_a_input {
                forward_encoder_extend(
                    &mut enc_scratch.hidden_b[..seq_len * hidden],
                    &enc_scratch.hidden_a[..seq_len * hidden],
                    lw,
                    &layer_cache,
                    weights,
                    text,
                    layer,
                    seq_len,
                    &positions,
                    kv_len_before,
                    layer_scratch,
                    engine,
                    &gpu_kv,
                )?;
            } else {
                forward_encoder_extend(
                    &mut enc_scratch.hidden_a[..seq_len * hidden],
                    &enc_scratch.hidden_b[..seq_len * hidden],
                    lw,
                    &layer_cache,
                    weights,
                    text,
                    layer,
                    seq_len,
                    &positions,
                    kv_len_before,
                    layer_scratch,
                    engine,
                    &gpu_kv,
                )?;
            }
        }
        if !weights.is_dgq() {
            weights.release_layer();
        }
        use_a_input = !use_a_input;
    }

    gpu_kv.advance_kv_len(seq_len)?;
    dec_scratch.gpu_kv = Some(gpu_kv);
    kv_cache.advance_kv_len(seq_len);
    Ok(())
}