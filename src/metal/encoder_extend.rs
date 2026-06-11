//! GPU encoder extend: append committed canvas tokens to the GPU KV prefix.

use crate::config::ModelConfig;
use crate::metal::decoder_layer::forward_encoder_extend;
use crate::metal::engine::GpuDecoderEngine;
use crate::metal::kv_cache::GpuKvCache;
use crate::metal::weights::GpuDecoderWeightCache;
use crate::model::embed::embed_tokens;
use crate::model::encoder::EncoderScratch;
use crate::model::kv_cache::KvCache;
use crate::model::layer_weights::DecoderLayerWeights;
use crate::safetensors::Error;
use crate::weights::WeightStore;

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

    let embed = store.tensor("model.decoder.embed_tokens.weight")?;
    embed.expect_shape(&[text.vocab_size as i64, hidden as i64])?;
    embed_tokens(
        &mut enc_scratch.embed_buf[..seq_len * hidden],
        token_ids,
        embed.bf16()?,
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
    for layer in 0..n_layers {
        let layer_weights = DecoderLayerWeights::load(store, layer, text)?;
        let layer_cache = weights.ensure_layer(store, text, layer)?;
        let layer_scratch =
            dec_scratch.ensure_layer_scratch(cfg, seq_len, kv_len_before, layer)?;
        if use_a_input {
            forward_encoder_extend(
                &mut enc_scratch.hidden_b[..seq_len * hidden],
                &enc_scratch.hidden_a[..seq_len * hidden],
                &layer_weights,
                layer_cache,
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
                &layer_weights,
                layer_cache,
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
        weights.release_layer();
        use_a_input = !use_a_input;
    }

    gpu_kv.advance_kv_len(seq_len)?;
    dec_scratch.gpu_kv = Some(gpu_kv);
    kv_cache.advance_kv_len(seq_len);
    Ok(())
}