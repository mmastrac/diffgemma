//! GPU encoder prefill and extend (KV stays on Metal buffers).

use crate::Error;
use crate::config::ModelConfig;
use crate::metal::decoder_layer::{forward_encoder_extend, forward_encoder_prefill};
use crate::metal::engine::GpuDecoderEngine;
use crate::metal::weights::GpuDecoderWeightCache;
use crate::model::embed::embed_tokens_from_store;
use crate::model::encoder::EncoderPrefillInput;
use crate::model::encoder::EncoderScratch;
use crate::model::kv_cache::KvCache;
use crate::model::layer_weights::DecoderLayerWeights;
use crate::weights::WeightStore;

use crate::flags::progress_enabled;

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
        return Err(Error::Runtime("prefill requires at least one token"));
    }
    let hidden = text.hidden_size;
    let embed_scale = (hidden as f32).sqrt();

    dec_scratch.ensure_gpu_kv(&engine.ctx.device, text, max_encoder_kv, max_canvas)?;
    let mut gpu_kv = dec_scratch
        .gpu_kv
        .take()
        .ok_or(Error::Gpu("gpu kv cache missing"))?;
    gpu_kv.reset_len();

    if let Some(embed) = weights.embed_q8() {
        let blob = weights
            .dgq_blob()
            .ok_or(Error::Format("dgq blob missing"))?;
        crate::metal::embed::embed_token_ids_q8_gpu(
            engine,
            embed,
            &blob,
            input.token_ids,
            seq_len,
            hidden,
            &mut enc_scratch.hidden_a[..seq_len * hidden],
        )?;
    } else {
        // Non-dgq (bf16 store) or dgq with bf16 (Raw) embed table: CPU gather.
        embed_tokens_from_store(
            store,
            &mut enc_scratch.embed_buf[..seq_len * hidden],
            input.token_ids,
            hidden,
            embed_scale,
        )?;
        enc_scratch.hidden_a[..seq_len * hidden]
            .copy_from_slice(&enc_scratch.embed_buf[..seq_len * hidden]);
    }

    let positions: Vec<i64> = (0..seq_len as i64)
        .map(|i| input.position_offset + i)
        .collect();

    let n_layers = max_layers
        .unwrap_or(text.num_hidden_layers)
        .min(text.num_hidden_layers);
    let prefill_started = std::time::Instant::now();

    // GPU-resident prefill (default): hidden state stays on GPU across all
    // layers — 2 syncs/layer (route flush + MoE batch) instead of 4 with
    // multi-MiB host round-trips. Bit-identical kernels/order to the classic
    // path. `DGQ_PREFILL_RESIDENT=0` falls back to the classic per-layer path.
    let resident =
        weights.is_dgq() && engine.encoder_gpu_moe() && crate::flags::prefill_resident_enabled();
    if resident {
        if progress_enabled() {
            eprintln!(
                "encoder: prefill starting ({n_layers} layers, {seq_len} tokens, gpu-resident)..."
            );
        }
        let bufs = crate::metal::decoder_layer::PrefillResidentBufs::new(engine, seq_len, hidden)?;
        bufs.upload_hidden(0, &enc_scratch.hidden_a[..seq_len * hidden])?;
        // Tiny reusable scratch — the resident path never touches the CPU
        // attention/MoE buffers a full per-layer GpuDecoderLayerScratch carries.
        let mut rope_freqs: Vec<f32> = Vec::new();
        let mut token_indices: Vec<u32> = Vec::new();
        let mut in_idx = 0usize;
        for layer in 0..n_layers {
            let layer_started = std::time::Instant::now();
            weights.ensure_layer(store, text, layer, &engine.ctx.device, &mut engine.pool)?;
            {
                let layer_cache = weights.layer_ref(layer);
                crate::metal::decoder_layer::forward_encoder_prefill_resident(
                    &bufs,
                    in_idx,
                    &layer_cache,
                    weights,
                    text,
                    layer,
                    seq_len,
                    &positions,
                    0,
                    &mut rope_freqs,
                    &mut token_indices,
                    engine,
                    &gpu_kv,
                )?;
            }
            if progress_enabled() && (layer == 0 || layer + 1 == n_layers || (layer + 1) % 5 == 0) {
                eprintln!(
                    "encoder: prefill layer {}/{} ({layer_elapsed:.2?}, cumulative {cum:.2?})",
                    layer + 1,
                    n_layers,
                    layer_elapsed = layer_started.elapsed(),
                    cum = prefill_started.elapsed(),
                );
            }
            in_idx = 1 - in_idx;
        }
        bufs.release(engine);
        engine.pool.trim(0);
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
        return Ok(kv_cache);
    }

    if progress_enabled() {
        eprintln!("encoder: prefill starting ({n_layers} layers, {seq_len} tokens)...");
    }
    let mut use_a_input = true;
    for layer in 0..n_layers {
        let layer_started = std::time::Instant::now();
        let layer_weights = if weights.is_dgq() {
            None
        } else {
            Some(DecoderLayerWeights::load(store, layer, text)?)
        };
        let lw = layer_weights.as_ref();
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
        if progress_enabled() && (layer == 0 || layer + 1 == n_layers || (layer + 1) % 5 == 0) {
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
        .ok_or(Error::Gpu("gpu kv cache missing"))?;
    let text = &cfg.text_config;
    let seq_len = token_ids.len();
    if seq_len == 0 {
        return Ok(());
    }
    let hidden = text.hidden_size;
    let embed_scale = (hidden as f32).sqrt();
    let kv_len_before = gpu_kv.kv_len;
    if kv_len_before != kv_cache.kv_len {
        return Err(Error::Gpu("gpu/cpu kv_len mismatch"));
    }

    if let Some(embed) = weights.embed_q8() {
        let blob = weights
            .dgq_blob()
            .ok_or(Error::Format("dgq blob missing"))?;
        crate::metal::embed::embed_token_ids_q8_gpu(
            engine,
            embed,
            &blob,
            token_ids,
            seq_len,
            hidden,
            &mut enc_scratch.hidden_a[..seq_len * hidden],
        )?;
    } else {
        // Non-dgq (bf16 store) or dgq with bf16 (Raw) embed table: CPU gather.
        embed_tokens_from_store(
            store,
            &mut enc_scratch.embed_buf[..seq_len * hidden],
            token_ids,
            hidden,
            embed_scale,
        )?;
        enc_scratch.hidden_a[..seq_len * hidden]
            .copy_from_slice(&enc_scratch.embed_buf[..seq_len * hidden]);
    }

    let positions: Vec<i64> = (0..seq_len as i64)
        .map(|i| kv_len_before as i64 + i)
        .collect();

    let n_layers = max_layers
        .unwrap_or(text.num_hidden_layers)
        .min(text.num_hidden_layers);

    // GPU-resident extend (default): same kernels/order as the classic path
    // below (bit-identical KV), hidden stays on GPU — 2 syncs/layer instead of
    // 4 with multi-MiB host round-trips. Mirrors `prefill_gpu`'s resident
    // branch with the EncoderExtend mask. `DGQ_PREFILL_RESIDENT=0` opts out.
    let resident =
        weights.is_dgq() && engine.encoder_gpu_moe() && crate::flags::prefill_resident_enabled();
    if resident {
        let bufs = crate::metal::decoder_layer::PrefillResidentBufs::new(engine, seq_len, hidden)?;
        bufs.upload_hidden(0, &enc_scratch.hidden_a[..seq_len * hidden])?;
        let mut rope_freqs: Vec<f32> = Vec::new();
        let mut token_indices: Vec<u32> = Vec::new();
        let mut in_idx = 0usize;
        for layer in 0..n_layers {
            weights.ensure_layer(store, text, layer, &engine.ctx.device, &mut engine.pool)?;
            let layer_cache = weights.layer_ref(layer);
            crate::metal::decoder_layer::forward_encoder_prefill_resident(
                &bufs,
                in_idx,
                &layer_cache,
                weights,
                text,
                layer,
                seq_len,
                &positions,
                kv_len_before,
                &mut rope_freqs,
                &mut token_indices,
                engine,
                &gpu_kv,
            )?;
            in_idx = 1 - in_idx;
        }
        // No pool trim here (unlike prefill): extends run in chunk loops —
        // keep the pool warm across chunks.
        bufs.release(engine);
        gpu_kv.advance_kv_len(seq_len)?;
        dec_scratch.gpu_kv = Some(gpu_kv);
        kv_cache.advance_kv_len(seq_len);
        return Ok(());
    }

    let mut use_a_input = true;
    for layer in 0..n_layers {
        let layer_weights = if weights.is_dgq() {
            None
        } else {
            Some(DecoderLayerWeights::load(store, layer, text)?)
        };
        let lw = layer_weights.as_ref();
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
