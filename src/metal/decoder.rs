use crate::config::ModelConfig;
use crate::metal::decoder_layer::{forward_decoder as layer_forward, GpuDecoderLayerScratch};
use crate::metal::engine::GpuDecoderEngine;
use crate::model::decoder::{DecoderForwardInput, DecoderForwardOutput, DecoderScratch};
use crate::model::embed::{embed_tokens, lm_head_tied_bf16, logit_softcapping};
use crate::model::kv_cache::LayerKvView;
use crate::model::layer_weights::DecoderLayerWeights;
use crate::model::self_conditioning::{apply as apply_self_conditioning, SelfConditioningWeights};
use crate::safetensors::Error;
use crate::metal::weights::GpuDecoderWeightCache;
use crate::weights::WeightStore;

pub fn load_weight_cache(
    store: &WeightStore,
    text: &crate::config::TextConfig,
) -> Result<GpuDecoderWeightCache, Error> {
    eprintln!(
        "warming GPU weight cache ({} layers)...",
        text.num_hidden_layers
    );
    let warm = std::time::Instant::now();
    let cache = GpuDecoderWeightCache::load(store, text)?;
    eprintln!("  weight cache ready in {:.2?}", warm.elapsed());
    Ok(cache)
}

pub struct GpuDecoderScratch {
    pub cpu: DecoderScratch,
    layer_kv_len: usize,
    layers: Vec<GpuDecoderLayerScratch>,
}

impl GpuDecoderScratch {
    pub fn new(seq_len: usize, cfg: &ModelConfig) -> Self {
        Self {
            cpu: DecoderScratch::new(seq_len, cfg),
            layer_kv_len: usize::MAX,
            layers: Vec::new(),
        }
    }

    fn ensure_layers(
        &mut self,
        cfg: &ModelConfig,
        seq_len: usize,
        kv_len: usize,
    ) -> Result<(), Error> {
        let n_layers = cfg.text_config.num_hidden_layers;
        if self.layer_kv_len == kv_len && self.layers.len() == n_layers {
            return Ok(());
        }
        self.layers.clear();
        for layer in 0..n_layers {
            self.layers.push(GpuDecoderLayerScratch::with_kv_len(
                seq_len,
                &cfg.text_config,
                layer,
                kv_len,
            )?);
        }
        self.layer_kv_len = kv_len;
        Ok(())
    }
}

pub struct BenchConfig {
    pub max_layers: usize,
}

pub fn bench_forward(
    store: &WeightStore,
    cfg: &ModelConfig,
    input: &DecoderForwardInput<'_>,
    scratch: &mut GpuDecoderScratch,
    weights: &GpuDecoderWeightCache,
    engine: &mut GpuDecoderEngine,
    bench: &BenchConfig,
) -> Result<DecoderForwardOutput, Error> {
    forward_inner(
        store,
        cfg,
        input,
        scratch,
        weights,
        engine,
        bench.max_layers,
    )
}

pub fn forward(
    store: &WeightStore,
    cfg: &ModelConfig,
    input: &DecoderForwardInput<'_>,
    scratch: &mut GpuDecoderScratch,
    weights: &GpuDecoderWeightCache,
    engine: &mut GpuDecoderEngine,
) -> Result<DecoderForwardOutput, Error> {
    let n = cfg.text_config.num_hidden_layers;
    forward_inner(store, cfg, input, scratch, weights, engine, n)
}

fn forward_inner(
    store: &WeightStore,
    cfg: &ModelConfig,
    input: &DecoderForwardInput<'_>,
    scratch: &mut GpuDecoderScratch,
    weights: &GpuDecoderWeightCache,
    engine: &mut GpuDecoderEngine,
    max_layers: usize,
) -> Result<DecoderForwardOutput, Error> {
    let text = &cfg.text_config;
    let seq_len = input.token_ids.len();
    let hidden = text.hidden_size;
    let vocab = text.vocab_size;
    let embed_scale = (hidden as f32).sqrt();

    let embed = store.tensor("model.decoder.embed_tokens.weight")?;
    embed.expect_shape(&[vocab as i64, hidden as i64])?;
    let embed_bf16 = embed.bf16()?;

    embed_tokens(
        &mut scratch.cpu.embed_buf,
        input.token_ids,
        embed_bf16,
        hidden,
        embed_scale,
    )?;

    let sc_weights = SelfConditioningWeights::load(store, text)?;
    match input.self_conditioning_logits {
        Some(logits) => {
            crate::model::embed::soft_embeddings_from_logits(
                &mut scratch.cpu.sc_signal,
                logits,
                embed_bf16,
                seq_len,
                vocab,
                hidden,
                embed_scale,
            );
        }
        None => scratch.cpu.sc_signal.fill(0.0),
    }
    apply_self_conditioning(
        &mut scratch.cpu.hidden_a,
        &scratch.cpu.embed_buf,
        &scratch.cpu.sc_signal,
        &sc_weights,
        text,
        seq_len,
        &mut scratch.cpu.self_cond,
    )?;

    let mask = input.mask.as_ref();
    let positions: Vec<i64> =
        (input.kv_cache.kv_len as i64..input.kv_cache.kv_len as i64 + seq_len as i64).collect();

    scratch.ensure_layers(cfg, seq_len, input.kv_cache.kv_len)?;

    let mut in_buf = &mut scratch.cpu.hidden_a;
    let mut out_buf = &mut scratch.cpu.hidden_b;

    let n_layers = max_layers.min(text.num_hidden_layers);
    for layer in 0..n_layers {
        let layer_weights = DecoderLayerWeights::load(store, layer, text)?;
        layer_forward(
            out_buf,
            in_buf,
            &layer_weights,
            &weights.layers[layer],
            text,
            layer,
            seq_len,
            &positions,
            LayerKvView::from_layer(
                input
                    .kv_cache
                    .layer(layer)
                    .ok_or(Error::Format("missing kv layer"))?,
                input.kv_cache.kv_len,
            ),
            mask,
            &mut scratch.layers[layer],
            engine,
        )?;
        std::mem::swap(&mut in_buf, &mut out_buf);
    }

    {
        use crate::metal::batched_kernels as bk;
        use crate::metal::batch::GpuBatch;
        let mut batch = GpuBatch::begin(&engine.ctx.queue, &mut engine.pool, &engine.ctx.device)?;
        bk::rms_norm_rows(
            &mut batch,
            &engine.kernels,
            out_buf,
            in_buf,
            &weights.final_norm,
            seq_len,
            hidden,
            text.rms_norm_eps as f32,
        )?;
        batch.end()?;
    }

    let mut logits = vec![0.0f32; seq_len * vocab];
    lm_head_tied_bf16(&mut logits, out_buf, embed, seq_len, hidden, vocab)?;
    logit_softcapping(&mut logits, text.final_logit_softcapping as f32);

    let mut hidden_out = vec![0.0f32; seq_len * hidden];
    hidden_out.copy_from_slice(out_buf);

    Ok(DecoderForwardOutput {
        hidden_states: hidden_out,
        logits,
    })
}
