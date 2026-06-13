use crate::config::{LayerType, ModelConfig, TextConfig};
use crate::metal::decoder_layer::{forward_decoder as layer_forward, GpuDecoderLayerScratch};
use crate::metal::engine::GpuDecoderEngine;
use crate::model::decoder::{DecoderForwardInput, DecoderForwardOutput, DecoderScratch};
use crate::model::embed::{embed_tokens_from_store, lm_head_tied_from_store, logit_softcapping};
use crate::model::kv_cache::LayerKvView;
use crate::model::self_conditioning::apply_from_store;
use crate::safetensors::Error;
use crate::metal::weights::GpuDecoderWeightCache;
use crate::metal::kv_cache::GpuKvCache;
use crate::weights::WeightStore;
use crate::kernels::cpu::rms_norm_no_scale;

pub fn load_weight_cache(
    store: &WeightStore,
    text: &crate::config::TextConfig,
    seq_len: usize,
    kv_len: usize,
) -> Result<GpuDecoderWeightCache, Error> {
    load_weight_cache_opt(store, text, seq_len, kv_len, None)
}

pub fn load_weight_cache_opt(
    store: &WeightStore,
    text: &crate::config::TextConfig,
    seq_len: usize,
    kv_len: usize,
    shared_dgq_blob: Option<std::sync::Arc<crate::metal::dgq_gpu::DgqGpuBlob>>,
) -> Result<GpuDecoderWeightCache, Error> {
    use crate::metal::device::MetalContext;
    use crate::metal::memory::{expert_cache_budget_bytes, log_gpu_memory_plan};

    eprintln!("initializing GPU weight cache...");
    let warm = std::time::Instant::now();
    let ctx = MetalContext::new()?;
    let expert_budget = if store.is_quantized() {
        0
    } else {
        let budget = expert_cache_budget_bytes(&ctx.device, text, seq_len, kv_len);
        log_gpu_memory_plan(&ctx.device, text, seq_len, kv_len, budget);
        budget
    };
    if store.is_quantized() {
        if shared_dgq_blob.is_some() {
            eprintln!("  mode: .dgq resident (shared step-kernel blob, zero-copy)");
        } else {
            eprintln!("  mode: .dgq resident (zero-copy blob, no expert LRU)");
        }
    } else {
        eprintln!("  mode: bf16 paged layers + expert LRU");
    }
    let cache = if let (WeightStore::Dgq(_), Some(blob)) = (store, shared_dgq_blob) {
        GpuDecoderWeightCache::load_with_dgq_blob(store, text, &ctx.device, blob)?
    } else {
        GpuDecoderWeightCache::load(store, text, expert_budget, &ctx.device)?
    };
    eprintln!("  cache ready in {:.2?}", warm.elapsed());
    Ok(cache)
}

fn example_layer(text: &TextConfig, kind: LayerType) -> usize {
    text.layer_types
        .iter()
        .position(|t| *t == kind)
        .unwrap_or(0)
}

/// Per-layer decoder scratch (do not share MoE/route arenas across layers).
struct LayerScratchReuse {
    seq_len: usize,
    kv_len: usize,
    layers: Vec<Option<GpuDecoderLayerScratch>>,
}

impl LayerScratchReuse {
    fn ensure(
        &mut self,
        cfg: &ModelConfig,
        seq_len: usize,
        kv_len: usize,
        layer: usize,
    ) -> Result<&mut GpuDecoderLayerScratch, Error> {
        if self.seq_len != seq_len || self.kv_len != kv_len {
            self.layers.clear();
            self.seq_len = seq_len;
            self.kv_len = kv_len;
        }
        if self.layers.len() <= layer {
            self.layers.resize_with(layer + 1, || None);
        }
        if self.layers[layer].is_none() {
            let text = &cfg.text_config;
            let kind = text
                .layer_types
                .get(layer)
                .ok_or(Error::Format("invalid layer"))?;
            let example = example_layer(text, *kind);
            self.layers[layer] = Some(GpuDecoderLayerScratch::with_kv_len(
                seq_len, text, example, kv_len,
            )?);
        }
        Ok(self.layers[layer].as_mut().expect("layer scratch"))
    }
}

pub struct GpuDecoderScratch {
    pub cpu: DecoderScratch,
    pub gpu_kv: Option<GpuKvCache>,
    pub sampler: crate::metal::sampler::GpuSamplerScratch,
    /// When true, lm_head + denoise sampler keep logits on GPU (`.dgq` path).
    pub use_gpu_sampler: bool,
    pub have_gpu_sc_logits: bool,
    layer: LayerScratchReuse,
}

impl GpuDecoderScratch {
    pub fn new(seq_len: usize, cfg: &ModelConfig) -> Self {
        Self {
            cpu: DecoderScratch::new(seq_len, cfg),
            gpu_kv: None,
            sampler: crate::metal::sampler::GpuSamplerScratch::new(
                cfg.canvas_length,
                cfg.text_config.vocab_size,
            ),
            use_gpu_sampler: false,
            have_gpu_sc_logits: false,
            layer: LayerScratchReuse {
                seq_len: 0,
                kv_len: 0,
                layers: Vec::new(),
            },
        }
    }

    pub fn ensure_gpu_kv(
        &mut self,
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        cfg: &TextConfig,
        max_encoder_kv: usize,
        max_canvas: usize,
    ) -> Result<(), Error> {
        if self.gpu_kv.is_none() {
            self.gpu_kv = Some(GpuKvCache::new(device, cfg, max_encoder_kv, max_canvas)?);
        }
        Ok(())
    }

    pub fn sync_gpu_kv_from_cpu(
        &mut self,
        cpu: &crate::model::kv_cache::KvCache,
        canvas_len: usize,
    ) -> Result<(), Error> {
        if let Some(gpu_kv) = self.gpu_kv.as_mut() {
            gpu_kv.sync_all_from_cpu(cpu, canvas_len)?;
        }
        Ok(())
    }

    pub fn ensure_layer_scratch(
        &mut self,
        cfg: &ModelConfig,
        seq_len: usize,
        kv_len: usize,
        layer: usize,
    ) -> Result<&mut GpuDecoderLayerScratch, Error> {
        self.layer.ensure(cfg, seq_len, kv_len, layer)
    }
}

pub struct BenchConfig {
    pub max_layers: usize,
}

pub fn bench_forward(
    store: &WeightStore,
    cfg: &ModelConfig,
    input: &mut DecoderForwardInput<'_>,
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
    input: &mut DecoderForwardInput<'_>,
    scratch: &mut GpuDecoderScratch,
    weights: &GpuDecoderWeightCache,
    engine: &mut GpuDecoderEngine,
    max_layers: Option<usize>,
) -> Result<DecoderForwardOutput, Error> {
    let n = max_layers
        .unwrap_or(cfg.text_config.num_hidden_layers)
        .min(cfg.text_config.num_hidden_layers);
    forward_inner(store, cfg, input, scratch, weights, engine, n)
}

fn forward_inner(
    store: &WeightStore,
    cfg: &ModelConfig,
    input: &mut DecoderForwardInput<'_>,
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

    embed_tokens_from_store(
        store,
        &mut scratch.cpu.embed_buf,
        input.token_ids,
        hidden,
        embed_scale,
    )?;

    let eps = text.rms_norm_eps as f32;
    if scratch.have_gpu_sc_logits {
        if let (Some(embed_q8), Some(logits_buf)) = (
            weights.embed_q8(),
            scratch.sampler.logits.as_buf(),
        ) {
            crate::metal::embed::soft_embeddings_q8_gpu_from_buf(
                engine,
                logits_buf,
                embed_q8,
                seq_len,
                vocab,
                hidden,
                embed_scale,
                &mut scratch.cpu.sc_signal,
            )?;
        } else {
            return Err(Error::Format("gpu sc logits missing"));
        }
    } else if let Some(logits) = input.self_conditioning_logits {
        if let Some(embed_q8) = weights.embed_q8() {
            crate::metal::embed::soft_embeddings_q8_gpu(
                engine,
                logits,
                embed_q8,
                seq_len,
                vocab,
                hidden,
                embed_scale,
                &mut scratch.cpu.sc_signal,
            )?;
        } else {
            crate::model::embed::soft_embeddings_from_logits_store(
                store,
                &mut scratch.cpu.sc_signal,
                logits,
                seq_len,
                vocab,
                hidden,
                embed_scale,
                &mut scratch.cpu.sc_probs,
            )?;
        }
        if let Some(sc) = weights.self_conditioning() {
            crate::metal::self_conditioning::apply_gpu(
                engine,
                sc,
                &mut scratch.cpu.hidden_a,
                &scratch.cpu.embed_buf,
                &scratch.cpu.sc_signal,
                seq_len,
                hidden,
                text.intermediate_size,
                eps,
            )?;
        } else {
            apply_from_store(
                &mut scratch.cpu.hidden_a,
                &scratch.cpu.embed_buf,
                &scratch.cpu.sc_signal,
                store,
                text,
                seq_len,
                &mut scratch.cpu.self_cond,
            )?;
        }
    } else {
        scratch.cpu.hidden_a.copy_from_slice(&scratch.cpu.embed_buf);
        scratch
            .cpu
            .self_cond
            .normed
            .resize(seq_len * hidden, 0.0);
        for s in 0..seq_len {
            let off = s * hidden;
            scratch.cpu.self_cond.normed[off..off + hidden]
                .copy_from_slice(&scratch.cpu.hidden_a[off..off + hidden]);
            rms_norm_no_scale(
                &mut scratch.cpu.hidden_a[off..off + hidden],
                &scratch.cpu.self_cond.normed[off..off + hidden],
                eps,
            );
        }
    }

    let mask = input.mask;
    let positions: Vec<i64> =
        (input.kv_cache.kv_len as i64..input.kv_cache.kv_len as i64 + seq_len as i64).collect();

    let mut in_buf = &mut scratch.cpu.hidden_a;
    let mut out_buf = &mut scratch.cpu.hidden_b;

    let n_layers = max_layers.min(text.num_hidden_layers);
    if !engine.use_mps_q4() {
        engine.pool.clear();
    }
    let expert_before = if engine.telemetry_enabled() {
        Some(weights.expert_cache_stats())
    } else {
        None
    };
    let gpu_kv_ref = scratch.gpu_kv.as_ref();
    // Canvas K/V are fully overwritten per layer in write_canvas_kv_pre_rope; do not
    // zero the suffix here — clearing all layers before each forward caused repeat drift.
    let mut bf16_layer_weights = None;
    for layer in 0..n_layers {
        let layer_scratch = scratch.layer.ensure(cfg, seq_len, input.kv_cache.kv_len, layer)?;
        let lw = if weights.is_dgq() {
            None
        } else {
            bf16_layer_weights = Some(crate::model::layer_weights::DecoderLayerWeights::load(
                store, layer, text,
            )?);
            bf16_layer_weights.as_ref()
        };
        weights.ensure_layer(store, text, layer, &engine.ctx.device, &mut engine.pool)?;
        let layer_cache = weights.layer_ref(layer);
        layer_forward(
            out_buf,
            in_buf,
            lw,
            &layer_cache,
            weights,
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
            layer_scratch,
            engine,
            gpu_kv_ref,
        )?;
        if !weights.is_dgq() {
            weights.release_layer();
        }
        std::mem::swap(&mut in_buf, &mut out_buf);
    }

    // Each layer writes to `out_buf`, then swap moves it to `in_buf`.
    let (norm_in, norm_out) = (in_buf, out_buf);

    {
        use crate::metal::batched_kernels as bk;
        use crate::metal::batch::begin_engine_batch;
        let telemetry = engine.batch_telemetry();
        let mut batch = begin_engine_batch(
            &engine.ctx.queue,
            &mut engine.pool,
            &engine.ctx.device,
            telemetry,
        )?;
        bk::rms_norm_rows(
            &mut batch,
            &engine.kernels,
            norm_in,
            norm_out,
            weights.final_norm(),
            seq_len,
            hidden,
            text.rms_norm_eps as f32,
        )?;
        batch.end()?;
    }
    engine.pool.trim(4);

    if let Some(before) = expert_before {
        let after = weights.expert_cache_stats();
        let handle = engine.telemetry_handle();
        let mut tel = handle.borrow_mut();
        tel.expert_hits += after.hits.saturating_sub(before.hits);
        tel.expert_misses += after.misses.saturating_sub(before.misses);
        tel.expert_upload_bytes += after.upload_bytes.saturating_sub(before.upload_bytes);
    }
    if engine.telemetry_enabled() && input.compute_logits {
        if !scratch.use_gpu_sampler {
            engine
                .telemetry_handle()
                .borrow_mut()
                .lm_head_logits_bytes = (seq_len * vocab * 4) as u64;
        }
    }

    if input.compute_logits {
        if scratch.use_gpu_sampler {
            if let Some(embed_q8) = weights.embed_q8() {
                use crate::metal::batch::begin_engine_batch;
                use crate::metal::lm_head::lm_head_tied_q8_gpu_buf;
                let total = seq_len * vocab;
                scratch
                    .sampler
                    .logits
                    .ensure(&engine.ctx.device, &mut engine.pool, total)?;
                let telemetry = engine.batch_telemetry();
                let mut batch = begin_engine_batch(
                    &engine.ctx.queue,
                    &mut engine.pool,
                    &engine.ctx.device,
                    telemetry,
                )?;
                lm_head_tied_q8_gpu_buf(
                    &mut batch,
                    &engine.f32_q8_linear_pipeline,
                    &engine.sampler_kernels,
                    &engine.kernels,
                    norm_out,
                    embed_q8,
                    &mut scratch.sampler.logits,
                    seq_len,
                    hidden,
                    vocab,
                )?;
                batch.end()?;
            } else {
                return Err(Error::Format("gpu sampler requires .dgq embed"));
            }
        } else if let Some(out) = input.logits_out.as_mut() {
            let need = seq_len * vocab;
            if out.len() != need {
                return Err(Error::Format("logits_out length mismatch"));
            }
            if let Some(embed_q8) = weights.embed_q8() {
                use crate::metal::batch::begin_engine_batch;
                use crate::metal::lm_head::lm_head_tied_q8_gpu;
                let telemetry = engine.batch_telemetry();
                let mut batch = begin_engine_batch(
                    &engine.ctx.queue,
                    &mut engine.pool,
                    &engine.ctx.device,
                    telemetry,
                )?;
                lm_head_tied_q8_gpu(
                    &mut batch,
                    &engine.f32_q8_linear_pipeline,
                    norm_out,
                    embed_q8,
                    seq_len,
                    hidden,
                    vocab,
                    out,
                )?;
                batch.end()?;
            } else {
                lm_head_tied_from_store(
                    store,
                    out,
                    norm_out,
                    seq_len,
                    hidden,
                    vocab,
                    &mut scratch.cpu.lm_head_chunk,
                )?;
            }
            logit_softcapping(out, text.final_logit_softcapping as f32);
        } else {
            scratch.cpu.logits_buf.ensure_len(seq_len * vocab);
            if let Some(embed_q8) = weights.embed_q8() {
                use crate::metal::batch::begin_engine_batch;
                use crate::metal::lm_head::lm_head_tied_q8_gpu;
                let telemetry = engine.batch_telemetry();
                let mut batch = begin_engine_batch(
                    &engine.ctx.queue,
                    &mut engine.pool,
                    &engine.ctx.device,
                    telemetry,
                )?;
                lm_head_tied_q8_gpu(
                    &mut batch,
                    &engine.f32_q8_linear_pipeline,
                    norm_out,
                    embed_q8,
                    seq_len,
                    hidden,
                    vocab,
                    scratch.cpu.logits_buf.as_slice_mut(),
                )?;
                batch.end()?;
            } else {
                lm_head_tied_from_store(
                    store,
                    scratch.cpu.logits_buf.as_slice_mut(),
                    norm_out,
                    seq_len,
                    hidden,
                    vocab,
                    &mut scratch.cpu.lm_head_chunk,
                )?;
            }
            logit_softcapping(
                scratch.cpu.logits_buf.as_slice_mut(),
                text.final_logit_softcapping as f32,
            );
        }
    }

    let hidden_states = if input.return_hidden {
        let mut hidden_out = vec![0.0f32; seq_len * hidden];
        hidden_out.copy_from_slice(norm_out);
        hidden_out
    } else {
        Vec::new()
    };

    let logits = if input.compute_logits && input.logits_out.is_none() {
        scratch.cpu.logits_buf.as_slice().to_vec()
    } else {
        Vec::new()
    };

    Ok(DecoderForwardOutput {
        hidden_states,
        logits,
    })
}
