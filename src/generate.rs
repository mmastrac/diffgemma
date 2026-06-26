//! End-to-end block diffusion generation (CPU decoder; optional Metal GPU decoder).

use crate::config::ModelConfig;
use crate::model::decoder::{DecoderForwardInput, DecoderScratch};
use crate::model::encoder::extend_prefill;
use crate::model::encoder::{prefill, EncoderPrefillInput, EncoderScratch};
use crate::model::kv_cache::KvCache;
use crate::model::mask::DecoderAttnMask;
use crate::sample::{
    accept_canvas, apply_temperature, argmax_canvas, denoise_steps_completed, initialize_canvas,
    renoise_canvas, sample_canvas, Rng, SamplerConfig, StableConfidentStopper,
};
use crate::safetensors::Error;
use crate::weights::WeightStore;

#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub sampler: SamplerConfig,
    pub max_new_tokens: usize,
    pub seed: u64,
    /// Limit decoder layers (None = full stack). For smoke tests only.
    pub max_layers: Option<usize>,
    /// When true, run every denoise step (disable stable/confident early stop).
    pub no_early_stop: bool,
    /// Parity / golden tests: native Q4 kernels + CPU sampler (deterministic, slower).
    pub deterministic: bool,
    /// Optional label stored in denoise trace JSON.
    pub trace_prompt: Option<String>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            sampler: SamplerConfig::default(),
            max_new_tokens: 256,
            seed: 42,
            max_layers: None,
            no_early_stop: false,
            deterministic: false,
            trace_prompt: None,
        }
    }
}

pub struct GenerateOutput {
    pub token_ids: Vec<u32>,
    pub denoise_steps_run: usize,
    pub blocks_committed: usize,
    /// True when generation ended because a committed block emitted an
    /// end-of-turn / EOS token (full-message mode) rather than exhausting the
    /// `max_new_tokens` budget.
    pub stopped_on_eot: bool,
    /// Effective denoise steps per committed block (monolithic path).
    pub block_steps_eff: Vec<u32>,
    /// Accepted positions per step in the last committed block.
    pub last_block_accept_hist: Vec<u32>,
    /// Min per-position entropy (nats) each denoise step in the last block.
    pub last_block_min_entropy_hist: Vec<f32>,
    pub prefill_elapsed: std::time::Duration,
    pub denoise_elapsed: std::time::Duration,
    pub extend_elapsed: std::time::Duration,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub session_telemetry: crate::metal::SessionTelemetry,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub denoise_trace: Option<crate::denoise_trace::DenoiseTrace>,
}

enum DecoderBackend<'a> {
    Cpu {
        store: &'a WeightStore,
        cfg: &'a ModelConfig,
        scratch: &'a mut DecoderScratch,
    },
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Gpu {
        store: &'a WeightStore,
        cfg: &'a ModelConfig,
        scratch: &'a mut crate::metal::GpuDecoderScratch,
        weights: &'a mut crate::metal::GpuDecoderWeightCache,
        engine: &'a mut crate::metal::GpuDecoderEngine,
    },
}

fn decoder_forward(
    backend: &mut DecoderBackend<'_>,
    input: &mut DecoderForwardInput<'_>,
    max_layers: Option<usize>,
) -> Result<(), Error> {
    match backend {
        DecoderBackend::Cpu { store, cfg, scratch } => {
            crate::model::decoder::forward(store, cfg, input, scratch, max_layers)?;
            Ok(())
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        DecoderBackend::Gpu {
            store,
            cfg,
            scratch,
            weights,
            engine,
        } => {
            crate::metal::decoder_forward(store, cfg, input, scratch, weights, engine, max_layers)?;
            Ok(())
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_encoder_prefill(
    store: &WeightStore,
    cfg: &ModelConfig,
    prompt_token_ids: &[u32],
    enc_scratch: &mut EncoderScratch,
    decoder: &mut DecoderBackend<'_>,
    gen_cfg: &GenerateConfig,
    canvas_len: usize,
) -> Result<(KvCache, std::time::Duration), Error> {
    let max_encoder_kv = prompt_token_ids.len() + gen_cfg.max_new_tokens;
    let input = EncoderPrefillInput {
        token_ids: prompt_token_ids,
        position_offset: 0,
    };
    let started = std::time::Instant::now();
    let kv_cache = match decoder {
        DecoderBackend::Gpu {
            scratch,
            engine,
            store,
            cfg,
            ..
        } => {
            if store.is_quantized() {
                return Err(Error::Format(
                    ".dgq encoder prefill for generate-gpu is removed; use generate-monolithic",
                ));
            }
            let out = prefill(store, cfg, &input, enc_scratch)?;
            scratch.ensure_gpu_kv(
                &engine.ctx.device,
                &cfg.text_config,
                max_encoder_kv,
                canvas_len,
            )?;
            scratch.sync_gpu_kv_from_cpu(&out.kv_cache, canvas_len)?;
            out.kv_cache
        }
        DecoderBackend::Cpu { .. } => {
            let out = prefill(store, cfg, &input, enc_scratch)?;
            out.kv_cache
        }
    };
    Ok((kv_cache, started.elapsed()))
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_encoder_prefill(
    store: &WeightStore,
    cfg: &ModelConfig,
    prompt_token_ids: &[u32],
    enc_scratch: &mut EncoderScratch,
    _decoder: &mut DecoderBackend<'_>,
    _gen_cfg: &GenerateConfig,
    _canvas_len: usize,
) -> Result<(KvCache, std::time::Duration), Error> {
    let started = std::time::Instant::now();
    let out = prefill(
        store,
        cfg,
        &EncoderPrefillInput {
            token_ids: prompt_token_ids,
            position_offset: 0,
        },
        enc_scratch,
    )?;
    Ok((out.kv_cache, started.elapsed()))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn extend_encoder_kv(
    backend: &mut DecoderBackend<'_>,
    _store: &WeightStore,
    _cfg: &ModelConfig,
    kv_cache: &mut KvCache,
    token_ids: &[u32],
    enc_scratch: &mut EncoderScratch,
    _max_layers: Option<usize>,
) -> Result<(), Error> {
    match backend {
        DecoderBackend::Gpu {
            scratch,
            engine,
            store,
            cfg,
            ..
        } => {
            if store.is_quantized() {
                return Err(Error::Format(
                    ".dgq encoder extend for generate-gpu is removed; use generate-monolithic",
                ));
            }
            extend_prefill(store, cfg, kv_cache, token_ids, enc_scratch)?;
            scratch.ensure_gpu_kv(
                &engine.ctx.device,
                &cfg.text_config,
                kv_cache.kv_len,
                cfg.canvas_length,
            )?;
            scratch.sync_gpu_kv_from_cpu(kv_cache, cfg.canvas_length)
        }
        DecoderBackend::Cpu { store, cfg, .. } => {
            extend_prefill(store, cfg, kv_cache, token_ids, enc_scratch)
        }
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn extend_encoder_kv(
    backend: &mut DecoderBackend<'_>,
    store: &WeightStore,
    cfg: &ModelConfig,
    kv_cache: &mut KvCache,
    token_ids: &[u32],
    enc_scratch: &mut EncoderScratch,
    _max_layers: Option<usize>,
) -> Result<(), Error> {
    let _ = backend;
    extend_prefill(store, cfg, kv_cache, token_ids, enc_scratch)
}

fn generate_progress_enabled() -> bool {
    match std::env::var("DGQ_QUIET") {
        Ok(v) => v != "1" && !v.eq_ignore_ascii_case("true"),
        Err(_) => true,
    }
}

fn generate_inner(
    store: &WeightStore,
    cfg: &ModelConfig,
    prompt_token_ids: &[u32],
    gen_cfg: &GenerateConfig,
    enc_scratch: &mut EncoderScratch,
    decoder: &mut DecoderBackend<'_>,
) -> Result<GenerateOutput, Error> {
    let text = &cfg.text_config;
    let canvas_len = cfg.canvas_length;
    let vocab = text.vocab_size;
    let max_blocks = gen_cfg.max_new_tokens.div_ceil(canvas_len).max(1);

    let (kv_cache, prefill_elapsed) =
        run_encoder_prefill(store, cfg, prompt_token_ids, enc_scratch, decoder, gen_cfg, canvas_len)?;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if generate_progress_enabled() {
        eprintln!(
            "generate-gpu: encoder prefill finished ({prefill_elapsed:.2?}, kv_len={})",
            kv_cache.kv_len
        );
    }

    let mut sequences = prompt_token_ids.to_vec();
    let mut kv_cache = kv_cache;
    let mut rng = Rng::new(gen_cfg.seed);
    let mut denoise_steps_run = 0usize;
    let mut blocks_committed = 0usize;
    let mut denoise_elapsed = std::time::Duration::ZERO;
    let mut extend_elapsed = std::time::Duration::ZERO;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let mut session_telemetry = crate::metal::SessionTelemetry::default();
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let mut step_traces = Vec::new();
    let max_denoise_steps = gen_cfg.sampler.max_denoising_steps.max(1);

    for _block in 0..max_blocks {
        if sequences.len() >= prompt_token_ids.len() + gen_cfg.max_new_tokens {
            break;
        }

        let remaining = prompt_token_ids.len() + gen_cfg.max_new_tokens - sequences.len();
        let is_last_block = remaining <= canvas_len;

        let mut current_canvas = initialize_canvas(canvas_len, vocab, &mut rng);
        let mut argmax_canvas_tokens = current_canvas.clone();
        let mut finished = false;
        let mut have_sc_logits = false;

        let mut stopper = StableConfidentStopper::with_plateau(
            gen_cfg.sampler.stability_threshold,
            gen_cfg.sampler.confidence_threshold,
            gen_cfg.sampler.accept_plateau_threshold,
            gen_cfg.sampler.plateau_prefix_mean_max,
            1,
        );
        stopper.reset();

        let mask = DecoderAttnMask::all_valid(canvas_len, kv_cache.kv_len);
        let mut processed_logits = vec![0.0f32; canvas_len * vocab];
        let mut sample_logits = vec![0.0f32; canvas_len * vocab];
        let mut sc_logits = vec![0.0f32; canvas_len * vocab];

        let denoise_started = std::time::Instant::now();
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let block_idx = blocks_committed + 1;
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let mut block_step_count = 0u32;
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let DecoderBackend::Gpu { scratch, .. } = decoder {
            scratch.have_gpu_sc_logits = false;
        }
        for cur_step in (1..=gen_cfg.sampler.max_denoising_steps).rev() {
            if finished {
                break;
            }

            #[cfg(all(feature = "metal", target_os = "macos"))]
            let gpu_sampler = matches!(
                decoder,
                DecoderBackend::Gpu { scratch, .. } if scratch.use_gpu_sampler
            );

            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            let gpu_sampler = false;

            if std::env::var("DGQ_LOG_SC").ok().as_deref() == Some("1") {
                eprintln!(
                    "engine denoise: cur_step={cur_step} have_sc_logits={have_sc_logits} gpu_sampler={gpu_sampler}"
                );
            }

            let decoder_started = std::time::Instant::now();
            #[cfg(all(feature = "metal", target_os = "macos"))]
            let (step_finished, step_argmax, forward_telemetry, sampler_ms) = if gpu_sampler {
                let mut decoder_input = DecoderForwardInput {
                    token_ids: &current_canvas,
                    kv_cache: &kv_cache,
                    self_conditioning_logits: None,
                    mask: Some(&mask),
                    logits_out: None,
                    compute_logits: true,
                    return_hidden: false,
                };
                if let DecoderBackend::Gpu { engine, .. } = decoder {
                    engine.reset_forward_telemetry();
                }
                decoder_forward(decoder, &mut decoder_input, gen_cfg.max_layers)?;
                let forward_telemetry = if let DecoderBackend::Gpu { engine, .. } = decoder {
                    engine.take_forward_telemetry()
                } else {
                    crate::metal::ForwardTelemetry::default()
                };
                let sampler_started = std::time::Instant::now();
                let step = if let DecoderBackend::Gpu { scratch, engine, .. } = decoder {
                    crate::metal::sampler_step_gpu(
                        engine,
                        &mut scratch.sampler,
                        &mut current_canvas,
                        cur_step,
                        canvas_len,
                        vocab,
                        text.final_logit_softcapping as f32,
                        &gen_cfg.sampler,
                        &mut stopper,
                        &mut rng,
                    )?
                } else {
                    return Err(Error::Format("gpu sampler backend mismatch"));
                };
                if let DecoderBackend::Gpu { scratch, engine, .. } = decoder {
                    if engine.telemetry_enabled() {
                        engine
                            .telemetry_handle()
                            .borrow_mut()
                            .lm_head_logits_bytes += step.readback_bytes;
                    }
                    scratch.have_gpu_sc_logits = true;
                }
                let sampler_ms = sampler_started.elapsed().as_secs_f64() * 1000.0;
                (step.finished, step.argmax, forward_telemetry, sampler_ms)
            } else {
                let mut decoder_input = DecoderForwardInput {
                    token_ids: &current_canvas,
                    kv_cache: &kv_cache,
                    self_conditioning_logits: if have_sc_logits {
                        Some(sc_logits.as_slice())
                    } else {
                        None
                    },
                    mask: Some(&mask),
                    logits_out: Some(&mut processed_logits),
                    compute_logits: true,
                    return_hidden: false,
                };
                if let DecoderBackend::Gpu { engine, .. } = decoder {
                    engine.reset_forward_telemetry();
                }
                decoder_forward(decoder, &mut decoder_input, gen_cfg.max_layers)?;
                let forward_telemetry = match decoder {
                    DecoderBackend::Gpu { engine, .. } => engine.take_forward_telemetry(),
                    _ => crate::metal::ForwardTelemetry::default(),
                };
                let sampler_started = std::time::Instant::now();
                apply_temperature(&mut processed_logits, cur_step, &gen_cfg.sampler);
                sample_logits.copy_from_slice(&processed_logits);
                let denoiser_canvas =
                    sample_canvas(&mut sample_logits, canvas_len, vocab, &mut rng);
                let new_argmax = argmax_canvas(&processed_logits, canvas_len, vocab);
                let (accepted, accepted_mask) = accept_canvas(
                    &current_canvas,
                    &denoiser_canvas,
                    &processed_logits,
                    canvas_len,
                    vocab,
                    gen_cfg.sampler.entropy_bound,
                );
                let canvas_for_stop = current_canvas.clone();
                let accept_u32: Vec<u32> = accepted_mask.iter().map(|&m| u32::from(m)).collect();
                current_canvas = renoise_canvas(&accepted, &accepted_mask, vocab, &mut rng);
                let steps_done =
                    denoise_steps_completed(gen_cfg.sampler.max_denoising_steps, cur_step);
                let entropies =
                    crate::sample::token_entropy(&processed_logits, canvas_len, vocab);
                let step_finished = stopper.should_stop_with_entropies(
                    &new_argmax,
                    &entropies,
                    &canvas_for_stop,
                    &accept_u32,
                    steps_done,
                );
                sc_logits.copy_from_slice(&processed_logits);
                have_sc_logits = true;
                let sampler_ms = sampler_started.elapsed().as_secs_f64() * 1000.0;
                (step_finished, new_argmax, forward_telemetry, sampler_ms)
            };

            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            let (step_finished, step_argmax, forward_telemetry, sampler_ms) = {
                let mut decoder_input = DecoderForwardInput {
                    token_ids: &current_canvas,
                    kv_cache: &kv_cache,
                    self_conditioning_logits: if have_sc_logits {
                        Some(sc_logits.as_slice())
                    } else {
                        None
                    },
                    mask: Some(&mask),
                    logits_out: Some(&mut processed_logits),
                    compute_logits: true,
                    return_hidden: false,
                };
                let decoder_started = std::time::Instant::now();
                decoder_forward(decoder, &mut decoder_input, gen_cfg.max_layers)?;
                let _decoder_ms = decoder_started.elapsed().as_secs_f64() * 1000.0;
                let forward_telemetry = crate::metal::ForwardTelemetry::default();
                let sampler_started = std::time::Instant::now();
                apply_temperature(&mut processed_logits, cur_step, &gen_cfg.sampler);
                sample_logits.copy_from_slice(&processed_logits);
                let denoiser_canvas =
                    sample_canvas(&mut sample_logits, canvas_len, vocab, &mut rng);
                let new_argmax = argmax_canvas(&processed_logits, canvas_len, vocab);
                let (accepted, accepted_mask) = accept_canvas(
                    &current_canvas,
                    &denoiser_canvas,
                    &processed_logits,
                    canvas_len,
                    vocab,
                    gen_cfg.sampler.entropy_bound,
                );
                let canvas_for_stop = current_canvas.clone();
                let accept_u32: Vec<u32> = accepted_mask.iter().map(|&m| u32::from(m)).collect();
                current_canvas = renoise_canvas(&accepted, &accepted_mask, vocab, &mut rng);
                let steps_done =
                    denoise_steps_completed(gen_cfg.sampler.max_denoising_steps, cur_step);
                let entropies =
                    crate::sample::token_entropy(&processed_logits, canvas_len, vocab);
                let step_finished = stopper.should_stop_with_entropies(
                    &new_argmax,
                    &entropies,
                    &canvas_for_stop,
                    &accept_u32,
                    steps_done,
                );
                sc_logits.copy_from_slice(&processed_logits);
                have_sc_logits = true;
                let sampler_ms = sampler_started.elapsed().as_secs_f64() * 1000.0;
                (step_finished, new_argmax, forward_telemetry, sampler_ms)
            };

            finished = step_finished;
            argmax_canvas_tokens = step_argmax;
            let decoder_ms = decoder_started.elapsed().as_secs_f64() * 1000.0;
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                use crate::denoise_trace::step_trace_from_stats;
                use crate::sample::{accept_mask_from_entropies, step_entropy_stats, token_entropy};
                block_step_count += 1;
                let (stats, argmax_for_trace, entropy_for_trace) = if gpu_sampler {
                    let DecoderBackend::Gpu { scratch, .. } = decoder else {
                        return Err(Error::Format("gpu sampler backend mismatch"));
                    };
                    let mask = accept_mask_from_entropies(
                        &scratch.sampler.entropies,
                        gen_cfg.sampler.entropy_bound,
                    );
                    let accept_u32: Vec<u32> = mask.iter().map(|&b| u32::from(b)).collect();
                    (
                        step_entropy_stats(&scratch.sampler.entropies, &accept_u32),
                        scratch.sampler.argmax.clone(),
                        if crate::metal::trace_entropy_enabled() {
                            Some(scratch.sampler.entropies.clone())
                        } else {
                            None
                        },
                    )
                } else {
                    let ent = token_entropy(&processed_logits, canvas_len, vocab);
                    let mask =
                        accept_mask_from_entropies(&ent, gen_cfg.sampler.entropy_bound);
                    let accept_u32: Vec<u32> = mask.iter().map(|&b| u32::from(b)).collect();
                    (
                        step_entropy_stats(&ent, &accept_u32),
                        argmax_canvas(&processed_logits, canvas_len, vocab),
                        if crate::metal::trace_entropy_enabled() {
                            Some(ent)
                        } else {
                            None
                        },
                    )
                };
                step_traces.push(step_trace_from_stats(
                    block_idx as u32,
                    block_step_count,
                    max_denoise_steps,
                    &stats,
                    &argmax_for_trace,
                    entropy_for_trace.as_deref(),
                    step_finished,
                ));
            }
            #[cfg(all(feature = "metal", target_os = "macos"))]
            if matches!(decoder, DecoderBackend::Gpu { .. }) {
                session_telemetry.steps.push(crate::metal::StepPhaseTelemetry {
                    decoder_ms,
                    sampler_ms,
                    forward: forward_telemetry,
                });
            }
            denoise_steps_run += 1;
            #[cfg(all(feature = "metal", target_os = "macos"))]
            if generate_progress_enabled() {
                let step_idx = max_denoise_steps - cur_step + 1;
                eprintln!(
                    "generate-gpu: block {block_idx} step {step_idx}/{max_denoise_steps} cur_step={cur_step} forward={decoder_ms:.0}ms sampler={sampler_ms:.0}ms stop={finished}"
                );
            }
        }
        denoise_elapsed += denoise_started.elapsed();

        sequences.extend_from_slice(&argmax_canvas_tokens);
        if !is_last_block {
            let extend_started = std::time::Instant::now();
            extend_encoder_kv(
                decoder,
                store,
                cfg,
                &mut kv_cache,
                &argmax_canvas_tokens,
                enc_scratch,
                gen_cfg.max_layers,
            )?;
            extend_elapsed += extend_started.elapsed();
        }
        blocks_committed += 1;
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    let (block_steps_eff, last_block_accept_hist, last_block_min_entropy_hist, denoise_trace) = {
        use crate::denoise_trace::{DenoiseTrace, SCHEMA_VERSION};
        let last_block = blocks_committed.max(1) as u32;
        let block_traces: Vec<_> = step_traces
            .iter()
            .filter(|s| s.block == last_block)
            .collect();
        let last_block_accept_hist = block_traces.iter().map(|s| s.accept_count).collect();
        let last_block_min_entropy_hist = block_traces.iter().map(|s| s.min_entropy).collect();
        let block_steps_eff = vec![block_traces.len() as u32];
        let layers = gen_cfg
            .max_layers
            .unwrap_or(cfg.text_config.num_hidden_layers);
        let trace = DenoiseTrace {
            schema_version: SCHEMA_VERSION,
            source: "rust-engine".into(),
            prompt: gen_cfg.trace_prompt.clone().unwrap_or_default(),
            prompt_token_ids: prompt_token_ids.to_vec(),
            seed: gen_cfg.seed,
            max_denoise_steps,
            layers,
            max_new_tokens: gen_cfg.max_new_tokens,
            weights_profile: Some(if store.is_quantized() {
                "dgq_q4".into()
            } else {
                "safetensors".into()
            }),
            entropy_bound: gen_cfg.sampler.entropy_bound,
            step_traces: step_traces.clone(),
            denoise_steps_run,
            blocks_committed,
            output_token_ids: sequences.clone(),
            initial_canvas_ids: None,
            canvas_rng: Some("rust-lcg".into()),
        };
        (
            block_steps_eff,
            last_block_accept_hist,
            last_block_min_entropy_hist,
            Some(trace),
        )
    };

    Ok(GenerateOutput {
        token_ids: sequences,
        denoise_steps_run,
        blocks_committed,
        stopped_on_eot: false,
        block_steps_eff: {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                block_steps_eff
            }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            {
                Vec::new()
            }
        },
        last_block_accept_hist: {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                last_block_accept_hist
            }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            {
                Vec::new()
            }
        },
        last_block_min_entropy_hist: {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                last_block_min_entropy_hist
            }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            {
                Vec::new()
            }
        },
        prefill_elapsed,
        denoise_elapsed,
        extend_elapsed,
        #[cfg(all(feature = "metal", target_os = "macos"))]
        session_telemetry,
        #[cfg(all(feature = "metal", target_os = "macos"))]
        denoise_trace,
    })
}

pub fn generate(
    store: &WeightStore,
    cfg: &ModelConfig,
    prompt_token_ids: &[u32],
    gen_cfg: &GenerateConfig,
    enc_scratch: &mut EncoderScratch,
    dec_scratch: &mut DecoderScratch,
) -> Result<GenerateOutput, Error> {
    let mut decoder = DecoderBackend::Cpu {
        store,
        cfg,
        scratch: dec_scratch,
    };
    generate_inner(store, cfg, prompt_token_ids, gen_cfg, enc_scratch, &mut decoder)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn generate_gpu(
    store: &WeightStore,
    cfg: &ModelConfig,
    prompt_token_ids: &[u32],
    gen_cfg: &GenerateConfig,
    enc_scratch: &mut EncoderScratch,
    dec_scratch: &mut crate::metal::GpuDecoderScratch,
    weights: &mut crate::metal::GpuDecoderWeightCache,
    engine: &mut crate::metal::GpuDecoderEngine,
) -> Result<GenerateOutput, Error> {
    dec_scratch.use_gpu_sampler = weights.is_dgq() && !gen_cfg.deterministic;
    engine.pool.clear();
    let mut decoder = DecoderBackend::Gpu {
        store,
        cfg,
        scratch: dec_scratch,
        weights,
        engine,
    };
    generate_inner(store, cfg, prompt_token_ids, gen_cfg, enc_scratch, &mut decoder)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn generate_monolithic_gpu(
    model_dir: &std::path::Path,
    prompt_token_ids: &[u32],
    gen_cfg: &GenerateConfig,
    max_seq: usize,
    prompt_label: &str,
) -> Result<GenerateOutput, Error> {
    use crate::metal::{generate_monolithic, validate_step_model, StepGenerateConfig};
    let validated = validate_step_model(model_dir)?;
    let layers = gen_cfg
        .max_layers
        .unwrap_or(validated.num_layers)
        .min(validated.num_layers);
    let cfg = StepGenerateConfig::from_generate(
        gen_cfg.seed,
        gen_cfg.max_new_tokens,
        max_seq,
        layers,
        gen_cfg.sampler.clone(),
        gen_cfg.no_early_stop,
    );
    generate_monolithic(model_dir, prompt_token_ids, &cfg, prompt_label)
}

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod gpu_determinism {
    use super::*;
    use crate::metal::{
        decoder_forward, load_weight_cache, prefill_gpu, GpuDecoderEngine, GpuDecoderScratch,
    };
    use crate::model::decoder::DecoderForwardInput;
    use crate::model::encoder::{EncoderPrefillInput, EncoderScratch};
    use crate::model::mask::DecoderAttnMask;
    use crate::sample::{argmax_canvas, initialize_canvas, Rng};
    use crate::weights::WeightStore;

    const DGQ_HELLO_PROMPT: [u32; 1] = [9259];
    const DGQ_MAX_LAYERS: usize = 3;

    fn dgq_fixture_dir() -> Option<std::path::PathBuf> {
        let dir = std::path::Path::new("/tmp/quantized-weights");
        if dir.join("model.dgq.json").exists() {
            Some(dir.to_path_buf())
        } else {
            None
        }
    }

    fn hello_gen_cfg() -> GenerateConfig {
        GenerateConfig {
            sampler: crate::sample::sampler_for_steps(1, false),
            max_new_tokens: 256,
            seed: 42,
            max_layers: Some(DGQ_MAX_LAYERS),
            no_early_stop: false,
            deterministic: true,
            trace_prompt: None,
        }
    }

    fn forward_logits_once(
        store: &WeightStore,
        cfg: &ModelConfig,
        weights: &crate::metal::GpuDecoderWeightCache,
        engine: &mut GpuDecoderEngine,
        dec: &mut GpuDecoderScratch,
        canvas_tokens: &[u32],
        kv_cache: &KvCache,
        max_layers: usize,
    ) -> Result<Vec<f32>, Error> {
        dec.use_gpu_sampler = false;
        let canvas = cfg.canvas_length;
        let vocab = cfg.text_config.vocab_size;
        let mut logits = vec![0.0f32; canvas * vocab];
        let mask = DecoderAttnMask::all_valid(canvas, kv_cache.kv_len);
        let mut input = DecoderForwardInput {
            token_ids: canvas_tokens,
            kv_cache,
            self_conditioning_logits: None,
            mask: Some(&mask),
            logits_out: Some(&mut logits),
            compute_logits: true,
            return_hidden: false,
        };
        decoder_forward(
            store,
            cfg,
            &mut input,
            dec,
            weights,
            engine,
            Some(max_layers),
        )?;
        Ok(logits)
    }

    fn prefill_and_canvas(
        store: &WeightStore,
        cfg: &ModelConfig,
        weights: &mut crate::metal::GpuDecoderWeightCache,
        engine: &mut GpuDecoderEngine,
        enc: &mut EncoderScratch,
        dec: &mut GpuDecoderScratch,
        max_kv: usize,
    ) -> Result<(KvCache, Vec<u32>), Error> {
        let canvas = cfg.canvas_length;
        let vocab = cfg.text_config.vocab_size;
        let input = EncoderPrefillInput {
            token_ids: &DGQ_HELLO_PROMPT,
            position_offset: 0,
        };
        dec.use_gpu_sampler = false;
        let kv = prefill_gpu(
            store,
            cfg,
            &input,
            enc,
            dec,
            weights,
            engine,
            max_kv,
            canvas,
            Some(DGQ_MAX_LAYERS),
        )?;
        let mut rng = Rng::new(42);
        let canvas_tokens = initialize_canvas(canvas, vocab, &mut rng);
        Ok((kv, canvas_tokens))
    }

    fn first_logit_diff(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
        a.iter()
            .zip(b.iter())
            .enumerate()
            .find_map(|(i, (&x, &y))| if x.to_bits() != y.to_bits() { Some((i, x, y)) } else { None })
    }

    /// Same KV + canvas: two decoder forwards back-to-back must match bit-for-bit.
    #[test]
    fn dgq_forward_logits_same_inputs_twice() {
        let Some(dgq_dir) = dgq_fixture_dir() else {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        };
        let store = WeightStore::open(&dgq_dir).expect("open");
        let cfg = crate::config::ModelConfig::load(&dgq_dir).expect("cfg");
        let canvas = cfg.canvas_length;
        let max_kv = DGQ_HELLO_PROMPT.len() + 256;

        let mut weights = load_weight_cache(&store, &cfg.text_config, canvas, max_kv).expect("cache");
        let mut engine = GpuDecoderEngine::new().expect("engine");
        let mut enc = EncoderScratch::new(canvas.max(1), &cfg);
        let mut dec = GpuDecoderScratch::new(canvas, &cfg);
        let (kv, canvas_tokens) =
            prefill_and_canvas(&store, &cfg, &mut weights, &mut engine, &mut enc, &mut dec, max_kv)
                .expect("prefill");

        let a = forward_logits_once(
            &store,
            &cfg,
            &weights,
            &mut engine,
            &mut dec,
            &canvas_tokens,
            &kv,
            DGQ_MAX_LAYERS,
        )
        .expect("forward a");
        let b = forward_logits_once(
            &store,
            &cfg,
            &weights,
            &mut engine,
            &mut dec,
            &canvas_tokens,
            &kv,
            DGQ_MAX_LAYERS,
        )
        .expect("forward b");
        if a != b {
            if let Some((idx, x, y)) = first_logit_diff(&a, &b) {
                panic!("forward drift on repeat: flat idx {idx}: {x} vs {y}");
            }
        }
    }

    /// Isolate engine pool: second forward uses a fresh engine (same dec/kv/canvas).
    #[test]
    fn dgq_forward_logits_fresh_engine_second_pass() {
        let Some(dgq_dir) = dgq_fixture_dir() else {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        };
        let store = WeightStore::open(&dgq_dir).expect("open");
        let cfg = crate::config::ModelConfig::load(&dgq_dir).expect("cfg");
        let canvas = cfg.canvas_length;
        let max_kv = DGQ_HELLO_PROMPT.len() + 256;

        let mut weights = load_weight_cache(&store, &cfg.text_config, canvas, max_kv).expect("cache");
        let mut engine_a = GpuDecoderEngine::new().expect("engine a");
        let mut enc = EncoderScratch::new(canvas.max(1), &cfg);
        let mut dec = GpuDecoderScratch::new(canvas, &cfg);
        let (kv, canvas_tokens) = prefill_and_canvas(
            &store,
            &cfg,
            &mut weights,
            &mut engine_a,
            &mut enc,
            &mut dec,
            max_kv,
        )
        .expect("prefill");

        let a = forward_logits_once(
            &store,
            &cfg,
            &weights,
            &mut engine_a,
            &mut dec,
            &canvas_tokens,
            &kv,
            DGQ_MAX_LAYERS,
        )
        .expect("forward a");

        let mut engine_b = GpuDecoderEngine::new().expect("engine b");
        let b = forward_logits_once(
            &store,
            &cfg,
            &weights,
            &mut engine_b,
            &mut dec,
            &canvas_tokens,
            &kv,
            DGQ_MAX_LAYERS,
        )
        .expect("forward b");

        if a != b {
            if let Some((idx, x, y)) = first_logit_diff(&a, &b) {
                panic!("forward drift with fresh engine: flat idx {idx}: {x} vs {y}");
            }
        }
    }

    /// Second pass repeats prefill (fresh gpu_kv) — if this passes, gpu_kv canvas corruption is the bug.
    #[test]
    fn dgq_forward_logits_fresh_prefill_second_pass() {
        let Some(dgq_dir) = dgq_fixture_dir() else {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        };
        let store = WeightStore::open(&dgq_dir).expect("open");
        let cfg = crate::config::ModelConfig::load(&dgq_dir).expect("cfg");
        let canvas = cfg.canvas_length;
        let max_kv = DGQ_HELLO_PROMPT.len() + 256;

        let mut weights = load_weight_cache(&store, &cfg.text_config, canvas, max_kv).expect("cache");

        let mut engine = GpuDecoderEngine::new().expect("engine");
        let mut enc = EncoderScratch::new(canvas.max(1), &cfg);
        let mut dec = GpuDecoderScratch::new(canvas, &cfg);
        let (kv, canvas_tokens) =
            prefill_and_canvas(&store, &cfg, &mut weights, &mut engine, &mut enc, &mut dec, max_kv)
                .expect("prefill a");
        let a = forward_logits_once(
            &store,
            &cfg,
            &weights,
            &mut engine,
            &mut dec,
            &canvas_tokens,
            &kv,
            DGQ_MAX_LAYERS,
        )
        .expect("forward a");

        let mut enc = EncoderScratch::new(canvas.max(1), &cfg);
        let mut dec = GpuDecoderScratch::new(canvas, &cfg);
        let (kv, canvas_tokens) =
            prefill_and_canvas(&store, &cfg, &mut weights, &mut engine, &mut enc, &mut dec, max_kv)
                .expect("prefill b");
        let b = forward_logits_once(
            &store,
            &cfg,
            &weights,
            &mut engine,
            &mut dec,
            &canvas_tokens,
            &kv,
            DGQ_MAX_LAYERS,
        )
        .expect("forward b");

        if a != b {
            if let Some((idx, x, y)) = first_logit_diff(&a, &b) {
                panic!("forward drift with fresh prefill: flat idx {idx}: {x} vs {y}");
            }
        }
    }

    /// Fully isolated second chain (fresh weights + engine + scratch).
    #[test]
    fn dgq_forward_logits_fully_isolated_chains() {
        let Some(dgq_dir) = dgq_fixture_dir() else {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        };
        let store = WeightStore::open(&dgq_dir).expect("open");
        let cfg = crate::config::ModelConfig::load(&dgq_dir).expect("cfg");
        let canvas = cfg.canvas_length;
        let max_kv = DGQ_HELLO_PROMPT.len() + 256;

        let run_chain = || -> Vec<f32> {
            let mut weights =
                load_weight_cache(&store, &cfg.text_config, canvas, max_kv).expect("cache");
            let mut engine = GpuDecoderEngine::new().expect("engine");
            let mut enc = EncoderScratch::new(canvas.max(1), &cfg);
            let mut dec = GpuDecoderScratch::new(canvas, &cfg);
            let (kv, canvas_tokens) = prefill_and_canvas(
                &store,
                &cfg,
                &mut weights,
                &mut engine,
                &mut enc,
                &mut dec,
                max_kv,
            )
            .expect("prefill");
            forward_logits_once(
                &store,
                &cfg,
                &weights,
                &mut engine,
                &mut dec,
                &canvas_tokens,
                &kv,
                DGQ_MAX_LAYERS,
            )
            .expect("forward")
        };

        let a = run_chain();
        let b = run_chain();
        if a != b {
            if let Some((idx, x, y)) = first_logit_diff(&a, &b) {
                panic!("forward drift isolated chains: flat idx {idx}: {x} vs {y}");
            }
        }
    }

    fn drift_survey_layers(layers: usize, trials: usize) -> std::collections::HashSet<u32> {
        let dgq_dir = dgq_fixture_dir().expect("dgq dir");
        let store = WeightStore::open(&dgq_dir).expect("open");
        let cfg = crate::config::ModelConfig::load(&dgq_dir).expect("cfg");
        let canvas = cfg.canvas_length;
        let vocab = cfg.text_config.vocab_size;
        let max_kv = DGQ_HELLO_PROMPT.len() + 256;
        let mut argmax_samples = std::collections::HashSet::new();
        for trial in 0..trials {
            let mut weights =
                load_weight_cache(&store, &cfg.text_config, canvas, max_kv).expect("cache");
            let mut engine = GpuDecoderEngine::new().expect("engine");
            let mut enc = EncoderScratch::new(canvas.max(1), &cfg);
            let mut dec = GpuDecoderScratch::new(canvas, &cfg);
                dec.use_gpu_sampler = false;
            let input = EncoderPrefillInput {
                token_ids: &DGQ_HELLO_PROMPT,
                position_offset: 0,
            };
            let kv = prefill_gpu(
                &store,
                &cfg,
                &input,
                &mut enc,
                &mut dec,
                &mut weights,
                &mut engine,
                max_kv,
                canvas,
                Some(layers),
            )
            .expect("prefill");
            let mut rng = Rng::new(42);
            let canvas_tokens = initialize_canvas(canvas, vocab, &mut rng);
            let logits = forward_logits_once(
                &store,
                &cfg,
                &weights,
                &mut engine,
                &mut dec,
                &canvas_tokens,
                &kv,
                layers,
            )
            .expect("forward");
            let nan_n = logits.iter().filter(|v| v.is_nan()).count();
            let argmax = argmax_canvas(&logits, canvas, vocab);
            eprintln!(
                "{layers}-layer trial {trial}: nan={nan_n} pos1_argmax={} logit0={}",
                argmax[1],
                logits[0]
            );
            argmax_samples.insert(argmax[1]);
        }
        argmax_samples
    }

    #[test]
    fn dgq_drift_prefill_vs_decoder_layers() {
        if dgq_fixture_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let dgq_dir = dgq_fixture_dir().unwrap();
        let store = WeightStore::open(&dgq_dir).expect("open");
        let cfg = crate::config::ModelConfig::load(&dgq_dir).expect("cfg");
        let canvas = cfg.canvas_length;
        let vocab = cfg.text_config.vocab_size;
        let max_kv = DGQ_HELLO_PROMPT.len() + 256;

        let run = |prefill_layers: usize, decoder_layers: usize| -> u32 {
            let mut weights =
                load_weight_cache(&store, &cfg.text_config, canvas, max_kv).expect("cache");
            let mut engine = GpuDecoderEngine::new().expect("engine");
            let mut enc = EncoderScratch::new(canvas.max(1), &cfg);
            let mut dec = GpuDecoderScratch::new(canvas, &cfg);
                dec.use_gpu_sampler = false;
            let input = EncoderPrefillInput {
                token_ids: &DGQ_HELLO_PROMPT,
                position_offset: 0,
            };
            let kv = prefill_gpu(
                &store,
                &cfg,
                &input,
                &mut enc,
                &mut dec,
                &mut weights,
                &mut engine,
                max_kv,
                canvas,
                Some(prefill_layers),
            )
            .expect("prefill");
            let mut rng = Rng::new(42);
            let canvas_tokens = initialize_canvas(canvas, vocab, &mut rng);
            let logits = forward_logits_once(
                &store,
                &cfg,
                &weights,
                &mut engine,
                &mut dec,
                &canvas_tokens,
                &kv,
                decoder_layers,
            )
            .expect("forward");
            argmax_canvas(&logits, canvas, vocab)[1]
        };

        let mut a = std::collections::HashSet::new();
        let mut b = std::collections::HashSet::new();
        let mut c = std::collections::HashSet::new();
        let mut d = std::collections::HashSet::new();
        let mut e = std::collections::HashSet::new();
        let mut f = std::collections::HashSet::new();
        for _ in 0..8 {
            a.insert(run(1, 1));
            b.insert(run(2, 1));
            c.insert(run(2, 2));
            d.insert(run(3, 1));
            e.insert(run(3, 2));
            f.insert(run(3, 3));
        }
        eprintln!(
            "prefill1/dec1={} prefill2/dec1={} prefill2/dec2={} prefill3/dec1={} prefill3/dec2={} prefill3/dec3={}",
            a.len(),
            b.len(),
            c.len(),
            d.len(),
            e.len(),
            f.len()
        );
        assert_eq!(a.len(), 1, "1/1 drift");
        assert_eq!(b.len(), 1, "2/1 drift");
        assert_eq!(c.len(), 1, "2/2 drift");
        assert_eq!(d.len(), 1, "3/1 drift");
        assert_eq!(e.len(), 1, "3/2 drift");
        assert_eq!(f.len(), 1, "3/3 drift");
    }

    #[test]
    fn dgq_drift_survey_two_layers() {
        if dgq_fixture_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let unique = drift_survey_layers(2, 8);
        eprintln!("2-layer unique pos1 argmax: {}", unique.len());
        assert_eq!(unique.len(), 1, "2-layer drift: {unique:?}");
    }

    #[test]
    fn dgq_drift_survey_one_layer() {
        if dgq_fixture_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let unique = drift_survey_layers(1, 8);
        eprintln!("1-layer unique pos1 argmax: {}", unique.len());
        assert_eq!(unique.len(), 1, "1-layer drift: {unique:?}");
    }

    #[test]
    fn dgq_drift_survey_three_layers() {
        if dgq_fixture_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let unique = drift_survey_layers(3, 8);
        eprintln!("3-layer unique pos1 argmax: {}", unique.len());
        assert_eq!(unique.len(), 1, "3-layer drift: {unique:?}");
    }

    #[test]
    #[ignore = "3-layer stack drift survey; use dgq_drift_survey_three_layers"]
    fn dgq_drift_survey() {
        let Some(dgq_dir) = dgq_fixture_dir() else {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        };
        let store = WeightStore::open(&dgq_dir).expect("open");
        let cfg = crate::config::ModelConfig::load(&dgq_dir).expect("cfg");
        let canvas = cfg.canvas_length;
        let vocab = cfg.text_config.vocab_size;
        let max_kv = DGQ_HELLO_PROMPT.len() + 256;

        let mut argmax_samples = Vec::new();
        for trial in 0..8 {
            let mut weights =
                load_weight_cache(&store, &cfg.text_config, canvas, max_kv).expect("cache");
            let mut engine = GpuDecoderEngine::new().expect("engine");
            let mut enc = EncoderScratch::new(canvas.max(1), &cfg);
            let mut dec = GpuDecoderScratch::new(canvas, &cfg);
            let (kv, canvas_tokens) = prefill_and_canvas(
                &store,
                &cfg,
                &mut weights,
                &mut engine,
                &mut enc,
                &mut dec,
                max_kv,
            )
            .expect("prefill");
            let logits = forward_logits_once(
                &store,
                &cfg,
                &weights,
                &mut engine,
                &mut dec,
                &canvas_tokens,
                &kv,
                DGQ_MAX_LAYERS,
            )
            .expect("forward");
            let nan_n = logits.iter().filter(|v| v.is_nan()).count();
            let argmax = argmax_canvas(&logits, canvas, vocab);
            eprintln!(
                "trial {trial}: nan={nan_n} canvas[0..3]={:?} pos1_argmax={} logit0={}",
                &canvas_tokens[0..3],
                argmax[1],
                logits[0]
            );
            argmax_samples.push(argmax[1]);
        }
        let unique: std::collections::HashSet<_> = argmax_samples.iter().copied().collect();
        eprintln!("unique pos1 argmax values in 8 trials: {}", unique.len());
        assert_eq!(
            unique.len(),
            1,
            "expected deterministic pos1 argmax, got {unique:?}"
        );
    }

    /// Full prefill + forward repeated in-process (reused engine pool).
    #[test]
    fn dgq_forward_logits_repeatable_with_reused_engine() {
        let Some(dgq_dir) = dgq_fixture_dir() else {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        };
        let store = WeightStore::open(&dgq_dir).expect("open");
        let cfg = crate::config::ModelConfig::load(&dgq_dir).expect("cfg");
        let canvas = cfg.canvas_length;
        let vocab = cfg.text_config.vocab_size;
        let max_kv = DGQ_HELLO_PROMPT.len() + 256;

        let mut weights = load_weight_cache(&store, &cfg.text_config, canvas, max_kv).expect("cache");
        let mut engine = GpuDecoderEngine::new().expect("engine");

        let mut enc = EncoderScratch::new(canvas.max(1), &cfg);
        let mut dec = GpuDecoderScratch::new(canvas, &cfg);
        let (kv, canvas_tokens) =
            prefill_and_canvas(&store, &cfg, &mut weights, &mut engine, &mut enc, &mut dec, max_kv)
                .expect("prefill baseline");
        let baseline = forward_logits_once(
            &store,
            &cfg,
            &weights,
            &mut engine,
            &mut dec,
            &canvas_tokens,
            &kv,
            DGQ_MAX_LAYERS,
        )
        .expect("baseline forward");
        let argmax1 = argmax_canvas(&baseline, canvas, vocab)[1];

        for trial in 1..=20 {
            let mut enc = EncoderScratch::new(canvas.max(1), &cfg);
            let mut dec = GpuDecoderScratch::new(canvas, &cfg);
            let (kv_t, canvas_t) = prefill_and_canvas(
                &store,
                &cfg,
                &mut weights,
                &mut engine,
                &mut enc,
                &mut dec,
                max_kv,
            )
            .expect("prefill");
            assert_eq!(canvas_t, canvas_tokens, "canvas init drift at trial {trial}");
            let logits = forward_logits_once(
                &store,
                &cfg,
                &weights,
                &mut engine,
                &mut dec,
                &canvas_t,
                &kv_t,
                DGQ_MAX_LAYERS,
            )
            .expect("forward");
            if logits != baseline {
                let argmax_t = argmax_canvas(&logits, canvas, vocab)[1];
                let detail = first_logit_diff(&baseline, &logits)
                    .map(|(i, x, y)| format!("flat idx {i}: {x} vs {y}"))
                    .unwrap_or_default();
                panic!(
                    "forward drift trial {trial}: pos1 argmax {argmax1} vs {argmax_t}; {detail}"
                );
            }
        }
    }

    /// Monolithic generate should be repeatable in deterministic mode.
    #[test]
    fn dgq_generate_stable_deterministic_mode() {
        let Some(dgq_dir) = dgq_fixture_dir() else {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        };
        let prompt = DGQ_HELLO_PROMPT.to_vec();
        let gen_cfg = hello_gen_cfg();
        let max_seq = prompt.len() + gen_cfg.max_new_tokens;
        let out_a = generate_monolithic_gpu(&dgq_dir, &prompt, &gen_cfg, max_seq, "hello")
            .expect("generate a");
        let out_b = generate_monolithic_gpu(&dgq_dir, &prompt, &gen_cfg, max_seq, "hello")
            .expect("generate b");
        if out_a.token_ids != out_b.token_ids {
            let idx = out_a
                .token_ids
                .iter()
                .zip(out_b.token_ids.iter())
                .position(|(a, b)| a != b);
            panic!(
                "token drift at index {idx:?}: a={:?} b={:?}",
                out_a.token_ids.get(1),
                out_b.token_ids.get(1)
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_config_defaults_match_model_card() {
        let cfg = GenerateConfig::default();
        assert_eq!(cfg.sampler.entropy_bound, 0.1);
        assert_eq!(cfg.sampler.max_denoising_steps, 48);
        assert!((cfg.sampler.t_max - 0.8).abs() < 1e-6);
        assert!((cfg.sampler.t_min - 0.4).abs() < 1e-6);
        assert_eq!(cfg.sampler.confidence_threshold, 0.005);
    }
}
