//! End-to-end block diffusion generation (CPU decoder; optional Metal GPU decoder).

use crate::config::ModelConfig;
use crate::model::decoder::{DecoderForwardInput, DecoderForwardOutput, DecoderScratch};
use crate::model::encoder::extend_prefill;
use crate::model::encoder::{prefill, EncoderPrefillInput, EncoderScratch};
use crate::model::mask::DecoderAttnMask;
use crate::sample::{
    accept_canvas, apply_temperature, argmax_canvas, initialize_canvas, renoise_canvas,
    sample_canvas, Rng, SamplerConfig, StableConfidentStopper,
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
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            sampler: SamplerConfig::default(),
            max_new_tokens: 256,
            seed: 42,
            max_layers: None,
        }
    }
}

pub struct GenerateOutput {
    pub token_ids: Vec<u32>,
    pub denoise_steps_run: usize,
    pub blocks_committed: usize,
    pub prefill_elapsed: std::time::Duration,
    pub denoise_elapsed: std::time::Duration,
    pub extend_elapsed: std::time::Duration,
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
    input: &DecoderForwardInput<'_>,
    max_layers: Option<usize>,
) -> Result<DecoderForwardOutput, Error> {
    match backend {
        DecoderBackend::Cpu { store, cfg, scratch } => {
            crate::model::decoder::forward(store, cfg, input, scratch, max_layers)
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        DecoderBackend::Gpu {
            store,
            cfg,
            scratch,
            weights,
            engine,
        } => crate::metal::decoder_forward(store, cfg, input, scratch, weights, engine, max_layers),
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

    let prefill_started = std::time::Instant::now();
    let prefill_out = prefill(
        store,
        cfg,
        &EncoderPrefillInput {
            token_ids: prompt_token_ids,
            position_offset: 0,
        },
        enc_scratch,
    )?;
    let prefill_elapsed = prefill_started.elapsed();

    let mut sequences = prompt_token_ids.to_vec();
    let mut kv_cache = prefill_out.kv_cache;
    let mut rng = Rng::new(gen_cfg.seed);
    let mut denoise_steps_run = 0usize;
    let mut blocks_committed = 0usize;
    let mut denoise_elapsed = std::time::Duration::ZERO;
    let mut extend_elapsed = std::time::Duration::ZERO;

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

        let mut stopper = StableConfidentStopper::new(
            gen_cfg.sampler.stability_threshold,
            gen_cfg.sampler.confidence_threshold,
        );
        stopper.reset();

        let mask = DecoderAttnMask::all_valid(canvas_len, kv_cache.kv_len);
        let mut processed_logits = vec![0.0f32; canvas_len * vocab];
        let mut sample_logits = vec![0.0f32; canvas_len * vocab];
        let mut sc_logits = vec![0.0f32; canvas_len * vocab];

        let denoise_started = std::time::Instant::now();
        for cur_step in (1..=gen_cfg.sampler.max_denoising_steps).rev() {
            if finished {
                break;
            }

            let decoder_input = DecoderForwardInput {
                token_ids: &current_canvas,
                kv_cache: &kv_cache,
                self_conditioning_logits: if have_sc_logits {
                    Some(sc_logits.as_slice())
                } else {
                    None
                },
                mask: Some(&mask),
            };
            let decoder_out = decoder_forward(decoder, &decoder_input, gen_cfg.max_layers)?;

            processed_logits.copy_from_slice(&decoder_out.logits);
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
            current_canvas = renoise_canvas(&accepted, &accepted_mask, vocab, &mut rng);
            argmax_canvas_tokens = new_argmax;

            finished = stopper.should_stop(
                &argmax_canvas_tokens,
                &processed_logits,
                canvas_len,
                vocab,
            );
            sc_logits.copy_from_slice(&processed_logits);
            have_sc_logits = true;
            denoise_steps_run += 1;
        }
        denoise_elapsed += denoise_started.elapsed();

        sequences.extend_from_slice(&argmax_canvas_tokens);
        if !is_last_block {
            let extend_started = std::time::Instant::now();
            extend_prefill(store, cfg, &mut kv_cache, &argmax_canvas_tokens, enc_scratch)?;
            extend_elapsed += extend_started.elapsed();
        }
        blocks_committed += 1;
    }

    Ok(GenerateOutput {
        token_ids: sequences,
        denoise_steps_run,
        blocks_committed,
        prefill_elapsed,
        denoise_elapsed,
        extend_elapsed,
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
    let mut decoder = DecoderBackend::Gpu {
        store,
        cfg,
        scratch: dec_scratch,
        weights,
        engine,
    };
    generate_inner(store, cfg, prompt_token_ids, gen_cfg, enc_scratch, &mut decoder)
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
