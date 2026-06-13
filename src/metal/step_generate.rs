//! M2/M4: end-to-end monolithic generate loop (prefill → denoise blocks → KV extend).

use crate::generate::GenerateOutput;
use crate::metal::step_kernel::{
    build_step_runtime, init_canvas_state_from_rng, step_params_from_sampler, StepFinishMode,
    StepRuntime, StepSmokeConfig, CANVAS, N_LAYERS, VOCAB,
};
use crate::metal::step_kv::{
    extend_monolithic_kv_with_cache, prefill_monolithic_kv_with_cache, MonolithicEncoderCache,
};
use crate::metal::{ForwardTelemetry, SessionTelemetry, StepPhaseTelemetry};
use crate::sample::{initialize_canvas, Rng, SamplerConfig, step_entropy_stats};
use crate::safetensors::Error;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct StepGenerateConfig {
    pub seed: u64,
    pub max_new_tokens: usize,
    pub max_seq: usize,
    pub layers: usize,
    pub sampler: SamplerConfig,
    pub no_early_stop: bool,
    pub use_mps_q4: Option<bool>,
}

impl StepGenerateConfig {
    pub fn from_generate(
        seed: u64,
        max_new_tokens: usize,
        max_seq: usize,
        layers: usize,
        sampler: SamplerConfig,
        no_early_stop: bool,
    ) -> Self {
        Self {
            seed,
            max_new_tokens,
            max_seq,
            layers,
            sampler,
            no_early_stop,
            use_mps_q4: None,
        }
    }
}

fn smoke_config(cfg: &StepGenerateConfig) -> StepSmokeConfig {
    StepSmokeConfig {
        layers: cfg.layers.min(N_LAYERS).max(1),
        steps: cfg.sampler.max_denoising_steps.max(1),
        kv_len: 0,
        seed: cfg.seed,
        max_seq: cfg.max_seq,
        finish: StepFinishMode::Full,
        use_mps_q4: cfg.use_mps_q4,
        prefill_token_ids: None,
    }
}

/// Reusable monolithic runtime across prompts (M4.3).
pub struct StepGenerateSession {
    rt: StepRuntime,
    model_dir: PathBuf,
    layers: usize,
    encoder: Option<MonolithicEncoderCache>,
}

impl StepGenerateSession {
    pub fn open(model_dir: &Path, cfg: &StepGenerateConfig) -> Result<(Self, Duration), Error> {
        let layers = cfg.layers.min(N_LAYERS).max(1);
        let (rt, build) = build_step_runtime(model_dir, &smoke_config(cfg))?;
        eprintln!(
            "step-generate: runtime ready (total={:.2?}, compile={:.2?})",
            build.total, build.compile
        );
        Ok((
            Self {
                rt,
                model_dir: model_dir.to_path_buf(),
                layers,
                encoder: None,
            },
            build.compile,
        ))
    }

    pub fn runtime(&self) -> &StepRuntime {
        &self.rt
    }

    pub fn runtime_mut(&mut self) -> &mut StepRuntime {
        &mut self.rt
    }
}

/// Monolithic generate: prefill prompt → denoise blocks → extend KV (matches `generate_inner` structure).
pub fn generate_monolithic(
    model_dir: &Path,
    prompt_token_ids: &[u32],
    cfg: &StepGenerateConfig,
) -> Result<GenerateOutput, Error> {
    let (mut session, _) = StepGenerateSession::open(model_dir, cfg)?;
    generate_with_session(&mut session, prompt_token_ids, cfg)
}

pub fn generate_with_session(
    session: &mut StepGenerateSession,
    prompt_token_ids: &[u32],
    cfg: &StepGenerateConfig,
) -> Result<GenerateOutput, Error> {
    if prompt_token_ids.is_empty() {
        return Err(Error::Format("generate requires a non-empty prompt"));
    }
    let canvas_len = CANVAS;
    let layers = session.layers;
    let max_blocks = cfg.max_new_tokens.div_ceil(canvas_len).max(1);
    let model_dir = session.model_dir.as_path();
    let shared_blob = session.rt.shared_dgq_blob();
    if session.encoder.is_none() {
        let encoder_started = Instant::now();
        session.encoder = Some(MonolithicEncoderCache::open_opt(
            model_dir,
            canvas_len,
            cfg.max_seq,
            Some(shared_blob),
            cfg.use_mps_q4,
        )?);
        eprintln!(
            "step-generate: encoder cache ready ({:.2?})",
            encoder_started.elapsed()
        );
    }
    let encoder = session.encoder.as_mut().expect("encoder cache");
    let rt = &mut session.rt;

    let prefill_started = Instant::now();
    let (kv_len, prefill_timing) = prefill_monolithic_kv_with_cache(
        encoder,
        prompt_token_ids,
        rt.kvcache(),
        rt.layout(),
        cfg.max_seq,
        layers,
    )?;
    rt.set_kv_len(kv_len as u32);
    let prefill_elapsed = prefill_started.elapsed();
    eprintln!(
        "step-generate: prefilled kv_len={kv_len} ({prefill_elapsed:.2?}, gpu_forward={:.1}ms kv_pack={:.1}ms)",
        prefill_timing.gpu_forward_ms,
        prefill_timing.kv_pack_ms
    );

    let mut sequences = prompt_token_ids.to_vec();
    let mut rng = Rng::new(cfg.seed);
    let mut denoise_steps_run = 0usize;
    let mut blocks_committed = 0usize;
    let mut block_steps_eff = Vec::new();
    let mut last_block_accept_hist = Vec::new();
    let mut last_block_min_entropy_hist = Vec::new();
    let mut denoise_elapsed = Duration::ZERO;
    let mut extend_elapsed = Duration::ZERO;
    let mut session_telemetry = SessionTelemetry::default();

    for _block in 0..max_blocks {
        if sequences.len() >= prompt_token_ids.len() + cfg.max_new_tokens {
            break;
        }

        let remaining = prompt_token_ids.len() + cfg.max_new_tokens - sequences.len();
        let is_last_block = remaining <= canvas_len;

        let params = step_params_from_sampler(
            &cfg.sampler,
            rt.read_params().kv_len,
            cfg.no_early_stop,
        );
        rt.reset_block(VOCAB, &mut rng, params);

        let block_started = Instant::now();
        let mut block_step_count = 0u32;
        let mut accept_hist = Vec::new();
        let mut min_entropy_hist = Vec::new();
        let mut low_ent_hist = Vec::new();
        loop {
            let step_started = Instant::now();
            rt.run_denoise_step()?;
            rt.check_logits_finite()?;
            let step_ms = step_started.elapsed().as_secs_f64() * 1000.0;
            session_telemetry.steps.push(StepPhaseTelemetry {
                decoder_ms: step_ms,
                sampler_ms: 0.0,
                forward: ForwardTelemetry::monolithic_gpu_step(),
            });
            denoise_steps_run += 1;
            block_step_count += 1;
            let st = rt.read_canvas_state();
            let stats = step_entropy_stats(&st.entropy, &st.accept);
            accept_hist.push(stats.accept_count);
            min_entropy_hist.push(stats.min_entropy);
            low_ent_hist.push(stats.low_entropy_positions);
            if st.stop_flag != 0 {
                break;
            }
            if st.step >= cfg.sampler.max_denoising_steps as u32 {
                break;
            }
        }
        let block_elapsed = block_started.elapsed();
        denoise_elapsed += block_elapsed;
        block_steps_eff.push(block_step_count);
        last_block_accept_hist = accept_hist.clone();
        last_block_min_entropy_hist = min_entropy_hist.clone();
        let late = accept_hist.len().saturating_sub(8);
        let late_accept: u32 = accept_hist.get(late..).unwrap_or(&[]).iter().sum();
        let late_min_ent = min_entropy_hist
            .get(late..)
            .and_then(|s| s.iter().copied().reduce(f32::min))
            .unwrap_or(f32::NAN);
        let late_low_ent = low_ent_hist
            .get(late..)
            .and_then(|s| s.iter().copied().reduce(u32::max))
            .unwrap_or(0);
        eprintln!(
            "step-generate: block {} denoise={block_elapsed:.2?} steps_eff={block_step_count} accept/step={accept_hist:?}",
            blocks_committed + 1
        );
        eprintln!(
            "step-generate: block {} min_ent/step={min_entropy_hist:?}",
            blocks_committed + 1
        );
        eprintln!(
            "step-generate: block {} low_ent(<0.1)/step={low_ent_hist:?}",
            blocks_committed + 1
        );
        eprintln!(
            "step-generate: block {} late-window (last 8 steps): accept_sum={late_accept} min_ent={late_min_ent:.4} max_low_ent={late_low_ent} (need low_ent~15-20 for accept~15-20)",
            blocks_committed + 1
        );

        let st = rt.read_canvas_state();
        let argmax_tokens: Vec<u32> = st.prev_argmax.to_vec();
        sequences.extend_from_slice(&argmax_tokens);
        blocks_committed += 1;

        if !is_last_block {
            let extend_started = Instant::now();
            let kv_before = rt.read_params().kv_len as usize;
            let new_kv_len = extend_monolithic_kv_with_cache(
                encoder,
                rt.kvcache(),
                rt.layout(),
                kv_before,
                &argmax_tokens,
                cfg.max_seq,
                layers,
            )?;
            rt.set_kv_len(new_kv_len as u32);
            let block_extend = extend_started.elapsed();
            extend_elapsed += block_extend;
            eprintln!(
                "step-generate: extended kv {kv_before} -> {new_kv_len} (+{} tokens) ({block_extend:.2?})",
                argmax_tokens.len()
            );
        }
    }

    Ok(GenerateOutput {
        token_ids: sequences,
        denoise_steps_run,
        blocks_committed,
        block_steps_eff,
        last_block_accept_hist,
        last_block_min_entropy_hist,
        prefill_elapsed,
        denoise_elapsed,
        extend_elapsed,
        #[cfg(all(feature = "metal", target_os = "macos"))]
        session_telemetry,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::initialize_canvas;

    #[test]
    fn block_reset_uses_fresh_canvas() {
        let mut rng = Rng::new(42);
        let a = initialize_canvas(CANVAS, VOCAB, &mut rng);
        let b = initialize_canvas(CANVAS, VOCAB, &mut rng);
        assert_ne!(a, b);
        let mut r = Rng::new(99);
        let st = init_canvas_state_from_rng(VOCAB, &mut r);
        assert_eq!(st.ids.len(), CANVAS);
    }
}
