//! M2: end-to-end monolithic generate loop (prefill → denoise blocks → KV extend).

use crate::generate::GenerateOutput;
use crate::metal::step_kernel::{
    build_step_runtime, init_canvas_state_from_rng, step_params_from_sampler, StepFinishMode,
    StepRuntime, StepSmokeConfig, CANVAS, N_LAYERS, VOCAB,
};
use crate::metal::step_kv::{extend_monolithic_kv, prefill_monolithic_kv};
use crate::sample::{initialize_canvas, Rng, SamplerConfig};
use crate::safetensors::Error;
use std::path::Path;
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

fn open_runtime(model_dir: &Path, cfg: &StepGenerateConfig) -> Result<StepRuntime, Error> {
    let smoke = StepSmokeConfig {
        layers: cfg.layers.min(N_LAYERS).max(1),
        steps: cfg.sampler.max_denoising_steps.max(1),
        kv_len: 0,
        seed: cfg.seed,
        max_seq: cfg.max_seq,
        finish: StepFinishMode::Full,
        use_mps_q4: cfg.use_mps_q4,
        prefill_token_ids: None,
    };
    let (rt, compile) = build_step_runtime(model_dir, &smoke)?;
    eprintln!("step-generate: runtime ready ({compile:.2?})");
    Ok(rt)
}

/// Monolithic generate: prefill prompt → denoise blocks → extend KV (matches `generate_inner` structure).
pub fn generate_monolithic(
    model_dir: &Path,
    prompt_token_ids: &[u32],
    cfg: &StepGenerateConfig,
) -> Result<GenerateOutput, Error> {
    if prompt_token_ids.is_empty() {
        return Err(Error::Format("generate requires a non-empty prompt"));
    }
    let canvas_len = CANVAS;
    let layers = cfg.layers.min(N_LAYERS).max(1);
    let max_blocks = cfg.max_new_tokens.div_ceil(canvas_len).max(1);

    let mut rt = open_runtime(model_dir, cfg)?;

    let prefill_started = Instant::now();
    let kv_len = prefill_monolithic_kv(
        model_dir,
        prompt_token_ids,
        rt.kvcache(),
        rt.layout(),
        cfg.max_seq,
        layers,
    )?;
    rt.set_kv_len(kv_len as u32);
    let prefill_elapsed = prefill_started.elapsed();
    eprintln!("step-generate: prefilled kv_len={kv_len}");

    let mut sequences = prompt_token_ids.to_vec();
    let mut rng = Rng::new(cfg.seed);
    let mut denoise_steps_run = 0usize;
    let mut blocks_committed = 0usize;
    let mut denoise_elapsed = Duration::ZERO;
    let mut extend_elapsed = Duration::ZERO;

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
        loop {
            rt.run_denoise_step()?;
            denoise_steps_run += 1;
            let st = rt.read_canvas_state();
            if st.stop_flag != 0 {
                break;
            }
            if st.step >= cfg.sampler.max_denoising_steps as u32 {
                break;
            }
        }
        denoise_elapsed += block_started.elapsed();

        let st = rt.read_canvas_state();
        let argmax_tokens: Vec<u32> = st.prev_argmax.to_vec();
        sequences.extend_from_slice(&argmax_tokens);
        blocks_committed += 1;

        if !is_last_block {
            let extend_started = Instant::now();
            let kv_before = rt.read_params().kv_len as usize;
            let new_kv_len = extend_monolithic_kv(
                model_dir,
                rt.kvcache(),
                rt.layout(),
                kv_before,
                &argmax_tokens,
                cfg.max_seq,
                layers,
            )?;
            rt.set_kv_len(new_kv_len as u32);
            extend_elapsed += extend_started.elapsed();
            eprintln!(
                "step-generate: extended kv {kv_before} -> {new_kv_len} (+{} tokens)",
                argmax_tokens.len()
            );
        }
    }

    Ok(GenerateOutput {
        token_ids: sequences,
        denoise_steps_run,
        blocks_committed,
        prefill_elapsed,
        denoise_elapsed,
        extend_elapsed,
        #[cfg(all(feature = "metal", target_os = "macos"))]
        session_telemetry: crate::metal::SessionTelemetry::default(),
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
