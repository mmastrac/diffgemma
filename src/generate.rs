//! End-to-end block diffusion generation (CPU decoder; optional Metal GPU decoder).

use crate::config::ModelConfig;
use crate::model::decoder::{DecoderForwardInput, DecoderScratch};
use crate::model::encoder::extend_prefill;
use crate::model::encoder::{EncoderPrefillInput, EncoderScratch, prefill};
use crate::model::kv_cache::KvCache;
use crate::model::mask::DecoderAttnMask;
use crate::safetensors::Error;
use crate::sample::{
    Rng, SamplerConfig, StableConfidentStopper, accept_canvas, apply_temperature, argmax_canvas,
    denoise_steps_completed, initialize_canvas, renoise_canvas, sample_canvas,
};
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
    /// Stop the whole reply as soon as a committed block emits a generation stop
    /// token (chat-like, multi-block-until-eos). Off = fixed `max_new_tokens`
    /// length (parity/golden). When on, the model's stop tokens are loaded from
    /// the model dir and `max_new_tokens` becomes a cap.
    pub full_message_stop: bool,
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
            full_message_stop: false,
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
    #[cfg(target_os = "macos")]
    pub session_telemetry: crate::metal::SessionTelemetry,
    #[cfg(target_os = "macos")]
    pub denoise_trace: Option<crate::denoise_trace::DenoiseTrace>,
}

#[cfg(target_os = "macos")]
pub fn generate_monolithic_gpu(
    model_dir: &std::path::Path,
    prompt_token_ids: &[u32],
    gen_cfg: &GenerateConfig,
    max_seq: usize,
    prompt_label: &str,
) -> Result<GenerateOutput, Error> {
    use crate::metal::{StepGenerateConfig, generate_monolithic, validate_step_model};
    let validated = validate_step_model(model_dir)?;
    let layers = gen_cfg
        .max_layers
        .unwrap_or(validated.num_layers)
        .min(validated.num_layers);
    let mut cfg = StepGenerateConfig::from_generate(
        gen_cfg.seed,
        gen_cfg.max_new_tokens,
        max_seq,
        layers,
        gen_cfg.sampler.clone(),
        gen_cfg.no_early_stop,
    );
    // Chat-like full-message stop: end the reply when a committed block emits a
    // generation stop token, so a multi-block reply terminates at eos instead of
    // running the whole `max_new_tokens` budget. Off → fixed length (parity).
    let stops = if gen_cfg.full_message_stop {
        crate::config::load_generation_stop_tokens(model_dir)
    } else {
        Vec::new()
    };
    cfg.stop_token_ids = stops.clone();
    // E6 empty/degenerate-reply canvas re-roll (only when enabled). Detects an
    // empty user-facing reply from the decoded+sanitized committed block.
    cfg.degenerate_reply_check = crate::chat_template::empty_reply_check(model_dir, stops);
    generate_monolithic(model_dir, prompt_token_ids, &cfg, prompt_label)
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
