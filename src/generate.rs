//! End-to-end block diffusion generation (CPU decoder; optional Metal GPU decoder).

use crate::Error;
use crate::sample::SamplerConfig;

#[derive(Debug, Clone)]
#[allow(dead_code)] // some fields are consumed only by parity/golden test paths
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
    /// True when a [`crate::metal::CancelToken`] stopped generation early: the
    /// in-flight canvas was abandoned uncommitted and `token_ids` holds only
    /// the prompt plus fully committed blocks (KV-consistent).
    pub cancelled: bool,
    /// True when generation ended because a committed block emitted an
    /// end-of-turn / EOS token (full-message mode) rather than exhausting the
    /// `max_new_tokens` budget.
    pub stopped_on_eot: bool,
    /// Stop-token id when [`Self::stopped_on_eot`], else `None`.
    pub stop_token_id: Option<u32>,
    /// 1-based block index that emitted the stop token.
    pub stop_block_idx: Option<usize>,
    /// Offset of the stop token within that block's committed canvas.
    pub stop_offset: Option<usize>,
    /// Per committed block: the final-token stats that the verbose step-generate
    /// log prints (accept/min_ent/mean_ent/low_ent histograms + late window).
    /// Plumbed through from `TurnState`; the live verbose log currently reads
    /// block stats off the pipeline event, so nothing consumes this field yet.
    #[allow(dead_code)]
    pub block_stats: Vec<BlockDenoiseStats>,
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

/// Per-block denoise summary (same content as the quiet-suppressed step-generate
/// block footer lines).
#[derive(Debug, Clone)]
pub struct BlockDenoiseStats {
    pub block_idx: usize,
    pub max_blocks: usize,
    pub steps_eff: u32,
    pub denoise_secs: f64,
    pub accept_per_step: Vec<u32>,
    pub min_ent_per_step: Vec<f32>,
    pub mean_ent_per_step: Vec<f32>,
    pub low_ent_per_step: Vec<u32>,
    pub late_accept_sum: u32,
    pub late_min_ent: f32,
    pub late_mean_ent: f32,
    pub late_max_low_ent: u32,
    /// `confident` | `plateau` | `max_steps`
    pub denoise_stop: String,
    /// Tokens kept from this block after stop-token trim (full canvas if none).
    pub kept_tokens: usize,
    /// Kept token ids for this block (same slice as Kv-extend / reply append).
    pub token_ids: Vec<u32>,
    pub stop_token_id: Option<u32>,
    pub stop_offset: Option<usize>,
    /// Stop token was trimmed but generation continued (incomplete tool call).
    pub continued_past_stop: bool,
}

/// Build the step-level config for a generate command (layers resolution,
/// full-message stop tokens, degenerate-reply check). Shared by the direct
/// path and the token-pipeline path so both run the identical policy.
#[cfg(target_os = "macos")]
pub fn step_config_for_generate(
    model_dir: &std::path::Path,
    gen_cfg: &GenerateConfig,
    max_seq: usize,
) -> Result<crate::metal::StepGenerateConfig, Error> {
    use crate::metal::{StepGenerateConfig, validate_step_model};
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
    // Empty/degenerate-reply canvas re-roll (only when enabled). Detects an
    // empty user-facing reply from the decoded+sanitized committed block.
    cfg.degenerate_reply_check = crate::chat_template::empty_reply_check(model_dir, stops);
    Ok(cfg)
}

#[cfg(target_os = "macos")]
pub fn generate_monolithic_gpu(
    model_dir: &std::path::Path,
    prompt_token_ids: &[u32],
    gen_cfg: &GenerateConfig,
    max_seq: usize,
    prompt_label: &str,
) -> Result<GenerateOutput, Error> {
    let cfg = step_config_for_generate(model_dir, gen_cfg, max_seq)?;
    crate::metal::generate_monolithic(model_dir, prompt_token_ids, &cfg, prompt_label)
}

/// `generate_monolithic_gpu` routed through the token pipeline (its first
/// production client): same config build, but the session lives on the
/// pipeline thread and the generate is a serialized op. Output equivalence
/// with the direct path is pinned by `ask_via_pipeline_matches_direct`.
#[cfg(target_os = "macos")]
pub fn generate_monolithic_gpu_pipeline(
    model_dir: &std::path::Path,
    prompt_token_ids: &[u32],
    gen_cfg: &GenerateConfig,
    max_seq: usize,
    prompt_label: &str,
) -> Result<GenerateOutput, Error> {
    let cfg = step_config_for_generate(model_dir, gen_cfg, max_seq)?;
    let pipeline = crate::pipeline::Pipeline::spawn(
        model_dir.to_path_buf(),
        max_seq,
        gen_cfg.sampler.max_denoising_steps.max(1),
    );
    crate::pipeline::generate(
        &pipeline,
        prompt_token_ids.to_vec(),
        Box::new(cfg),
        prompt_label,
    )
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
