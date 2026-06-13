//! M2/M4: end-to-end monolithic generate loop (prefill → denoise blocks → KV extend).

use crate::denoise_trace::{step_trace_from_stats, DenoiseTrace, SCHEMA_VERSION};
use crate::generate::GenerateOutput;
use crate::metal::step_kernel::{
    build_step_runtime, init_canvas_state_from_rng, step_params_from_sampler, StepFinishMode,
    StepRuntime, StepSmokeConfig, CANVAS, N_LAYERS, VOCAB,
};
use crate::metal::step_kv::{
    extend_monolithic_kv_with_cache, prefill_monolithic_kv_with_cache, MonolithicEncoderCache,
};
use crate::metal::{ForwardTelemetry, SessionTelemetry, StepPhaseTelemetry};
use crate::sample::{Rng, SamplerConfig, step_entropy_stats};
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
    /// Step-kernel dense GEMM (`DGQ_STEP_MPS_Q4`, default native Q4).
    pub step_use_mps_q4: Option<bool>,
    /// Encoder prefill/extend (`DGQ_MPS_Q4` when `None`).
    pub encoder_use_mps_q4: Option<bool>,
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
            step_use_mps_q4: None,
            encoder_use_mps_q4: None,
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
        use_mps_q4: cfg.step_use_mps_q4,
        prefill_token_ids: None,
        no_early_stop: false,
        encoder_use_mps_q4: cfg.encoder_use_mps_q4,
    }
}

fn progress_enabled() -> bool {
    match std::env::var("DGQ_QUIET") {
        Ok(v) => v != "1" && !v.eq_ignore_ascii_case("true"),
        Err(_) => true,
    }
}

fn log_denoise_step_progress(
    block_idx: usize,
    max_blocks: usize,
    step_idx: u32,
    max_steps: usize,
    stats: &crate::sample::StepEntropyStats,
    step_elapsed: Duration,
    block_elapsed: Duration,
    denoise_elapsed: Duration,
    early_stop: bool,
) {
    if !progress_enabled() {
        return;
    }
    let stop_note = if early_stop { " early_stop" } else { "" };
    eprintln!(
        "step-generate: block {block_idx}/{max_blocks} step {step_idx}/{max_steps} accept={} low_ent={} min_ent={:.4} step={step_elapsed:.2?} block={block_elapsed:.2?} denoise={denoise_elapsed:.2?}{stop_note}",
        stats.accept_count,
        stats.low_entropy_positions,
        stats.min_entropy,
    );
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
        let step_mps_q4 = rt.use_mps_q4();
        eprintln!(
            "step-generate: runtime ready (total={:.2?}, compile={:.2?}, step_use_mps_q4={step_mps_q4})",
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

    pub fn step_use_mps_q4(&self) -> bool {
        self.rt.use_mps_q4()
    }
}

/// Monolithic generate: prefill prompt → denoise blocks → extend KV (matches `generate_inner` structure).
pub fn generate_monolithic(
    model_dir: &Path,
    prompt_token_ids: &[u32],
    cfg: &StepGenerateConfig,
    prompt_label: &str,
) -> Result<GenerateOutput, Error> {
    let (mut session, _) = StepGenerateSession::open(model_dir, cfg)?;
    generate_with_session(&mut session, prompt_token_ids, cfg, prompt_label)
}

pub fn generate_with_session(
    session: &mut StepGenerateSession,
    prompt_token_ids: &[u32],
    cfg: &StepGenerateConfig,
    prompt_label: &str,
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
            cfg.encoder_use_mps_q4,
        )?);
        let enc_mps = session
            .encoder
            .as_ref()
            .expect("encoder cache")
            .use_mps_q4();
        eprintln!(
            "step-generate: encoder cache ready ({:.2?}, encoder_use_mps_q4={enc_mps})",
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
    let mut step_traces = Vec::new();
    let max_steps = cfg.sampler.max_denoising_steps.max(1);

    for _block in 0..max_blocks {
        if sequences.len() >= prompt_token_ids.len() + cfg.max_new_tokens {
            break;
        }

        let remaining = prompt_token_ids.len() + cfg.max_new_tokens - sequences.len();
        let is_last_block = remaining <= canvas_len;
        let block_idx = blocks_committed + 1;

        let params = step_params_from_sampler(
            &cfg.sampler,
            rt.read_params().kv_len,
            cfg.no_early_stop,
        );
        rt.reset_block(VOCAB, &mut rng, params);

        if progress_enabled() {
            eprintln!(
                "step-generate: block {block_idx}/{max_blocks} starting denoise (kv_len={}, max_steps={max_steps}, new_tokens_remaining={remaining})",
                rt.read_params().kv_len
            );
        }

        let block_started = Instant::now();
        let mut block_step_count = 0u32;
        let mut accept_hist = Vec::new();
        let mut min_entropy_hist = Vec::new();
        let mut low_ent_hist = Vec::new();
        let mut last_st = None;
        loop {
            let step_started = Instant::now();
            rt.run_denoise_step()?;
            let check_logits = crate::metal::step_kernel::logits_finite_check_enabled();
            rt.check_logits_finite()?;
            let step_elapsed = step_started.elapsed();
            let step_ms = step_elapsed.as_secs_f64() * 1000.0;
            let readback_bytes = StepRuntime::denoise_step_host_readback_bytes(check_logits);
            session_telemetry.steps.push(StepPhaseTelemetry {
                decoder_ms: step_ms,
                sampler_ms: 0.0,
                forward: ForwardTelemetry::monolithic_gpu_step(readback_bytes),
            });
            denoise_steps_run += 1;
            block_step_count += 1;
            let st = rt.read_canvas_state();
            last_st = Some(st);
            if std::env::var("DGQ_LOG_SC").ok().as_deref() == Some("1") {
                eprintln!(
                    "monolithic denoise: step_index={block_step_count} st.step={} sc_active_next={}",
                    st.step,
                    st.step >= 1
                );
            }
            let stats = step_entropy_stats(&st.entropy, &st.accept);
            accept_hist.push(stats.accept_count);
            min_entropy_hist.push(stats.min_entropy);
            low_ent_hist.push(stats.low_entropy_positions);
            let early_stop = st.stop_flag != 0;
            step_traces.push(step_trace_from_stats(
                block_idx as u32,
                block_step_count,
                max_steps,
                &stats,
                &st.prev_argmax,
                early_stop,
            ));
            log_denoise_step_progress(
                block_idx,
                max_blocks,
                block_step_count,
                max_steps,
                &stats,
                step_elapsed,
                block_started.elapsed(),
                denoise_elapsed + block_started.elapsed(),
                early_stop,
            );
            if early_stop {
                break;
            }
            if st.step >= max_steps as u32 {
                break;
            }
        }
        let st = last_st.expect("denoise loop ran at least one step");
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
            block_idx
        );
        eprintln!(
            "step-generate: block {} min_ent/step={min_entropy_hist:?}",
            block_idx
        );
        eprintln!(
            "step-generate: block {} low_ent(<0.1)/step={low_ent_hist:?}",
            block_idx
        );
        eprintln!(
            "step-generate: block {} late-window (last 8 steps): accept_sum={late_accept} min_ent={late_min_ent:.4} max_low_ent={late_low_ent} (need low_ent~15-20 for accept~15-20)",
            block_idx
        );

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

    if progress_enabled() && !session_telemetry.steps.is_empty() {
        let n = session_telemetry.steps.len().max(1) as f64;
        let agg = session_telemetry.aggregate_forward();
        eprintln!(
            "step-generate: P2.1 hot path mean {:.2} syncs/step, {:.1} KiB readback/step (DGQ_CHECK_LOGITS for opt-in logits scan)",
            agg.gpu_syncs as f64 / n,
            agg.gpu_readback_bytes as f64 / 1024.0 / n
        );
    }

    Ok(GenerateOutput {
        token_ids: sequences.clone(),
        denoise_steps_run,
        blocks_committed,
        block_steps_eff,
        last_block_accept_hist,
        last_block_min_entropy_hist,
        prefill_elapsed,
        denoise_elapsed,
        extend_elapsed,
        session_telemetry,
        denoise_trace: Some(DenoiseTrace {
            schema_version: SCHEMA_VERSION,
            source: "rust-monolithic".into(),
            prompt: prompt_label.to_string(),
            prompt_token_ids: prompt_token_ids.to_vec(),
            seed: cfg.seed,
            max_denoise_steps: max_steps,
            layers,
            max_new_tokens: cfg.max_new_tokens,
            weights_profile: Some(crate::generate_golden::monolithic_weights_profile().into()),
            entropy_bound: cfg.sampler.entropy_bound,
            step_traces,
            denoise_steps_run,
            blocks_committed,
            output_token_ids: sequences.clone(),
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::step_kernel::{logits_finite_check_enabled, StepRuntime};
    use crate::sample::initialize_canvas;

    #[test]
    fn p21_denoise_readback_under_1mb() {
        let bytes = StepRuntime::denoise_step_host_readback_bytes(false);
        assert!(
            bytes <= 1024 * 1024,
            "hot-path readback {bytes} B exceeds 1 MiB"
        );
        assert_eq!(bytes, (StepRuntime::CANVAS_STATE_BYTES * 2) as u64);
        if logits_finite_check_enabled() {
            let with_check = StepRuntime::denoise_step_host_readback_bytes(true);
            assert!(with_check <= 1024 * 1024);
        }
    }

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
