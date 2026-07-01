//! M2/M4: end-to-end monolithic generate loop (prefill → denoise blocks → KV extend).

use crate::denoise_trace::{step_trace_from_stats, DenoiseTrace, SCHEMA_VERSION};
use crate::generate::GenerateOutput;
use crate::metal::step_kernel::{
    build_step_runtime, denoise_parity_log_enabled, final_entropy_log_enabled,
    log_denoise_parity_step, log_final_per_token_entropy, step_text_log_enabled,
    step_params_from_sampler, trace_entropy_enabled, StepFinishMode, StepRuntime, StepSmokeConfig,
    CANVAS, N_LAYERS, VOCAB,
};
use crate::metal::step_kv::{
    extend_monolithic_kv_with_cache, prefill_monolithic_kv_with_cache, MonolithicEncoderCache,
    MonolithicPrefillTiming,
};
use crate::metal::{ForwardTelemetry, SessionTelemetry, StepPhaseTelemetry};
use crate::sample::{Rng, SamplerConfig, step_entropy_stats};
use crate::safetensors::Error;
use crate::tokenizer::Tokenizer;
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
    /// Override random canvas (256 ids) for parity with MLX/HF traces.
    pub initial_canvas_ids: Option<Vec<u32>>,
    /// End-of-turn / EOS ids that terminate a "full message". When non-empty,
    /// generation stops (and the sequence is truncated) as soon as a committed
    /// block emits any of these. Empty preserves the fixed `max_new_tokens`
    /// budget behavior used by parity/golden paths.
    pub stop_token_ids: Vec<u32>,
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
            initial_canvas_ids: None,
            stop_token_ids: Vec::new(),
        }
    }
}

fn smoke_config(cfg: &StepGenerateConfig, prefill_token_ids: Option<Vec<u32>>) -> StepSmokeConfig {
    StepSmokeConfig {
        layers: cfg.layers.min(N_LAYERS).max(1),
        steps: cfg.sampler.max_denoising_steps.max(1),
        kv_len: 0,
        seed: cfg.seed,
        max_seq: cfg.max_seq,
        finish: StepFinishMode::Full,
        prefill_token_ids,
        no_early_stop: false,
    }
}

fn progress_enabled() -> bool {
    match std::env::var("DGQ_QUIET") {
        Ok(v) => v != "1" && !v.eq_ignore_ascii_case("true"),
        Err(_) => true,
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DenoiseStopReason {
    None,
    Confident,
    Plateau,
    MaxSteps,
}

fn log_denoise_step_progress(
    block_idx: usize,
    max_blocks: usize,
    step_idx: u32,
    max_steps: usize,
    stats: &crate::sample::StepEntropyStats,
    mean_entropy_gpu: f32,
    prefix_mean: Option<f32>,
    region_end: Option<usize>,
    answer_text: Option<&str>,
    canvas_stable: u32,
    prefix_stable: u32,
    full_argmax_diff: Option<usize>,
    prefix_argmax_diff: Option<usize>,
    argmax_hist_len: u32,
    accept_plateau: u32,
    step_elapsed: Duration,
    block_elapsed: Duration,
    denoise_elapsed: Duration,
    stop: DenoiseStopReason,
) {
    if !progress_enabled() {
        return;
    }
    let stop_note = match stop {
        DenoiseStopReason::None => "",
        DenoiseStopReason::Confident => " confident_stop",
        DenoiseStopReason::Plateau => " plateau_stop",
        DenoiseStopReason::MaxSteps => " max_steps",
    };
    let mut extra = String::new();
    if let Some(pm) = prefix_mean {
        extra.push_str(&format!(" prefix_mean={pm:.4}"));
    }
    if let Some(re) = region_end {
        extra.push_str(&format!(" ans_len={re}"));
    }
    if let Some(text) = answer_text {
        let one_line = text.replace(['\n', '\r'], "\\n");
        let shown = if one_line.len() > 72 {
            format!("{}...", &one_line[..72])
        } else {
            one_line
        };
        extra.push_str(&format!(" text={shown:?}"));
    }
    if let Some(d) = full_argmax_diff {
        extra.push_str(&format!(" full_diff={d}"));
    }
    if let Some(d) = prefix_argmax_diff {
        extra.push_str(&format!(" prefix_diff={d}"));
    }
    eprintln!(
        "step-generate: block {block_idx}/{max_blocks} step {step_idx}/{max_steps} accept={} low_ent={} min_ent={:.4} mean_ent={mean_entropy_gpu:.4} canvas_stable={canvas_stable} prefix_stable={prefix_stable} hist_len={argmax_hist_len} plateau={accept_plateau}{extra} step={step_elapsed:.2?} block={block_elapsed:.2?} denoise={denoise_elapsed:.2?}{stop_note}",
        stats.accept_count,
        stats.low_entropy_positions,
        stats.min_entropy,
    );
}

fn step_answer_text(
    tokenizer: Option<&Tokenizer>,
    prev_argmax: &[u32],
    ids: &[u32],
    eos_token_id: u32,
) -> (usize, Option<String>) {
    let region_end = crate::sample::answer_region_end(ids, eos_token_id);
    let prefix = crate::sample::answer_prefix_ids(prev_argmax, ids, eos_token_id);
    let text = tokenizer.map(|tok| tok.decode(prefix));
    (region_end, text)
}

/// Reusable monolithic runtime across prompts (M4.3).
pub struct StepGenerateSession {
    rt: StepRuntime,
    model_dir: PathBuf,
    layers: usize,
    encoder: Option<MonolithicEncoderCache>,
    step_text_tokenizer: Option<Tokenizer>,
}

impl StepGenerateSession {
    pub fn open(
        model_dir: &Path,
        cfg: &StepGenerateConfig,
        prefill_token_ids: Option<Vec<u32>>,
    ) -> Result<(Self, Duration), Error> {
        let layers = cfg.layers.min(N_LAYERS).max(1);
        let (rt, build) = build_step_runtime(model_dir, &smoke_config(cfg, prefill_token_ids))?;
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
                step_text_tokenizer: None,
            },
            build.compile,
        ))
    }

    /// Drop the prefilled KV so the next `generate_with_session` re-prefills from
    /// scratch. Use for *independent* prompts (smoketest); chat relies on the
    /// KV-reuse continuation path instead.
    pub fn reset_kv(&mut self) {
        self.rt.set_kv_len(0);
    }
}

/// Monolithic generate: prefill prompt → denoise blocks → extend KV (matches `generate_inner` structure).
pub fn generate_monolithic(
    model_dir: &Path,
    prompt_token_ids: &[u32],
    cfg: &StepGenerateConfig,
    prompt_label: &str,
) -> Result<GenerateOutput, Error> {
    let (mut session, _) = StepGenerateSession::open(
        model_dir,
        cfg,
        Some(prompt_token_ids.to_vec()),
    )?;
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
    let rt = &mut session.rt;

    if step_text_log_enabled() && session.step_text_tokenizer.is_none() {
        let tok_path = session.model_dir.join("tokenizer.json");
        match Tokenizer::load(&tok_path) {
            Ok(tok) => session.step_text_tokenizer = Some(tok),
            Err(err) => {
                eprintln!(
                    "step-generate: DGQ_LOG_STEP_TEXT=1 but failed to load {}: {err}",
                    tok_path.display()
                );
            }
        }
    }

    let prefill_started = Instant::now();
    let existing_kv = rt.read_params().kv_len as usize;
    let (kv_len, prefill_timing, prefill_elapsed) = if existing_kv >= prompt_token_ids.len()
        && existing_kv > 0
    {
        eprintln!("step-generate: using step-kernel prefill kv_len={existing_kv}");
        (
            existing_kv,
            MonolithicPrefillTiming::default(),
            Duration::ZERO,
        )
    } else if crate::metal::step_kernel::should_fast_prefill(prompt_token_ids.len()) {
        // Fast monolithic prefill: quantized + causal forward over prompt chunks,
        // writing the b4 KV directly (no f32 engine, no pack conversion).
        let kv_len = rt.prefill_chunks(prompt_token_ids)?;
        let prefill_elapsed = prefill_started.elapsed();
        eprintln!(
            "step-generate: fast-prefill kv_len={kv_len} ({prefill_elapsed:.2?})"
        );
        (kv_len, MonolithicPrefillTiming::default(), prefill_elapsed)
    } else {
        if session.encoder.is_none() {
            let encoder_started = Instant::now();
            session.encoder = Some(MonolithicEncoderCache::open_opt(
                model_dir,
                canvas_len,
                cfg.max_seq,
                Some(std::sync::Arc::clone(&shared_blob)),
            )?);
            eprintln!(
                "step-generate: encoder cache ready ({:.2?})",
                encoder_started.elapsed()
            );
        }
        let encoder = session.encoder.as_mut().expect("encoder cache");
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
        (kv_len, prefill_timing, prefill_elapsed)
    };
    if prefill_elapsed > Duration::ZERO {
        eprintln!(
            "step-generate: prefilled kv_len={kv_len} ({prefill_elapsed:.2?}, gpu_forward={:.1}ms kv_pack={:.1}ms)",
            prefill_timing.gpu_forward_ms,
            prefill_timing.kv_pack_ms
        );
    }

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
    let mut initial_canvas_ids: Option<Vec<u32>> = None;
    let mut stopped_on_eot = false;

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
            rt.read_params().eos_token_id,
        );
        rt.reset_block(VOCAB, &mut rng, params);
        if let Some(ref ids) = cfg.initial_canvas_ids {
            rt.set_canvas_ids(ids)?;
        }
        if initial_canvas_ids.is_none() {
            initial_canvas_ids = Some(rt.read_canvas_state().ids.to_vec());
            if denoise_parity_log_enabled() {
                let c = initial_canvas_ids.as_ref().expect("initial canvas");
                eprintln!(
                    "denoise-parity: initial_canvas[:8]={:?}",
                    &c[..8.min(c.len())]
                );
            }
        }

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
        let mut mean_entropy_hist = Vec::new();
        let mut low_ent_hist = Vec::new();
        let mut last_st;
        let mut prev_step_argmax: Option<[u32; CANVAS]> = None;
        let mut prefix_stable_streak = 0u32;
        loop {
            let step_started = Instant::now();
            rt.run_denoise_step()?;
            let check_logits = crate::metal::step_kernel::logits_finite_check_enabled();
            rt.check_logits_finite()?;
            let step_elapsed = step_started.elapsed();
            let step_ms = step_elapsed.as_secs_f64() * 1000.0;
            let readback_bytes = StepRuntime::denoise_step_host_readback_bytes(check_logits);
            let mut forward = ForwardTelemetry::monolithic_gpu_step(readback_bytes);
            rt.fill_expert_forward_telemetry(&mut forward);
            session_telemetry.steps.push(StepPhaseTelemetry {
                decoder_ms: step_ms,
                sampler_ms: 0.0,
                forward,
            });
            denoise_steps_run += 1;
            block_step_count += 1;
            let st = rt.read_canvas_state();
            last_st = st;
            if denoise_parity_log_enabled() {
                log_denoise_parity_step(
                    &format!("block={block_idx} step_index={block_step_count}"),
                    &st,
                    &rt.read_params(),
                    rt.logits(),
                );
            }
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
            mean_entropy_hist.push(st.mean_entropy);
            low_ent_hist.push(stats.low_entropy_positions);
            let max_steps_reached = st.step >= max_steps as u32;
            let params = rt.read_params();
            let region_end = crate::sample::answer_region_end(&st.ids, params.eos_token_id);
            let (full_diff, prefix_diff, prefix_stable_streak) = match prev_step_argmax {
                Some(prev) => {
                    let full =
                        crate::sample::count_argmax_diff(&st.prev_argmax, &prev, CANVAS);
                    let prefix =
                        crate::sample::count_argmax_diff(&st.prev_argmax, &prev, region_end);
                    let streak = if prefix == 0 {
                        prefix_stable_streak.saturating_add(1)
                    } else {
                        0
                    };
                    (Some(full), Some(prefix), streak)
                }
                None => (None, None, 0),
            };
            prev_step_argmax = Some(st.prev_argmax);
            let early_stop = crate::sample::decode_early_stop_flag(st.stop_flag);
            let snap = crate::sample::EarlyStopSnapshot {
                canvas_stable: st.canvas_stable,
                mean_entropy: st.mean_entropy,
                accept_plateau: st.accept_plateau,
                conf_threshold: params.conf_threshold,
                accept_plateau_threshold: params.accept_plateau_threshold,
                plateau_prefix_mean_max: params.plateau_prefix_mean_max,
            };
            let cpu_early =
                !cfg.no_early_stop && crate::sample::early_stop_from_snapshot(&snap);
            let gpu_early = !cfg.no_early_stop && crate::sample::is_early_stop_flag(st.stop_flag);
            let stop_reason = match early_stop {
                Some(crate::sample::EarlyStopKind::Confident) => DenoiseStopReason::Confident,
                Some(crate::sample::EarlyStopKind::Plateau) => DenoiseStopReason::Plateau,
                Some(crate::sample::EarlyStopKind::MaxSteps) => DenoiseStopReason::MaxSteps,
                None if max_steps_reached => DenoiseStopReason::MaxSteps,
                None => DenoiseStopReason::None,
            };
            if std::env::var("DGQ_LOG_EARLY_STOP")
                .ok()
                .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            {
                if gpu_early != cpu_early {
                    eprintln!(
                        "step-generate: early-stop mismatch step={} gpu_flag={} gpu_early={gpu_early} cpu_early={cpu_early} accept_plateau={} mean_ent={:.4} stable={} threshold={:.4}",
                        st.step,
                        st.stop_flag,
                        st.accept_plateau,
                        st.mean_entropy,
                        st.canvas_stable,
                        params.conf_threshold,
                    );
                } else if gpu_early {
                    let kind = match early_stop {
                        Some(crate::sample::EarlyStopKind::Confident) => "confident_stable",
                        Some(crate::sample::EarlyStopKind::Plateau) => "plateau_stop",
                        _ => "early_stop",
                    };
                    eprintln!(
                        "step-generate: early-stop step={} reason={kind} stop_flag={} accept_plateau={} mean_ent={:.4} stable={} accept={}",
                        st.step,
                        st.stop_flag,
                        st.accept_plateau,
                        st.mean_entropy,
                        st.canvas_stable,
                        stats.accept_count,
                    );
                }
            }
            let (prefix_mean_log, region_end_log, answer_text_log) = if step_text_log_enabled() {
                let pm = crate::sample::mean_entropy_answer_prefix(
                    &st.entropy,
                    &st.ids,
                    params.eos_token_id,
                );
                let (re, text) = step_answer_text(
                    session.step_text_tokenizer.as_ref(),
                    &st.prev_argmax,
                    &st.ids,
                    params.eos_token_id,
                );
                (Some(pm), Some(re), text)
            } else {
                (None, None, None)
            };
            let answer_text_ref = answer_text_log.as_deref();
            step_traces.push(step_trace_from_stats(
                block_idx as u32,
                block_step_count,
                max_steps,
                &stats,
                &st.prev_argmax,
                if trace_entropy_enabled() {
                    Some(&st.entropy)
                } else {
                    None
                },
                stop_reason != DenoiseStopReason::None,
            ));
            log_denoise_step_progress(
                block_idx,
                max_blocks,
                block_step_count,
                max_steps,
                &stats,
                st.mean_entropy,
                prefix_mean_log,
                region_end_log,
                answer_text_ref,
                st.canvas_stable,
                prefix_stable_streak,
                full_diff,
                prefix_diff,
                st.argmax_hist_len,
                st.accept_plateau,
                step_elapsed,
                block_started.elapsed(),
                denoise_elapsed + block_started.elapsed(),
                stop_reason,
            );
            if stop_reason != DenoiseStopReason::None {
                break;
            }
        }
        let st = last_st;
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
            "step-generate: block {} mean_ent/step={mean_entropy_hist:?}",
            block_idx
        );
        eprintln!(
            "step-generate: block {} low_ent(<0.1)/step={low_ent_hist:?}",
            block_idx
        );
        let late_mean_ent = mean_entropy_hist
            .get(late..)
            .and_then(|s| s.iter().copied().reduce(f32::min))
            .unwrap_or(f32::NAN);
        eprintln!(
            "step-generate: block {} late-window (last 8 steps): accept_sum={late_accept} min_ent={late_min_ent:.4} mean_ent={late_mean_ent:.4} max_low_ent={late_low_ent} (early stop: accept plateau>={} or prefix mean_ent<{:.4} + stable argmax)",
            block_idx,
            crate::sample::ACCEPT_PLATEAU_THRESHOLD,
            cfg.sampler.confidence_threshold,
        );
        if final_entropy_log_enabled() {
            log_final_per_token_entropy(
                &format!("block {block_idx} final"),
                &st,
                st.stop_flag,
                rt.read_params().eos_token_id,
            );
        }

        let argmax_tokens: Vec<u32> = st.prev_argmax.to_vec();
        let block_base = sequences.len();
        sequences.extend_from_slice(&argmax_tokens);
        blocks_committed += 1;

        // Full-message stop: end the turn as soon as the committed block emits a
        // stop token (e.g. <turn|> or <eos>). Trim it and everything after so the
        // reply is exactly the model's turn, and skip the KV extend.
        if !cfg.stop_token_ids.is_empty() {
            if let Some(rel) = argmax_tokens
                .iter()
                .position(|id| cfg.stop_token_ids.contains(id))
            {
                sequences.truncate(block_base + rel);
                stopped_on_eot = true;
                if progress_enabled() {
                    eprintln!(
                        "step-generate: block {block_idx} hit stop token {} at offset {rel}; ending turn ({} new tokens)",
                        argmax_tokens[rel],
                        sequences.len() - prompt_token_ids.len()
                    );
                }
                break;
            }
        }

        if !is_last_block {
            let extend_started = Instant::now();
            let kv_before = rt.read_params().kv_len as usize;
            if session.encoder.is_none() {
                session.encoder = Some(MonolithicEncoderCache::open_opt(
                    model_dir,
                    canvas_len,
                    cfg.max_seq,
                    Some(std::sync::Arc::clone(&shared_blob)),
                )?);
            }
            let encoder = session.encoder.as_mut().expect("encoder cache");
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
        stopped_on_eot,
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
            initial_canvas_ids,
            canvas_rng: Some("rust-lcg".into()),
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::step_kernel::{logits_finite_check_enabled, init_canvas_state_from_rng, StepRuntime};
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
