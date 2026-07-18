use crate::Error;
use crate::denoise_trace::{DenoiseTrace, SCHEMA_VERSION, step_trace_from_stats};
use crate::generate::GenerateOutput;
use crate::metal::step_kernel::{
    CANVAS, StepRuntime, VOCAB, denoise_parity_log_enabled, final_entropy_log_enabled,
    log_denoise_parity_step, log_final_per_token_entropy, step_params_from_sampler,
    step_text_log_enabled, trace_entropy_enabled,
};
use crate::metal::step_kv::{
    MonolithicEncoderCache, MonolithicPrefillTiming, extend_monolithic_kv_chunked,
    extend_monolithic_kv_with_cache, prefill_monolithic_kv_with_cache,
};
use crate::metal::{ForwardTelemetry, SessionTelemetry, StepPhaseTelemetry};
use crate::sample::{Rng, step_entropy_stats};
use crate::tokenizer::Tokenizer;
use std::path::Path;
use std::time::{Duration, Instant};

use super::config::{StepGenerateConfig, StepProgressEvent};
use super::progress::{DenoiseStopReason, log_denoise_step_progress, step_answer_text};
use super::session::{StepGenerateSession, longest_common_prefix, perturb_live_kv_f16};
use crate::flags::progress_enabled;

/// Monolithic generate: prefill prompt → denoise blocks → extend KV (matches `generate_inner` structure).
pub fn generate_monolithic(
    model_dir: &Path,
    prompt_token_ids: &[u32],
    cfg: &StepGenerateConfig,
    prompt_label: &str,
) -> Result<GenerateOutput, Error> {
    let (mut session, _) =
        StepGenerateSession::open(model_dir, cfg, Some(prompt_token_ids.to_vec()))?;
    generate_with_session(&mut session, prompt_token_ids, cfg, prompt_label)
}

/// One block's uncommitted denoise result: the active-canvas argmax plus the
/// stats the monolithic path records. Produced by [`propose_block`]; the
/// caller decides `kept_len` (stop-scan, validators) and applies it via
/// [`commit_block`] — or drops the proposal, and the next [`propose_block`]
/// re-rolls from fresh noise (the canvas rng advances every attempt).
pub struct ProposedBlock {
    /// Active-canvas argmax (shrink-on-retry may narrow it below `CANVAS`).
    pub token_ids: Vec<u32>,
    /// Stats for this block; `kept_tokens`/`token_ids` are finalized at
    /// commit, the stop fields by the caller's policy.
    pub stats: crate::generate::BlockDenoiseStats,
    /// 1-based index this block would commit as.
    pub block_idx: usize,
}

/// Outcome of one [`propose_block`] call.
pub enum BlockOutcome {
    /// A canvas converged and awaits the commit decision.
    Proposal(Box<ProposedBlock>),
    /// Token budget or block count already spent; nothing was denoised.
    Exhausted,
    /// The commit guard abandoned the turn (failed block's stats recorded).
    Abandoned,
    /// A [`CancelToken`] stopped the turn; any mid-flight canvas was dropped.
    Cancelled,
}

/// Accumulating state of one generation turn, block by block: created by
/// [`begin_turn`] (prefill + bookkeeping), advanced by
/// [`propose_block`]/[`commit_block`], closed by [`finish_turn`]
/// (`kv_valid_tokens` + `GenerateOutput`). [`generate_with_session`] is a
/// driver over exactly these primitives; the pipeline exposes them as
/// per-block ops.
pub struct TurnState {
    prompt_token_ids: Vec<u32>,
    label: String,
    sequences: Vec<u32>,
    rng: Rng,
    max_steps: usize,
    max_blocks: usize,
    denoise_steps_run: usize,
    blocks_committed: usize,
    block_steps_eff: Vec<u32>,
    last_block_accept_hist: Vec<u32>,
    last_block_min_entropy_hist: Vec<f32>,
    prefill_elapsed: Duration,
    denoise_elapsed: Duration,
    extend_elapsed: Duration,
    session_telemetry: SessionTelemetry,
    step_traces: Vec<crate::denoise_trace::DenoiseStepTrace>,
    initial_canvas_ids: Option<Vec<u32>>,
    cancelled: bool,
    stopped_on_eot: bool,
    stop_token_id: Option<u32>,
    stop_block_idx: Option<usize>,
    stop_offset: Option<usize>,
    block_stats: Vec<crate::generate::BlockDenoiseStats>,
}

impl TurnState {
    /// New (non-prompt) tokens committed so far.
    pub fn new_tokens(&self) -> usize {
        self.sequences.len() - self.prompt_token_ids.len()
    }

    /// The full committed token log (prompt + committed blocks).
    pub fn sequences(&self) -> &[u32] {
        &self.sequences
    }
}

pub fn generate_with_session(
    session: &mut StepGenerateSession,
    prompt_token_ids: &[u32],
    cfg: &StepGenerateConfig,
    prompt_label: &str,
) -> Result<GenerateOutput, Error> {
    let mut ts = begin_turn(session, prompt_token_ids, cfg, prompt_label)?;
    loop {
        let pb = match propose_block(session, cfg, &mut ts)? {
            BlockOutcome::Proposal(pb) => pb,
            BlockOutcome::Exhausted | BlockOutcome::Abandoned | BlockOutcome::Cancelled => break,
        };
        if !default_commit_policy(session, cfg, &mut ts, *pb)? {
            break;
        }
    }
    finish_turn(session, cfg, ts)
}

/// The monolithic commit policy, unchanged from the pre-decomposition path:
/// full-message stop-scan (with the tool-mode defer), the whitespace-collapse
/// stopgap, and the causal KV extend for every non-final block. Returns
/// whether the turn continues.
fn default_commit_policy(
    session: &mut StepGenerateSession,
    cfg: &StepGenerateConfig,
    ts: &mut TurnState,
    mut pb: ProposedBlock,
) -> Result<bool, Error> {
    let remaining = ts.prompt_token_ids.len() + cfg.max_new_tokens - ts.sequences.len();
    let is_last_block = remaining <= CANVAS;
    let block_idx = pb.block_idx;
    let argmax_tokens = pb.token_ids.clone();
    let block_base = ts.sequences.len();

    // Full-message stop: end the turn as soon as the committed block emits a
    // stop token (e.g. <turn|> or <eos>). Trim it and everything after so the
    // reply is exactly the model's turn, and skip the KV extend — unless
    // `continue_incomplete_tool_calls` and the reply still looks unfinished
    // (open `call:NAME{…}`, or trailing prose after a closed tool call).
    let mut kept = argmax_tokens.len();
    let mut end_turn = false;
    if !cfg.stop_token_ids.is_empty()
        && let Some(rel) = argmax_tokens
            .iter()
            .position(|id| cfg.stop_token_ids.contains(id))
    {
        kept = rel;
        pb.stats.stop_token_id = Some(argmax_tokens[rel]);
        pb.stats.stop_offset = Some(rel);

        let defer_stop = cfg.continue_incomplete_tool_calls && {
            if session.step_text_tokenizer.is_none() {
                let tok_path = session.model_dir.join("tokenizer.json");
                if let Ok(tok) = Tokenizer::load(&tok_path) {
                    session.step_text_tokenizer = Some(tok);
                }
            }
            let mut reply = ts.sequences[ts.prompt_token_ids.len()..].to_vec();
            reply.extend_from_slice(&argmax_tokens[..rel]);
            let cleaned = crate::sample::strip_degenerate_token_ids(&reply);
            session
                .step_text_tokenizer
                .as_ref()
                .is_some_and(|tok| crate::tools::should_continue_past_stop(&tok.decode(&cleaned)))
        };

        if defer_stop {
            pb.stats.continued_past_stop = true;
            eprintln!(
                "serve: unfinished tool turn after stop token {}; continuing to next block",
                argmax_tokens[rel],
            );
        } else {
            ts.stopped_on_eot = true;
            ts.stop_token_id = Some(argmax_tokens[rel]);
            ts.stop_block_idx = Some(block_idx);
            ts.stop_offset = Some(rel);
            end_turn = true;
            if progress_enabled() {
                eprintln!(
                    "step-generate: block {block_idx} hit stop token {} at offset {rel}; ending turn ({} new tokens)",
                    argmax_tokens[rel],
                    ts.new_tokens() + rel
                );
            }
        }
    }

    // Whitespace-collapse STOPGAP (opt-in, see flags::ws_block_stop_enabled:
    // the attractor is being treated as an unfixed bug and a default-on
    // stopper would hide the evidence): a committed block whose text is
    // pure whitespace / all pad-filler ends the turn instead of crawling
    // toward the context wall at max_steps per block.
    let mut ws_stop = false;
    if !end_turn && crate::flags::ws_block_stop_enabled() {
        let cleaned = crate::sample::strip_degenerate_token_ids(&argmax_tokens[..kept]);
        let all_ws = cleaned.is_empty()
            || session
                .step_text_tokenizer
                .as_ref()
                .is_some_and(|tok| tok.decode(&cleaned).trim().is_empty());
        if all_ws {
            ws_stop = true;
        }
    }

    let extend = !end_turn && !ws_stop && !is_last_block && kept > 0;
    commit_block(session, cfg, ts, pb, kept, extend)?;
    if ws_stop {
        ts.sequences.truncate(block_base);
        if progress_enabled() {
            eprintln!(
                "step-generate: block {block_idx} committed pure whitespace; ending turn (DGQ_WS_BLOCK_STOP, {} new tokens kept)",
                ts.new_tokens()
            );
        }
        return Ok(false);
    }
    Ok(!end_turn)
}

/// Open a turn: select and run the prefill path (cross-turn reuse / fast
/// quantized / engine), leaving the session ready for the first
/// [`propose_block`].
pub fn begin_turn(
    session: &mut StepGenerateSession,
    prompt_token_ids: &[u32],
    cfg: &StepGenerateConfig,
    prompt_label: &str,
) -> Result<TurnState, Error> {
    if prompt_token_ids.is_empty() {
        return Err(Error::Runtime("generate requires a non-empty prompt"));
    }
    let canvas_len = CANVAS;
    let layers = session.layers;
    let mut max_blocks = cfg.max_new_tokens.div_ceil(canvas_len).max(1);
    if crate::flags::commit_conf_trim() > 0.0 || crate::flags::prefix_exit() > 0.0 {
        // Trimmed / prefix-exited blocks commit fewer than CANVAS tokens;
        // keep the TOKEN budget as the binding limit with 2x block headroom
        // so partial commits don't silently truncate long replies (both
        // floors still guarantee >=16 tokens of progress per block).
        max_blocks *= 2;
    }
    let model_dir = session.model_dir.as_path();
    let shared_blob = session.rt.shared_dgq_blob();
    let rt = &mut session.rt;

    // The step-text log AND the whitespace-collapse guard both need to decode
    // canvas tokens; load the tokenizer lazily once per session for either.
    if (step_text_log_enabled() || crate::flags::ws_block_stop_enabled())
        && session.step_text_tokenizer.is_none()
    {
        let tok_path = session.model_dir.join("tokenizer.json");
        match Tokenizer::load(&tok_path) {
            Ok(tok) => session.step_text_tokenizer = Some(tok),
            Err(err) => {
                eprintln!(
                    "step-generate: step-text/ws-guard tokenizer load failed for {}: {err}",
                    tok_path.display()
                );
            }
        }
    }

    // Owned: the rewind guard below needs `&mut session` before the encoder
    // paths use the dir again.
    let model_dir = model_dir.to_path_buf();
    let model_dir = model_dir.as_path();
    let prefill_started = Instant::now();
    let existing_kv = rt.read_params().kv_len as usize;
    let n_prompt = prompt_token_ids.len();
    // Cross-turn KV reuse: the session's `kv_valid_tokens` are already causal in
    // the KV. Reuse the longest common prefix (capped at what the runtime
    // actually holds) and prefill only the delta at that offset.
    let mut reuse = if crate::flags::kv_reuse_enabled() {
        longest_common_prefix(&session.kv_valid_tokens, prompt_token_ids).min(existing_kv)
    } else {
        0
    };
    // Rewind/divergence guard (task #93): a resident causal log that extends
    // past — or diverges from — the prompt's common prefix is stale for this
    // turn and must go before any reuse. O(1) truncate inside the ring slack;
    // a DEEP rewind resets instead (no-rebuild-to-salvage: a ring rebuild
    // costs ≈ a fresh prefill of the kept prefix and yields lineage KV — see
    // memory kv-reuse-delta-divergence). The session-open prefill (empty
    // kv_valid log, primed KV) is exempt: it was prefilled for exactly this
    // prompt.
    if !session.kv_valid_tokens.is_empty() && session.kv_valid_tokens.len() > reuse {
        if session.truncate_needs_rebuild(reuse) {
            if progress_enabled() {
                eprintln!(
                    "step-generate: deep rewind {} -> {reuse}; dropping KV for fresh prefill (no-rebuild-to-salvage)",
                    session.kv_valid_tokens.len(),
                );
            }
            session.reset_kv();
            reuse = 0;
        } else {
            session.truncate_kv_to(reuse)?;
        }
    }
    let rt = &mut session.rt;
    // Post-guard truth: a reset zeroed it; a truncate clamped it to `reuse`.
    let existing_kv = rt.read_params().kv_len as usize;
    let open_prefill = session.kv_valid_tokens.is_empty() && existing_kv >= n_prompt;
    let (kv_len, prefill_timing, prefill_elapsed) = if (open_prefill || reuse >= n_prompt)
        && existing_kv > 0
    {
        // Whole prompt already prefilled (session-open prefill, or full reuse).
        if progress_enabled() {
            eprintln!("step-generate: using step-kernel prefill kv_len={existing_kv}");
        }
        (
            existing_kv.min(n_prompt),
            MonolithicPrefillTiming::default(),
            Duration::ZERO,
        )
    } else if reuse > 0 {
        // Keep KV[0..reuse] (causally valid), prefill the delta [reuse..] at
        // that offset. Short deltas take the fast quantized resume; a delta
        // past the fast-prefill trust cap (task #64: the bf16 stream
        // accumulates error with length) extends via the f32 engine in
        // canvas-sized blocks instead — slower, correct.
        let delta = &prompt_token_ids[reuse..];
        let max = crate::flags::fast_prefill_max_tokens();
        let kv_len = if max == 0 || delta.len() <= max {
            rt.prefill_chunks_from(reuse, delta)?
        } else {
            if session.encoder.is_none() {
                session.encoder = Some(MonolithicEncoderCache::open_opt(
                    model_dir,
                    canvas_len,
                    cfg.max_seq,
                    Some(std::sync::Arc::clone(&shared_blob)),
                )?);
            }
            let encoder = session.encoder.as_mut().expect("encoder cache");
            let off = extend_monolithic_kv_chunked(
                encoder,
                rt.kvcache(),
                rt.layout(),
                reuse,
                delta,
                cfg.max_seq,
                layers,
            )?;
            rt.set_kv_len(off as u32);
            off
        };
        let prefill_elapsed = prefill_started.elapsed();
        if progress_enabled() {
            eprintln!(
                "step-generate: cross-turn KV reuse: kept {reuse}/{n_prompt}, prefilled {} delta tokens ({prefill_elapsed:.2?})",
                n_prompt - reuse
            );
        }
        (kv_len, MonolithicPrefillTiming::default(), prefill_elapsed)
    } else if crate::metal::step_kernel::should_fast_prefill(prompt_token_ids.len()) {
        // Fast monolithic prefill: quantized + causal forward over prompt chunks,
        // writing the b4 KV directly (no f32 engine, no pack conversion).
        let kv_len = rt.prefill_chunks(prompt_token_ids)?;
        let prefill_elapsed = prefill_started.elapsed();
        if progress_enabled() {
            eprintln!("step-generate: fast-prefill kv_len={kv_len} ({prefill_elapsed:.2?})");
        }
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
            if progress_enabled() {
                eprintln!(
                    "step-generate: encoder cache ready ({:.2?})",
                    encoder_started.elapsed()
                );
            }
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
    // DIAGNOSTIC (task #67 sensitivity probe): DGQ_KV_NOISE=<rel eps>
    // perturbs every live f16 KV value after prefill (whichever path ran) by
    // a random relative factor — separates "the model is knife-edge
    // sensitive to KV noise at this length" from "the fast path has a real
    // defect".
    if let Some(eps) = crate::flags::kv_noise() {
        perturb_live_kv_f16(rt, kv_len, eps, cfg.seed);
    }
    // Canvas denoise writes at [kv_len..kv_len+CANVAS]; ensure the runtime's
    // kv_len is exactly the prompt length (the reuse/extend paths update KV but
    // not params; a full-reuse KV may also hold stale tokens past the prompt).
    rt.set_kv_len(kv_len as u32);
    if progress_enabled() && prefill_elapsed > Duration::ZERO {
        eprintln!(
            "step-generate: prefilled kv_len={kv_len} ({prefill_elapsed:.2?}, gpu_forward={:.1}ms kv_pack={:.1}ms)",
            prefill_timing.gpu_forward_ms, prefill_timing.kv_pack_ms
        );
    }

    Ok(TurnState {
        prompt_token_ids: prompt_token_ids.to_vec(),
        label: prompt_label.to_string(),
        sequences: prompt_token_ids.to_vec(),
        rng: Rng::new(cfg.seed),
        max_steps: cfg.sampler.max_denoising_steps.max(1),
        max_blocks,
        denoise_steps_run: 0,
        blocks_committed: 0,
        block_steps_eff: Vec::new(),
        last_block_accept_hist: Vec::new(),
        last_block_min_entropy_hist: Vec::new(),
        prefill_elapsed,
        denoise_elapsed: Duration::ZERO,
        extend_elapsed: Duration::ZERO,
        session_telemetry: SessionTelemetry::default(),
        step_traces: Vec::new(),
        initial_canvas_ids: None,
        cancelled: false,
        stopped_on_eot: false,
        stop_token_id: None,
        stop_block_idx: None,
        stop_offset: None,
        block_stats: Vec::new(),
    })
}

/// One block's denoise: reset the canvas, run the attempt loop (E6
/// empty-retry + commit-guard re-rolls), return the uncommitted proposal or a
/// terminal outcome. Commits NOTHING — the caller inspects the proposal and
/// either [`commit_block`]s a kept prefix or drops it.
pub fn propose_block(
    session: &mut StepGenerateSession,
    cfg: &StepGenerateConfig,
    ts: &mut TurnState,
) -> Result<BlockOutcome, Error> {
    if ts.blocks_committed >= ts.max_blocks
        || ts.sequences.len() >= ts.prompt_token_ids.len() + cfg.max_new_tokens
    {
        return Ok(BlockOutcome::Exhausted);
    }
    if cfg.cancel_requested() {
        ts.cancelled = true;
        return Ok(BlockOutcome::Cancelled);
    }
    let max_steps = ts.max_steps;
    let max_blocks = ts.max_blocks;
    let rt = &mut session.rt;

    let remaining = ts.prompt_token_ids.len() + cfg.max_new_tokens - ts.sequences.len();
    let block_idx = ts.blocks_committed + 1;

    let params = step_params_from_sampler(
        &cfg.sampler,
        rt.read_params().kv_len,
        cfg.no_early_stop,
        rt.read_params().eos_token_id,
    );
    rt.reset_block(VOCAB, &mut ts.rng, params);
    if let Some(ref ids) = cfg.initial_canvas_ids {
        rt.set_canvas_ids(ids)?;
    }
    if ts.initial_canvas_ids.is_none() {
        ts.initial_canvas_ids = Some(rt.read_canvas_state().ids.to_vec());
        if denoise_parity_log_enabled() {
            let c = ts.initial_canvas_ids.as_ref().expect("initial canvas");
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
    // E6: empty/degenerate-reply retry — first block only. If the committed
    // canvas would trim to empty (position 0 is a stop/eos/control token),
    // re-roll the canvas and re-denoise up to N times (DGQ_EMPTY_REPLY_RETRY).
    let empty_retry_max = if block_idx == 1 {
        crate::flags::empty_reply_retry()
    } else {
        0
    };
    let mut empty_retry_attempt = 0u32;
    // Non-convergence commit guard (`DGQ_BLOCK_COMMIT_MAX_ENT`): re-roll
    // budget, and the abandon verdict when the budget is spent.
    let mut nc_retry_attempt = 0u32;
    let mut abandon_turn = false;
    // Head width committed by a PrefixExit stop (None = full active canvas).
    // Outlives the 'attempt loop; reset at each attempt start.
    let mut prefix_exit_head: Option<usize> = None;
    let (
        st,
        block_step_count,
        accept_hist,
        min_entropy_hist,
        mean_entropy_hist,
        low_ent_hist,
        denoise_stop,
    ) = 'attempt: loop {
        // Shrink-on-retry (E3/E6): attempt 0 uses the full canvas (handles
        // long answers); each degenerate retry narrows it (256→128→64),
        // where the empty/ceremony attractor is far weaker (72%→3% by width).
        let attempt_canvas = match empty_retry_attempt {
            0 => CANVAS,
            1 => 128,
            _ => 64,
        };
        rt.set_active_canvas(attempt_canvas);
        let active = rt.active_canvas();
        prefix_exit_head = None;
        let mut block_step_count = 0u32;
        let mut accept_hist = Vec::new();
        let mut min_entropy_hist = Vec::new();
        let mut mean_entropy_hist = Vec::new();
        let mut low_ent_hist = Vec::new();
        let mut last_st;
        let mut prev_step_argmax: Option<[u32; CANVAS]> = None;
        let prefix_stable_streak = 0u32;
        let last_denoise_stop = loop {
            let step_started = Instant::now();
            rt.run_denoise_step()?;
            let check_logits = crate::metal::step_kernel::logits_finite_check_enabled();
            rt.check_logits_finite()?;
            let step_elapsed = step_started.elapsed();
            let step_ms = step_elapsed.as_secs_f64() * 1000.0;
            let readback_bytes = StepRuntime::denoise_step_host_readback_bytes(check_logits);
            let mut forward = ForwardTelemetry::monolithic_gpu_step(readback_bytes);
            rt.fill_expert_forward_telemetry(&mut forward);
            ts.session_telemetry.steps.push(StepPhaseTelemetry {
                decoder_ms: step_ms,
                sampler_ms: 0.0,
                forward,
            });
            ts.denoise_steps_run += 1;
            block_step_count += 1;
            let st = rt.read_canvas_state();
            last_st = st;
            if let Some(path) = crate::flags::trace_pmax_jsonl() {
                trace_pmax_append(
                    &path,
                    serde_json::json!({
                        "kind": "step",
                        "block": block_idx,
                        "step_index": block_step_count,
                        "st_step": st.step,
                        "active": active,
                        "mean_entropy": st.mean_entropy,
                        "stop_flag": st.stop_flag,
                        "step_ms": step_ms,
                        "pmax": pmax_from_rowstats(&rt.read_sample_rowstats(active)),
                        "entropy": &st.entropy[..active],
                        "accept": &st.accept[..active],
                        "argmax": &st.prev_argmax[..active],
                    }),
                );
            }
            if denoise_parity_log_enabled() {
                log_denoise_parity_step(
                    &format!("block={block_idx} step_index={block_step_count}"),
                    &st,
                    &rt.read_params(),
                    rt.logits(),
                );
            }
            if crate::flags::sc_log_enabled() {
                eprintln!(
                    "monolithic denoise: step_index={block_step_count} st.step={} sc_active_next={}",
                    st.step,
                    st.step >= 1
                );
            }
            // Slice canvas stats/argmax to the ACTIVE rows — stale rows
            // [active..CANVAS) hold reset sentinels (u32::MAX / 0) that would
            // skew stats and (on decode) corrupt the reply.
            let stats = step_entropy_stats(&st.entropy[..active], &st.accept[..active]);
            accept_hist.push(stats.accept_count);
            min_entropy_hist.push(stats.min_entropy);
            mean_entropy_hist.push(st.mean_entropy);
            low_ent_hist.push(stats.low_entropy_positions);
            let max_steps_reached = st.step >= max_steps as u32;
            let params = rt.read_params();
            let region_end =
                crate::sample::answer_region_end(&st.ids[..active], params.eos_token_id);
            let (full_diff, prefix_diff, prefix_stable_streak) = match prev_step_argmax {
                Some(prev) => {
                    let full = crate::sample::count_argmax_diff(&st.prev_argmax, &prev, active);
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
            let prefix_exit_candidate = prefix_exit_candidate(
                &st.entropy[..active],
                &st.prev_argmax[..active],
                prev_step_argmax.as_ref().map(|p| &p[..active]),
                block_step_count,
            );
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
            let cpu_early = !cfg.no_early_stop && crate::sample::early_stop_from_snapshot(&snap);
            let gpu_early = !cfg.no_early_stop && crate::sample::is_early_stop_flag(st.stop_flag);
            let mut stop_reason = match early_stop {
                Some(crate::sample::EarlyStopKind::Confident) => DenoiseStopReason::Confident,
                Some(crate::sample::EarlyStopKind::Plateau) => DenoiseStopReason::Plateau,
                Some(crate::sample::EarlyStopKind::MaxSteps) => DenoiseStopReason::MaxSteps,
                None if max_steps_reached => DenoiseStopReason::MaxSteps,
                None => DenoiseStopReason::None,
            };
            // Prefix-exit: only when no natural stop fires this step (a
            // whole-canvas stop always commits more for the same steps).
            if stop_reason == DenoiseStopReason::None
                && let Some(p) = prefix_exit_candidate
            {
                prefix_exit_head = Some(p);
                stop_reason = DenoiseStopReason::PrefixExit;
                if progress_enabled() {
                    eprintln!(
                        "step-generate: block {block_idx} prefix-exit at step {block_step_count}: head {p}/{active} settled, tail still churning",
                    );
                }
            }
            if crate::flags::log_early_stop_enabled() {
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
                // Slice to the ACTIVE canvas: the readback buffers are
                // PREFILL_M-sized, and rows [active..] hold stale data
                // (was: ans_len=1024 + stale positions polluting
                // prefix_mean/text on every step line).
                let ids = &st.ids[..active.min(st.ids.len())];
                let entropy = &st.entropy[..active.min(st.entropy.len())];
                let pm =
                    crate::sample::mean_entropy_answer_prefix(entropy, ids, params.eos_token_id);
                let (re, text) = step_answer_text(
                    session.step_text_tokenizer.as_ref(),
                    &st.prev_argmax,
                    ids,
                    params.eos_token_id,
                );
                (Some(pm), Some(re), text)
            } else {
                (None, None, None)
            };
            let answer_text_ref = answer_text_log.as_deref();
            ts.step_traces.push(step_trace_from_stats(
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
                ts.denoise_elapsed + block_started.elapsed(),
                stop_reason,
            );
            if let Some(ref observer) = cfg.step_observer {
                // Never mark block_done here: E6 may still discard this canvas.
                // Active slice only — rows past `active` are stale after shrink.
                observer(&StepProgressEvent {
                    block_idx,
                    max_blocks,
                    step_in_block: block_step_count,
                    max_steps,
                    argmax: &st.prev_argmax[..active],
                    accept_count: stats.accept_count,
                    mean_entropy: st.mean_entropy,
                    block_done: false,
                });
            }
            if stop_reason != DenoiseStopReason::None {
                break stop_reason;
            }
            if cfg.cancel_requested() {
                break DenoiseStopReason::Cancelled;
            }
        };
        let st = last_st;
        // Cancelled mid-block: abandon this canvas uncommitted (the E6 /
        // commit-guard checks below are moot for a reply nobody will read).
        if last_denoise_stop == DenoiseStopReason::Cancelled {
            ts.cancelled = true;
            abandon_turn = true;
            break 'attempt (
                st,
                block_step_count,
                accept_hist,
                min_entropy_hist,
                mean_entropy_hist,
                low_ent_hist,
                last_denoise_stop,
            );
        }
        // Empty/degenerate-reply detection: does the committed argmax render as
        // an empty user-facing reply (eos-first canvas OR `<|channel>thought`
        // ceremony)? Checked against the real decoded+sanitized output (E6
        // attractor). Re-roll the canvas from the advancing rng and retry.
        let degenerate = cfg
            .degenerate_reply_check
            .as_ref()
            .is_some_and(|check| check(&st.prev_argmax[..active]));
        if empty_retry_attempt < empty_retry_max && degenerate {
            empty_retry_attempt += 1;
            if progress_enabled() {
                eprintln!(
                    "step-generate: block {block_idx} empty/degenerate reply; re-rolling canvas (attempt {empty_retry_attempt}/{empty_retry_max})"
                );
            }
            let params = step_params_from_sampler(
                &cfg.sampler,
                rt.read_params().kv_len,
                cfg.no_early_stop,
                rt.read_params().eos_token_id,
            );
            rt.reset_block(VOCAB, &mut ts.rng, params);
            continue 'attempt;
        }
        // Non-convergence commit guard: a block that burned the whole step
        // schedule and still shows late-window mean entropy above the floor
        // committed ~45%-accepted garble in the OpenCode collapse — and the
        // committed flood is self-consistent, so later blocks converge onto
        // it. Re-roll with fresh noise; if it still cannot converge, end
        // the turn instead of committing the attractor.
        let commit_max_ent = crate::flags::block_commit_max_ent();
        let non_converged =
            commit_max_ent > 0.0 && last_denoise_stop == DenoiseStopReason::MaxSteps && {
                let late0 = mean_entropy_hist.len().saturating_sub(8);
                let late_min = mean_entropy_hist[late0..]
                    .iter()
                    .copied()
                    .fold(f32::INFINITY, f32::min);
                late_min.is_finite() && late_min > commit_max_ent
            };
        if non_converged {
            if nc_retry_attempt < crate::flags::block_commit_retry() {
                nc_retry_attempt += 1;
                if progress_enabled() {
                    eprintln!(
                        "step-generate: block {block_idx} ended max_steps NON-CONVERGED (late mean_ent > {commit_max_ent}); re-rolling canvas (attempt {nc_retry_attempt}/{})",
                        crate::flags::block_commit_retry()
                    );
                }
                let params = step_params_from_sampler(
                    &cfg.sampler,
                    rt.read_params().kv_len,
                    cfg.no_early_stop,
                    rt.read_params().eos_token_id,
                );
                rt.reset_block(VOCAB, &mut ts.rng, params);
                continue 'attempt;
            }
            // Budget spent: abandon without the block_done notification —
            // this canvas must never reach a streamer as final.
            abandon_turn = true;
            break 'attempt (
                st,
                block_step_count,
                accept_hist,
                min_entropy_hist,
                mean_entropy_hist,
                low_ent_hist,
                last_denoise_stop,
            );
        }
        // Accepted: notify streamers that this block's active argmax is final.
        // A prefix-exited block's final draft is only the settled head — the
        // churning tail rows must never stream as final text.
        if let Some(ref observer) = cfg.step_observer {
            observer(&StepProgressEvent {
                block_idx,
                max_blocks,
                step_in_block: block_step_count,
                max_steps,
                argmax: &st.prev_argmax[..prefix_exit_head.unwrap_or(active).min(active)],
                accept_count: accept_hist.last().copied().unwrap_or(0),
                mean_entropy: st.mean_entropy,
                block_done: true,
            });
        }
        break 'attempt (
            st,
            block_step_count,
            accept_hist,
            min_entropy_hist,
            mean_entropy_hist,
            low_ent_hist,
            last_denoise_stop,
        );
    };
    let block_elapsed = block_started.elapsed();
    ts.denoise_elapsed += block_elapsed;
    ts.block_steps_eff.push(block_step_count);
    ts.last_block_accept_hist = accept_hist.clone();
    ts.last_block_min_entropy_hist = min_entropy_hist.clone();
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
    let late_mean_ent = mean_entropy_hist
        .get(late..)
        .and_then(|s| s.iter().copied().reduce(f32::min))
        .unwrap_or(f32::NAN);
    if progress_enabled() {
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
        eprintln!(
            "step-generate: block {} late-window (last 8 steps): accept_sum={late_accept} min_ent={late_min_ent:.4} mean_ent={late_mean_ent:.4} max_low_ent={late_low_ent} (early stop: accept plateau>={} or prefix mean_ent<{:.4} + stable argmax)",
            block_idx,
            crate::sample::ACCEPT_PLATEAU_THRESHOLD,
            cfg.sampler.confidence_threshold,
        );
    }
    if final_entropy_log_enabled() {
        log_final_per_token_entropy(
            &format!("block {block_idx} final"),
            &st,
            st.stop_flag,
            rt.read_params().eos_token_id,
        );
    }

    // Commit-guard abandon: the block ran out of schedule AND retries while
    // non-converged. Commit nothing (the canvas is the garble attractor),
    // record the failed block's stats for diagnosis, and end the turn with
    // whatever earlier blocks produced.
    if abandon_turn {
        if ts.cancelled {
            eprintln!(
                "step-generate: block {block_idx} cancelled after {block_step_count} steps; ending turn without committing ({} new tokens kept)",
                ts.new_tokens()
            );
        } else {
            eprintln!(
                "step-generate: block {block_idx} NON-CONVERGED after {} re-rolls (late mean_ent={late_mean_ent:.3} > {}); ending turn without committing ({} new tokens kept)",
                crate::flags::block_commit_retry(),
                crate::flags::block_commit_max_ent(),
                ts.new_tokens()
            );
        }
        let stats = crate::generate::BlockDenoiseStats {
            block_idx,
            max_blocks,
            steps_eff: block_step_count,
            denoise_secs: block_elapsed.as_secs_f64(),
            accept_per_step: accept_hist,
            min_ent_per_step: min_entropy_hist,
            mean_ent_per_step: mean_entropy_hist,
            low_ent_per_step: low_ent_hist,
            late_accept_sum: late_accept,
            late_min_ent,
            late_mean_ent,
            late_max_low_ent: late_low_ent,
            denoise_stop: if ts.cancelled {
                "cancelled".to_string()
            } else {
                "non_converged_abandon".to_string()
            },
            kept_tokens: 0,
            token_ids: Vec::new(),
            stop_token_id: None,
            stop_offset: None,
            continued_past_stop: false,
        };
        if let Some(ref obs) = cfg.block_commit_observer {
            obs(&stats);
        }
        ts.block_stats.push(stats);
        return Ok(if ts.cancelled {
            BlockOutcome::Cancelled
        } else {
            BlockOutcome::Abandoned
        });
    }

    // The proposed block is only the ACTIVE canvas rows (shrink-on-retry
    // narrows a degenerate first block); rows [active..CANVAS) are stale
    // sentinels. `rt.active_canvas()` after the loop is the successful
    // attempt's width (no set_active_canvas ran after its break).
    let committed_canvas = rt.active_canvas();
    let mut argmax_tokens: Vec<u32> = st.prev_argmax[..committed_canvas].to_vec();

    // Prefix-exit commit: only the settled head; the churning tail rows are
    // dropped and re-denoise next block against this committed context.
    if denoise_stop == DenoiseStopReason::PrefixExit
        && let Some(p) = prefix_exit_head
    {
        argmax_tokens.truncate(p);
    }

    // Confidence trim (`DGQ_COMMIT_CONF_TRIM=τ`, default off): drop the canvas
    // tail from the first row that is BOTH low-confidence (final-step p_max
    // < τ) AND forming a duplication (argmax equals a neighbor's argmax) —
    // the duplication micro-stutter signature: an unresolved row's marginal
    // copies its neighbor (`the the`, `<|"|><|"|>`; every strain-battery
    // stutter committed at p_max 0.40-0.86). The conjunction is what makes
    // the rule surgical: benign creative soft rows overlap the same p_max
    // range but do NOT duplicate their neighbor — a p_max-only trim churned
    // half-canvases on creative prompts and blew the step budget (smoketest
    // 15-16/17), while the conjunctive rule fires exactly on the 5 real
    // stutter formations across 51 traced turns and nowhere else. The next
    // block re-denoises the dropped tail against committed left context.
    // Floor: rows inside the first MIN_CONF_KEEP never trim (progress
    // guarantee; the E6/commit-guard paths own those failures).
    let mut conf_trim_row: Option<usize> = None;
    let mut conf_tau = crate::flags::commit_conf_trim();
    if conf_tau <= 0.0 && denoise_stop == DenoiseStopReason::PrefixExit {
        // A prefix-exited head shares the mean's blind spot (one unresolved
        // row hides in a settled head's mean) — always dup-scan it.
        conf_tau = 0.9;
    }
    if conf_tau > 0.0 {
        const MIN_CONF_KEEP: usize = 16;
        let pmax = pmax_from_rowstats(&rt.read_sample_rowstats(committed_canvas));
        // Scan only the ANSWER region: everything at/after the first
        // stop/pad/filler row belongs to the stop-scan, and eos-padding runs
        // are structural duplicates (adjacent identical <eos> at 0.76-0.87
        // confidence trip the dup rule and cost half a canvas for nothing —
        // the seed-123 sky_blue step blowup).
        let eos = rt.read_params().eos_token_id;
        let region_end = argmax_tokens
            .iter()
            .position(|t| {
                cfg.stop_token_ids.contains(t)
                    || *t == eos
                    || *t == crate::sample::PAD_TOKEN_ID
                    || *t == crate::sample::FILLER_TOKEN_ID
            })
            .unwrap_or(argmax_tokens.len());
        let dup_at = |i: usize| {
            (i > 0 && argmax_tokens[i] == argmax_tokens[i - 1])
                || (i + 1 < region_end && argmax_tokens[i] == argmax_tokens[i + 1])
        };
        if let Some(first_bad) =
            (MIN_CONF_KEEP..region_end).find(|&i| pmax[i] < conf_tau && dup_at(i))
        {
            if progress_enabled() {
                eprintln!(
                    "step-generate: block {block_idx} confidence-trim at row {first_bad}/{committed_canvas} (dup token {} at p_max {:.3} < {conf_tau})",
                    argmax_tokens[first_bad], pmax[first_bad],
                );
            }
            argmax_tokens.truncate(first_bad);
            conf_trim_row = Some(first_bad);
        }
    }

    if progress_enabled() {
        let toks = argmax_tokens.len();
        let secs = block_elapsed.as_secs_f64().max(1e-9);
        eprintln!(
            "step-generate: block {block_idx} commit {toks}/{committed_canvas} tokens after {block_step_count} steps (stop={}) {:.1} tok/s{}",
            denoise_stop.as_str(),
            toks as f64 / secs,
            if toks < committed_canvas {
                " [early exit / trim]"
            } else {
                ""
            },
        );
    }

    if let Some(path) = crate::flags::trace_pmax_jsonl() {
        trace_pmax_append(
            &path,
            serde_json::json!({
                "kind": "block_commit",
                "block": block_idx,
                // Turn identity + position: label changes per request, and
                // block_base (prompt + committed so far) orders blocks within
                // a turn — the trace is analyzable without stderr logs.
                "label": ts.label,
                "block_base": ts.sequences.len(),
                "steps": block_step_count,
                "denoise_secs": block_elapsed.as_secs_f64(),
                "attempt_empty_retries": empty_retry_attempt,
                "attempt_nc_retries": nc_retry_attempt,
                "active": committed_canvas,
                // Post-confidence-trim proposal length (== active when off).
                "kept": argmax_tokens.len(),
                "conf_trim_row": conf_trim_row,
                "stop_reason": denoise_stop.as_str(),
                "eos": rt.read_params().eos_token_id,
                "mean_entropy": st.mean_entropy,
                "pmax": pmax_from_rowstats(&rt.read_sample_rowstats(committed_canvas)),
                "entropy": &st.entropy[..committed_canvas],
                "argmax": &st.prev_argmax[..committed_canvas],
            }),
        );
    }

    let stats = crate::generate::BlockDenoiseStats {
        block_idx,
        max_blocks,
        steps_eff: block_step_count,
        denoise_secs: block_elapsed.as_secs_f64(),
        accept_per_step: accept_hist,
        min_ent_per_step: min_entropy_hist,
        mean_ent_per_step: mean_entropy_hist,
        low_ent_per_step: low_ent_hist,
        late_accept_sum: late_accept,
        late_min_ent,
        late_mean_ent,
        late_max_low_ent: late_low_ent,
        denoise_stop: denoise_stop.as_str().to_string(),
        kept_tokens: argmax_tokens.len(),
        token_ids: Vec::new(),
        stop_token_id: None,
        stop_offset: None,
        continued_past_stop: false,
    };
    Ok(BlockOutcome::Proposal(Box::new(ProposedBlock {
        token_ids: argmax_tokens,
        stats,
        block_idx,
    })))
}

/// Commit `kept` tokens of a proposal: append them to the token log, finalize
/// and record the block stats, and (when `extend`) re-encode the kept prefix
/// causally into KV. `extend: false` matches the monolithic contract for
/// turn-final blocks — the canvas KV is bidirectional, so the next turn
/// re-prefills it as prompt content.
pub fn commit_block(
    session: &mut StepGenerateSession,
    cfg: &StepGenerateConfig,
    ts: &mut TurnState,
    mut pb: ProposedBlock,
    kept: usize,
    extend: bool,
) -> Result<(), Error> {
    if kept > pb.token_ids.len() {
        return Err(Error::Runtime("commit_block: kept_len exceeds proposal"));
    }
    let canvas_len = CANVAS;
    let layers = session.layers;
    let block_base = ts.sequences.len();
    ts.sequences.extend_from_slice(&pb.token_ids[..kept]);
    ts.blocks_committed += 1;
    pb.stats.kept_tokens = kept;
    pb.stats.token_ids = pb.token_ids[..kept].to_vec();
    if let Some(ref obs) = cfg.block_commit_observer {
        obs(&pb.stats);
    }
    ts.block_stats.push(pb.stats);

    if extend && kept > 0 {
        let model_dir = session.model_dir.as_path();
        let shared_blob = session.rt.shared_dgq_blob();
        let rt = &mut session.rt;
        let extend_tokens = &ts.sequences[block_base..];
        let extend_started = Instant::now();
        let kv_before = rt.read_params().kv_len as usize;
        let new_kv_len = if crate::flags::fast_block_extend_enabled() {
            // Fast quantized causal extend of the committed canvas — the
            // same offset-resume as the cross-turn delta prefill. The
            // engine extend costs ~10s per 256-token block (the dominant
            // cost of multi-block replies); this is one prefill-chunk
            // forward (~0.85s).
            rt.prefill_chunks_from(kv_before, extend_tokens)?
        } else {
            if session.encoder.is_none() {
                session.encoder = Some(MonolithicEncoderCache::open_opt(
                    model_dir,
                    canvas_len,
                    cfg.max_seq,
                    Some(std::sync::Arc::clone(&shared_blob)),
                )?);
            }
            let encoder = session.encoder.as_mut().expect("encoder cache");
            extend_monolithic_kv_with_cache(
                encoder,
                rt.kvcache(),
                rt.layout(),
                kv_before,
                extend_tokens,
                cfg.max_seq,
                layers,
            )?
        };
        rt.set_kv_len(new_kv_len as u32);
        let block_extend = extend_started.elapsed();
        ts.extend_elapsed += block_extend;
        if progress_enabled() {
            eprintln!(
                "step-generate: extended kv {kv_before} -> {new_kv_len} (+{} tokens) ({block_extend:.2?})",
                extend_tokens.len()
            );
        }
    }
    Ok(())
}

/// Close the turn: record `kv_valid_tokens` (the causal contract the next
/// turn's cross-turn reuse diffs against) and assemble the output.
pub fn finish_turn(
    session: &mut StepGenerateSession,
    cfg: &StepGenerateConfig,
    ts: TurnState,
) -> Result<GenerateOutput, Error> {
    if progress_enabled() && !ts.session_telemetry.steps.is_empty() {
        let n = ts.session_telemetry.steps.len().max(1) as f64;
        let agg = ts.session_telemetry.aggregate_forward();
        eprintln!(
            "step-generate: P2.1 hot path mean {:.2} syncs/step, {:.1} KiB readback/step (DGQ_CHECK_LOGITS for opt-in logits scan)",
            agg.gpu_syncs as f64 / n,
            agg.gpu_readback_bytes as f64 / 1024.0 / n
        );
    }

    // Record the tokens whose causal KV is now valid (== the runtime kv_len):
    // the prompt plus any committed blocks that were causally extended. The
    // final (last/stopped) block's canvas KV is bidirectional, so it is NOT
    // counted — the next turn re-prefills it as prompt content. This is what
    // the next turn's cross-turn reuse diffs against.
    let valid_len = (session.rt.read_params().kv_len as usize).min(ts.sequences.len());
    session.kv_valid_tokens = ts.sequences[..valid_len].to_vec();

    Ok(GenerateOutput {
        token_ids: ts.sequences.clone(),
        denoise_steps_run: ts.denoise_steps_run,
        blocks_committed: ts.blocks_committed,
        cancelled: ts.cancelled,
        stopped_on_eot: ts.stopped_on_eot,
        stop_token_id: ts.stop_token_id,
        stop_block_idx: ts.stop_block_idx,
        stop_offset: ts.stop_offset,
        block_stats: ts.block_stats,
        block_steps_eff: ts.block_steps_eff,
        last_block_accept_hist: ts.last_block_accept_hist,
        last_block_min_entropy_hist: ts.last_block_min_entropy_hist,
        prefill_elapsed: ts.prefill_elapsed,
        denoise_elapsed: ts.denoise_elapsed,
        extend_elapsed: ts.extend_elapsed,
        session_telemetry: ts.session_telemetry,
        denoise_trace: Some(DenoiseTrace {
            schema_version: SCHEMA_VERSION,
            source: "rust-monolithic".into(),
            prompt: ts.label,
            prompt_token_ids: ts.prompt_token_ids,
            seed: cfg.seed,
            max_denoise_steps: ts.max_steps,
            layers: session.layers,
            max_new_tokens: cfg.max_new_tokens,
            weights_profile: Some(crate::generate_golden::monolithic_weights_profile().into()),
            entropy_bound: cfg.sampler.entropy_bound,
            step_traces: ts.step_traces,
            denoise_steps_run: ts.denoise_steps_run,
            blocks_committed: ts.blocks_committed,
            output_token_ids: ts.sequences,
            initial_canvas_ids: ts.initial_canvas_ids,
            canvas_rng: Some("rust-lcg".into()),
        }),
    })
}

/// Minimum steps invested before a prefix exit may fire. Higher than the
/// offline sim's 3: cheap full convergence typically lands at 5-12 steps on
/// short answers, and preempting it buys nothing.
const PREFIX_EXIT_MIN_STEPS: u32 = 5;
/// Absolute tail mean-entropy floor for an exit: the tail must be GENUINELY
/// stuck (strain stuck-tails sit at 1-3 nats), not merely a step or two from
/// converging (~0.1 nats). A relative 2τ bar fired on imminently-converging
/// tails and cascaded into mini-block churn.
const PREFIX_EXIT_TAIL_MIN_ENT: f32 = 0.3;

/// `DGQ_PREFIX_EXIT`: the settled active/2 head this step, or None. Live A/B
/// killed smaller heads (/4, /8): an early exit reshapes the NEXT block's
/// trajectory so it, too, settles a small head first — a cascade of
/// mini-blocks that each re-pay minimum steps (smoke 3-seed: blocks 19→23,
/// steps NET WORSE; the offline sim missed this trajectory feedback because
/// it never charged the successor block). An exit must be worth a block:
/// half-canvas head, hard-stuck tail, ≥2-step head stability.
fn prefix_exit_candidate(
    entropy: &[f32],
    cur_argmax: &[u32],
    prev_argmax: Option<&[u32]>,
    step_in_block: u32,
) -> Option<usize> {
    let tau = crate::flags::prefix_exit();
    if tau <= 0.0 || step_in_block < PREFIX_EXIT_MIN_STEPS {
        return None;
    }
    let prev = prev_argmax?;
    let active = entropy.len();
    let p = active / 2;
    if p < 16 || p >= active {
        return None;
    }
    let head = entropy[..p].iter().sum::<f32>() / p as f32;
    let tail = entropy[p..].iter().sum::<f32>() / (active - p) as f32;
    (head < tau
        && tail >= (tau * 2.0).max(PREFIX_EXIT_TAIL_MIN_ENT)
        && cur_argmax[..p] == prev[..p])
        .then_some(p)
}

/// `p_max = 1/sum` from the sampler rowstat plane `{mx, sum}` (the softmax is
/// centered on the max logit, so the max token's unnormalized weight is 1).
fn pmax_from_rowstats(rowstats: &[[f32; 2]]) -> Vec<f32> {
    rowstats
        .iter()
        .map(|s| if s[1] > 0.0 { 1.0 / s[1] } else { 0.0 })
        .collect()
}

/// Append one line to the `DGQ_TRACE_PMAX_JSONL` trace (E7 M0). Best-effort:
/// trace I/O must never fail a generation.
fn trace_pmax_append(path: &str, v: serde_json::Value) {
    use std::io::Write as _;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(f, "{v}");
    }
}
