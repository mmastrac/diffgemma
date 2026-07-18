//! `Worker`: the single GPU-owning thread that drains the job queue and runs
//! generation (`run` / `handle` / tool-compaction). Extracted from server.rs
//! (backlog item 10); the `Worker` struct + wire/job/event types stay in the
//! parent, so `use super::*` + ancestry reach them (incl. private fields).

use std::sync::atomic::AtomicUsize;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use super::*;

/// Loud tripwire for the token-injection hardening: client text carried
/// special-token literals that `encode_prompt` refused to promote to
/// protocol tokens (they encoded as plain text instead).
fn log_neutralized(n: usize) {
    if n > 0 {
        eprintln!("serve: neutralized {n} special-token literal(s) in client text");
    }
}

impl Worker {
    /// The one render+encode seam: guarded tool-grammar render of
    /// `messages`+`tools`, encoded with client-text special literals
    /// neutralized. Returns (token ids, neutralized count, guarded render) —
    /// callers decide whether to log the count (the sizing closure renders in
    /// a loop and stays silent).
    fn render_prompt(
        &self,
        messages: &[serde_json::Value],
        tools: &[serde_json::Value],
        add_generation_prompt: bool,
        thinking: bool,
    ) -> (Vec<u32>, usize, String) {
        let g = crate::tools::render_conversation_guarded(
            messages,
            tools,
            add_generation_prompt,
            thinking,
        );
        let (ids, neutralized) = self.tokenizer.encode_prompt(&g);
        (ids, neutralized, g)
    }

    /// Open the session (Metal objects are not `Send`, so they must be created
    /// on this thread), signal readiness, then drain the job queue one at a time.
    pub(crate) fn run(self, ready: mpsc::Sender<Result<(), String>>, jobs: mpsc::Receiver<Job>) {
        // Quiet is thread-local: the accept-thread set_quiet never reaches this
        // worker, so flip it here unless DGQ_QUIET was set in the env.
        if !crate::flags::quiet_set_by_user() {
            crate::flags::set_quiet(true);
        }
        let open_started = std::time::Instant::now();
        // The token pipeline owns the session + conversation registry on its
        // own thread; this worker is a message-layer client speaking ops.
        // Stage chains (tool compaction, op-log, span handles) insert between
        // the two without either side knowing.
        let pipeline =
            crate::pipeline::Pipeline::spawn(self.model_dir.clone(), self.max_seq, self.steps);
        // First chain link: with --log-dir, every op + event is journaled to
        // ops.jsonl beside the turn dumps (the replayable op-log).
        let stage: Box<dyn crate::pipeline::PipelineStage> = match &self.log_dir {
            Some(dir) => match crate::pipeline::OpLogStage::new(pipeline, &dir.join("ops.jsonl")) {
                Ok(logged) => {
                    // Session header: what `replay` needs to respawn the
                    // pipeline this log ran against.
                    logged.log_meta(
                        &self.model_dir.display().to_string(),
                        self.max_seq,
                        self.steps,
                    );
                    Box::new(logged)
                }
                Err(err) => {
                    let _ = ready.send(Err(format!("op-log open failed: {err}")));
                    return;
                }
            },
            None => Box::new(pipeline),
        };
        // Repair OUTSIDE the op-log: its Mark/Generate/Rewind choreography
        // flows through the log, so a repaired session's ops.jsonl replays
        // faithfully.
        let stage: Box<dyn crate::pipeline::PipelineStage> = if self.tool_repair {
            eprintln!(
                "serve: tool-repair ON (error tool-response -> corrected call -> corrupt exchange rewound)"
            );
            Box::new(crate::pipeline::ToolRepairStage::new(
                stage,
                Arc::clone(&self.tokenizer),
                1,
            ))
        } else {
            stage
        };
        // Validator OUTSIDE the op-log: its Mark/Rewind/retry ops flow through
        // the log, so a validated session's ops.jsonl replays faithfully.
        let stage: Box<dyn crate::pipeline::PipelineStage> = if self.tool_validate {
            eprintln!("serve: tool-validate ON (rewind + regenerate on malformed tool grammar)");
            Box::new(crate::pipeline::ToolValidatorStage::new(
                stage,
                Arc::clone(&self.tokenizer),
                1,
            ))
        } else {
            stage
        };
        match stage.call(crate::pipeline::PipelineOp::Ping) {
            crate::pipeline::PipelineEvent::Pong => {}
            ev => {
                let _ = ready.send(Err(format!("pipeline open failed: {ev:?}")));
                return;
            }
        }
        let ram_bytes = crate::flags::conv_cache_bytes();
        let disk_bytes = crate::flags::conv_disk_bytes();
        let mut tool_store = self.tool_compact.as_ref().map(|cc| {
            eprintln!(
                "serve: tool-compact ON (threshold {} tokens, {} expand rounds)",
                cc.threshold, cc.max_expand_rounds
            );
            crate::toolcompact::ToolOutputStore::new(crate::flags::tool_compact_dir())
        });
        let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
        eprintln!(
            "serve: model ready ({:.1}s, ctx={}, conv-cache={:.1} GiB RAM + {:.1} GiB SSD)",
            open_started.elapsed().as_secs_f64(),
            self.max_seq,
            gib(ram_bytes),
            gib(disk_bytes),
        );
        if ready.send(Ok(())).is_err() {
            return;
        }
        for job in jobs {
            self.handle(&*stage, tool_store.as_mut(), job);
        }
    }

    fn handle(
        &self,
        stage: &dyn crate::pipeline::PipelineStage,
        store: Option<&mut crate::toolcompact::ToolOutputStore>,
        job: Job,
    ) {
        use crate::pipeline::{PipelineEvent, PipelineOp};
        let tool_mode = needs_tool_rendering(&job.messages, &job.tools);
        if tool_mode && let (Some(cc), Some(store)) = (&self.tool_compact, store) {
            return self.handle_tool_compact(stage, store, cc, job);
        }
        let thinking = job.enable_thinking;
        // Plain turns for the non-tool prompt/finalize path (gate-validated).
        let history = build_turns(&job.messages);

        let (prompt, prompt_text) = if tool_mode {
            let (ids, neutralized, g) =
                self.render_prompt(&job.messages, &job.tools, true, thinking);
            log_neutralized(neutralized);
            (ids, crate::tools::strip_client_guards(&g))
        } else {
            let opts = crate::chat_template::ChatFormatOptions {
                add_generation_prompt: true,
                enable_thinking: thinking,
            };
            match crate::chat_template::format_chat_token_ids(&self.tokenizer, &history, &opts) {
                Ok(ids) => {
                    let s = self.tokenizer.decode(&ids);
                    (ids, s)
                }
                Err(err) => {
                    let _ = job.resp.send(ServerEvent::Error(format!("prompt: {err}")));
                    return;
                }
            }
        };
        let prompt_len = prompt.len();

        // Reserve one CANVAS block so the KV never overflows max_seq.
        let budget = self
            .max_seq
            .saturating_sub(prompt_len + crate::metal::CANVAS);
        if budget == 0 {
            let _ = job.resp.send(ServerEvent::Error(format!(
                "prompt ({prompt_len} tokens) leaves no room for a reply within the {}-token context",
                self.max_seq
            )));
            return;
        }

        let mut cfg = self.per_request_cfg(&job, budget);
        let mapper = self.make_mapper(&job);
        let log_seq = self.allocate_log_seq();
        let file_block = Arc::new(AtomicUsize::new(0));
        self.attach_block_commit_logger(&mut cfg, log_seq, &file_block);

        // Route to the conversation first so the resident KV is the right
        // prefix; the incremental log is the prompt delta beyond that.
        let (conv_id, reuse) = match stage.call(PipelineOp::Activate {
            prompt: prompt.clone(),
        }) {
            PipelineEvent::Activated {
                conv_id, reused, ..
            } => (conv_id, reused),
            ev => {
                let _ = job
                    .resp
                    .send(ServerEvent::Error(format!("activate: {ev:?}")));
                return;
            }
        };
        let delta_ids = &prompt[reuse..];
        let delta_text = self.tokenizer.decode(delta_ids);
        eprintln!(
            "serve: in +{}tok (reused {reuse}) {}",
            delta_ids.len(),
            trunc_preview(&delta_text, 120)
        );

        // Dry-start status: a big prompt delta means a long, otherwise-silent
        // prefill before the first streamed delta. Emit a synthetic reasoning
        // status + heartbeat; the stream observer ends it at the first
        // denoise step.
        let prefill_status = (job.stream
            && crate::flags::prefill_status_enabled()
            && delta_ids.len() >= PREFILL_STATUS_MIN_TOKENS)
            .then(|| Arc::new(std::sync::atomic::AtomicBool::new(false)));
        if let Some(ref started) = prefill_status {
            let rate = match self.prefill_rate_tok_s.load(Ordering::Relaxed) {
                0 => PREFILL_RATE_DEFAULT_TOK_S,
                r => r as f64,
            };
            eprintln!(
                "serve: prefill heartbeat ON ({} new tokens, ~{:.0}s expected at {rate:.0} tok/s)",
                delta_ids.len(),
                delta_ids.len() as f64 / rate.max(1.0),
            );
            spawn_prefill_status(&job.resp, delta_ids.len(), rate, started);
        }
        attach_stream_observer(
            &mut cfg,
            &mapper,
            &job.resp,
            tool_mode,
            true,
            prefill_status,
        );

        let out = match stage.call(PipelineOp::Generate {
            prompt: prompt.clone(),
            cfg: Box::new(cfg),
            label: "serve".into(),
        }) {
            PipelineEvent::Generated { out, .. } => Ok(*out),
            PipelineEvent::Error(err) => Err(crate::Error::Pipeline(err)),
            ev => Err(crate::Error::Pipeline(format!("unexpected event {ev:?}"))),
        };
        match out {
            Ok(out) => {
                flush_paced(&mapper, &job.resp, tool_mode);
                let raw_text = mapper.lock().unwrap().content().to_string();
                let reasoning = mapper.lock().unwrap().reasoning().to_string();
                let completion_tokens = out.token_ids.len().saturating_sub(prompt_len);
                let raw_ids_decode =
                    self.tokenizer
                        .decode(&crate::sample::strip_degenerate_token_ids(
                            &out.token_ids[prompt_len..],
                        ));

                // In tool mode the committed text may contain `<|tool_call>…` spans:
                // parse them out; `content` becomes the preamble before the first
                // call. A tool_call does NOT end the turn, so the model may emit
                // several calls + interleaved text — we capture all calls.
                let (content, tool_calls) = if tool_mode {
                    let mut calls = crate::tools::parse_tool_calls(&raw_text);
                    if calls.is_empty() && raw_text.contains("call:") {
                        eprintln!(
                            "serve: tool-mode: raw has call: but parse returned 0 ({})",
                            trunc_preview(&raw_text, 160)
                        );
                    }
                    // Splitter-disagreement recovery: the raw reply carries a
                    // well-formed call the mapper's channel split kept out of
                    // content (e.g. an unclosed thought span around the call —
                    // repair-round conditioning provokes these; the repair
                    // validator judges such replies clean, so dropping the
                    // call here handed the client prose with no tool_calls,
                    // field incident 2026-07-18). Adopt the calls from the
                    // thought-stripped raw decode instead of only logging.
                    if calls.is_empty() {
                        // Unclosed span: strip_thinking keeps its text.
                        let mut salvage = crate::tools::parse_tool_calls(
                            &crate::tools::strip_thinking(&raw_ids_decode),
                        );
                        if salvage.is_empty() {
                            // Closed span: the call lives in reasoning text.
                            salvage = crate::tools::parse_tool_calls(&reasoning);
                        }
                        if !salvage.is_empty() {
                            eprintln!(
                                "serve: CHANNEL MISMATCH: {} call(s) lost by the stream split; recovered from raw reply ({})",
                                salvage.len(),
                                trunc_preview(&raw_ids_decode, 160)
                            );
                            calls = salvage;
                        }
                    }
                    (
                        crate::tools::content_before_tool_calls(&raw_text),
                        crate::tools::to_openai_tool_calls(&calls),
                    )
                } else {
                    (raw_text.clone(), Vec::new())
                };

                let finish_reason = finish_reason_for(&tool_calls, out.stopped_on_eot);
                if let (Some(dir), Some(seq)) = (&self.log_dir, log_seq) {
                    let stop_token = out.stop_token_id.and_then(|id| {
                        self.tokenizer
                            .id_to_token(id)
                            .map(|s| s.to_string())
                            .or_else(|| Some(format!("id:{id}")))
                    });
                    let record = serde_json::json!({
                        "seq": seq,
                        "ts_unix_ms": now_ms(),
                        "tool_mode": tool_mode,
                        "tool_compact": false,
                        "seed": job.seed,
                        "max_tokens": job.max_tokens,
                        "messages": &job.messages,
                        "tools": &job.tools,
                        "prompt_text": &prompt_text,
                        "prompt_tokens": prompt_len,
                        "completion_tokens": completion_tokens,
                        "raw_text": &raw_text,
                        "raw_ids_decode": &raw_ids_decode,
                        "reasoning": &reasoning,
                        "content": &content,
                        "tool_calls": &tool_calls,
                        "stopped": out.stopped_on_eot,
                        "stop_token_id": out.stop_token_id,
                        "stop_token": stop_token,
                        "stop_block": out.stop_block_idx,
                        "stop_offset": out.stop_offset,
                        "finish_reason": finish_reason,
                        "cancelled": out.cancelled,
                        "blocks_committed": out.blocks_committed,
                        "prefill_secs": out.prefill_elapsed.as_secs_f64(),
                        "denoise_secs": out.denoise_elapsed.as_secs_f64(),
                        "extend_secs": out.extend_elapsed.as_secs_f64(),
                    });
                    self.write_serve_turn_log(dir, seq, &record);
                }

                let prefill_toks = delta_ids.len().max(1) as f64;
                let prefill_tps = if out.prefill_elapsed.as_secs_f64() > 0.0 {
                    prefill_toks / out.prefill_elapsed.as_secs_f64()
                } else {
                    0.0
                };
                // Feed the dry-start status's duration estimate; only real
                // prefills (skip reuse-hit no-ops whose rates are absurd).
                if out.prefill_elapsed.as_secs_f64() > 0.25 && prefill_tps > 1.0 {
                    self.prefill_rate_tok_s
                        .store(prefill_tps as u64, Ordering::Relaxed);
                }
                let denoise_tps = if out.denoise_elapsed.as_secs_f64() > 0.0 {
                    completion_tokens as f64 / out.denoise_elapsed.as_secs_f64()
                } else {
                    0.0
                };
                let call_note = if tool_calls.is_empty() {
                    String::new()
                } else {
                    let names: Vec<&str> = tool_calls
                        .iter()
                        .filter_map(|tc| {
                            tc.get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(|n| n.as_str())
                        })
                        .collect();
                    format!(" tools=[{}]", names.join(","))
                };
                let preview = if content.trim().is_empty() && !tool_calls.is_empty() {
                    "(tool_calls only)".to_string()
                } else {
                    trunc_preview(&content, 120)
                };
                eprintln!(
                    "serve: out {}tok {}{call_note}  prefill={prefill_tps:.0} tok/s  denoise={denoise_tps:.0} tok/s",
                    completion_tokens, preview,
                );
                if out.cancelled {
                    // Client hung up mid-generate (stream observer cancelled the
                    // token). Skip the finalize KV rebuild — nothing will
                    // continue this partial turn; the next request re-activates.
                    eprintln!(
                        "serve: client disconnected; generate cancelled after {completion_tokens} tokens"
                    );
                    return;
                }

                // Finalize to the canonical completed-turns log so the snapshot is a
                // clean prefix of the next turn's prompt (reasoning never persists;
                // reuse survives a swap). Tool turns re-derive the assistant message
                // WITH its tool_calls via the tool-aware renderer.
                let canonical = if tool_mode {
                    let mut completed = job.messages.clone();
                    // KV-reuse-first: a tool-calling turn's trailing prose
                    // renders AFTER the (future) tool responses in the
                    // grammar, so baking it into the canonical would make the
                    // canonical structurally non-prefix of the next request —
                    // a full re-prefill every tool turn. Finalize through the
                    // calls + response opener only (content empty in the
                    // CANONICAL render). The conversation itself is
                    // untouched: the client echoes the content back and the
                    // next render places it after the responses.
                    let mut assistant = serde_json::json!({
                        "role": "assistant",
                        "content": if tool_calls.is_empty() { content.as_str() } else { "" },
                    });
                    if !tool_calls.is_empty() {
                        assistant["tool_calls"] = serde_json::Value::Array(tool_calls.clone());
                    }
                    completed.push(assistant);
                    Some(
                        self.render_prompt(&completed, &job.tools, false, thinking)
                            .0,
                    )
                } else {
                    let mut completed = history.clone();
                    completed.push(crate::chat_template::ChatTurn::model(content.clone()));
                    let opts = crate::chat_template::ChatFormatOptions {
                        add_generation_prompt: false,
                        enable_thinking: false,
                    };
                    crate::chat_template::format_chat_token_ids(&self.tokenizer, &completed, &opts)
                        .map_err(|err| eprintln!("serve: canonical prompt build failed: {err}"))
                        .ok()
                };
                // Unblock the client BEFORE the finalize KV rebuild: finalize is
                // ~a prefill-chunk of GPU work the reply doesn't depend on, and
                // stalling the finish chunk on it made serve feel slower than
                // chat. The single worker still finishes it before the next job.
                let _ = job.resp.send(ServerEvent::Done {
                    content,
                    reasoning,
                    tool_calls,
                    prompt_tokens: prompt_len,
                    completion_tokens,
                    stopped: out.stopped_on_eot,
                });

                if let Some(canonical) = canonical
                    && let PipelineEvent::Error(err) =
                        stage.call(PipelineOp::Finalize { conv_id, canonical })
                {
                    // A failed finalize only costs reuse (next turn re-prefills);
                    // the reply is already correct.
                    eprintln!("serve: conversation finalize failed: {err}");
                }
            }
            Err(err) => {
                let _ = job.resp.send(ServerEvent::Error(format!("{err}")));
            }
        }
    }

    /// Per-request generation config shared by every serve path.
    fn per_request_cfg(&self, job: &Job, budget: usize) -> crate::metal::StepGenerateConfig {
        let mut cfg = self.base_cfg.clone();
        cfg.sampler = crate::sample::sampler_for_steps(self.steps, self.no_early_stop);
        cfg.max_new_tokens = job.max_tokens.map_or(budget, |c| c.min(budget));
        cfg.seed = job.seed.unwrap_or(self.base_cfg.seed);
        cfg.stop_token_ids = self.stop_token_ids.clone();
        // Continue-past-stop is OFF by default (2026-07-17 strain-battery
        // finding): a stop token inside an open tool call is a DEGRADATION
        // SYMPTOM — the deterministic collapse cell shows the deferred
        // continuation flooding for 8 blocks and never rescuing the call.
        // Honor the model's stop; repair belongs to the tool-repair stage
        // (feedback + rewind), not forced continuation. `DGQ_CONTINUE_PAST_STOP=1`
        // restores the old bet.
        cfg.continue_incomplete_tool_calls = crate::flags::continue_past_stop_enabled()
            && needs_tool_rendering(&job.messages, &job.tools);
        cfg.degenerate_reply_check =
            crate::chat_template::empty_reply_check(&self.model_dir, self.stop_token_ids.clone());
        cfg
    }

    fn make_mapper(&self, job: &Job) -> ServeMapper {
        Arc::new(Mutex::new(DiffusionStreamMapper::new(
            Arc::clone(&self.tokenizer),
            self.stop_token_ids.clone(),
            self.channel_open,
            self.channel_close,
            job.enable_thinking,
            job.emit_drafts,
            crate::flags::paced_stream_enabled(),
        )))
    }

    /// Tool-mode request with compaction on (the KV rewinder). Over-threshold
    /// tool responses are summarized by the model (checkpoint → summarize pass
    /// → rollback) and replaced in every render by `{summary, full_output_id}`;
    /// the model can retrieve slices of the stored full output via the built-in
    /// `expand_summary` tool, executed server-side in a bounded in-turn loop.
    /// Excerpts are turn-scoped: finalize rebuilds the canonical compacted log,
    /// evicting them from KV.
    fn handle_tool_compact(
        &self,
        stage: &dyn crate::pipeline::PipelineStage,
        store: &mut crate::toolcompact::ToolOutputStore,
        cc: &ToolCompactCfg,
        job: Job,
    ) {
        use crate::toolcompact as tc;
        let thinking = job.enable_thinking;
        let canvas = crate::metal::CANVAS;

        // The retrieval tool is appended (last) on EVERY render — prompt,
        // summarize passes, and canonical — so the system-turn prefix stays
        // stable for KV reuse.
        let mut tools_aug = job.tools.clone();
        tools_aug.push(tc::expand_summary_tool());

        let count = |s: &str| self.tokenizer.encode(s, false).len();
        let substitute =
            |store: &tc::ToolOutputStore, msgs: &[serde_json::Value]| -> Vec<serde_json::Value> {
                tc::compact_messages(msgs, cc.threshold, &count, &|h| {
                    store.get(h).map(|o| (o.id.clone(), o.summary.clone()))
                })
            };

        // Route with the current-store substitution: completed turns are
        // already compacted in the conversation's canonical log, so the
        // longest-prefix match holds whether or not this turn's NEW tool
        // responses (still verbose here) have been summarized yet.
        let routing_messages = substitute(store, &job.messages);
        let routing_prompt = {
            let (ids, neutralized, _) =
                self.render_prompt(&routing_messages, &tools_aug, true, thinking);
            log_neutralized(neutralized);
            ids
        };
        use crate::pipeline::{PipelineEvent, PipelineOp};
        let (conv_id, reuse) = match stage.call(PipelineOp::Activate {
            prompt: routing_prompt.clone(),
        }) {
            PipelineEvent::Activated {
                conv_id, reused, ..
            } => (conv_id, reused),
            ev => {
                let _ = job
                    .resp
                    .send(ServerEvent::Error(format!("activate: {ev:?}")));
                return;
            }
        };
        let delta_ids = &routing_prompt[reuse..];
        eprintln!(
            "serve: in +{}tok {}",
            delta_ids.len(),
            trunc_preview(&self.tokenizer.decode(delta_ids), 120)
        );

        // Summarize passes for new over-threshold responses, in message order
        // (each sees prior substitutions). Failures degrade to a mechanical
        // head+tail digest — compaction never blocks the reply.
        let mut summarized_any = false;
        for cand in tc::find_compactable(&job.messages, cc.threshold, &count) {
            if store.get(cand.hash).is_some() {
                continue;
            }
            let started = std::time::Instant::now();
            let mut ctx = substitute(store, &job.messages[..=cand.idx]);
            ctx.push(serde_json::json!({
                "role": "user", "content": tc::summarize_instruction(),
            }));
            let too_big = |ctx: &[serde_json::Value]| {
                // NOTE: thinking=false here where siblings pass the request
                // flag — preserved as-is (sizing-only render; PLAN item).
                self.render_prompt(ctx, &tools_aug, true, false).0.len() + canvas + 64
                    > self.max_seq
            };
            if too_big(&ctx) {
                // The verbose response alone blows the context: summarize a
                // head+tail slice instead (the stored full output keeps every
                // byte and stays reachable via expand_summary).
                ctx[cand.idx]["content"] =
                    serde_json::Value::String(tc::mechanical_summary(&cand.text, cc.threshold * 8));
            }
            let summary = if too_big(&ctx) {
                None
            } else {
                run_summarize_pass(
                    stage,
                    &self.tokenizer,
                    &self.base_cfg,
                    self.steps,
                    &self.stop_token_ids,
                    &self.model_dir,
                    self.max_seq,
                    cc.summarize_max_new,
                    &ctx,
                    &tools_aug,
                )
            };
            let summary = summary.unwrap_or_else(|| tc::mechanical_summary(&cand.text, 1024));
            eprintln!(
                "serve: tool-compact: summarized {} tokens as {} ({:.1}s):\n  | {}",
                count(&cand.text),
                tc::output_id(cand.hash),
                started.elapsed().as_secs_f64(),
                summary.replace('\n', "\n  | "),
            );
            if let Err(err) = store.put(cand.hash, &cand.text, summary) {
                // No store entry → the resolver misses → verbose passthrough.
                eprintln!(
                    "serve: tool-compact: store write failed for {}: {err}",
                    tc::output_id(cand.hash)
                );
            }
            summarized_any = true;
        }

        // Main generation over the fully substituted prompt, with a bounded
        // server-side expand_summary loop. When no summarize pass ran, the
        // routing render IS the main prompt — skip the duplicate render+encode.
        let (messages_sub, prompt) = if summarized_any {
            let msgs = substitute(store, &job.messages);
            let p = self.render_prompt(&msgs, &tools_aug, true, thinking).0;
            (msgs, p)
        } else {
            (routing_messages, routing_prompt)
        };
        let prompt_len = prompt.len();
        let budget = self.max_seq.saturating_sub(prompt_len + canvas);
        if budget == 0 {
            let _ = job.resp.send(ServerEvent::Error(format!(
                "prompt ({prompt_len} tokens) leaves no room for a reply within the {}-token context",
                self.max_seq
            )));
            return;
        }

        let mut content_pieces: Vec<String> = Vec::new();
        let mut reasoning = String::new();
        let mut completion_tokens = 0usize;
        let mut tool_calls_out: Vec<serde_json::Value> = Vec::new();
        let mut stopped = false;
        let mut stop_token_id: Option<u32> = None;
        let mut stop_block_idx: Option<usize> = None;
        let mut stop_offset: Option<usize> = None;
        let mut round_prompt = prompt;
        let mut raw_rounds: Vec<String> = Vec::new();
        let log_seq = self.allocate_log_seq();
        let file_block = Arc::new(AtomicUsize::new(0));

        for round in 0..=cc.max_expand_rounds {
            let remaining = job.max_tokens.map(|m| m.saturating_sub(completion_tokens));
            let room = self.max_seq.saturating_sub(round_prompt.len() + canvas);
            // Cap CUMULATIVE completion at the main prompt's budget so the
            // canonical log (prompt + all rounds' content) stays finalizable
            // within the context.
            let cum_room = budget.saturating_sub(completion_tokens);
            if remaining == Some(0) || room == 0 || cum_room == 0 {
                break;
            }
            let mut cfg = self.per_request_cfg(&job, room);
            cfg.max_new_tokens = cfg.max_new_tokens.min(cum_room);
            if let Some(r) = remaining {
                cfg.max_new_tokens = cfg.max_new_tokens.min(r);
            }
            // Fresh mapper per round: block indices restart at 1 on every
            // generation, and a prior round's stop token must not eat this
            // round's text.
            let mapper = self.make_mapper(&job);
            attach_stream_observer(&mut cfg, &mapper, &job.resp, true, true, None);
            self.attach_block_commit_logger(&mut cfg, log_seq, &file_block);

            let out = match stage.call(PipelineOp::Generate {
                prompt: round_prompt.clone(),
                cfg: Box::new(cfg),
                label: "serve".into(),
            }) {
                PipelineEvent::Generated { out, .. } => *out,
                ev => {
                    let _ = job.resp.send(ServerEvent::Error(format!("{ev:?}")));
                    return;
                }
            };
            flush_paced(&mapper, &job.resp, true);
            let raw = mapper.lock().unwrap().content().to_string();
            reasoning.push_str(mapper.lock().unwrap().reasoning());
            completion_tokens += out.token_ids.len().saturating_sub(round_prompt.len());
            stopped = out.stopped_on_eot;
            stop_token_id = out.stop_token_id;
            stop_block_idx = out.stop_block_idx;
            stop_offset = out.stop_offset;
            raw_rounds.push(raw.clone());
            if out.cancelled {
                // Client hung up mid-generate: end the turn here; skip the
                // finalize rebuild (next request re-activates the conversation).
                eprintln!("serve: client disconnected; generate cancelled mid-turn");
                return;
            }

            let mut calls = crate::tools::parse_tool_calls(&raw);
            if calls.is_empty() {
                // Splitter-disagreement recovery (see the non-compact path):
                // a call misclassified into the round's reasoning must not
                // silently end the turn as a plain answer.
                let salvage = crate::tools::parse_tool_calls(&mapper.lock().unwrap().reasoning());
                if !salvage.is_empty() {
                    eprintln!(
                        "serve: CHANNEL MISMATCH: {} call(s) lost by the stream split; recovered from round reasoning",
                        salvage.len(),
                    );
                    calls = salvage;
                }
            }
            let piece = crate::tools::content_before_tool_calls(&raw);
            if !piece.is_empty() {
                content_pieces.push(piece);
            }
            let (expand_calls, user_calls): (Vec<_>, Vec<_>) = calls
                .into_iter()
                .partition(|c| c.name == tc::EXPAND_TOOL_NAME);

            // Only a pure expand round continues the turn; anything else (user
            // tool calls, a plain answer) finishes it. expand_summary never
            // reaches the client.
            if !user_calls.is_empty() || expand_calls.is_empty() {
                if !expand_calls.is_empty() {
                    eprintln!(
                        "serve: tool-compact: dropped {} expand_summary call(s) mixed with user tool calls",
                        expand_calls.len()
                    );
                }
                tool_calls_out = crate::tools::to_openai_tool_calls(&user_calls);
                break;
            }
            if round == cc.max_expand_rounds {
                eprintln!(
                    "serve: tool-compact: expand round cap reached; {} call(s) unanswered",
                    expand_calls.len()
                );
                break;
            }

            // Serve the expand calls locally and continue the turn: extend KV
            // with the model's own emitted tokens + the canonical tool
            // responses (its actual ids — no decode/re-encode drift), then
            // re-enter generation with prompt == kv_valid_tokens (total reuse,
            // fresh block).
            let mut resp_text = String::new();
            for call in &expand_calls {
                let id = call
                    .arguments
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("");
                let excerpt = match store.read_full(id) {
                    Some(full) => tc::dispatch_expand(&call.arguments, &full),
                    None => tc::expand_error(&format!("unknown id {id:?}")),
                };
                let excerpt = cap_tokens(&self.tokenizer, &excerpt, cc.threshold / 2);
                eprintln!(
                    "serve: tool-compact: expand_summary({id}) round {} -> {} chars",
                    round + 1,
                    excerpt.len()
                );
                resp_text.push_str(&crate::tools::render_tool_response_guarded(
                    tc::EXPAND_TOOL_NAME,
                    &serde_json::json!({ "content": excerpt }),
                ));
            }
            let mut ext = out.token_ids.clone();
            ext.extend(self.tokenizer.encode_prompt(&resp_text).0);
            if ext.len() + canvas >= self.max_seq {
                eprintln!(
                    "serve: tool-compact: expand response would overflow the context; finishing turn"
                );
                break;
            }
            if let PipelineEvent::Error(err) = stage.call(PipelineOp::AlignTo {
                target: ext.clone(),
            }) {
                eprintln!("serve: tool-compact: expand KV align failed: {err}");
                break;
            }
            round_prompt = ext;
        }

        // Finalize with the substituted canonical log. The LCP truncation
        // evicts every ephemeral expand excerpt from KV (they diverge from the
        // canonical at the assistant-turn boundary).
        let content = content_pieces.join("\n\n").trim().to_string();
        let mut completed = messages_sub;
        // Same prefix-stable canonical rule as the plain path: a tool-calling
        // turn finalizes without its trailing prose (see handle()).
        let mut assistant = serde_json::json!({
            "role": "assistant",
            "content": if tool_calls_out.is_empty() { content.as_str() } else { "" },
        });
        if !tool_calls_out.is_empty() {
            assistant["tool_calls"] = serde_json::Value::Array(tool_calls_out.clone());
        }
        completed.push(assistant);
        let canonical = {
            let g =
                crate::tools::render_conversation_guarded(&completed, &tools_aug, false, thinking);
            self.tokenizer.encode_prompt(&g).0
        };

        // Unblock the client before the finalize KV rebuild (same reasoning as
        // the plain path — the reply doesn't depend on it).
        let finish_reason = finish_reason_for(&tool_calls_out, stopped);
        if let (Some(dir), Some(seq)) = (&self.log_dir, log_seq) {
            let stop_token = stop_token_id.and_then(|id| {
                self.tokenizer
                    .id_to_token(id)
                    .map(|s| s.to_string())
                    .or_else(|| Some(format!("id:{id}")))
            });
            let record = serde_json::json!({
                "seq": seq,
                "ts_unix_ms": now_ms(),
                "tool_mode": true,
                "tool_compact": true,
                "seed": job.seed,
                "max_tokens": job.max_tokens,
                "messages": &job.messages,
                "tools": &job.tools,
                "prompt_tokens": prompt_len,
                "completion_tokens": completion_tokens,
                "raw_rounds": &raw_rounds,
                "raw_text": raw_rounds.last().cloned().unwrap_or_default(),
                "reasoning": &reasoning,
                "content": &content,
                "tool_calls": &tool_calls_out,
                "stopped": stopped,
                "stop_token_id": stop_token_id,
                "stop_token": stop_token,
                "stop_block": stop_block_idx,
                "stop_offset": stop_offset,
                "finish_reason": finish_reason,
                "blocks_committed": file_block.load(Ordering::Relaxed),
            });
            self.write_serve_turn_log(dir, seq, &record);
        }
        eprintln!(
            "serve: out {}tok {}",
            completion_tokens,
            trunc_preview(&content, 120)
        );
        let _ = job.resp.send(ServerEvent::Done {
            content,
            reasoning,
            tool_calls: tool_calls_out,
            prompt_tokens: prompt_len,
            completion_tokens,
            stopped,
        });

        if let PipelineEvent::Error(err) = stage.call(PipelineOp::Finalize {
            conv_id,
            canonical: canonical.clone(),
        }) {
            eprintln!("serve: conversation finalize failed: {err}");
        }
    }

    fn allocate_log_seq(&self) -> Option<u64> {
        self.log_dir
            .as_ref()
            .map(|_| self.turn_seq.fetch_add(1, Ordering::Relaxed))
    }

    /// Flush each committed denoise block to `model-NNNNN-MMMMM.json` as it
    /// finishes (fsync'd), so killing the server mid-turn still leaves
    /// completed blocks on disk. `file_block` counts across expand rounds.
    fn attach_block_commit_logger(
        &self,
        cfg: &mut crate::metal::StepGenerateConfig,
        log_seq: Option<u64>,
        file_block: &Arc<AtomicUsize>,
    ) {
        let (Some(seq), Some(dir)) = (log_seq, self.log_dir.clone()) else {
            return;
        };
        let tok = Arc::clone(&self.tokenizer);
        let file_block = Arc::clone(file_block);
        cfg.block_commit_observer = Some(Arc::new(move |stats| {
            let n = file_block.fetch_add(1, Ordering::Relaxed) + 1;
            let stop_token = stats.stop_token_id.and_then(|id| {
                tok.id_to_token(id)
                    .map(|s| s.to_string())
                    .or_else(|| Some(format!("id:{id}")))
            });
            let mut record = block_stats_to_json(stats, &tok, stop_token.as_deref());
            if let Some(obj) = record.as_object_mut() {
                obj.insert("file_block".into(), serde_json::json!(n));
            }
            match write_model_block_log(&dir, seq, n, &record) {
                Ok(path) => eprintln!("serve: wrote {}", path.display()),
                Err(err) => eprintln!("serve: model log failed: {err}"),
            }
        }));
    }

    fn write_serve_turn_log(
        &self,
        dir: &std::path::Path,
        seq: u64,
        serve_record: &serde_json::Value,
    ) {
        match write_serve_log(dir, seq, serve_record) {
            Ok(path) => eprintln!("serve: wrote {}", path.display()),
            Err(err) => eprintln!("serve: serve log failed: {err}"),
        }
    }
}
