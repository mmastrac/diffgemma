//! `Worker`: the single GPU-owning thread that drains the job queue and runs
//! generation (`run` / `handle` / tool-compaction). Extracted from server.rs
//! (backlog item 10); the `Worker` struct + wire/job/event types stay in the
//! parent, so `use super::*` + ancestry reach them (incl. private fields).

use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use super::*;

impl Worker {
    /// Open the session (Metal objects are not `Send`, so they must be created
    /// on this thread), signal readiness, then drain the job queue one at a time.
    pub(crate) fn run(self, ready: mpsc::Sender<Result<(), String>>, jobs: mpsc::Receiver<Job>) {
        let open_started = std::time::Instant::now();
        let session =
            match crate::metal::StepGenerateSession::open(&self.model_dir, &self.base_cfg, None) {
                Ok((s, _compile)) => s,
                Err(err) => {
                    let _ = ready.send(Err(err.to_string()));
                    return;
                }
            };
        let ram_bytes = crate::flags::conv_cache_bytes();
        let disk_bytes = crate::flags::conv_disk_bytes();
        let disk_dir = crate::flags::conv_cache_dir();
        let mut manager =
            crate::conversation::ConversationManager::new(session, ram_bytes, disk_bytes, disk_dir);
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
            self.handle(&mut manager, tool_store.as_mut(), job);
        }
    }

    fn handle(
        &self,
        manager: &mut crate::conversation::ConversationManager,
        store: Option<&mut crate::toolcompact::ToolOutputStore>,
        job: Job,
    ) {
        let tool_mode = needs_tool_rendering(&job.messages, &job.tools);
        if tool_mode && let (Some(cc), Some(store)) = (&self.tool_compact, store) {
            return self.handle_tool_compact(manager, store, cc, job);
        }
        let thinking = job.enable_thinking;
        // Plain turns for the non-tool prompt/finalize path (gate-validated).
        let history = build_turns(&job.messages);

        let prompt = if tool_mode {
            let s = crate::tools::render_conversation(&job.messages, &job.tools, true, thinking);
            self.tokenizer.encode_with_specials(&s)
        } else {
            let opts = crate::chat_template::ChatFormatOptions {
                add_generation_prompt: true,
                enable_thinking: thinking,
            };
            match crate::chat_template::format_chat_token_ids(&self.tokenizer, &history, &opts) {
                Ok(p) => p,
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
        attach_stream_observer(&mut cfg, &mapper, &job.resp);

        // Route this prompt to its conversation (longest-prefix match), loading
        // that conversation's KV into the hot buffer. `generate_with_session`
        // then reuses the conversation's prefix and prefills only the new-turn
        // delta (or re-prefills whole for a new/evicted conversation).
        let conv_id = manager.activate(&prompt);

        let out =
            crate::metal::generate_with_session(manager.session_mut(), &prompt, &cfg, "serve");
        cfg.step_observer = None;
        match out {
            Ok(out) => {
                let raw = mapper.lock().unwrap().content().to_string();
                let reasoning = mapper.lock().unwrap().reasoning().to_string();
                let completion_tokens = out.token_ids.len().saturating_sub(prompt_len);

                // In tool mode the committed text may contain `<|tool_call>…` spans:
                // parse them out; `content` becomes the preamble before the first
                // call. A tool_call does NOT end the turn, so the model may emit
                // several calls + interleaved text — we capture all calls.
                let (content, tool_calls) = if tool_mode {
                    let calls = crate::tools::parse_tool_calls(&raw);
                    (
                        crate::tools::content_before_tool_calls(&raw),
                        crate::tools::to_openai_tool_calls(&calls),
                    )
                } else {
                    (raw, Vec::new())
                };

                // Finalize to the canonical completed-turns log so the snapshot is a
                // clean prefix of the next turn's prompt (reasoning never persists;
                // reuse survives a swap). Tool turns re-derive the assistant message
                // WITH its tool_calls via the tool-aware renderer.
                let canonical = if tool_mode {
                    let mut completed = job.messages.clone();
                    let mut assistant =
                        serde_json::json!({"role": "assistant", "content": content});
                    if !tool_calls.is_empty() {
                        assistant["tool_calls"] = serde_json::Value::Array(tool_calls.clone());
                    }
                    completed.push(assistant);
                    let s =
                        crate::tools::render_conversation(&completed, &job.tools, false, thinking);
                    Some(self.tokenizer.encode_with_specials(&s))
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

                if let Some(canonical) = canonical {
                    if let Err(err) = manager.finalize(conv_id, &canonical) {
                        // A failed finalize only costs reuse (next turn re-prefills);
                        // the reply is already correct.
                        eprintln!("serve: conversation finalize failed: {err}");
                    }
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
        manager: &mut crate::conversation::ConversationManager,
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
        let routing_prompt =
            self.tokenizer
                .encode_with_specials(&crate::tools::render_conversation(
                    &routing_messages,
                    &tools_aug,
                    true,
                    thinking,
                ));
        let conv_id = manager.activate(&routing_prompt);

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
                let p = self
                    .tokenizer
                    .encode_with_specials(&crate::tools::render_conversation(
                        ctx, &tools_aug, true, false,
                    ));
                p.len() + canvas + 64 > self.max_seq
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
                    manager.session_mut(),
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
            let p = self
                .tokenizer
                .encode_with_specials(&crate::tools::render_conversation(
                    &msgs, &tools_aug, true, thinking,
                ));
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
        let mut round_prompt = prompt;

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
            attach_stream_observer(&mut cfg, &mapper, &job.resp);

            let out = crate::metal::generate_with_session(
                manager.session_mut(),
                &round_prompt,
                &cfg,
                "serve",
            );
            let out = match out {
                Ok(o) => o,
                Err(err) => {
                    let _ = job.resp.send(ServerEvent::Error(format!("{err}")));
                    return;
                }
            };
            let raw = mapper.lock().unwrap().content().to_string();
            reasoning.push_str(mapper.lock().unwrap().reasoning());
            completion_tokens += out.token_ids.len().saturating_sub(round_prompt.len());
            stopped = out.stopped_on_eot;

            let calls = crate::tools::parse_tool_calls(&raw);
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
                resp_text.push_str(&crate::tools::render_tool_response(
                    tc::EXPAND_TOOL_NAME,
                    &serde_json::json!({ "content": excerpt }),
                ));
            }
            let mut ext = out.token_ids.clone();
            ext.extend(self.tokenizer.encode_with_specials(&resp_text));
            if ext.len() + canvas >= self.max_seq {
                eprintln!(
                    "serve: tool-compact: expand response would overflow the context; finishing turn"
                );
                break;
            }
            let session = manager.session_mut();
            let reuse = lcp(session.kv_valid_tokens(), &ext);
            session.truncate_kv_to(reuse);
            if let Err(err) = session.extend_kv(&ext[reuse..]) {
                eprintln!("serve: tool-compact: expand KV extend failed: {err}");
                break;
            }
            round_prompt = ext;
        }

        // Finalize with the substituted canonical log. The LCP truncation
        // evicts every ephemeral expand excerpt from KV (they diverge from the
        // canonical at the assistant-turn boundary).
        let content = content_pieces.join("\n\n").trim().to_string();
        let mut completed = messages_sub;
        let mut assistant = serde_json::json!({"role": "assistant", "content": content});
        if !tool_calls_out.is_empty() {
            assistant["tool_calls"] = serde_json::Value::Array(tool_calls_out.clone());
        }
        completed.push(assistant);
        let canonical = self
            .tokenizer
            .encode_with_specials(&crate::tools::render_conversation(
                &completed, &tools_aug, false, thinking,
            ));

        // Unblock the client before the finalize KV rebuild (same reasoning as
        // the plain path — the reply doesn't depend on it).
        let _ = job.resp.send(ServerEvent::Done {
            content,
            reasoning,
            tool_calls: tool_calls_out,
            prompt_tokens: prompt_len,
            completion_tokens,
            stopped,
        });

        if let Err(err) = manager.finalize(conv_id, &canonical) {
            eprintln!("serve: conversation finalize failed: {err}");
        }
    }
}
