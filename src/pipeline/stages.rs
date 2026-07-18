use super::{KvId, PipelineEvent, PipelineOp, PipelineStage};

/// Chain stage — the model-guided tool-call repair loop (user design
/// 2026-07-17): when a tool-mode reply ends with INVALID tool grammar
/// (incomplete call — the premature-stop strain symptom — or unparseable
/// calls), do not blindly re-roll. Feed the model a synthetic tool RESPONSE
/// describing the error and let it emit a corrected call; then Rewind to the
/// pre-turn mark so the corrupt call AND the error exchange never enter the
/// causal KV. The caller receives `prompt + the new reply`; finalize extends
/// the clean canonical from the mark. Every op flows through the inner stage,
/// so the whole choreography is op-logged and replayable.
pub struct ToolRepairStage<S> {
    pub(super) inner: S,
    tokenizer: std::sync::Arc<crate::tokenizer::Tokenizer>,
    attempts: u64,
}

impl<S: PipelineStage> ToolRepairStage<S> {
    pub fn new(
        inner: S,
        tokenizer: std::sync::Arc<crate::tokenizer::Tokenizer>,
        attempts: u64,
    ) -> Self {
        Self {
            inner,
            tokenizer,
            attempts,
        }
    }

    fn reply_verdict(
        &self,
        out: &crate::generate::GenerateOutput,
        prompt_len: usize,
    ) -> Result<(), &'static str> {
        let start = prompt_len.min(out.token_ids.len());
        let cleaned = crate::sample::strip_degenerate_token_ids(&out.token_ids[start..]);
        crate::tools::validate_tool_reply(&self.tokenizer.decode(&cleaned))
    }

    /// The `<|tool_response>` opener's token id (when it encodes to a single
    /// special), used to (a) let the repair regeneration run past it to a
    /// natural `<eos>` and (b) trim any hallucinated response tail after.
    fn response_opener_id(&self) -> Option<u32> {
        let ids = self.tokenizer.encode_with_specials("<|tool_response>");
        (ids.len() == 1).then(|| ids[0])
    }

    /// One error tool-response per invalid call attempt in `reply` (generic
    /// name when unrecoverable); a tailored error when the call was emitted
    /// inside the thinking block; a single generic error when the reply is
    /// invalid without any recognizable call attempt.
    pub(super) fn error_responses(reply: &str) -> String {
        const ERR: &str = "error: this tool call was malformed and has been discarded. \
             Regenerate your narration and tool calls, corrected, in full, then \
             continue as before.";
        const ERR_THINKING: &str = "error: you emitted this tool call inside your thinking \
             block, where it cannot be executed. It has been discarded. Regenerate \
             your reply with the narration and the tool call in the visible \
             response, then continue as before.";
        // Guarded renders: the call NAME is model-authored text — a name
        // containing special literals must not encode as protocol tokens.
        let mut out = String::new();
        if crate::tools::tool_call_lost_in_thinking(reply) {
            for name in crate::tools::thinking_call_names(reply) {
                let name = if name.is_empty() { "tool" } else { &name };
                out.push_str(&crate::tools::render_tool_response_guarded(
                    name,
                    &serde_json::json!({"content": ERR_THINKING}),
                ));
            }
        }
        // Malformed-call errors are judged on the VISIBLE text only: a
        // fragment rehearsed inside thought is not an attempt to act.
        for (name, valid) in crate::tools::scan_call_attempts(&crate::tools::strip_thinking(reply))
        {
            if valid {
                continue;
            }
            let name = if name.is_empty() { "tool" } else { &name };
            out.push_str(&crate::tools::render_tool_response_guarded(
                name,
                &serde_json::json!({"content": ERR}),
            ));
        }
        if out.is_empty() {
            out = crate::tools::render_tool_response_guarded(
                "tool",
                &serde_json::json!({"content": ERR}),
            );
        }
        out
    }
}

impl<S: PipelineStage> PipelineStage for ToolRepairStage<S> {
    fn call(&self, op: PipelineOp) -> PipelineEvent {
        let PipelineOp::Generate { prompt, cfg, label } = op else {
            return self.inner.call(op);
        };
        let mut mark = match self.inner.call(PipelineOp::Mark) {
            PipelineEvent::Marked { kv } => kv,
            ev => return ev,
        };
        let mut event = self.inner.call(PipelineOp::Generate {
            prompt: prompt.clone(),
            cfg: cfg.clone(),
            label: label.clone(),
        });
        // Set at the first invalid detection; a later valid verdict prints
        // the repaired-in line against it.
        let mut first_invalid_at: Option<std::time::Instant> = None;
        for attempt in 1..=self.attempts {
            let PipelineEvent::Generated { out, .. } = &event else {
                return event;
            };
            let Err(reason) = self.reply_verdict(out, prompt.len()) else {
                if let Some(t0) = first_invalid_at {
                    eprintln!(
                        "tool-repair: repaired — clean reply after {} attempt(s) ({:.1?})",
                        attempt - 1,
                        t0.elapsed()
                    );
                }
                return event;
            };
            first_invalid_at.get_or_insert_with(std::time::Instant::now);
            let reply_text = self
                .tokenizer
                .decode(&crate::sample::strip_degenerate_token_ids(
                    &out.token_ids[prompt.len().min(out.token_ids.len())..],
                ));
            eprintln!(
                "tool-repair: invalid reply ({reason}); returning an error tool response and regenerating (attempt {attempt}/{})",
                self.attempts
            );
            // Streaming clients have already seen the bad attempt: surface
            // the do-over as a thinking block instead of silent dead air.
            if let Some(notify) = cfg.status_notify.as_ref() {
                notify("I made a mistake with my tool call, let me try again.");
            }
            let err_text = Self::error_responses(&reply_text);
            // The repair continuation: the model's own turn as generated,
            // plus the error response(s) — a delta prefill over resident KV.
            let mut repair_prompt = out.token_ids.clone();
            repair_prompt.extend(self.tokenizer.encode_prompt(&err_text).0);
            // Let the regeneration run to a natural <eos>: the response-opener
            // stop is a forced early cut, and forced cuts are their own
            // degeneracy source. Hallucinated response text past the opener
            // is trimmed below instead.
            let mut regen_cfg = cfg.clone();
            if let Some(open) = self.response_opener_id() {
                regen_cfg.stop_token_ids.retain(|&id| id != open);
            }
            let repair_ev = self.inner.call(PipelineOp::Generate {
                prompt: repair_prompt.clone(),
                cfg: regen_cfg,
                label: label.clone(),
            });
            let PipelineEvent::Generated { out: repaired, .. } = repair_ev else {
                eprintln!("tool-repair: repair generate failed; keeping the invalid reply");
                return event;
            };
            // Surgical removal: rewind the corrupt call + error exchange out
            // of KV; the stitched output is original prompt + the NEW reply,
            // and finalize extends the clean canonical from there. KV-reuse-
            // first: the prompt's prefill is a valid causal prefix — rewind
            // to its end, not the pre-generate mark (which may be 0 on a
            // fresh conversation and would discard the whole prefill).
            let target = KvId {
                epoch: mark.epoch,
                pos: mark.pos.max(prompt.len()),
            };
            let rewound = match self.inner.call(PipelineOp::Rewind(target)) {
                PipelineEvent::Rewound { kv } => kv,
                ev => {
                    eprintln!("tool-repair: rewind failed ({ev:?}); keeping the invalid reply");
                    return event;
                }
            };
            mark = rewound;
            let mut stitched = *repaired;
            let new_reply_start = repair_prompt.len().min(stitched.token_ids.len());
            let mut new_reply = stitched.token_ids[new_reply_start..].to_vec();
            // Keep narration + calls; drop everything from the first response
            // opener on (the model ran to eos, so any response content there
            // is hallucinated — the REAL response arrives from the client).
            if let Some(open) = self.response_opener_id()
                && let Some(cut) = new_reply.iter().position(|&id| id == open)
            {
                new_reply.truncate(cut);
            }
            stitched.token_ids = prompt.clone();
            stitched.token_ids.extend(new_reply);
            event = PipelineEvent::Generated {
                out: Box::new(stitched),
                kv: rewound,
            };
        }
        if let PipelineEvent::Generated { out, .. } = &event {
            match self.reply_verdict(out, prompt.len()) {
                Err(reason) => eprintln!(
                    "tool-repair: still invalid after {} repair attempt(s) ({reason}); passing through",
                    self.attempts
                ),
                Ok(()) => {
                    if let Some(t0) = first_invalid_at {
                        eprintln!(
                            "tool-repair: repaired — clean reply after {} attempt(s) ({:.1?})",
                            self.attempts,
                            t0.elapsed()
                        );
                    }
                }
            }
        }
        event
    }
}

/// Chain stage: tool-grammar validation with the anti-collapse retry. On a
/// `Generate` whose reply speaks malformed tool grammar (incomplete call, or
/// `call:` that parses to nothing), Rewind to the pre-generate mark and
/// re-Generate at a bumped seed — the failed attempt never enters the causal
/// context (the surgical-removal move from the collapse root-cause). Retries
/// are ordinary ops through the inner stage, so they land in the op-log and
/// replay faithfully. All other ops pass through untouched.
pub struct ToolValidatorStage<S> {
    pub(super) inner: S,
    tokenizer: std::sync::Arc<crate::tokenizer::Tokenizer>,
    retries: u64,
}

impl<S: PipelineStage> ToolValidatorStage<S> {
    pub fn new(
        inner: S,
        tokenizer: std::sync::Arc<crate::tokenizer::Tokenizer>,
        retries: u64,
    ) -> Self {
        Self {
            inner,
            tokenizer,
            retries,
        }
    }

    fn reply_verdict(
        &self,
        out: &crate::generate::GenerateOutput,
        prompt_len: usize,
    ) -> Result<(), &'static str> {
        let start = prompt_len.min(out.token_ids.len());
        let cleaned = crate::sample::strip_degenerate_token_ids(&out.token_ids[start..]);
        crate::tools::validate_tool_reply(&self.tokenizer.decode(&cleaned))
    }
}

impl<S: PipelineStage> PipelineStage for ToolValidatorStage<S> {
    fn call(&self, op: PipelineOp) -> PipelineEvent {
        let PipelineOp::Generate { prompt, cfg, label } = op else {
            return self.inner.call(op);
        };
        let mut mark = match self.inner.call(PipelineOp::Mark) {
            PipelineEvent::Marked { kv } => kv,
            ev => return ev,
        };
        let mut event = self.inner.call(PipelineOp::Generate {
            prompt: prompt.clone(),
            cfg: cfg.clone(),
            label: label.clone(),
        });
        for attempt in 1..=self.retries {
            let PipelineEvent::Generated { out, .. } = &event else {
                return event;
            };
            let Err(reason) = self.reply_verdict(out, prompt.len()) else {
                return event;
            };
            eprintln!(
                "tool-validate: malformed reply ({reason}); rewinding and regenerating (attempt {attempt}/{})",
                self.retries
            );
            if let Some(notify) = cfg.status_notify.as_ref() {
                notify("I made a mistake with my tool call, let me try again.");
            }
            match self.inner.call(PipelineOp::Rewind(mark)) {
                // Rewind bumps the epoch; refresh the mark for a later retry.
                PipelineEvent::Rewound { kv } => mark = kv,
                ev => {
                    eprintln!("tool-validate: rewind failed ({ev:?}); keeping original reply");
                    return event;
                }
            }
            let mut retry_cfg = cfg.clone();
            retry_cfg.seed = cfg.seed.wrapping_add(0x9e37 * attempt);
            event = self.inner.call(PipelineOp::Generate {
                prompt: prompt.clone(),
                cfg: retry_cfg,
                label: label.clone(),
            });
        }
        if let PipelineEvent::Generated { out, .. } = &event
            && let Err(reason) = self.reply_verdict(out, prompt.len())
        {
            eprintln!(
                "tool-validate: still malformed after {} retries ({reason}); passing through",
                self.retries
            );
        }
        event
    }
}

/// The first wrapper stage: append every op + event to a JSONL op-log — the
/// design's durable replay artifact. Ops are logged at FULL fidelity (every
/// id, plus the replayable core of a Generate cfg); events carry digests
/// (token counts, ids FNV, fingerprints) so `replay` can re-execute the ops
/// and diff outcomes line by line.
pub struct OpLogStage<S> {
    inner: S,
    log: std::sync::Mutex<std::io::BufWriter<std::fs::File>>,
}

impl<S: PipelineStage> OpLogStage<S> {
    pub fn new(inner: S, path: &std::path::Path) -> std::io::Result<Self> {
        Ok(Self {
            inner,
            log: std::sync::Mutex::new(std::io::BufWriter::new(
                std::fs::File::options()
                    .create(true)
                    .append(true)
                    .open(path)?,
            )),
        })
    }

    /// Write the session header (`{"meta":…}`): the spawn parameters a replay
    /// needs to reconstruct the pipeline. Call once, before the first op.
    pub fn log_meta(&self, model: &str, max_seq: usize, steps: usize) {
        use std::io::Write;
        if let Ok(mut w) = self.log.lock() {
            let _ = writeln!(
                w,
                "{}",
                serde_json::json!({"meta": {"model": model, "max_seq": max_seq, "steps": steps}})
            );
            let _ = w.flush();
        }
    }
}

impl<S: PipelineStage> PipelineStage for OpLogStage<S> {
    fn call(&self, op: PipelineOp) -> PipelineEvent {
        use std::io::Write;
        let op_json = op.log_json();
        let event = self.inner.call(op);
        if let Ok(mut w) = self.log.lock() {
            let _ = writeln!(
                w,
                "{}",
                serde_json::json!({"op": op_json, "event": event.log_json()})
            );
            let _ = w.flush();
        }
        event
    }
}
