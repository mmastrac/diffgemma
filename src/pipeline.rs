//! Token pipeline — the serialized op-stream core (design: PLAN.md
//! "Token pipeline").
//!
//! One dedicated thread owns the model session AND the multi-conversation
//! registry (GPU + KV); clients speak token IDS only — never strings —
//! through an input queue of ops and an output queue of events. Current
//! clients: `ask` (Generate), `chat` (Generate per turn), `serve`
//! (Activate / Generate / Finalize / AlignTo / Mark+Rewind for
//! tool-compact). The per-block protocol
//! (BeginTurn/ProposeBlock/CommitBlock/DiscardBlock/EndTurn) exposes each
//! uncommitted canvas for inspection: partial commit covers
//! good-prefix/bad-tail, discard re-rolls from fresh noise, and `Generate`
//! remains the whole-turn composite over the identical primitives
//! (equivalence pinned byte-exactly by `per_block_ops_match_monolithic_generate`).
//!
//! KV identity: [`KvId`] = (epoch, position). Lineage-invalidating ops
//! (today: `Rewind`) bump the epoch; a rewind to a stale-epoch id fails
//! loudly instead of silently landing on different-lineage KV — the
//! OpenCode-collapse drift class, type-checked.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};

/// A KV position bound to its lineage epoch. Issued by pipeline events;
/// consumed by `Rewind`. Stale-epoch ids are rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvId {
    pub epoch: u64,
    pub pos: usize,
}

/// Input ops, applied strictly in order by the pipeline thread.
pub enum PipelineOp {
    /// Append tokens to the causal KV (prefill/extend path).
    Extend(Vec<u32>),
    /// Generate a reply from `prompt` (prefill delta + denoise). `cfg.layers`
    /// and `cfg.max_seq` are session-owned and overwritten with the pipeline's
    /// open-time values — the session's buffers are sized once at spawn.
    Generate {
        prompt: Vec<u32>,
        cfg: Box<crate::metal::StepGenerateConfig>,
        label: String,
    },
    /// Truncate the causal KV to `id.pos` (token-granular; O(1) below a
    /// sliding-ring wrap, ring-rebuild re-prefill past one). Bumps the epoch.
    Rewind(KvId),
    /// Replace the session state with a deterministic pseudorandom KV declared
    /// `tokens` long (~1 s for 100k vs a ~7-minute real prefill). The bytes are
    /// meaningless but finite and bit-deterministic — the substrate for the
    /// long-context order-of-operations gates. Bumps the epoch (the previous
    /// lineage is destroyed). Test infrastructure.
    SyntheticFill {
        tokens: usize,
        seed: u64,
    },
    /// FNV-1a of the READABLE state (causal token log + live KV: linear
    /// layers whole prefix, ring layers window-only) — the rewind
    /// byte-consistency probe. Ring residue no future read can observe is
    /// deliberately excluded.
    KvFingerprint,
    /// Readiness / liveness probe. The first `Pong` proves the model session
    /// opened (serve blocks its "listening" print on it).
    Ping,
    /// Capture the current lineage position (the checkpoint half of the
    /// checkpoint→generate→rollback pattern; the rollback half is `Rewind`).
    Mark,
    /// Route `prompt` to its conversation (longest-prefix match), swapping KV
    /// through the multi-conversation registry as needed. Bumps the epoch
    /// (activation may replace the whole resident state).
    Activate {
        prompt: Vec<u32>,
    },
    /// Finalize a conversation to its canonical (thought-free) token log:
    /// truncate to the common prefix, extend the canonical tail, record the
    /// log in the registry. Bumps the epoch.
    Finalize {
        conv_id: u64,
        canonical: Vec<u32>,
    },
    /// Align the resident KV to `target`: truncate to the longest common
    /// prefix with the causal log, extend the remainder. The tool-compact
    /// expand loop's primitive (and the session-level core of `Finalize`,
    /// without the registry write). Bumps the epoch.
    AlignTo {
        target: Vec<u32>,
    },
    /// Surgically replace `[start..end)` of the causal token log with
    /// `replacement`, re-encoding the tail at its new offset. The
    /// anti-collapse move: failed attempts, malformed calls, and expired
    /// transient hints are REMOVED from context instead of accumulating
    /// (tool-compact substitution and span handles are Splice clients).
    /// Composite of Rewind(start) + Extend(replacement + tail). Bumps the
    /// epoch.
    Splice {
        start: usize,
        end: usize,
        replacement: Vec<u32>,
    },
    /// Open a per-block turn: prefill `prompt`, then drive generation with
    /// `ProposeBlock`/`CommitBlock`/`DiscardBlock`, closing with `EndTurn`.
    /// While a turn is open, lineage-mutating ops are rejected. `Generate`
    /// remains the whole-turn composite (identical machinery, default policy).
    BeginTurn {
        prompt: Vec<u32>,
        cfg: Box<crate::metal::StepGenerateConfig>,
        label: String,
    },
    /// Denoise one block WITHOUT committing it. The `Proposed` event carries
    /// the canvas argmax for inspection; nothing lands in the token log or
    /// causal KV until `CommitBlock`.
    ProposeBlock,
    /// Commit `kept_len` tokens of the pending proposal (partial commit =
    /// good-prefix/bad-tail). `extend: false` skips the causal KV re-encode —
    /// the monolithic contract for turn-final blocks.
    CommitBlock {
        kept_len: usize,
        extend: bool,
    },
    /// Drop the pending proposal; the next `ProposeBlock` re-rolls the canvas
    /// from fresh noise (the rng advances every attempt).
    DiscardBlock,
    /// Close the open turn: record `kv_valid_tokens` and return the assembled
    /// `Generated` output (same shape as the whole-turn op).
    EndTurn,
    Shutdown,
}

/// One event per op, in op order.
pub enum PipelineEvent {
    Extended {
        kv: KvId,
    },
    /// `out` = the full generate output (token_ids includes the prompt); `kv`
    /// = the causally-resident KV position (the final block is committed to
    /// the reply but only causally extended when a later op needs it — same
    /// contract as the session).
    Generated {
        out: Box<crate::generate::GenerateOutput>,
        kv: KvId,
    },
    Rewound {
        kv: KvId,
    },
    Filled {
        kv: KvId,
    },
    Fingerprint {
        fnv: u64,
        kv: KvId,
    },
    Pong,
    Marked {
        kv: KvId,
    },
    /// `reused` = longest common prefix of the resident causal log and the
    /// activating prompt (the cross-turn reuse the serve log reports).
    Activated {
        conv_id: u64,
        kv: KvId,
        reused: usize,
    },
    Finalized {
        kv: KvId,
    },
    Aligned {
        kv: KvId,
        reused: usize,
    },
    /// `removed`/`inserted` = the splice delta; `kv` = the new log end.
    Spliced {
        kv: KvId,
        removed: usize,
        inserted: usize,
    },
    /// A turn is open; block ops drive it until `EndTurn`.
    TurnStarted {
        kv: KvId,
    },
    /// An uncommitted block awaits the commit decision. `stop` is the
    /// advisory stop-token scan over `cfg.stop_token_ids`: (offset, id).
    Proposed {
        ids: Vec<u32>,
        stop: Option<(usize, u32)>,
        steps_eff: u32,
        late_mean_ent: f32,
        kv: KvId,
    },
    /// No further proposals will come (budget spent, commit-guard abandon, or
    /// cancel); `EndTurn` collects the output.
    TurnStalled {
        reason: &'static str,
        kv: KvId,
    },
    BlockCommitted {
        kv: KvId,
        new_tokens: usize,
    },
    BlockDiscarded {
        kv: KvId,
    },
    Error(String),
    ShutDown,
}

impl std::fmt::Debug for PipelineEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Extended { kv } => write!(f, "Extended {{ kv: {kv:?} }}"),
            Self::Generated { out, kv } => write!(
                f,
                "Generated {{ tokens: {}, blocks: {}, cancelled: {}, kv: {kv:?} }}",
                out.token_ids.len(),
                out.blocks_committed,
                out.cancelled
            ),
            Self::Rewound { kv } => write!(f, "Rewound {{ kv: {kv:?} }}"),
            Self::Filled { kv } => write!(f, "Filled {{ kv: {kv:?} }}"),
            Self::Fingerprint { fnv, kv } => {
                write!(f, "Fingerprint {{ fnv: {fnv:#x}, kv: {kv:?} }}")
            }
            Self::Pong => write!(f, "Pong"),
            Self::Marked { kv } => write!(f, "Marked {{ kv: {kv:?} }}"),
            Self::Activated {
                conv_id,
                kv,
                reused,
            } => write!(
                f,
                "Activated {{ conv: {conv_id}, kv: {kv:?}, reused: {reused} }}"
            ),
            Self::Finalized { kv } => write!(f, "Finalized {{ kv: {kv:?} }}"),
            Self::Aligned { kv, reused } => {
                write!(f, "Aligned {{ kv: {kv:?}, reused: {reused} }}")
            }
            Self::Spliced {
                kv,
                removed,
                inserted,
            } => write!(
                f,
                "Spliced {{ kv: {kv:?}, removed: {removed}, inserted: {inserted} }}"
            ),
            Self::TurnStarted { kv } => write!(f, "TurnStarted {{ kv: {kv:?} }}"),
            Self::Proposed {
                ids,
                stop,
                steps_eff,
                late_mean_ent,
                kv,
            } => write!(
                f,
                "Proposed {{ tokens: {}, stop: {stop:?}, steps_eff: {steps_eff}, late_mean_ent: {late_mean_ent:.3}, kv: {kv:?} }}",
                ids.len()
            ),
            Self::TurnStalled { reason, kv } => {
                write!(f, "TurnStalled {{ reason: {reason:?}, kv: {kv:?} }}")
            }
            Self::BlockCommitted { kv, new_tokens } => {
                write!(
                    f,
                    "BlockCommitted {{ kv: {kv:?}, new_tokens: {new_tokens} }}"
                )
            }
            Self::BlockDiscarded { kv } => write!(f, "BlockDiscarded {{ kv: {kv:?} }}"),
            Self::Error(msg) => write!(f, "Error({msg:?})"),
            Self::ShutDown => write!(f, "ShutDown"),
        }
    }
}

/// One link in the op chain. The terminal stage is [`Pipeline`] (the model
/// thread); wrappers — tool compaction, span handles, validators, the op-log —
/// implement the same trait around an inner stage and compose freely. Clients
/// (ask/chat/serve) talk to the top of whatever chain they were given.
pub trait PipelineStage {
    fn call(&self, op: PipelineOp) -> PipelineEvent;
}

impl<S: PipelineStage + ?Sized> PipelineStage for Box<S> {
    fn call(&self, op: PipelineOp) -> PipelineEvent {
        (**self).call(op)
    }
}

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
    inner: S,
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
    fn error_responses(reply: &str) -> String {
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
    inner: S,
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

/// FNV-1a over token ids (the event digest `replay` diffs against).
pub(crate) fn ids_fnv(ids: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &id in ids {
        for b in id.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x1_0000_0000_01b3);
        }
    }
    h
}

impl PipelineOp {
    /// Short op name for error messages and progress lines.
    fn name(&self) -> &'static str {
        match self {
            Self::Extend(_) => "extend",
            Self::Generate { .. } => "generate",
            Self::Rewind(_) => "rewind",
            Self::SyntheticFill { .. } => "synthetic_fill",
            Self::KvFingerprint => "fingerprint",
            Self::Ping => "ping",
            Self::Mark => "mark",
            Self::Activate { .. } => "activate",
            Self::Finalize { .. } => "finalize",
            Self::AlignTo { .. } => "align_to",
            Self::Splice { .. } => "splice",
            Self::BeginTurn { .. } => "begin_turn",
            Self::ProposeBlock => "propose_block",
            Self::CommitBlock { .. } => "commit_block",
            Self::DiscardBlock => "discard_block",
            Self::EndTurn => "end_turn",
            Self::Shutdown => "shutdown",
        }
    }

    /// Full-fidelity JSON for the op-log: every id, plus the replayable core
    /// of a Generate/BeginTurn cfg. Observers and cancel tokens don't
    /// serialize — they never alter the token trajectory. `replay`
    /// reconstructs ops from exactly this shape (see `op_from_log_json`).
    pub fn log_json(&self) -> serde_json::Value {
        use serde_json::json;
        fn cfg_json(cfg: &crate::metal::StepGenerateConfig) -> serde_json::Value {
            json!({
                "seed": cfg.seed,
                "max_new_tokens": cfg.max_new_tokens,
                "steps": cfg.sampler.max_denoising_steps,
                "no_early_stop": cfg.no_early_stop,
                "sampler_no_early_stop": cfg.sampler.confidence_threshold == f32::MAX,
                "stop_token_ids": cfg.stop_token_ids,
                "continue_incomplete_tool_calls": cfg.continue_incomplete_tool_calls,
                "degenerate_reply_check": cfg.degenerate_reply_check.is_some(),
            })
        }
        match self {
            Self::Extend(ids) => json!({"extend": {"ids": ids}}),
            Self::Generate { prompt, cfg, label } => {
                json!({"generate": {"prompt": prompt, "label": label, "cfg": cfg_json(cfg)}})
            }
            Self::Rewind(id) => json!({"rewind": {"epoch": id.epoch, "pos": id.pos}}),
            Self::SyntheticFill { tokens, seed } => {
                json!({"synthetic_fill": {"tokens": tokens, "seed": seed}})
            }
            Self::KvFingerprint => json!("fingerprint"),
            Self::Ping => json!("ping"),
            Self::Mark => json!("mark"),
            Self::Activate { prompt } => json!({"activate": {"prompt": prompt}}),
            Self::Finalize { conv_id, canonical } => {
                json!({"finalize": {"conv_id": conv_id, "canonical": canonical}})
            }
            Self::AlignTo { target } => json!({"align_to": {"target": target}}),
            Self::Splice {
                start,
                end,
                replacement,
            } => json!({"splice": {"start": start, "end": end, "replacement": replacement}}),
            Self::BeginTurn { prompt, cfg, label } => {
                json!({"begin_turn": {"prompt": prompt, "label": label, "cfg": cfg_json(cfg)}})
            }
            Self::ProposeBlock => json!("propose_block"),
            Self::CommitBlock { kept_len, extend } => {
                json!({"commit_block": {"kept_len": kept_len, "extend": extend}})
            }
            Self::DiscardBlock => json!("discard_block"),
            Self::EndTurn => json!("end_turn"),
            Self::Shutdown => json!("shutdown"),
        }
    }
}

impl PipelineOp {
    /// Reconstruct an op from its [`Self::log_json`] form (the replay half).
    /// `model_dir` rebuilds the non-serializable cfg parts (the
    /// degenerate-reply check needs the tokenizer). `None` = unknown shape.
    pub fn from_log_json(v: &serde_json::Value, model_dir: &std::path::Path) -> Option<PipelineOp> {
        fn ids(v: &serde_json::Value) -> Option<Vec<u32>> {
            v.as_array()?
                .iter()
                .map(|x| x.as_u64().map(|n| n as u32))
                .collect()
        }
        fn cfg_from(
            v: &serde_json::Value,
            model_dir: &std::path::Path,
        ) -> Option<crate::metal::StepGenerateConfig> {
            let seed = v.get("seed")?.as_u64()?;
            let max_new = v.get("max_new_tokens")?.as_u64()? as usize;
            let steps = v.get("steps")?.as_u64()? as usize;
            let no_early = v.get("no_early_stop")?.as_bool()?;
            let sampler_ne = v.get("sampler_no_early_stop")?.as_bool()?;
            let stop_ids = ids(v.get("stop_token_ids")?)?;
            let cont = v.get("continue_incomplete_tool_calls")?.as_bool()?;
            let degen = v.get("degenerate_reply_check")?.as_bool()?;
            let sampler = crate::sample::sampler_for_steps(steps, sampler_ne);
            // layers/max_seq are session-owned and overwritten by the pipeline.
            let mut cfg = crate::metal::StepGenerateConfig::from_generate(
                seed, max_new, 0, 0, sampler, no_early,
            );
            cfg.stop_token_ids = stop_ids.clone();
            cfg.continue_incomplete_tool_calls = cont;
            if degen {
                cfg.degenerate_reply_check =
                    crate::chat_template::empty_reply_check(model_dir, stop_ids);
            }
            Some(cfg)
        }
        if let Some(s) = v.as_str() {
            return match s {
                "fingerprint" => Some(Self::KvFingerprint),
                "ping" => Some(Self::Ping),
                "mark" => Some(Self::Mark),
                "propose_block" => Some(Self::ProposeBlock),
                "discard_block" => Some(Self::DiscardBlock),
                "end_turn" => Some(Self::EndTurn),
                "shutdown" => Some(Self::Shutdown),
                _ => None,
            };
        }
        let obj = v.as_object()?;
        let (key, body) = obj.iter().next()?;
        match key.as_str() {
            "extend" => Some(Self::Extend(ids(body.get("ids")?)?)),
            "generate" => Some(Self::Generate {
                prompt: ids(body.get("prompt")?)?,
                cfg: Box::new(cfg_from(body.get("cfg")?, model_dir)?),
                label: body.get("label")?.as_str()?.to_string(),
            }),
            "rewind" => Some(Self::Rewind(KvId {
                epoch: body.get("epoch")?.as_u64()?,
                pos: body.get("pos")?.as_u64()? as usize,
            })),
            "synthetic_fill" => Some(Self::SyntheticFill {
                tokens: body.get("tokens")?.as_u64()? as usize,
                seed: body.get("seed")?.as_u64()?,
            }),
            "activate" => Some(Self::Activate {
                prompt: ids(body.get("prompt")?)?,
            }),
            "finalize" => Some(Self::Finalize {
                conv_id: body.get("conv_id")?.as_u64()?,
                canonical: ids(body.get("canonical")?)?,
            }),
            "align_to" => Some(Self::AlignTo {
                target: ids(body.get("target")?)?,
            }),
            "splice" => Some(Self::Splice {
                start: body.get("start")?.as_u64()? as usize,
                end: body.get("end")?.as_u64()? as usize,
                replacement: ids(body.get("replacement")?)?,
            }),
            "begin_turn" => Some(Self::BeginTurn {
                prompt: ids(body.get("prompt")?)?,
                cfg: Box::new(cfg_from(body.get("cfg")?, model_dir)?),
                label: body.get("label")?.as_str()?.to_string(),
            }),
            "commit_block" => Some(Self::CommitBlock {
                kept_len: body.get("kept_len")?.as_u64()? as usize,
                extend: body.get("extend")?.as_bool()?,
            }),
            _ => None,
        }
    }
}

impl PipelineEvent {
    /// Digest JSON for the op-log: enough to DIFF a replay against (token
    /// counts, ids FNV, KV fingerprints, lineage ids) without storing every
    /// generated id.
    pub fn log_json(&self) -> serde_json::Value {
        use serde_json::json;
        fn kv_json(kv: &KvId) -> serde_json::Value {
            json!({"epoch": kv.epoch, "pos": kv.pos})
        }
        match self {
            Self::Extended { kv } => json!({"extended": {"kv": kv_json(kv)}}),
            Self::Generated { out, kv } => json!({"generated": {
                "tokens": out.token_ids.len(),
                "blocks": out.blocks_committed,
                "cancelled": out.cancelled,
                "ids_fnv": format!("{:#x}", ids_fnv(&out.token_ids)),
                "kv": kv_json(kv),
            }}),
            Self::Rewound { kv } => json!({"rewound": {"kv": kv_json(kv)}}),
            Self::Filled { kv } => json!({"filled": {"kv": kv_json(kv)}}),
            Self::Fingerprint { fnv, kv } => {
                json!({"fingerprint": {"fnv": format!("{fnv:#x}"), "kv": kv_json(kv)}})
            }
            Self::Pong => json!("pong"),
            Self::Marked { kv } => json!({"marked": {"kv": kv_json(kv)}}),
            Self::Activated {
                conv_id,
                kv,
                reused,
            } => json!({"activated": {"conv_id": conv_id, "reused": reused, "kv": kv_json(kv)}}),
            Self::Finalized { kv } => json!({"finalized": {"kv": kv_json(kv)}}),
            Self::Aligned { kv, reused } => {
                json!({"aligned": {"reused": reused, "kv": kv_json(kv)}})
            }
            Self::Spliced {
                kv,
                removed,
                inserted,
            } => json!({"spliced": {"removed": removed, "inserted": inserted, "kv": kv_json(kv)}}),
            Self::TurnStarted { kv } => json!({"turn_started": {"kv": kv_json(kv)}}),
            Self::Proposed {
                ids,
                stop,
                steps_eff,
                late_mean_ent,
                kv,
            } => json!({"proposed": {
                "tokens": ids.len(),
                "ids_fnv": format!("{:#x}", ids_fnv(ids)),
                "stop": stop.map(|(off, id)| json!([off, id])),
                "steps_eff": steps_eff,
                "late_mean_ent": late_mean_ent,
                "kv": kv_json(kv),
            }}),
            Self::TurnStalled { reason, kv } => {
                json!({"turn_stalled": {"reason": reason, "kv": kv_json(kv)}})
            }
            Self::BlockCommitted { kv, new_tokens } => {
                json!({"block_committed": {"new_tokens": new_tokens, "kv": kv_json(kv)}})
            }
            Self::BlockDiscarded { kv } => json!({"block_discarded": {"kv": kv_json(kv)}}),
            Self::Error(msg) => json!({"error": msg}),
            Self::ShutDown => json!("shutdown"),
        }
    }
}

pub struct Pipeline {
    tx: Sender<PipelineOp>,
    rx: Receiver<PipelineEvent>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl PipelineStage for Pipeline {
    fn call(&self, op: PipelineOp) -> PipelineEvent {
        Pipeline::call(self, op)
    }
}

impl Pipeline {
    /// Spawn the pipeline thread and open the model session on it (Metal
    /// objects are created and stay on that thread; nothing GPU-owned
    /// crosses the channel).
    pub fn spawn(model_dir: PathBuf, max_seq: usize, steps: usize) -> Self {
        let (tx, op_rx) = channel();
        let (ev_tx, rx) = channel();
        // Quiet is thread-local: inherit the spawner's effective choice, or
        // serve/chat's set_quiet never reaches the thread that actually runs
        // generate (which is how the per-step debug spew came back in serve).
        let inherit_quiet = !crate::flags::progress_enabled();
        let join = std::thread::Builder::new()
            .name("token-pipeline".into())
            .spawn(move || {
                crate::flags::set_quiet(inherit_quiet);
                run_pipeline(&model_dir, max_seq, steps, &op_rx, &ev_tx)
            })
            .expect("spawn token-pipeline thread");
        Self {
            tx,
            rx,
            join: Some(join),
        }
    }

    /// Send one op and wait for its event (ops are strictly ordered, so this
    /// is a synchronous facade over the queue pair).
    pub fn call(&self, op: PipelineOp) -> PipelineEvent {
        if self.tx.send(op).is_err() {
            return PipelineEvent::Error("pipeline thread gone".into());
        }
        self.rx
            .recv()
            .unwrap_or_else(|_| PipelineEvent::Error("pipeline thread gone".into()))
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        let _ = self.tx.send(PipelineOp::Shutdown);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

fn run_pipeline(
    model_dir: &std::path::Path,
    max_seq: usize,
    steps: usize,
    op_rx: &Receiver<PipelineOp>,
    ev_tx: &Sender<PipelineEvent>,
) {
    use crate::metal::{StepGenerateConfig, StepGenerateSession, generate_with_session};

    let layers = match crate::commands::resolve_model_layers(model_dir, None) {
        Ok(n) => n,
        Err(err) => {
            let _ = ev_tx.send(PipelineEvent::Error(format!("model layers: {err}")));
            return;
        }
    };
    let open_cfg = StepGenerateConfig::from_generate(
        7,
        256,
        max_seq,
        layers,
        crate::sample::sampler_for_steps(steps, false),
        false,
    );
    let session = match StepGenerateSession::open(model_dir, &open_cfg, None) {
        Ok((s, _)) => s,
        Err(err) => {
            let _ = ev_tx.send(PipelineEvent::Error(format!("session open: {err}")));
            return;
        }
    };
    // The multi-conversation registry lives ON the pipeline thread with the
    // session it wraps (serve's Activate/Finalize become ops; single-client
    // callers like ask/chat simply never Activate).
    let mut manager = crate::conversation::ConversationManager::new(
        session,
        crate::flags::conv_cache_bytes(),
        crate::flags::conv_disk_bytes(),
        crate::flags::conv_cache_dir(),
    );

    let mut epoch: u64 = 0;
    let kv_id = |epoch: u64, manager: &mut crate::conversation::ConversationManager| KvId {
        epoch,
        pos: manager.session_mut().kv_valid_tokens().len(),
    };

    // Per-block turn state: at most one open turn, at most one pending
    // proposal. Lineage-mutating ops are rejected while a turn is open —
    // the turn's TurnState and the session would silently diverge otherwise.
    let mut turn: Option<Box<crate::metal::TurnState>> = None;
    let mut turn_cfg: Option<Box<crate::metal::StepGenerateConfig>> = None;
    let mut pending: Option<Box<crate::metal::ProposedBlock>> = None;
    fn conflicts_with_open_turn(op: &PipelineOp) -> bool {
        matches!(
            op,
            PipelineOp::Extend(_)
                | PipelineOp::Generate { .. }
                | PipelineOp::Rewind(_)
                | PipelineOp::SyntheticFill { .. }
                | PipelineOp::Activate { .. }
                | PipelineOp::Finalize { .. }
                | PipelineOp::AlignTo { .. }
                | PipelineOp::Splice { .. }
                | PipelineOp::BeginTurn { .. }
        )
    }

    while let Ok(op) = op_rx.recv() {
        if turn.is_some() && conflicts_with_open_turn(&op) {
            let refused = PipelineEvent::Error(format!(
                "op rejected while a turn is open (end or discard first): {}",
                op.name()
            ));
            if ev_tx.send(refused).is_err() {
                return;
            }
            continue;
        }
        let event = match op {
            PipelineOp::Extend(ids) => match manager.session_mut().extend_kv(&ids) {
                Ok(()) => PipelineEvent::Extended {
                    kv: kv_id(epoch, &mut manager),
                },
                Err(err) => PipelineEvent::Error(format!("extend: {err}")),
            },
            PipelineOp::Generate { prompt, cfg, label } => {
                let mut cfg = *cfg;
                cfg.layers = layers;
                cfg.max_seq = max_seq;
                match generate_with_session(manager.session_mut(), &prompt, &cfg, &label) {
                    Ok(out) => PipelineEvent::Generated {
                        out: Box::new(out),
                        kv: kv_id(epoch, &mut manager),
                    },
                    Err(err) => PipelineEvent::Error(format!("generate: {err}")),
                }
            }
            PipelineOp::Rewind(id) => {
                if id.epoch != epoch {
                    PipelineEvent::Error(format!(
                        "rewind: stale epoch {} (current {epoch})",
                        id.epoch
                    ))
                } else {
                    match manager.session_mut().truncate_kv_to(id.pos) {
                        Ok(()) => {
                            epoch += 1;
                            PipelineEvent::Rewound {
                                kv: kv_id(epoch, &mut manager),
                            }
                        }
                        Err(err) => PipelineEvent::Error(format!("rewind: {err}")),
                    }
                }
            }
            PipelineOp::SyntheticFill { tokens, seed } => {
                match manager.session_mut().synthetic_fill_kv(tokens, seed) {
                    Ok(()) => {
                        epoch += 1;
                        PipelineEvent::Filled {
                            kv: kv_id(epoch, &mut manager),
                        }
                    }
                    Err(err) => PipelineEvent::Error(format!("synthetic fill: {err}")),
                }
            }
            PipelineOp::KvFingerprint => PipelineEvent::Fingerprint {
                fnv: manager.session_mut().live_kv_fingerprint(),
                kv: kv_id(epoch, &mut manager),
            },
            PipelineOp::Ping => PipelineEvent::Pong,
            PipelineOp::Mark => PipelineEvent::Marked {
                kv: kv_id(epoch, &mut manager),
            },
            PipelineOp::Activate { prompt } => {
                let conv_id = manager.activate(&prompt);
                epoch += 1;
                let session = manager.session_mut();
                let reused = session
                    .kv_valid_tokens()
                    .iter()
                    .zip(prompt.iter())
                    .take_while(|(a, b)| a == b)
                    .count();
                // KV-reuse-first tail salvage: routing may match a
                // conversation whose log diverges from the prompt in its
                // tail. Truncate the resident KV to the common prefix so
                // every downstream consumer (reuse guard, delta prefill)
                // sees a clean extend — O(1) inside the routing slack.
                let salvage = if reused < session.kv_valid_tokens().len() {
                    session.truncate_kv_to(reused).err()
                } else {
                    None
                };
                match salvage {
                    None => PipelineEvent::Activated {
                        conv_id,
                        kv: kv_id(epoch, &mut manager),
                        reused,
                    },
                    Some(err) => PipelineEvent::Error(format!("activate salvage: {err}")),
                }
            }
            PipelineOp::Finalize { conv_id, canonical } => {
                match manager.finalize(conv_id, &canonical) {
                    Ok(()) => {
                        epoch += 1;
                        PipelineEvent::Finalized {
                            kv: kv_id(epoch, &mut manager),
                        }
                    }
                    Err(err) => PipelineEvent::Error(format!("finalize: {err}")),
                }
            }
            PipelineOp::AlignTo { target } => {
                let session = manager.session_mut();
                let reused = session
                    .kv_valid_tokens()
                    .iter()
                    .zip(target.iter())
                    .take_while(|(a, b)| a == b)
                    .count();
                let aligned = session
                    .truncate_kv_to(reused)
                    .and_then(|()| session.extend_kv(&target[reused..]));
                match aligned {
                    Ok(()) => {
                        epoch += 1;
                        PipelineEvent::Aligned {
                            kv: kv_id(epoch, &mut manager),
                            reused,
                        }
                    }
                    Err(err) => PipelineEvent::Error(format!("align: {err}")),
                }
            }
            PipelineOp::Splice {
                start,
                end,
                replacement,
            } => {
                let session = manager.session_mut();
                let log_len = session.kv_valid_tokens().len();
                if start > end || end > log_len {
                    PipelineEvent::Error(format!(
                        "splice: range {start}..{end} out of bounds (log {log_len})"
                    ))
                } else {
                    let inserted = replacement.len();
                    let removed = end - start;
                    let mut tail = replacement;
                    tail.extend_from_slice(&session.kv_valid_tokens()[end..]);
                    let spliced = session
                        .truncate_kv_to(start)
                        .and_then(|()| session.extend_kv(&tail));
                    match spliced {
                        Ok(()) => {
                            epoch += 1;
                            PipelineEvent::Spliced {
                                kv: kv_id(epoch, &mut manager),
                                removed,
                                inserted,
                            }
                        }
                        Err(err) => PipelineEvent::Error(format!("splice: {err}")),
                    }
                }
            }
            PipelineOp::BeginTurn { prompt, cfg, label } => {
                let mut cfg = *cfg;
                cfg.layers = layers;
                cfg.max_seq = max_seq;
                match crate::metal::begin_turn(manager.session_mut(), &prompt, &cfg, &label) {
                    Ok(ts) => {
                        turn = Some(Box::new(ts));
                        turn_cfg = Some(Box::new(cfg));
                        PipelineEvent::TurnStarted {
                            kv: kv_id(epoch, &mut manager),
                        }
                    }
                    Err(err) => PipelineEvent::Error(format!("begin_turn: {err}")),
                }
            }
            PipelineOp::ProposeBlock => match (turn.as_mut(), turn_cfg.as_ref()) {
                _ if pending.is_some() => PipelineEvent::Error(
                    "propose_block: proposal pending (commit or discard first)".into(),
                ),
                (Some(ts), Some(cfg)) => {
                    match crate::metal::propose_block(manager.session_mut(), cfg, ts) {
                        Ok(crate::metal::BlockOutcome::Proposal(pb)) => {
                            let stop = pb
                                .token_ids
                                .iter()
                                .position(|id| cfg.stop_token_ids.contains(id))
                                .map(|off| (off, pb.token_ids[off]));
                            let ev = PipelineEvent::Proposed {
                                ids: pb.token_ids.clone(),
                                stop,
                                steps_eff: pb.stats.steps_eff,
                                late_mean_ent: pb.stats.late_mean_ent,
                                kv: kv_id(epoch, &mut manager),
                            };
                            pending = Some(pb);
                            ev
                        }
                        Ok(outcome) => PipelineEvent::TurnStalled {
                            reason: match outcome {
                                crate::metal::BlockOutcome::Exhausted => "exhausted",
                                crate::metal::BlockOutcome::Abandoned => "abandoned",
                                crate::metal::BlockOutcome::Cancelled => "cancelled",
                                crate::metal::BlockOutcome::Proposal(_) => unreachable!(),
                            },
                            kv: kv_id(epoch, &mut manager),
                        },
                        Err(err) => PipelineEvent::Error(format!("propose_block: {err}")),
                    }
                }
                _ => PipelineEvent::Error("propose_block: no open turn".into()),
            },
            PipelineOp::CommitBlock { kept_len, extend } => {
                if pending
                    .as_ref()
                    .is_some_and(|pb| kept_len > pb.token_ids.len())
                {
                    PipelineEvent::Error(format!(
                        "commit_block: kept_len {kept_len} exceeds proposal ({})",
                        pending.as_ref().map_or(0, |pb| pb.token_ids.len())
                    ))
                } else {
                    match (turn.as_mut(), turn_cfg.as_ref(), pending.take()) {
                        (Some(ts), Some(cfg), Some(pb)) => match crate::metal::commit_block(
                            manager.session_mut(),
                            cfg,
                            ts,
                            *pb,
                            kept_len,
                            extend,
                        ) {
                            Ok(()) => PipelineEvent::BlockCommitted {
                                kv: kv_id(epoch, &mut manager),
                                new_tokens: turn.as_ref().expect("turn open").new_tokens(),
                            },
                            Err(err) => PipelineEvent::Error(format!("commit_block: {err}")),
                        },
                        (_, _, None) => {
                            PipelineEvent::Error("commit_block: no pending proposal".into())
                        }
                        _ => PipelineEvent::Error("commit_block: no open turn".into()),
                    }
                }
            }
            PipelineOp::DiscardBlock => {
                if pending.take().is_some() {
                    PipelineEvent::BlockDiscarded {
                        kv: kv_id(epoch, &mut manager),
                    }
                } else {
                    PipelineEvent::Error("discard_block: no pending proposal".into())
                }
            }
            PipelineOp::EndTurn => {
                if pending.is_some() {
                    PipelineEvent::Error(
                        "end_turn: proposal pending (commit or discard first)".into(),
                    )
                } else if let (Some(ts), Some(cfg)) = (turn.take(), turn_cfg.take()) {
                    match crate::metal::finish_turn(manager.session_mut(), &cfg, *ts) {
                        Ok(out) => PipelineEvent::Generated {
                            out: Box::new(out),
                            kv: kv_id(epoch, &mut manager),
                        },
                        Err(err) => PipelineEvent::Error(format!("end_turn: {err}")),
                    }
                } else {
                    PipelineEvent::Error("end_turn: no open turn".into())
                }
            }
            PipelineOp::Shutdown => {
                let _ = ev_tx.send(PipelineEvent::ShutDown);
                return;
            }
        };
        if ev_tx.send(event).is_err() {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scripted inner stage for validator tests: returns canned events in
    /// order and records the op names it saw.
    struct ScriptedStage {
        script: std::cell::RefCell<Vec<PipelineEvent>>,
        seen: std::cell::RefCell<Vec<&'static str>>,
    }

    impl PipelineStage for ScriptedStage {
        fn call(&self, op: PipelineOp) -> PipelineEvent {
            self.seen.borrow_mut().push(op.name());
            self.script.borrow_mut().remove(0)
        }
    }

    fn generated_event(token_ids: Vec<u32>, kv: KvId) -> PipelineEvent {
        PipelineEvent::Generated {
            out: Box::new(crate::generate::GenerateOutput {
                token_ids,
                denoise_steps_run: 1,
                blocks_committed: 1,
                cancelled: false,
                stopped_on_eot: true,
                stop_token_id: None,
                stop_block_idx: None,
                stop_offset: None,
                block_stats: Vec::new(),
                block_steps_eff: vec![1],
                last_block_accept_hist: Vec::new(),
                last_block_min_entropy_hist: Vec::new(),
                prefill_elapsed: std::time::Duration::ZERO,
                denoise_elapsed: std::time::Duration::ZERO,
                extend_elapsed: std::time::Duration::ZERO,
                #[cfg(target_os = "macos")]
                session_telemetry: crate::metal::SessionTelemetry::default(),
                #[cfg(target_os = "macos")]
                denoise_trace: None,
            }),
            kv,
        }
    }

    /// Scripted inner for the repair test: echoes each Generate's prompt with
    /// a canned reply suffix (so the stage's self-built repair prompt flows
    /// through naturally), and records ops + Generate prompt lengths.
    struct RepairScript {
        replies: std::cell::RefCell<Vec<Vec<u32>>>,
        seen: std::cell::RefCell<Vec<&'static str>>,
        gen_prompt_lens: std::cell::RefCell<Vec<usize>>,
        rewind_pos: std::cell::RefCell<Vec<usize>>,
    }

    impl PipelineStage for RepairScript {
        fn call(&self, op: PipelineOp) -> PipelineEvent {
            self.seen.borrow_mut().push(op.name());
            // Fresh conversation: the pre-generate mark sits at position 0.
            let kv = KvId { epoch: 0, pos: 0 };
            match op {
                PipelineOp::Mark => PipelineEvent::Marked { kv },
                PipelineOp::Rewind(id) => {
                    self.rewind_pos.borrow_mut().push(id.pos);
                    PipelineEvent::Rewound { kv: id }
                }
                PipelineOp::Generate { prompt, .. } => {
                    self.gen_prompt_lens.borrow_mut().push(prompt.len());
                    let reply = self.replies.borrow_mut().remove(0);
                    let mut ids = prompt;
                    ids.extend(reply);
                    generated_event(ids, kv)
                }
                _ => PipelineEvent::Error("unexpected op".into()),
            }
        }
    }

    /// A call emitted inside the thinking block gets the tailored error
    /// (naming the lost call), not the generic malformed-call text.
    #[test]
    fn repair_error_responses_for_thinking_call() {
        let lost = "<|channel>thought\n<|tool_call>call:write{path:<|\"|>/tmp/x<|\"|>}<tool_call|><channel|>";
        let errs =
            crate::tools::strip_client_guards(&ToolRepairStage::<RepairScript>::error_responses(
                lost,
            ));
        assert!(
            errs.contains("inside your thinking block"),
            "missing thinking-specific error: {errs}"
        );
        assert!(
            errs.contains("response:write"),
            "error must name the lost call: {errs}"
        );

        let malformed = "<|tool_call>call:write{path:";
        let errs =
            crate::tools::strip_client_guards(&ToolRepairStage::<RepairScript>::error_responses(
                malformed,
            ));
        assert!(
            errs.contains("was malformed"),
            "generic path broken: {errs}"
        );
    }

    /// The repair stage's choreography, pinned against a scripted inner: an
    /// invalid first reply (incomplete call) triggers error-response feedback
    /// (the second Generate's prompt = the failed turn + the rendered error
    /// tool response), then a Rewind — and the caller receives the ORIGINAL
    /// prompt + the corrected reply, with the corrupt exchange gone.
    #[test]
    fn tool_repair_feeds_error_and_rewinds_corrupt_call() {
        let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
            return;
        };
        let tokenizer = std::sync::Arc::new(
            crate::tokenizer::Tokenizer::load(&dir.join("tokenizer.json")).expect("tokenizer"),
        );
        let prompt: Vec<u32> = vec![10, 11, 12];
        let bad_reply = tokenizer.encode_with_specials("<|tool_call>call:write{path:");
        let good_reply = tokenizer
            .encode_with_specials("<|tool_call>call:write{path:<|\"|>/tmp/x.rs<|\"|>}<tool_call|>");
        // The regeneration runs to eos, so it may append a hallucinated tool
        // response — the stage must trim from the opener on.
        let mut regen_reply = good_reply.clone();
        regen_reply.extend(
            tokenizer
                .encode_with_specials("<|tool_response>response:write{value:<|\"|>fake<|\"|>}"),
        );
        let inner = RepairScript {
            replies: std::cell::RefCell::new(vec![bad_reply.clone(), regen_reply]),
            seen: std::cell::RefCell::new(Vec::new()),
            gen_prompt_lens: std::cell::RefCell::new(Vec::new()),
            rewind_pos: std::cell::RefCell::new(Vec::new()),
        };
        let repair = ToolRepairStage::new(inner, tokenizer, 1);
        let mut cfg = crate::metal::StepGenerateConfig::from_generate(
            42,
            64,
            1024,
            0,
            crate::sample::sampler_for_steps(8, false),
            false,
        );
        // The repair must announce itself through the status hook exactly
        // once (serve renders this as a thinking block on the stream).
        let statuses = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let statuses_sink = std::sync::Arc::clone(&statuses);
        cfg.status_notify = Some(std::sync::Arc::new(move |msg: &str| {
            statuses_sink.lock().unwrap().push(msg.to_string());
        }));
        let ev = repair.call(PipelineOp::Generate {
            prompt: prompt.clone(),
            cfg: Box::new(cfg),
            label: "repair-test".into(),
        });
        assert_eq!(
            *statuses.lock().unwrap(),
            vec!["I made a mistake with my tool call, let me try again.".to_string()],
            "repair must emit exactly one status notification"
        );
        let PipelineEvent::Generated { out, .. } = ev else {
            panic!("repair swallowed the Generated event: {ev:?}");
        };
        let mut expect = prompt.clone();
        expect.extend(good_reply);
        assert_eq!(
            out.token_ids, expect,
            "caller must receive original prompt + the corrected reply, hallucinated response trimmed"
        );
        assert_eq!(
            *repair.inner.seen.borrow(),
            vec!["mark", "generate", "generate", "rewind"],
            "repair choreography mismatch"
        );
        let lens = repair.inner.gen_prompt_lens.borrow();
        assert!(
            lens[1] > prompt.len() + bad_reply.len(),
            "repair prompt must include the failed turn + the error response ({} vs {})",
            lens[1],
            prompt.len() + bad_reply.len()
        );
        // KV-reuse-first: the rewind must land at the end of the prompt (its
        // prefill is a valid causal prefix), not the position-0 mark.
        assert_eq!(*repair.inner.rewind_pos.borrow(), vec![prompt.len()]);
    }

    /// The validator's retry choreography, pinned against a scripted inner
    /// stage: malformed first reply -> Mark, Generate, Rewind, Generate — and
    /// the second (clean) reply is what the caller receives. No GPU; the
    /// tokenizer comes from the model dir.
    #[test]
    fn tool_validator_rewinds_and_retries_malformed_reply() {
        let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
            return;
        };
        let tokenizer = std::sync::Arc::new(
            crate::tokenizer::Tokenizer::load(&dir.join("tokenizer.json")).expect("tokenizer"),
        );
        let prompt: Vec<u32> = vec![10, 11, 12];
        let malformed: Vec<u32> = {
            let mut ids = prompt.clone();
            ids.extend(tokenizer.encode_with_specials("<|tool_call>call:list_dir{path:"));
            ids
        };
        let clean: Vec<u32> = {
            let mut ids = prompt.clone();
            ids.extend(tokenizer.encode_with_specials("All done."));
            ids
        };
        let kv = |pos| KvId { epoch: 0, pos };
        let inner = ScriptedStage {
            script: std::cell::RefCell::new(vec![
                PipelineEvent::Marked { kv: kv(3) },
                generated_event(malformed, kv(3)),
                PipelineEvent::Rewound { kv: kv(3) },
                generated_event(clean.clone(), kv(3)),
            ]),
            seen: std::cell::RefCell::new(Vec::new()),
        };
        let validator = ToolValidatorStage::new(inner, tokenizer, 1);
        let cfg = crate::metal::StepGenerateConfig::from_generate(
            42,
            64,
            1024,
            0,
            crate::sample::sampler_for_steps(8, false),
            false,
        );
        let ev = validator.call(PipelineOp::Generate {
            prompt,
            cfg: Box::new(cfg),
            label: "validator-test".into(),
        });
        let PipelineEvent::Generated { out, .. } = ev else {
            panic!("validator swallowed the Generated event: {ev:?}");
        };
        assert_eq!(
            out.token_ids, clean,
            "caller must receive the retried reply"
        );
        assert_eq!(
            *validator.inner.seen.borrow(),
            vec!["mark", "generate", "rewind", "generate"],
            "retry choreography mismatch"
        );
    }

    /// The op-log round trip: ops journaled by OpLogStage reconstruct via
    /// `from_log_json` and re-execute on a fresh pipeline to bit-identical
    /// event digests (fingerprints, generated ids FNV, lineage ids). This is
    /// the standing replay gate — a field ops.jsonl IS a repro artifact.
    #[test]
    fn oplog_roundtrip_replays_bit_identically() {
        let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
            return;
        };
        let log_path =
            std::env::temp_dir().join(format!("dgq-oplog-gate-{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&log_path);
        let ids: Vec<u32> = (0..300u32).map(|i| 1000 + (i * 5087) % 30000).collect();
        let cfg = || {
            crate::metal::StepGenerateConfig::from_generate(
                42,
                192,
                4096,
                0,
                crate::sample::sampler_for_steps(24, false),
                false,
            )
        };

        // Record a session through the op-log.
        {
            let stage = OpLogStage::new(Pipeline::spawn(dir.clone(), 4096, 24), &log_path)
                .expect("op-log open");
            let PipelineEvent::Extended { .. } = stage.call(PipelineOp::Extend(ids.clone())) else {
                panic!("extend failed");
            };
            let PipelineEvent::Marked { kv: mark } = stage.call(PipelineOp::Mark) else {
                panic!("mark failed");
            };
            let PipelineEvent::Fingerprint { .. } = stage.call(PipelineOp::KvFingerprint) else {
                panic!("fingerprint failed");
            };
            let PipelineEvent::Generated { .. } = stage.call(PipelineOp::Generate {
                prompt: ids.clone(),
                cfg: Box::new(cfg()),
                label: "oplog-gate".into(),
            }) else {
                panic!("generate failed");
            };
            let PipelineEvent::Rewound { .. } = stage.call(PipelineOp::Rewind(mark)) else {
                panic!("rewind failed");
            };
            let PipelineEvent::Fingerprint { .. } = stage.call(PipelineOp::KvFingerprint) else {
                panic!("fingerprint failed");
            };
            // Stage (and its pipeline) drop here — the GPU session closes
            // before the replay pipeline opens.
        }

        // Replay the journal on a fresh pipeline; every event digest must match.
        let text = std::fs::read_to_string(&log_path).expect("read op-log");
        let p = Pipeline::spawn(dir.clone(), 4096, 24);
        for (lineno, line) in text.lines().enumerate() {
            let v: serde_json::Value = serde_json::from_str(line).expect("op-log line parses");
            let (Some(op_json), Some(recorded)) = (v.get("op"), v.get("event")) else {
                panic!("line {} has no op/event: {line}", lineno + 1);
            };
            let op = PipelineOp::from_log_json(op_json, &dir)
                .unwrap_or_else(|| panic!("line {}: op did not reconstruct", lineno + 1));
            let got = p.call(op).log_json();
            assert_eq!(
                &got,
                recorded,
                "replay diverged at line {} ({})",
                lineno + 1,
                op_json
            );
        }
        let _ = std::fs::remove_file(&log_path);
    }

    /// Wrap-crossing rewind gate (the PLAN follow-up, now load-bearing):
    /// every other byte-consistency gate runs below the sliding-ring wrap,
    /// but real sessions rewind — guard re-rolls, cancels, salvage truncates
    /// — at 5-13k tokens, past it. At ring 4096: (a) generate → rewind at
    /// ~5k takes the past-wrap O(1) truncate and must restore the base
    /// fingerprint; (b) a rewind deeper than the slack takes the ring-REBUILD
    /// path and must land bit-identical to a fresh build of the same prefix.
    #[test]
    fn wrap_crossing_rewind_restores_state() {
        let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
            return;
        };
        let p = Pipeline::spawn(dir, 8192, 8);
        // 5,000 tokens: past the default 4096 ring wrap.
        let ids: Vec<u32> = (0..5000u32).map(|i| 1000 + (i * 2477) % 30000).collect();
        let PipelineEvent::Extended { kv: mark } = p.call(PipelineOp::Extend(ids.clone())) else {
            panic!("extend failed");
        };
        let PipelineEvent::Fingerprint { fnv: fp_base, .. } = p.call(PipelineOp::KvFingerprint)
        else {
            panic!("fingerprint failed");
        };

        let cfg = crate::metal::StepGenerateConfig::from_generate(
            42,
            384,
            8192,
            0,
            crate::sample::sampler_for_steps(8, false),
            false,
        );
        let PipelineEvent::Generated { .. } = p.call(PipelineOp::Generate {
            prompt: ids.clone(),
            cfg: Box::new(cfg),
            label: "wrap-gate".into(),
        }) else {
            panic!("generate failed");
        };
        // (a) Past-wrap rewind of the generated turn: O(1) truncate territory.
        let ev = p.call(PipelineOp::Rewind(mark));
        let PipelineEvent::Rewound { kv } = ev else {
            panic!("rewind failed: {ev:?}");
        };
        let PipelineEvent::Fingerprint { fnv, .. } = p.call(PipelineOp::KvFingerprint) else {
            panic!("fingerprint failed");
        };
        assert_eq!(fnv, fp_base, "past-wrap generate->rewind left residue");

        // (b) Deep rewind (5000 -> 1500 = 3500 back, beyond the 2817 slack):
        // the ring-rebuild path. Must equal a fresh build of ids[..1500].
        let deep = KvId {
            epoch: kv.epoch,
            pos: 1500,
        };
        let ev = p.call(PipelineOp::Rewind(deep));
        let PipelineEvent::Rewound { kv } = ev else {
            panic!("deep rewind failed: {ev:?}");
        };
        let PipelineEvent::Fingerprint {
            fnv: fp_rebuilt, ..
        } = p.call(PipelineOp::KvFingerprint)
        else {
            panic!("fingerprint failed");
        };
        let zero = KvId {
            epoch: kv.epoch,
            pos: 0,
        };
        let PipelineEvent::Rewound { .. } = p.call(PipelineOp::Rewind(zero)) else {
            panic!("rewind to 0 failed");
        };
        let PipelineEvent::Extended { .. } = p.call(PipelineOp::Extend(ids[..1500].to_vec()))
        else {
            panic!("fresh extend failed");
        };
        let PipelineEvent::Fingerprint { fnv: fp_fresh, .. } = p.call(PipelineOp::KvFingerprint)
        else {
            panic!("fingerprint failed");
        };
        assert_eq!(
            fp_rebuilt, fp_fresh,
            "ring-rebuild rewind diverged from a fresh build of the same prefix"
        );
    }

    /// In-block re-roll residue gate: an E6-discarded canvas attempt must
    /// leave NO state the retry or later rewinds can observe. Forces exactly
    /// one degenerate-reply re-roll per round (via cfg.degenerate_reply_check)
    /// inside the standing generate → rewind loop: regeneration must stay
    /// bit-identical across rounds and every rewind must restore the base
    /// fingerprint. Also exercises shrink-on-retry (the retry canvas is 128).
    #[test]
    fn forced_reroll_leaves_no_residue() {
        let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
            return;
        };
        let p = Pipeline::spawn(dir, 4096, 8);
        let ids: Vec<u32> = (0..400u32).map(|i| 1000 + (i * 6659) % 30000).collect();
        let PipelineEvent::Extended { kv: mut mark } = p.call(PipelineOp::Extend(ids.clone()))
        else {
            panic!("extend failed");
        };
        let PipelineEvent::Fingerprint { fnv: fp0, .. } = p.call(PipelineOp::KvFingerprint) else {
            panic!("fingerprint failed");
        };

        let mut first_reply: Option<Vec<u32>> = None;
        for round in 0..2 {
            let mut cfg = crate::metal::StepGenerateConfig::from_generate(
                42,
                192,
                4096,
                0,
                crate::sample::sampler_for_steps(8, false),
                false,
            );
            // Fires once per round: attempt 0's canvas is declared degenerate
            // and discarded; attempt 1 (shrunk canvas) is the reply.
            let fired = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            cfg.degenerate_reply_check = Some(std::sync::Arc::new(move |_ids: &[u32]| {
                !fired.swap(true, std::sync::atomic::Ordering::Relaxed)
            }));
            let ev = p.call(PipelineOp::Generate {
                prompt: ids.clone(),
                cfg: Box::new(cfg),
                label: "reroll-gate".into(),
            });
            let PipelineEvent::Generated { out, .. } = ev else {
                panic!("generate failed at round {round}: {ev:?}");
            };
            match &first_reply {
                None => first_reply = Some(out.token_ids.clone()),
                Some(f) => assert_eq!(
                    f, &out.token_ids,
                    "re-rolled regeneration diverged (round {round})"
                ),
            }
            let ev = p.call(PipelineOp::Rewind(mark));
            let PipelineEvent::Rewound { kv } = ev else {
                panic!("rewind failed at round {round}: {ev:?}");
            };
            mark = kv;
            let PipelineEvent::Fingerprint { fnv, .. } = p.call(PipelineOp::KvFingerprint) else {
                panic!("fingerprint failed at round {round}");
            };
            assert_eq!(
                fnv, fp0,
                "discarded re-roll attempt left KV residue (round {round})"
            );
        }
    }

    /// The KV-reuse-first salvage gate: a follow-up prompt that diverges from
    /// the finalized conversation only in its tail must route BACK to that
    /// conversation with the shared prefix reused (truncate-to-LCP on
    /// activate) — not fork a fresh conversation with reused: 0, which was
    /// the OpenCode full-re-prefill-per-tool-turn bug. Deep divergence still
    /// declines (a rebuild-depth truncate costs a fresh prefill anyway).
    #[test]
    fn activate_salvages_tail_divergent_conversation() {
        let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
            return;
        };
        let p = Pipeline::spawn(dir, 4096, 24);
        let canonical: Vec<u32> = (0..600u32).map(|i| 1000 + (i * 3571) % 30000).collect();

        let ev = p.call(PipelineOp::Activate {
            prompt: canonical.clone(),
        });
        let PipelineEvent::Activated {
            conv_id, reused, ..
        } = ev
        else {
            panic!("first activate failed: {ev:?}");
        };
        assert_eq!(reused, 0);
        let PipelineEvent::Finalized { kv } = p.call(PipelineOp::Finalize {
            conv_id,
            canonical: canonical.clone(),
        }) else {
            panic!("finalize failed");
        };
        assert_eq!(kv.pos, canonical.len());

        // Follow-up whose last 40 canonical tokens were re-rendered.
        let mut next = canonical[..560].to_vec();
        next.extend((0..240u32).map(|i| 7000 + i));
        let ev = p.call(PipelineOp::Activate { prompt: next });
        let PipelineEvent::Activated {
            conv_id: id2,
            reused,
            kv,
        } = ev
        else {
            panic!("salvage activate failed: {ev:?}");
        };
        assert_eq!(id2, conv_id, "tail divergence must salvage, not fork");
        assert_eq!(reused, 560, "shared prefix not reused");
        assert_eq!(
            kv.pos, 560,
            "resident KV not truncated to the shared prefix"
        );

        // Divergence far above the tail slack: fresh conversation.
        let mut deep = canonical[..100].to_vec();
        deep.extend((0..300u32).map(|i| 8000 + i));
        let ev = p.call(PipelineOp::Activate { prompt: deep });
        let PipelineEvent::Activated {
            conv_id: id3,
            reused,
            ..
        } = ev
        else {
            panic!("deep activate failed: {ev:?}");
        };
        assert_ne!(
            id3, conv_id,
            "deep divergence must not steal the conversation"
        );
        assert_eq!(reused, 0);
    }

    /// The Splice gate: surgically replacing a mid-log span must leave state
    /// byte-identical to building the target log fresh — no residue from the
    /// removed span, and the tail re-encode at its new offset must be
    /// bit-deterministic (the offset-resume property, exercised at a splice
    /// boundary instead of a turn boundary).
    #[test]
    fn splice_matches_fresh_build() {
        let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
            return;
        };
        let p = Pipeline::spawn(dir, 4096, 24);
        let ids: Vec<u32> = (0..600u32).map(|i| 1000 + (i * 6151) % 30000).collect();
        let replacement: Vec<u32> = (0..50u32).map(|i| 9000 + i * 17).collect();
        let mut target = ids[..200].to_vec();
        target.extend_from_slice(&replacement);
        target.extend_from_slice(&ids[300..]);

        let PipelineEvent::Extended { .. } = p.call(PipelineOp::Extend(ids.clone())) else {
            panic!("extend failed");
        };
        let ev = p.call(PipelineOp::Splice {
            start: 200,
            end: 300,
            replacement,
        });
        let PipelineEvent::Spliced {
            kv,
            removed,
            inserted,
        } = ev
        else {
            panic!("splice failed: {ev:?}");
        };
        assert_eq!((removed, inserted), (100, 50));
        assert_eq!(kv.pos, target.len(), "spliced log length wrong");
        let PipelineEvent::Fingerprint {
            fnv: fp_spliced, ..
        } = p.call(PipelineOp::KvFingerprint)
        else {
            panic!("fingerprint failed");
        };

        // Fresh build of the identical target log on the same session.
        let zero = KvId {
            epoch: kv.epoch,
            pos: 0,
        };
        let PipelineEvent::Rewound { .. } = p.call(PipelineOp::Rewind(zero)) else {
            panic!("rewind to 0 failed");
        };
        let PipelineEvent::Extended { .. } = p.call(PipelineOp::Extend(target)) else {
            panic!("fresh extend failed");
        };
        let PipelineEvent::Fingerprint { fnv: fp_fresh, .. } = p.call(PipelineOp::KvFingerprint)
        else {
            panic!("fingerprint failed");
        };
        assert_eq!(
            fp_spliced, fp_fresh,
            "splice left residue vs a fresh build of the same log"
        );

        // Range validation fails loudly.
        match p.call(PipelineOp::Splice {
            start: 10_000,
            end: 10_001,
            replacement: vec![1],
        }) {
            PipelineEvent::Error(msg) => {
                assert!(msg.contains("out of bounds"), "wrong error: {msg}")
            }
            ev => panic!("out-of-range splice must fail, got {ev:?}"),
        }
    }

    /// The P2 equivalence gate: a client driving
    /// BeginTurn/ProposeBlock/CommitBlock/EndTurn with the default policy
    /// (commit everything, mirror the monolithic extend rule) must produce
    /// byte-identical token ids to the whole-turn Generate op at the same
    /// seed. Pins the per-block protocol to the path it decomposed.
    #[test]
    fn per_block_ops_match_monolithic_generate() {
        let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
            return;
        };
        let p = Pipeline::spawn(dir, 4096, 24);
        let ids: Vec<u32> = (0..400u32).map(|i| 1000 + (i * 7919) % 30000).collect();
        let PipelineEvent::Extended { kv: mark } = p.call(PipelineOp::Extend(ids.clone())) else {
            panic!("extend failed");
        };

        let max_new = 512usize; // two blocks
        let cfg = || {
            crate::metal::StepGenerateConfig::from_generate(
                42,
                max_new,
                4096,
                0, // session-owned; overwritten by the pipeline
                crate::sample::sampler_for_steps(24, false),
                false,
            )
        };
        let ev = p.call(PipelineOp::Generate {
            prompt: ids.clone(),
            cfg: Box::new(cfg()),
            label: "p2-mono".into(),
        });
        let PipelineEvent::Generated { out: mono, .. } = ev else {
            panic!("monolithic generate failed: {ev:?}");
        };

        let PipelineEvent::Rewound { .. } = p.call(PipelineOp::Rewind(mark)) else {
            panic!("rewind failed");
        };

        let ev = p.call(PipelineOp::BeginTurn {
            prompt: ids.clone(),
            cfg: Box::new(cfg()),
            label: "p2-ops".into(),
        });
        let PipelineEvent::TurnStarted { .. } = ev else {
            panic!("begin_turn failed: {ev:?}");
        };
        let mut new_tokens = 0usize;
        loop {
            match p.call(PipelineOp::ProposeBlock) {
                PipelineEvent::Proposed {
                    ids: block, stop, ..
                } => {
                    let kept = stop.map_or(block.len(), |(off, _)| off);
                    let end = stop.is_some();
                    let remaining = max_new - new_tokens;
                    let is_last = remaining <= 256;
                    let extend = !end && !is_last && kept > 0;
                    let ev = p.call(PipelineOp::CommitBlock {
                        kept_len: kept,
                        extend,
                    });
                    let PipelineEvent::BlockCommitted { new_tokens: n, .. } = ev else {
                        panic!("commit failed: {ev:?}");
                    };
                    new_tokens = n;
                    if end {
                        break;
                    }
                }
                PipelineEvent::TurnStalled { .. } => break,
                ev => panic!("unexpected propose event: {ev:?}"),
            }
        }
        let ev = p.call(PipelineOp::EndTurn);
        let PipelineEvent::Generated { out: ops, .. } = ev else {
            panic!("end_turn failed: {ev:?}");
        };
        assert_eq!(
            mono.token_ids, ops.token_ids,
            "per-block ops diverged from the monolithic path"
        );
        assert_eq!(mono.blocks_committed, ops.blocks_committed);
    }

    /// Partial commit (good-prefix/bad-tail) + discard leave a consistent
    /// lineage: EndTurn reports exactly the committed prefix, a second
    /// proposal generates beyond it, lineage-mutating ops are rejected while
    /// the turn is open, and a rewind past the whole turn restores the base
    /// fingerprint bit-exactly.
    #[test]
    fn per_block_partial_commit_and_discard_consistency() {
        let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
            return;
        };
        let p = Pipeline::spawn(dir, 4096, 24);
        let ids: Vec<u32> = (0..400u32).map(|i| 2000 + (i * 4093) % 30000).collect();
        let PipelineEvent::Extended { kv: mark } = p.call(PipelineOp::Extend(ids.clone())) else {
            panic!("extend failed");
        };
        let PipelineEvent::Fingerprint { fnv: fp0, .. } = p.call(PipelineOp::KvFingerprint) else {
            panic!("fingerprint failed");
        };

        let cfg = crate::metal::StepGenerateConfig::from_generate(
            7,
            512,
            4096,
            0,
            crate::sample::sampler_for_steps(24, false),
            false,
        );
        let ev = p.call(PipelineOp::BeginTurn {
            prompt: ids.clone(),
            cfg: Box::new(cfg),
            label: "p2-partial".into(),
        });
        let PipelineEvent::TurnStarted { .. } = ev else {
            panic!("begin_turn failed: {ev:?}");
        };

        // Rewind is a lineage op: rejected while the turn is open.
        match p.call(PipelineOp::Rewind(mark)) {
            PipelineEvent::Error(msg) => {
                assert!(msg.contains("turn is open"), "wrong error: {msg}")
            }
            ev => panic!("rewind during open turn must fail, got {ev:?}"),
        }

        let PipelineEvent::Proposed { ids: block, .. } = p.call(PipelineOp::ProposeBlock) else {
            panic!("propose failed");
        };
        assert!(!block.is_empty());
        let kept = block.len() / 2;
        let ev = p.call(PipelineOp::CommitBlock {
            kept_len: kept,
            extend: true,
        });
        let PipelineEvent::BlockCommitted { new_tokens, .. } = ev else {
            panic!("partial commit failed: {ev:?}");
        };
        assert_eq!(new_tokens, kept);

        // The next proposal builds on the committed prefix; reject it.
        let PipelineEvent::Proposed { .. } = p.call(PipelineOp::ProposeBlock) else {
            panic!("second propose failed");
        };
        let PipelineEvent::BlockDiscarded { .. } = p.call(PipelineOp::DiscardBlock) else {
            panic!("discard failed");
        };

        let PipelineEvent::Generated { out, .. } = p.call(PipelineOp::EndTurn) else {
            panic!("end_turn failed");
        };
        assert_eq!(
            out.token_ids.len(),
            ids.len() + kept,
            "committed prefix mismatch"
        );
        assert_eq!(out.blocks_committed, 1);
        assert_eq!(&out.token_ids[ids.len()..], &block[..kept]);

        let PipelineEvent::Rewound { .. } = p.call(PipelineOp::Rewind(mark)) else {
            panic!("rewind after turn failed");
        };
        let PipelineEvent::Fingerprint { fnv, .. } = p.call(PipelineOp::KvFingerprint) else {
            panic!("fingerprint failed");
        };
        assert_eq!(fnv, fp0, "partial-commit turn left KV residue");
    }

    /// The standing rewind byte-consistency gate (PLAN "Token pipeline").
    /// Seeded generate -> rewind loops must (a) restore the KV snapshot hash
    /// exactly after every rewind and (b) regenerate bit-identical replies at
    /// the same seed. Any lineage residue a rewind leaves behind fails one of
    /// the two. Runs below the sliding-ring wrap (prompt 400 + canvas 256
    /// excursions stay inside the window), so every rewind takes the O(1)
    /// truncate path; a wrap-crossing variant is a follow-up.
    #[test]
    fn pipeline_rewind_kv_byte_consistency() {
        let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
            return;
        };
        let p = Pipeline::spawn(dir, 4096, 24);
        let ids: Vec<u32> = (0..400u32).map(|i| 1000 + (i * 7919) % 30000).collect();

        let PipelineEvent::Extended { kv: mut mark } = p.call(PipelineOp::Extend(ids.clone()))
        else {
            panic!("extend failed");
        };
        let PipelineEvent::Fingerprint { fnv: fp0, .. } = p.call(PipelineOp::KvFingerprint) else {
            panic!("fingerprint failed");
        };

        let mut first_reply: Option<Vec<u32>> = None;
        for round in 0..3 {
            let cfg = crate::metal::StepGenerateConfig::from_generate(
                42,
                192,
                4096,
                0, // session-owned; overwritten by the pipeline
                crate::sample::sampler_for_steps(24, false),
                false,
            );
            let ev = p.call(PipelineOp::Generate {
                prompt: ids.clone(),
                cfg: Box::new(cfg),
                label: "rewind-gate".into(),
            });
            let PipelineEvent::Generated { out, .. } = ev else {
                panic!("generate failed at round {round}: {ev:?}");
            };
            match &first_reply {
                None => first_reply = Some(out.token_ids.clone()),
                Some(f) => assert_eq!(
                    f, &out.token_ids,
                    "regeneration diverged after rewind (round {round})"
                ),
            }
            let ev = p.call(PipelineOp::Rewind(mark));
            let PipelineEvent::Rewound { kv } = ev else {
                panic!("rewind failed at round {round}: {ev:?}");
            };
            mark = kv;
            let PipelineEvent::Fingerprint { fnv, .. } = p.call(PipelineOp::KvFingerprint) else {
                panic!("fingerprint failed at round {round}");
            };
            assert_eq!(fnv, fp0, "KV bytes diverged after rewind (round {round})");
        }
    }

    /// Long-context extension byte-consistency on a synthetic 100k KV (PLAN
    /// "Token pipeline"): a pseudorandom KV declared 100k tokens costs ~1 s
    /// instead of a ~7-minute prefill, and the gates only assert
    /// order-of-operations bit-identity, never semantics. For each delta
    /// (1 token, then 256 = one full chunk): extend → fingerprint; rewind →
    /// must restore the base fingerprint (O(1) truncate — deltas stay inside
    /// the sliding-ring slack, ~769 tokens at ring 2048/window 1024, so no
    /// rebuild replaces the synthetic bytes); re-extend the same ids → must
    /// reproduce the post-extend fingerprint (extend-path recompute
    /// determinism at a 100k offset).
    #[test]
    fn synthetic_kv_extension_byte_consistency_100k() {
        let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
            return;
        };
        let p = Pipeline::spawn(dir, 101_000, 24);
        let PipelineEvent::Filled { kv: mut base } = p.call(PipelineOp::SyntheticFill {
            tokens: 100_000,
            seed: 7,
        }) else {
            panic!("synthetic fill failed");
        };
        let PipelineEvent::Fingerprint { fnv: fp_base, .. } = p.call(PipelineOp::KvFingerprint)
        else {
            panic!("fingerprint failed");
        };

        for delta in [1usize, 256] {
            let ids: Vec<u32> = (0..delta as u32).map(|i| 5000 + i * 11).collect();
            let PipelineEvent::Extended { .. } = p.call(PipelineOp::Extend(ids.clone())) else {
                panic!("extend {delta} failed");
            };
            let PipelineEvent::Fingerprint { fnv: fp_add, .. } = p.call(PipelineOp::KvFingerprint)
            else {
                panic!("fingerprint failed");
            };
            assert_ne!(fp_add, fp_base, "extend {delta} did not change the KV");

            let PipelineEvent::Rewound { kv } = p.call(PipelineOp::Rewind(base)) else {
                panic!("rewind after extend {delta} failed");
            };
            base = kv;
            let PipelineEvent::Fingerprint { fnv, .. } = p.call(PipelineOp::KvFingerprint) else {
                panic!("fingerprint failed");
            };
            assert_eq!(fnv, fp_base, "rewind after extend {delta} left residue");

            let PipelineEvent::Extended { .. } = p.call(PipelineOp::Extend(ids)) else {
                panic!("re-extend {delta} failed");
            };
            let PipelineEvent::Fingerprint { fnv, .. } = p.call(PipelineOp::KvFingerprint) else {
                panic!("fingerprint failed");
            };
            assert_eq!(
                fnv, fp_add,
                "re-extend {delta} at 100k offset was not bit-deterministic"
            );
            let PipelineEvent::Rewound { kv } = p.call(PipelineOp::Rewind(base)) else {
                panic!("rewind back to base failed");
            };
            base = kv;
        }
    }

    /// Deep-extension determinism at 100k (ignored: ~2× 10k-token extends).
    /// A 10k truncate would cross the ring slack and REBUILD (re-embedding the
    /// synthetic ids — premise destroyed), so determinism is asserted by
    /// refilling the identical synthetic base and re-running the identical
    /// extend: same op sequence, same bytes. Exercises the batched super-chunk
    /// extend path (M=1024) at a 100k offset.
    /// Run: `cargo test --release synthetic_kv_deep_extend -- --ignored`
    #[test]
    #[ignore = "long: two 10k-token extends at 100k offset"]
    fn synthetic_kv_deep_extend_redo_determinism() {
        let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
            return;
        };
        let p = Pipeline::spawn(dir, 111_000, 24);
        let ids: Vec<u32> = (0..10_000u32).map(|i| 3000 + (i * 97) % 150_000).collect();
        let mut fps = Vec::new();
        for round in 0..2 {
            let PipelineEvent::Filled { .. } = p.call(PipelineOp::SyntheticFill {
                tokens: 100_000,
                seed: 7,
            }) else {
                panic!("synthetic fill failed (round {round})");
            };
            let PipelineEvent::Extended { .. } = p.call(PipelineOp::Extend(ids.clone())) else {
                panic!("10k extend failed (round {round})");
            };
            let PipelineEvent::Fingerprint { fnv, .. } = p.call(PipelineOp::KvFingerprint) else {
                panic!("fingerprint failed (round {round})");
            };
            fps.push(fnv);
        }
        assert_eq!(
            fps[0], fps[1],
            "identical synthetic-fill + 10k-extend op sequences diverged"
        );
    }

    /// The ask reroute's equivalence gate: `generate_monolithic_gpu` (direct:
    /// session-open prefill) and `generate_monolithic_gpu_pipeline` (pipeline
    /// thread: generate-time prefill) must produce identical token ids for the
    /// same inputs. Pins the first production client of the pipeline to the
    /// path it replaced.
    #[test]
    fn ask_via_pipeline_matches_direct() {
        let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
            return;
        };
        let gen_cfg = crate::generate::GenerateConfig {
            seed: 42,
            max_new_tokens: 64,
            full_message_stop: true,
            ..Default::default()
        };
        let ids: Vec<u32> = (0..48u32).map(|i| 1000 + i * 13).collect();
        let direct =
            crate::generate::generate_monolithic_gpu(&dir, &ids, &gen_cfg, 1024, "eq-gate")
                .expect("direct generate");
        let piped = crate::generate::generate_monolithic_gpu_pipeline(
            &dir, &ids, &gen_cfg, 1024, "eq-gate",
        )
        .expect("pipeline generate");
        assert_eq!(
            direct.token_ids, piped.token_ids,
            "pipeline ask path diverged from the direct path"
        );
    }

    /// The cancel gate: a [`crate::metal::CancelToken`] riding in the Generate
    /// cfg (the same way observer Arcs do) stops an in-flight generation
    /// between denoise steps, and the abandoned canvas leaves NO residue — a
    /// rewind to the pre-generate mark must restore the base fingerprint
    /// exactly. This is the serve dead-socket fix's core contract (the stream
    /// observer cancels when the client hangs up).
    #[test]
    fn pipeline_cancel_stops_generate_kv_clean() {
        let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
            return;
        };
        let p = Pipeline::spawn(dir, 4096, 24);
        let ids: Vec<u32> = (0..400u32).map(|i| 1000 + (i * 7919) % 30000).collect();
        let PipelineEvent::Extended { kv: mark } = p.call(PipelineOp::Extend(ids.clone())) else {
            panic!("extend failed");
        };
        let PipelineEvent::Fingerprint { fnv: fp0, .. } = p.call(PipelineOp::KvFingerprint) else {
            panic!("fingerprint failed");
        };

        let mut cfg = crate::metal::StepGenerateConfig::from_generate(
            42,
            2048, // 8 blocks — far more than can denoise before the cancel lands
            4096,
            0, // session-owned; overwritten by the pipeline
            crate::sample::sampler_for_steps(24, false),
            false,
        );
        let cancel = crate::metal::CancelToken::new();
        cfg.cancel = Some(cancel.clone());
        let canceller = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(1500));
            cancel.cancel();
        });
        let ev = p.call(PipelineOp::Generate {
            prompt: ids.clone(),
            cfg: Box::new(cfg),
            label: "cancel-gate".into(),
        });
        canceller.join().expect("canceller thread");
        let PipelineEvent::Generated { out, .. } = ev else {
            panic!("generate failed: {ev:?}");
        };
        assert!(
            out.cancelled,
            "generate ran to completion before the cancel"
        );
        assert!(
            out.token_ids.len() < ids.len() + 2048,
            "cancelled generate still produced the full budget"
        );

        let PipelineEvent::Rewound { .. } = p.call(PipelineOp::Rewind(mark)) else {
            panic!("rewind after cancel failed");
        };
        let PipelineEvent::Fingerprint { fnv, .. } = p.call(PipelineOp::KvFingerprint) else {
            panic!("fingerprint failed");
        };
        assert_eq!(fnv, fp0, "cancelled generate left KV residue past the mark");
    }

    /// Stale-epoch rewinds must fail loudly (the drift class, type-checked):
    /// after a rewind bumps the epoch, an id captured before it is dead.
    #[test]
    fn pipeline_rejects_stale_epoch_rewind() {
        let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
            return;
        };
        let p = Pipeline::spawn(dir, 2048, 24);
        let ids: Vec<u32> = (0..64u32).map(|i| 2000 + i * 3).collect();
        let PipelineEvent::Extended { kv: old } = p.call(PipelineOp::Extend(ids)) else {
            panic!("extend failed");
        };
        let PipelineEvent::Rewound { .. } = p.call(PipelineOp::Rewind(old)) else {
            panic!("first rewind failed");
        };
        // `old` belongs to the pre-rewind epoch now.
        match p.call(PipelineOp::Rewind(old)) {
            PipelineEvent::Error(msg) => assert!(msg.contains("stale epoch"), "wrong error: {msg}"),
            other => panic!("stale-epoch rewind must fail, got {other:?}"),
        }
    }
}
