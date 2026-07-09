//! OpenAI-compatible HTTP server for local chat completions.
//!
//! Hand-rolled blocking HTTP/1.1 on `std::net` — no async runtime, no framework.
//! That is the right shape here, not a compromise: **the GPU is single-tenant by
//! hard rule**, so generation is serialized no matter what. One worker thread
//! owns the `StepGenerateSession` (Metal objects never cross threads) and drains
//! a queue; each HTTP connection thread parses a request, enqueues a job, and
//! streams the reply back over a channel. Requests queue; the GPU never runs two
//! generations at once.
//!
//! ## What's OpenAI-compatible, and what's ours
//!
//! `POST /v1/chat/completions` speaks the OpenAI schema (`messages`, `stream`,
//! `max_tokens`, `seed`, SSE `chat.completion.chunk`s ending in `[DONE]`), so any
//! OpenAI client works. On top of that we add two channels a block-diffusion
//! model has that an autoregressive one does not:
//!
//! - **`reasoning_content`** — the model's private thought channel
//!   (`<|channel>thought…<channel|>`), streamed on its own delta field the way
//!   DeepSeek/vLLM/OpenWebUI expect. Rendered by clients as a collapsible
//!   "thinking" block. Opt-in per request (`enable_thinking:true`): the default
//!   keeps the gate-validated prompt (empty thought channel seeded), and enabling
//!   thinking unseeds it.
//! - **`x-diffusion-draft`** — the *shimmering private messages*: as the canvas
//!   denoises, positions flip in place before they commit. Standard `content`
//!   only ever carries block-**committed** (immutable) text, so any OpenAI client
//!   sees a correct append-only reply; the speculative, still-converging draft
//!   rides entirely in this `x-` extension delta (replace-semantics), so a
//!   diffusion-aware client can render the live shimmer. On by default
//!   (per-request `x_diffusion_drafts:false` opts out).

use crate::chat_protocol::TextDecoder;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

/// Consecutive-step repeats before a canvas position is treated as stable enough
/// to stream (matches the terminal chat renderer's `STABLE_STREAK`).
const STABLE_STREAK: u32 = 2;

/// Reject request bodies larger than this (basic abuse guard).
const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

// ===========================================================================
// Wire protocol (request in, chunks out).
// ===========================================================================

#[derive(Deserialize)]
struct ChatRequest {
    /// Raw OpenAI messages, kept as JSON so the tool-aware renderer sees
    /// `tool_calls` / `tool` roles / content-parts verbatim. `content` (string or
    /// parts array) is flattened by `tools::message_text` for the plain path.
    #[serde(default)]
    messages: Vec<serde_json::Value>,
    /// OpenAI `tools` (function definitions). Present → tool-aware prompt.
    #[serde(default)]
    tools: Vec<serde_json::Value>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    /// Accepted for compatibility. The diffusion sampler uses its own
    /// temperature schedule, so this is not applied in v1 (documented).
    #[serde(default)]
    #[allow(dead_code)]
    temperature: Option<f32>,
    /// Expose the private thought channel as `reasoning_content`. Default: the
    /// server default (on). Accepts `chat_template_kwargs.enable_thinking` too.
    #[serde(default)]
    enable_thinking: Option<bool>,
    /// Stream speculative pre-commit canvas drafts as `x-diffusion-draft`.
    /// Default: on.
    #[serde(default)]
    x_diffusion_drafts: Option<bool>,
}

/// One streamed delta produced by the mapper, before OpenAI-envelope framing.
#[derive(Debug, Clone, PartialEq)]
pub enum WireDelta {
    /// Append to `reasoning_content` (the private thought channel).
    Reasoning(String),
    /// Append to `content` (block-committed, immutable answer text).
    Content(String),
    /// Replace the speculative draft — the shimmering, still-converging answer.
    Draft {
        text: String,
        committed: usize,
        block: usize,
        step: u32,
    },
}

// ===========================================================================
// DiffusionStreamMapper — the pure core (no GPU, no sockets; unit-tested).
//
// Turns the raw per-step canvas argmax stream into `WireDelta`s, splitting the
// harmony thought channel from the answer and separating block-committed text
// from the speculative draft.
// ===========================================================================

pub struct DiffusionStreamMapper<D: TextDecoder> {
    decoder: D,
    stops: Vec<u32>,
    channel_open: Option<u32>,
    channel_close: Option<u32>,
    thinking: bool,
    emit_drafts: bool,

    block_idx: usize,
    last_argmax: Vec<u32>,
    streak: Vec<u32>,
    /// Block-committed token ids accumulated across all committed blocks.
    committed_ids: Vec<u32>,
    /// Reasoning / content text already emitted as deltas (append cursors).
    emitted_reasoning: String,
    emitted_content: String,
    /// Last draft text emitted (dedupe replace-semantics chunks).
    last_draft: String,
    ended: bool,
}

/// Reasoning (thought-channel) and answer text split out of a committed id list.
struct Split {
    reasoning: String,
    content: String,
}

impl<D: TextDecoder> DiffusionStreamMapper<D> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        decoder: D,
        stops: Vec<u32>,
        channel_open: Option<u32>,
        channel_close: Option<u32>,
        thinking: bool,
        emit_drafts: bool,
    ) -> Self {
        Self {
            decoder,
            stops,
            channel_open,
            channel_close,
            thinking,
            emit_drafts,
            block_idx: 0,
            last_argmax: Vec::new(),
            streak: Vec::new(),
            committed_ids: Vec::new(),
            emitted_reasoning: String::new(),
            emitted_content: String::new(),
            last_draft: String::new(),
            ended: false,
        }
    }

    /// Committed (immutable) answer text emitted so far.
    pub fn content(&self) -> &str {
        &self.emitted_content
    }

    /// Committed reasoning text emitted so far.
    pub fn reasoning(&self) -> &str {
        &self.emitted_reasoning
    }

    pub fn ended(&self) -> bool {
        self.ended
    }

    /// Split committed ids into (reasoning, content). With thinking off, or once
    /// no channel is present, everything is content. While thinking and before
    /// the channel closes, everything committed so far is reasoning.
    fn split(&self, ids: &[u32]) -> Split {
        let close_at = self
            .channel_close
            .and_then(|c| ids.iter().position(|&id| id == c));
        match (self.thinking, close_at) {
            (true, Some(k)) => {
                let reasoning = self.clean_reasoning(&ids[..k]);
                let content = crate::chat_template::sanitize_model_reply(
                    &self
                        .decoder
                        .decode(&crate::sample::strip_degenerate_token_ids(&ids[k + 1..])),
                );
                Split { reasoning, content }
            }
            (true, None) => Split {
                reasoning: self.clean_reasoning(ids),
                content: String::new(),
            },
            (false, _) => Split {
                reasoning: String::new(),
                content: crate::chat_template::sanitize_model_reply(
                    &self
                        .decoder
                        .decode(&crate::sample::strip_degenerate_token_ids(ids)),
                ),
            },
        }
    }

    /// Decode a reasoning-region id slice and strip the `<|channel>thought\n`
    /// opener scaffold the model emits when thinking is enabled.
    fn clean_reasoning(&self, ids: &[u32]) -> String {
        // Drop a leading channel-open special id if present.
        let start = match (self.channel_open, ids.first()) {
            (Some(open), Some(&first)) if first == open => 1,
            _ => 0,
        };
        let raw = self
            .decoder
            .decode(&crate::sample::strip_degenerate_token_ids(&ids[start..]));
        raw.trim_start_matches("<|channel>")
            .trim_start()
            .trim_start_matches("thought")
            .trim()
            .to_string()
    }

    /// The current speculative answer draft: the stable-prefix answer region,
    /// including the still-converging tail, sanitized to visible text.
    fn draft_text(&self, stable_ids: &[u32]) -> String {
        let mut all = self.committed_ids.clone();
        all.extend_from_slice(stable_ids);
        self.split(&all).content
    }

    pub fn on_step(&mut self, ev: &crate::metal::StepProgressEvent<'_>) -> Vec<WireDelta> {
        let mut out = Vec::new();
        if self.ended {
            return out;
        }

        if ev.block_idx != self.block_idx {
            self.block_idx = ev.block_idx;
            self.last_argmax.clear();
            self.streak.clear();
        }

        // Per-position stable-streak update.
        if self.last_argmax.len() != ev.argmax.len() {
            self.last_argmax = ev.argmax.to_vec();
            self.streak = vec![0; ev.argmax.len()];
        } else {
            for i in 0..ev.argmax.len() {
                if ev.argmax[i] == self.last_argmax[i] {
                    self.streak[i] = self.streak[i].saturating_add(1);
                } else {
                    self.streak[i] = 0;
                    self.last_argmax[i] = ev.argmax[i];
                }
            }
        }

        // Stable prefix (whole canvas on commit), cut at the first stop token.
        let prefix_end = if ev.block_done {
            ev.argmax.len()
        } else {
            self.streak
                .iter()
                .position(|&k| k < STABLE_STREAK)
                .unwrap_or(ev.argmax.len())
        };
        let mut stable_ids = Vec::with_capacity(prefix_end);
        let mut hit_stop = false;
        for &id in &ev.argmax[..prefix_end] {
            if self.stops.contains(&id) {
                hit_stop = true;
                break;
            }
            stable_ids.push(id);
        }

        // Draft (replace-semantics), emitted every step it changes.
        if self.emit_drafts {
            let text = self.draft_text(&stable_ids);
            if text != self.last_draft {
                out.push(WireDelta::Draft {
                    committed: self.emitted_content.len(),
                    block: ev.block_idx,
                    step: ev.step_in_block,
                    text: text.clone(),
                });
                self.last_draft = text;
            }
        }

        // Commit: fold this block's stable ids into the committed stream and
        // emit any newly-committed reasoning / content as append deltas.
        if ev.block_done {
            self.committed_ids.extend_from_slice(&stable_ids);
            let split = self.split(&self.committed_ids);
            if let Some(delta) = append_delta(&self.emitted_reasoning, &split.reasoning) {
                out.push(WireDelta::Reasoning(delta));
                self.emitted_reasoning = split.reasoning;
            }
            if let Some(delta) = append_delta(&self.emitted_content, &split.content) {
                out.push(WireDelta::Content(delta));
                self.emitted_content = split.content;
            }
            self.last_argmax.clear();
            self.streak.clear();
            if hit_stop {
                self.ended = true;
            }
        }
        out
    }
}

/// If `next` extends `prev` (or diverges), return the suffix to append. Returns
/// `None` when nothing new. On a rare non-prefix revision, re-emit the whole new
/// text (append-only sinks can't retract; committed text almost never revises).
fn append_delta(prev: &str, next: &str) -> Option<String> {
    if next == prev {
        None
    } else if let Some(suffix) = next.strip_prefix(prev) {
        (!suffix.is_empty()).then(|| suffix.to_string())
    } else {
        Some(next.to_string())
    }
}

// ===========================================================================
// OpenAI SSE / JSON envelope framing.
// ===========================================================================

#[derive(Serialize)]
struct DeltaBody {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(rename = "x-diffusion-draft", skip_serializing_if = "Option::is_none")]
    draft: Option<DraftBody>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<serde_json::Value>>,
}

#[derive(Serialize)]
struct DraftBody {
    text: String,
    committed: usize,
    block: usize,
    step: u32,
}

#[derive(Serialize)]
struct ChunkChoice {
    index: u32,
    delta: DeltaBody,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
struct Chunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
}

fn empty_delta() -> DeltaBody {
    DeltaBody {
        role: None,
        content: None,
        reasoning_content: None,
        draft: None,
        tool_calls: None,
    }
}

impl WireDelta {
    fn into_delta_body(self) -> DeltaBody {
        match self {
            WireDelta::Reasoning(s) => DeltaBody {
                reasoning_content: Some(s),
                ..empty_delta()
            },
            WireDelta::Content(s) => DeltaBody {
                content: Some(s),
                ..empty_delta()
            },
            WireDelta::Draft {
                text,
                committed,
                block,
                step,
            } => DeltaBody {
                draft: Some(DraftBody {
                    text,
                    committed,
                    block,
                    step,
                }),
                ..empty_delta()
            },
        }
    }
}

// ===========================================================================
// Worker job protocol (HTTP thread <-> GPU worker thread).
// ===========================================================================

/// A unit of work handed to the GPU worker (all fields `Send`).
struct Job {
    /// Raw OpenAI messages (JSON) and `tools`, rendered on the worker thread.
    messages: Vec<serde_json::Value>,
    tools: Vec<serde_json::Value>,
    stream: bool,
    max_tokens: Option<usize>,
    seed: Option<u64>,
    enable_thinking: bool,
    emit_drafts: bool,
    resp: mpsc::Sender<ServerEvent>,
}

/// Events the worker streams back to the connection thread.
enum ServerEvent {
    Delta(WireDelta),
    Done {
        content: String,
        reasoning: String,
        tool_calls: Vec<serde_json::Value>,
        prompt_tokens: usize,
        completion_tokens: usize,
        stopped: bool,
    },
    Error(String),
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ===========================================================================
// GPU worker: owns the session, drains the job queue one at a time.
// ===========================================================================

struct Worker {
    model_dir: std::path::PathBuf,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
    stop_token_ids: Vec<u32>,
    channel_open: Option<u32>,
    channel_close: Option<u32>,
    max_seq: usize,
    steps: usize,
    no_early_stop: bool,
    base_cfg: crate::metal::StepGenerateConfig,
    tool_compact: Option<ToolCompactCfg>,
}

/// Tool-output compaction settings (`--tool-compact` / `DGQ_TOOL_COMPACT=1`).
struct ToolCompactCfg {
    /// Tool responses over this many tokens get summarized + substituted.
    threshold: usize,
    /// Max server-side `expand_summary` round-trips per request.
    max_expand_rounds: usize,
    /// Token budget for one summarize pass's reply.
    summarize_max_new: usize,
}

impl Worker {
    /// Open the session (Metal objects are not `Send`, so they must be created
    /// on this thread), signal readiness, then drain the job queue one at a time.
    fn run(self, ready: mpsc::Sender<Result<(), String>>, jobs: mpsc::Receiver<Job>) {
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

type ServeMapper = Arc<Mutex<DiffusionStreamMapper<Arc<crate::tokenizer::Tokenizer>>>>;

fn attach_stream_observer(
    cfg: &mut crate::metal::StepGenerateConfig,
    mapper: &ServeMapper,
    resp: &mpsc::Sender<ServerEvent>,
) {
    let mapper = Arc::clone(mapper);
    let resp = resp.clone();
    cfg.step_observer = Some(Arc::new(move |ev: &crate::metal::StepProgressEvent<'_>| {
        let deltas = mapper.lock().unwrap().on_step(ev);
        for d in deltas {
            let _ = resp.send(ServerEvent::Delta(d));
        }
    }));
}

/// One checkpoint-bracketed summarize generation: render `messages_ctx`
/// (conversation up to and including the verbose tool response + the summarize
/// instruction), generate a bounded reply with NO step observer, extract the
/// `<summarize>…</summarize>` span, and roll the KV back so nothing of the
/// pass persists. Returns None when the prompt doesn't fit, generation fails,
/// or the model produced no usable tags (caller falls back mechanically).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_summarize_pass(
    session: &mut crate::metal::StepGenerateSession,
    tokenizer: &crate::tokenizer::Tokenizer,
    base_cfg: &crate::metal::StepGenerateConfig,
    steps: usize,
    stop_token_ids: &[u32],
    model_dir: &std::path::Path,
    max_seq: usize,
    summarize_max_new: usize,
    messages_ctx: &[serde_json::Value],
    tools: &[serde_json::Value],
) -> Option<String> {
    let s = crate::tools::render_conversation(messages_ctx, tools, true, false);
    let prompt = tokenizer.encode_with_specials(&s);
    let room = max_seq.saturating_sub(prompt.len() + crate::metal::CANVAS);
    if room < 64 {
        return None;
    }
    let mut cfg = base_cfg.clone();
    cfg.sampler = crate::sample::sampler_for_steps(steps, false);
    cfg.max_new_tokens = summarize_max_new.min(room);
    cfg.stop_token_ids = stop_token_ids.to_vec();
    cfg.degenerate_reply_check =
        crate::chat_template::empty_reply_check(model_dir, stop_token_ids.to_vec());
    cfg.step_observer = None;

    let cp = session.checkpoint();
    let result = crate::metal::generate_with_session(session, &prompt, &cfg, "tool-compact");
    // Always rewind — the verbose response, the instruction, and the summary
    // itself must never persist in the conversation's KV.
    session.rollback_to(&cp);
    let out = match result {
        Ok(o) => o,
        Err(err) => {
            eprintln!("serve: tool-compact: summarize pass failed: {err}");
            return None;
        }
    };
    let generated = &out.token_ids[prompt.len().min(out.token_ids.len())..];
    let text = crate::chat_template::sanitize_model_reply(
        &tokenizer.decode(&crate::sample::strip_degenerate_token_ids(generated)),
    );
    crate::toolcompact::extract_summary(&text)
}

/// Longest common prefix (in tokens).
fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Cap an expand excerpt to a token budget so retrieval can't blow the
/// context the compactor just reclaimed.
fn cap_tokens(tokenizer: &crate::tokenizer::Tokenizer, text: &str, max_tokens: usize) -> String {
    let max_tokens = max_tokens.max(16);
    let ids = tokenizer.encode(text, false);
    if ids.len() <= max_tokens {
        return text.to_string();
    }
    let mut s = tokenizer.decode(&ids[..max_tokens]);
    s.push_str("\n[excerpt truncated]");
    s
}

// ===========================================================================
// HTTP layer.
// ===========================================================================

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

/// Parse an HTTP/1.1 request: request line, headers, and (Content-Length) body.
fn read_request(stream: &mut TcpStream) -> std::io::Result<Option<Request>> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(None); // connection closed
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();

    let mut content_length = 0usize;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 {
            break;
        }
        let h = h.trim_end();
        if h.is_empty() {
            break; // end of headers
        }
        if let Some(v) = h
            .split_once(':')
            .filter(|(k, _)| k.trim().eq_ignore_ascii_case("content-length"))
            .map(|(_, v)| v.trim())
        {
            content_length = v.parse().unwrap_or(0);
        }
    }
    if content_length > MAX_BODY_BYTES {
        return Ok(Some(Request {
            method,
            path,
            body: Vec::new(),
        }));
    }
    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;
    Ok(Some(Request { method, path, body }))
}

fn write_response(stream: &mut TcpStream, status: &str, content_type: &str, body: &[u8]) {
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

fn write_json(stream: &mut TcpStream, status: &str, value: &serde_json::Value) {
    let body = serde_json::to_vec(value).unwrap_or_default();
    write_response(stream, status, "application/json", &body);
}

fn error_json(msg: &str, kind: &str) -> serde_json::Value {
    serde_json::json!({ "error": { "message": msg, "type": kind } })
}

fn handle_connection(mut stream: TcpStream, jobs: mpsc::Sender<Job>, model_name: Arc<str>) {
    let req = match read_request(&mut stream) {
        Ok(Some(r)) => r,
        _ => return,
    };
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => {
            write_json(
                &mut stream,
                "200 OK",
                &serde_json::json!({ "status": "ok" }),
            );
        }
        ("GET", "/v1/models") => {
            write_json(
                &mut stream,
                "200 OK",
                &serde_json::json!({
                    "object": "list",
                    "data": [{ "id": &*model_name, "object": "model", "owned_by": "local" }],
                }),
            );
        }
        ("POST", "/v1/chat/completions") => {
            handle_chat(&mut stream, &req.body, &jobs, &model_name);
        }
        ("OPTIONS", _) => {
            write_response(&mut stream, "204 No Content", "text/plain", b"");
        }
        _ => {
            write_json(
                &mut stream,
                "404 Not Found",
                &error_json("unknown route", "invalid_request_error"),
            );
        }
    }
}

fn handle_chat(
    stream: &mut TcpStream,
    body: &[u8],
    jobs: &mpsc::Sender<Job>,
    model_name: &Arc<str>,
) {
    let req: ChatRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(err) => {
            write_json(
                stream,
                "400 Bad Request",
                &error_json(
                    &format!("invalid JSON body: {err}"),
                    "invalid_request_error",
                ),
            );
            return;
        }
    };

    if req.messages.is_empty() {
        write_json(
            stream,
            "400 Bad Request",
            &error_json("no messages", "invalid_request_error"),
        );
        return;
    }

    let (resp_tx, resp_rx) = mpsc::channel();
    let job = Job {
        messages: req.messages,
        tools: req.tools,
        stream: req.stream,
        max_tokens: req.max_tokens,
        seed: req.seed,
        // Default OFF: matches the gate-validated prompt (which seeds an empty
        // thought channel). Enabling thinking unseeds it — an unmeasured prompt
        // change — and this checkpoint does not reliably emit a thought channel
        // anyway, so reasoning_content stays empty. Clients opt in per request;
        // the reasoning_content plumbing is fully wired for when it fires.
        enable_thinking: req.enable_thinking.unwrap_or(false),
        // Draft deltas only exist for streaming responses; a non-streaming
        // request would decode the full canvas every step just to have
        // respond_json discard the events.
        emit_drafts: req.x_diffusion_drafts.unwrap_or(true) && req.stream,
        resp: resp_tx,
    };
    let streaming = job.stream;
    if jobs.send(job).is_err() {
        write_json(
            stream,
            "503 Service Unavailable",
            &error_json("generation worker unavailable", "server_error"),
        );
        return;
    }

    if streaming {
        stream_sse(stream, resp_rx, model_name);
    } else {
        respond_json(stream, resp_rx, model_name);
    }
}

/// Build plain `ChatTurn`s from raw messages for the non-tool prompt path
/// (`format_chat_template` — the gate-validated encoding). system→user,
/// assistant→model; content flattened via `tools::message_text`.
fn build_turns(messages: &[serde_json::Value]) -> Vec<crate::chat_template::ChatTurn> {
    messages
        .iter()
        .filter_map(|m| {
            let role = match m.get("role").and_then(|r| r.as_str()) {
                Some("user") | Some("system") | Some("developer") => {
                    crate::chat_template::ChatRole::User
                }
                Some("assistant") | Some("model") => crate::chat_template::ChatRole::Model,
                _ => return None,
            };
            Some(crate::chat_template::ChatTurn {
                role,
                content: crate::tools::message_text(m),
            })
        })
        .collect()
}

/// True when a request needs the tool-aware renderer: `tools` present, or any
/// message is a `tool` result or carries `tool_calls`.
fn needs_tool_rendering(messages: &[serde_json::Value], tools: &[serde_json::Value]) -> bool {
    !tools.is_empty()
        || messages.iter().any(|m| {
            m.get("role").and_then(|r| r.as_str()) == Some("tool") || m.get("tool_calls").is_some()
        })
}

/// Stream Server-Sent Events: one `chat.completion.chunk` per delta, ending with
/// a `finish_reason` chunk and `data: [DONE]`.
fn stream_sse(stream: &mut TcpStream, rx: mpsc::Receiver<ServerEvent>, model_name: &Arc<str>) {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
    if stream.write_all(head.as_bytes()).is_err() {
        return;
    }
    let id = format!("chatcmpl-{}", next_id());
    let created = now_secs();
    let model = model_name.to_string();

    // Opening role chunk (OpenAI convention).
    let _ = send_chunk(
        stream,
        &id,
        created,
        &model,
        ChunkChoice {
            index: 0,
            delta: DeltaBody {
                role: Some("assistant"),
                ..empty_delta()
            },
            finish_reason: None,
        },
    );

    let mut finish = "stop";
    for ev in rx {
        match ev {
            ServerEvent::Delta(d) => {
                if send_chunk(
                    stream,
                    &id,
                    created,
                    &model,
                    ChunkChoice {
                        index: 0,
                        delta: d.into_delta_body(),
                        finish_reason: None,
                    },
                )
                .is_err()
                {
                    return; // client hung up
                }
            }
            ServerEvent::Done {
                stopped,
                tool_calls,
                ..
            } => {
                if !tool_calls.is_empty() {
                    finish = "tool_calls";
                    let _ = send_chunk(
                        stream,
                        &id,
                        created,
                        &model,
                        ChunkChoice {
                            index: 0,
                            delta: DeltaBody {
                                tool_calls: Some(tool_calls),
                                ..empty_delta()
                            },
                            finish_reason: None,
                        },
                    );
                } else if !stopped {
                    finish = "length";
                }
                break;
            }
            ServerEvent::Error(msg) => {
                let _ = write_sse_data(stream, &error_json(&msg, "server_error"));
                let _ = stream.write_all(b"data: [DONE]\n\n");
                let _ = stream.flush();
                return;
            }
        }
    }

    let _ = send_chunk(
        stream,
        &id,
        created,
        &model,
        ChunkChoice {
            index: 0,
            delta: empty_delta(),
            finish_reason: Some(finish),
        },
    );
    let _ = stream.write_all(b"data: [DONE]\n\n");
    let _ = stream.flush();
}

fn send_chunk(
    stream: &mut TcpStream,
    id: &str,
    created: u64,
    model: &str,
    choice: ChunkChoice,
) -> std::io::Result<()> {
    let chunk = Chunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![choice],
    };
    write_sse_data(stream, &chunk)
}

fn write_sse_data<T: Serialize>(stream: &mut TcpStream, value: &T) -> std::io::Result<()> {
    let json = serde_json::to_string(value).unwrap_or_default();
    stream.write_all(b"data: ")?;
    stream.write_all(json.as_bytes())?;
    stream.write_all(b"\n\n")?;
    stream.flush()
}

/// Non-streaming: drain the worker to completion, then write one JSON response.
fn respond_json(stream: &mut TcpStream, rx: mpsc::Receiver<ServerEvent>, model_name: &Arc<str>) {
    for ev in rx {
        match ev {
            ServerEvent::Delta(_) => {}
            ServerEvent::Done {
                content,
                reasoning,
                tool_calls,
                prompt_tokens,
                completion_tokens,
                stopped,
            } => {
                let has_calls = !tool_calls.is_empty();
                let mut message = serde_json::json!({
                    "role": "assistant",
                    // OpenAI convention: content is null when only tool_calls.
                    "content": if has_calls && content.is_empty() {
                        serde_json::Value::Null
                    } else {
                        serde_json::Value::String(content)
                    },
                });
                if !reasoning.is_empty() {
                    message["reasoning_content"] = serde_json::Value::String(reasoning);
                }
                if has_calls {
                    message["tool_calls"] = serde_json::Value::Array(tool_calls);
                }
                let finish_reason = if has_calls {
                    "tool_calls"
                } else if stopped {
                    "stop"
                } else {
                    "length"
                };
                let value = serde_json::json!({
                    "id": format!("chatcmpl-{}", next_id()),
                    "object": "chat.completion",
                    "created": now_secs(),
                    "model": &**model_name,
                    "choices": [{
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                });
                write_json(stream, "200 OK", &value);
                return;
            }
            ServerEvent::Error(msg) => {
                write_json(
                    stream,
                    "500 Internal Server Error",
                    &error_json(&msg, "server_error"),
                );
                return;
            }
        }
    }
}

static ID_COUNTER: AtomicU64 = AtomicU64::new(1);
fn next_id() -> u64 {
    ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

// ===========================================================================
// serve entry point.
// ===========================================================================

pub fn run_serve(
    model_dir: &std::path::Path,
    addr: &str,
    ctx: usize,
    seed: u64,
    steps: usize,
    max_layers: Option<usize>,
    tool_compact: bool,
) -> Result<(), String> {
    use crate::metal::StepGenerateConfig;

    if !crate::dgq::store::looks_like_dgq_dir(model_dir) {
        return Err(format!(
            "serve requires a .dgq directory (-m /path/to/quantized-weights); got {}",
            model_dir.display()
        ));
    }
    let layers = crate::resolve_model_layers(model_dir, max_layers).map_err(|e| e.to_string())?;

    // Fail-fast context budget guard before loading weights (mirrors chat/ask).
    crate::check_ctx_budget(ctx)?;

    let stop_token_ids = crate::config::load_generation_stop_tokens(model_dir);
    let sampler = crate::sample::sampler_for_steps(steps, false);
    let mut base_cfg = StepGenerateConfig::from_generate(seed, ctx, ctx, layers, sampler, false);
    base_cfg.stop_token_ids = stop_token_ids.clone();

    let tokenizer = Arc::new(
        crate::tokenizer::Tokenizer::load(model_dir.join("tokenizer.json"))
            .map_err(|e| e.to_string())?,
    );
    let channel_open = tokenizer.special_token_id("<|channel>");
    let channel_close = tokenizer.special_token_id("<channel|>");
    let model_name: Arc<str> = Arc::from(
        model_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("diffusiongemma"),
    );

    // The Metal session is not `Send`, so the worker thread opens it itself and
    // signals readiness before we start accepting connections.
    let (job_tx, job_rx) = mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
    let tool_compact = tool_compact || crate::flags::tool_compact_enabled();
    let worker = Worker {
        model_dir: model_dir.to_path_buf(),
        tokenizer,
        stop_token_ids,
        channel_open,
        channel_close,
        max_seq: ctx,
        steps,
        no_early_stop: false,
        base_cfg,
        tool_compact: tool_compact.then(|| ToolCompactCfg {
            threshold: crate::flags::tool_compact_threshold(),
            max_expand_rounds: 4,
            summarize_max_new: 512,
        }),
    };
    eprintln!("serve: loading model…");
    let worker_handle = std::thread::spawn(move || worker.run(ready_tx, job_rx));
    match ready_rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(msg)) => return Err(msg),
        Err(_) => return Err("generation worker exited during startup".to_string()),
    }

    let listener = TcpListener::bind(addr).map_err(|e| format!("bind {addr}: {e}"))?;
    let local = listener.local_addr().map_err(|e| e.to_string())?;
    eprintln!("serve: listening on http://{local}  (POST /v1/chat/completions)");

    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let jobs = job_tx.clone();
                let name = Arc::clone(&model_name);
                std::thread::spawn(move || handle_connection(stream, jobs, name));
            }
            Err(err) => eprintln!("serve: accept error: {err}"),
        }
    }
    drop(job_tx);
    let _ = worker_handle.join();
    Ok(())
}

// ===========================================================================
// Tests — the pure mapper, no GPU or sockets.
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::StepProgressEvent;

    /// Trivial decoder: printable-ASCII ids map to their char; others (special
    /// tokens) decode to nothing. Enough to exercise the id-level split logic.
    struct FakeDecoder;
    impl TextDecoder for FakeDecoder {
        fn decode(&self, ids: &[u32]) -> String {
            let mut s = String::new();
            self.decode_append(&mut s, ids);
            s
        }
        fn decode_append(&self, out: &mut String, ids: &[u32]) {
            for &id in ids {
                if (32..127).contains(&id) {
                    out.push(id as u8 as char);
                }
            }
        }
    }

    fn step<'a>(block: usize, step: u32, argmax: &'a [u32], done: bool) -> StepProgressEvent<'a> {
        StepProgressEvent {
            block_idx: block,
            step_in_block: step,
            max_steps: 48,
            argmax,
            accept_count: 0,
            block_done: done,
        }
    }

    fn c(ch: char) -> u32 {
        ch as u32
    }

    #[test]
    fn content_only_commits_answer_text() {
        let mut m = DiffusionStreamMapper::new(FakeDecoder, vec![], None, None, false, false);
        let hi = [c('H'), c('i')];
        // Not committed yet: no content deltas.
        assert!(m.on_step(&step(1, 1, &hi, false)).is_empty());
        // Commit the block -> content delta "Hi".
        let out = m.on_step(&step(1, 2, &hi, true));
        assert_eq!(out, vec![WireDelta::Content("Hi".to_string())]);
        assert_eq!(m.content(), "Hi");
    }

    #[test]
    fn draft_streams_stable_prefix_before_commit() {
        let mut m = DiffusionStreamMapper::new(FakeDecoder, vec![], None, None, false, true);
        let hi = [c('H'), c('i')];
        // Streak must reach STABLE_STREAK before the draft shows anything.
        assert!(draft_of(&m.on_step(&step(1, 1, &hi, false))).is_none());
        let _ = m.on_step(&step(1, 2, &hi, false));
        let d = draft_of(&m.on_step(&step(1, 3, &hi, false)));
        assert_eq!(d.as_deref(), Some("Hi"));
    }

    #[test]
    fn thinking_splits_reasoning_from_content() {
        // channel_open=1, channel_close=2. Canvas: <open> t h o u g h t <close> H i
        let open = 1u32;
        let close = 2u32;
        let mut m =
            DiffusionStreamMapper::new(FakeDecoder, vec![], Some(open), Some(close), true, false);
        let canvas = [
            open,
            c('t'),
            c('h'),
            c('o'),
            c('u'),
            c('g'),
            c('h'),
            c('t'),
            close,
            c('H'),
            c('i'),
        ];
        let out = m.on_step(&step(1, 2, &canvas, true));
        // "thought" opener scaffold stripped from reasoning; answer is "Hi".
        assert!(out.contains(&WireDelta::Content("Hi".to_string())));
        assert_eq!(m.content(), "Hi");
        assert!(m.reasoning().is_empty() || !m.reasoning().contains("Hi"));
    }

    #[test]
    fn stop_token_ends_and_cuts() {
        let mut m = DiffusionStreamMapper::new(FakeDecoder, vec![99], None, None, false, false);
        let canvas = [c('O'), c('k'), 99, c('X')];
        let out = m.on_step(&step(1, 2, &canvas, true));
        assert_eq!(out, vec![WireDelta::Content("Ok".to_string())]);
        assert!(m.ended());
        // Further steps are inert.
        assert!(m.on_step(&step(1, 3, &canvas, true)).is_empty());
    }

    #[test]
    fn append_delta_semantics() {
        assert_eq!(append_delta("", "Hi"), Some("Hi".to_string()));
        assert_eq!(append_delta("Hi", "Hi there"), Some(" there".to_string()));
        assert_eq!(append_delta("Hi", "Hi"), None);
        assert_eq!(append_delta("Hello", "Help"), Some("Help".to_string()));
    }

    fn draft_of(deltas: &[WireDelta]) -> Option<String> {
        deltas.iter().find_map(|d| match d {
            WireDelta::Draft { text, .. } => Some(text.clone()),
            _ => None,
        })
    }
}

// ===========================================================================
// Model-gated tool-call smoke tests. These drive the server's *exact* tool path
// in-process — canonical render → encode-with-specials → generate → parse — so a
// regression in any of those (or the model's tool-call quality) fails the gate.
// They skip when the quantized weights aren't present, and run single-threaded
// (shared GPU). Assertions are type-level (robust to seed/version drift), which
// is what the degradation experiment showed is the stable invariant.
// ===========================================================================
#[cfg(all(test, target_os = "macos"))]
mod tool_smoke {
    use crate::metal::{StepGenerateConfig, StepGenerateSession, generate_with_session};
    use crate::tools::ParsedToolCall;
    use serde_json::{Value, json};
    use std::path::{Path, PathBuf};

    fn model_dir() -> Option<PathBuf> {
        for p in ["model/diffusiongemma-q4emb", "/tmp/quantized-weights"] {
            let d = PathBuf::from(p);
            if d.join("model.dgq.json").exists() {
                return Some(d);
            }
        }
        None
    }

    /// One tool turn, driven exactly like the server worker: render the messages
    /// + tools in the model's canonical format, tokenize the specials, generate
    /// (stop at eos), then parse the calls back. Returns (raw reply, calls).
    fn tool_turn(dir: &Path, messages: &[Value], tools: &[Value]) -> (String, Vec<ParsedToolCall>) {
        let tok = crate::tokenizer::Tokenizer::load(dir.join("tokenizer.json")).unwrap();
        let layers = crate::resolve_model_layers(dir, None).unwrap();
        let stop = crate::config::load_generation_stop_tokens(dir);
        let sampler = crate::sample::sampler_for_steps(24, false);
        const MAX_SEQ: usize = 2048;
        let mut cfg = StepGenerateConfig::from_generate(7, 512, MAX_SEQ, layers, sampler, false);
        cfg.stop_token_ids = stop.clone();
        cfg.degenerate_reply_check = crate::chat_template::empty_reply_check(dir, stop);
        let (mut session, _) = StepGenerateSession::open(dir, &cfg, None).unwrap();

        let prompt_str = crate::tools::render_conversation(messages, tools, true, false);
        let prompt = tok.encode_with_specials(&prompt_str);
        cfg.max_new_tokens = 512.min(MAX_SEQ.saturating_sub(prompt.len()).max(1));
        let out = generate_with_session(&mut session, &prompt, &cfg, "tool-smoke").unwrap();
        let new_ids = crate::sample::strip_degenerate_token_ids(
            out.token_ids.get(prompt.len()..).unwrap_or(&[]),
        );
        let text = tok.decode(&new_ids);
        let calls = crate::tools::parse_tool_calls(&text);
        (text, calls)
    }

    fn list_dir_tool() -> Value {
        json!({"type":"function","function":{
            "name":"list_dir",
            "description":"Lists all files and subdirectories in a specified directory path",
            "parameters":{"type":"object","properties":{
                "path":{"type":"string","description":"The absolute path to the directory to list (e.g., /tmp)"}},
                "required":["path"]}}})
    }

    fn search_tool() -> Value {
        json!({"type":"function","function":{
            "name":"search",
            "description":"Search the web",
            "parameters":{"type":"object","properties":{
                "query":{"type":"string","description":"the query"},
                "count":{"type":"integer","description":"number of results"},
                "safe":{"type":"boolean","description":"safe search"},
                "sort":{"type":"string","description":"sort order","enum":["relevance","date"]}},
                "required":["query"]}}})
    }

    #[test]
    fn tool_smoke_emits_a_well_formed_call() {
        let Some(dir) = model_dir() else {
            eprintln!("skip tool_smoke_emits_a_well_formed_call: quantized model not present");
            return;
        };
        let msgs = [json!({"role":"user","content":"List the files in /tmp using the tool."})];
        let (text, calls) = tool_turn(&dir, &msgs, &[list_dir_tool()]);
        assert!(
            !calls.is_empty(),
            "expected a tool call, got reply: {text:?}"
        );
        assert_eq!(calls[0].name, "list_dir", "reply: {text:?}");
        let path = calls[0].arguments.get("path").and_then(Value::as_str);
        assert!(
            path.is_some_and(|p| p.contains("tmp")),
            "path arg should reference /tmp, got {:?} (reply: {text:?})",
            calls[0].arguments
        );
    }

    #[test]
    fn tool_smoke_argument_types_match_schema() {
        let Some(dir) = model_dir() else {
            eprintln!("skip tool_smoke_argument_types_match_schema: quantized model not present");
            return;
        };
        let msgs = [json!({"role":"user","content":
            "Search for \"rust async\" with 5 results, safe search on, sorted by date. Use the tool."})];
        let (text, calls) = tool_turn(&dir, &msgs, &[search_tool()]);
        assert!(
            !calls.is_empty(),
            "expected a tool call, got reply: {text:?}"
        );
        let a = &calls[0].arguments;
        assert_eq!(calls[0].name, "search", "reply: {text:?}");
        // The stable invariant (per the degradation experiment): the model emits
        // schema-typed args — integer/boolean/valid-enum, not stringified.
        assert!(
            a.get("count").is_some_and(Value::is_number),
            "count should parse as a number, got {a:?} (reply: {text:?})"
        );
        assert!(
            a.get("safe").is_some_and(Value::is_boolean),
            "safe should parse as a bool, got {a:?} (reply: {text:?})"
        );
        if let Some(sort) = a.get("sort").and_then(Value::as_str) {
            assert!(
                ["relevance", "date"].contains(&sort),
                "sort should be a declared enum member, got {sort:?} (reply: {text:?})"
            );
        }
    }
}

// ===========================================================================
// Model-gated compaction smoke tests (the KV rewinder). These drive the
// summarize-pass rewind, the substituted-prompt reuse, the expand re-entry,
// and the finalize eviction against the real model — the invariants the pure
// unit tests can't reach. Skip when the weights aren't present; run
// single-threaded (shared GPU). Assertions are structural (KV lengths, token
// logs), never on summary wording (seed-robust).
// ===========================================================================
#[cfg(all(test, target_os = "macos"))]
mod tool_compact_smoke {
    use crate::metal::{StepGenerateConfig, StepGenerateSession, generate_with_session};
    use crate::toolcompact as tc;
    use serde_json::{Value, json};
    use std::path::{Path, PathBuf};

    // The verbose fixture renders ~1800 tokens; 4096 leaves room for the
    // summarize prompt + a canvas block at every step.
    const MAX_SEQ: usize = 4096;
    const THRESHOLD: usize = 64;

    fn model_dir() -> Option<PathBuf> {
        for p in ["model/diffusiongemma-q4emb", "/tmp/quantized-weights"] {
            let d = PathBuf::from(p);
            if d.join("model.dgq.json").exists() {
                return Some(d);
            }
        }
        None
    }

    fn open(
        dir: &Path,
    ) -> (
        crate::tokenizer::Tokenizer,
        StepGenerateConfig,
        StepGenerateSession,
    ) {
        let tok = crate::tokenizer::Tokenizer::load(dir.join("tokenizer.json")).unwrap();
        let layers = crate::resolve_model_layers(dir, None).unwrap();
        let stop = crate::config::load_generation_stop_tokens(dir);
        let sampler = crate::sample::sampler_for_steps(24, false);
        let mut cfg = StepGenerateConfig::from_generate(7, 512, MAX_SEQ, layers, sampler, false);
        cfg.stop_token_ids = stop.clone();
        cfg.degenerate_reply_check = crate::chat_template::empty_reply_check(dir, stop);
        let (session, _) = StepGenerateSession::open(dir, &cfg, None).unwrap();
        (tok, cfg, session)
    }

    fn read_file_tool() -> Value {
        json!({"type":"function","function":{
            "name":"read_file",
            "description":"Read a file from disk",
            "parameters":{"type":"object","properties":{
                "path":{"type":"string","description":"absolute path"}},
                "required":["path"]}}})
    }

    /// A fake verbose tool output, comfortably over THRESHOLD tokens.
    fn big_output() -> String {
        let mut s = String::from("directory listing of /data:\n");
        for i in 1..=60 {
            s.push_str(&format!(
                "file_{i:03}.txt  {}00 bytes  2026-07-0{}\n",
                i,
                i % 9 + 1
            ));
        }
        s.push_str("secret_marker=zebra42\n");
        s
    }

    fn base_messages(output: &str) -> Vec<Value> {
        vec![
            json!({"role":"user","content":"Read /data and tell me what's inside."}),
            json!({"role":"assistant","content":"","tool_calls":[
                {"id":"call_0","type":"function",
                 "function":{"name":"read_file","arguments":"{\"path\":\"/data\"}"}}]}),
            json!({"role":"tool","tool_call_id":"call_0","content":output}),
        ]
    }

    /// The full M1 loop: route → summarize pass (checkpoint/rollback) →
    /// substituted main generation → finalize. Asserts the KV-state invariants
    /// at each step.
    #[test]
    fn tool_compact_rewind_smoke() {
        let Some(dir) = model_dir() else {
            eprintln!("skip tool_compact_rewind_smoke: quantized model not present");
            return;
        };
        let (tok, cfg, session) = open(&dir);
        let conv_dir =
            std::env::temp_dir().join(format!("dgq-compact-smoke-{}", std::process::id()));
        let mut manager = crate::conversation::ConversationManager::new(session, 0, 0, conv_dir);

        let output = big_output();
        let messages = base_messages(&output);
        let mut tools = vec![read_file_tool()];
        tools.push(tc::expand_summary_tool());
        let count = |s: &str| tok.encode(s, false).len();
        assert!(
            count(&output) > THRESHOLD,
            "fixture must exceed the threshold"
        );

        // Route + prefill the (still verbose) routing prompt, as the server does.
        let routing = tok.encode_with_specials(&crate::tools::render_conversation(
            &messages, &tools, true, false,
        ));
        let conv_id = manager.activate(&routing);
        manager.session_mut().extend_kv(&routing).unwrap();
        let before = manager.session_mut().kv_valid_tokens().to_vec();
        assert!(!before.is_empty());

        // Summarize pass: generates, then must roll the KV back exactly.
        let mut ctx = messages.clone();
        ctx.push(json!({"role":"user","content": tc::summarize_instruction()}));
        let stop = crate::config::load_generation_stop_tokens(&dir);
        let summary_opt = super::run_summarize_pass(
            manager.session_mut(),
            &tok,
            &cfg,
            24,
            &stop,
            &dir,
            MAX_SEQ,
            256,
            &ctx,
            &tools,
        );
        assert_eq!(
            manager.session_mut().kv_valid_tokens(),
            &before[..],
            "summarize pass must leave the KV exactly as it found it"
        );
        // Extracted or mechanical — either way compaction proceeds.
        let summary = summary_opt.unwrap_or_else(|| tc::mechanical_summary(&output, 1024));
        assert!(!summary.is_empty());

        // Substituted main generation (delta-prefills from the divergence point).
        let hash = tc::fnv1a64(&output);
        let resolve = |h: u64| (h == hash).then(|| ("tr_smoke".to_string(), summary.clone()));
        let messages_sub = tc::compact_messages(&messages, THRESHOLD, &count, &resolve);
        assert_ne!(messages_sub[2]["content"], messages[2]["content"]);
        let prompt = tok.encode_with_specials(&crate::tools::render_conversation(
            &messages_sub,
            &tools,
            true,
            false,
        ));
        let mut cfg_main = cfg.clone();
        cfg_main.max_new_tokens =
            256.min(MAX_SEQ.saturating_sub(prompt.len() + crate::metal::CANVAS));
        let out = generate_with_session(manager.session_mut(), &prompt, &cfg_main, "compact-smoke")
            .expect("substituted generation must succeed");
        let reply = tok.decode(&crate::sample::strip_degenerate_token_ids(
            out.token_ids.get(prompt.len()..).unwrap_or(&[]),
        ));
        let content = crate::tools::content_before_tool_calls(
            &crate::chat_template::sanitize_model_reply(&reply),
        );

        // Finalize with the substituted canonical: the resident KV becomes
        // exactly the compacted completed-turns log (verbose text evicted).
        let mut completed = messages_sub.clone();
        completed.push(json!({"role":"assistant","content": content}));
        let canonical = tok.encode_with_specials(&crate::tools::render_conversation(
            &completed, &tools, false, false,
        ));
        manager.finalize(conv_id, &canonical).unwrap();
        assert_eq!(manager.session_mut().kv_valid_tokens(), &canonical[..]);
        // The canonical log must not contain the verbose response (it holds the
        // substituted summary object instead) — the whole point of compaction.
        let canonical_text = tok.decode(&canonical);
        assert!(
            !canonical_text.contains("file_042.txt"),
            "verbose output leaked into canonical KV"
        );
        assert!(
            canonical_text.contains("tr_smoke"),
            "substituted id missing from canonical KV"
        );
    }

    /// The M2 machinery: after a generation, extend the KV with the model's own
    /// tokens + a server-rendered expand_summary response, re-enter generation
    /// with prompt == kv_valid_tokens (fresh block), and verify finalize evicts
    /// the excerpt.
    #[test]
    fn tool_compact_expand_reentry_smoke() {
        let Some(dir) = model_dir() else {
            eprintln!("skip tool_compact_expand_reentry_smoke: quantized model not present");
            return;
        };
        let (tok, cfg, mut session) = open(&dir);

        let output = big_output();
        let messages = base_messages(&output);
        let mut tools = vec![read_file_tool()];
        tools.push(tc::expand_summary_tool());
        let count = |s: &str| tok.encode(s, false).len();
        let hash = tc::fnv1a64(&output);
        let resolve = |h: u64| {
            (h == hash).then(|| ("tr_smoke".to_string(), "a directory listing".to_string()))
        };
        let messages_sub = tc::compact_messages(&messages, THRESHOLD, &count, &resolve);

        let prompt = tok.encode_with_specials(&crate::tools::render_conversation(
            &messages_sub,
            &tools,
            true,
            false,
        ));
        let mut cfg_r0 = cfg.clone();
        cfg_r0.max_new_tokens = 256;
        let out = generate_with_session(&mut session, &prompt, &cfg_r0, "expand-smoke").unwrap();

        // Hand-build the expand round exactly as handle_tool_compact does:
        // model's own token ids + the canonical rendered tool response.
        let excerpt =
            tc::dispatch_expand(&json!({"mode":"grep","pattern":"secret_marker"}), &output);
        assert!(excerpt.contains("zebra42"));
        let resp_text =
            crate::tools::render_tool_response(tc::EXPAND_TOOL_NAME, &json!({"content": excerpt}));
        let mut ext = out.token_ids.clone();
        ext.extend(tok.encode_with_specials(&resp_text));
        assert!(ext.len() + crate::metal::CANVAS < MAX_SEQ);
        let reuse = ext
            .iter()
            .zip(session.kv_valid_tokens())
            .take_while(|(a, b)| a == b)
            .count();
        session.truncate_kv_to(reuse);
        session.extend_kv(&ext[reuse..]).unwrap();
        assert_eq!(session.kv_valid_tokens(), &ext[..]);

        // Re-entry: prompt == kv_valid_tokens → no prefill, fresh block denoise.
        let round_prompt = session.kv_valid_tokens().to_vec();
        let mut cfg_r1 = cfg.clone();
        cfg_r1.max_new_tokens =
            256.min(MAX_SEQ.saturating_sub(round_prompt.len() + crate::metal::CANVAS));
        let out2 = generate_with_session(&mut session, &round_prompt, &cfg_r1, "expand-smoke")
            .expect("re-entry after expand extension must succeed");
        assert!(
            out2.token_ids.len() > round_prompt.len(),
            "re-entry must denoise new tokens"
        );

        // Finalize-equivalent: rebuild the canonical (no expand round in it) and
        // verify the excerpt tokens are gone from the causal KV.
        let mut completed = messages_sub.clone();
        completed.push(json!({"role":"assistant","content":"done"}));
        let canonical = tok.encode_with_specials(&crate::tools::render_conversation(
            &completed, &tools, false, false,
        ));
        let reuse = canonical
            .iter()
            .zip(session.kv_valid_tokens())
            .take_while(|(a, b)| a == b)
            .count();
        session.truncate_kv_to(reuse);
        session.extend_kv(&canonical[reuse..]).unwrap();
        assert_eq!(session.kv_valid_tokens(), &canonical[..]);
        assert!(
            !tok.decode(session.kv_valid_tokens()).contains("zebra42"),
            "expand excerpt must be evicted from the canonical KV"
        );
    }

    /// Regression for the KV-overflow panic hit in production (2026-07-09): a
    /// 13-block reply overshot the token budget by up to a block, and finalize's
    /// extend of the ~context-full canonical tripped the `set_kv_len` assert.
    /// Now: `extend_kv` past capacity is a typed error, and `finalize` trims its
    /// extend to the longest fitting prefix (full canonical kept for routing).
    #[test]
    fn tool_compact_overlong_finalize_is_trimmed_not_panic() {
        let Some(dir) = model_dir() else {
            eprintln!(
                "skip tool_compact_overlong_finalize_is_trimmed_not_panic: quantized model not present"
            );
            return;
        };
        let (tok, _cfg, mut session) = open(&dir);
        let capacity = session.extend_capacity();
        assert_eq!(capacity, MAX_SEQ - crate::metal::CANVAS);

        // Raw extend past capacity: typed error, no partial write, no panic.
        let overlong: Vec<u32> = tok
            .encode(&"inventory line item alpha beta ".repeat(900), false)
            .into_iter()
            .take(capacity + 100)
            .collect();
        assert!(overlong.len() > capacity, "fixture must exceed capacity");
        assert!(session.extend_kv(&overlong).is_err());
        assert_eq!(session.kv_valid_tokens().len(), 0);

        // Finalize with the same overlong canonical: Ok, KV holds the longest
        // fitting prefix, routing log keeps the full canonical.
        let conv_dir =
            std::env::temp_dir().join(format!("dgq-compact-smoke-fin-{}", std::process::id()));
        let mut manager = crate::conversation::ConversationManager::new(session, 0, 0, conv_dir);
        let conv_id = manager.activate(&overlong);
        manager.finalize(conv_id, &overlong).unwrap();
        assert_eq!(
            manager.session_mut().kv_valid_tokens(),
            &overlong[..capacity]
        );
    }
}
