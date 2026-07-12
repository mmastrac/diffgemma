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
use std::net::TcpListener;
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

    #[allow(dead_code)]
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
pub(crate) struct Job {
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

#[path = "server_worker.rs"]
mod worker;

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
#[path = "server_http.rs"]
mod http;
pub(crate) use http::*;
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
    let layers =
        crate::commands::resolve_model_layers(model_dir, max_layers).map_err(|e| e.to_string())?;

    // Fail-fast context budget guard before loading weights (mirrors chat/ask).
    crate::commands::check_ctx_budget(ctx)?;

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
#[path = "server_tests.rs"]
mod tests;

// ===========================================================================
// Model-gated tool-call smoke tests. These drive the server's *exact* tool path
// in-process — canonical render → encode-with-specials → generate → parse — so a
// regression in any of those (or the model's tool-call quality) fails the gate.
// They skip when the quantized weights aren't present, and run single-threaded
// (shared GPU). Assertions are type-level (robust to seed/version drift), which
// is what the degradation experiment showed is the stable invariant.
// ===========================================================================
#[cfg(all(test, target_os = "macos"))]
#[path = "server_tool_smoke_tests.rs"]
mod tool_smoke;

// ===========================================================================
// Model-gated compaction smoke tests (the KV rewinder). These drive the
// summarize-pass rewind, the substituted-prompt reuse, the expand re-entry,
// and the finalize eviction against the real model — the invariants the pure
// unit tests can't reach. Skip when the weights aren't present; run
// single-threaded (shared GPU). Assertions are structural (KV lengths, token
// logs), never on summary wording (seed-robust).
// ===========================================================================
#[cfg(all(test, target_os = "macos"))]
#[path = "server_tool_compact_tests.rs"]
mod tool_compact_smoke;
