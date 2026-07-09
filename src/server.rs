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
            self.handle(&mut manager, job);
        }
    }

    fn handle(&self, manager: &mut crate::conversation::ConversationManager, job: Job) {
        let tool_mode = needs_tool_rendering(&job.messages, &job.tools);
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

        let mut cfg = self.base_cfg.clone();
        cfg.sampler = crate::sample::sampler_for_steps(self.steps, self.no_early_stop);
        cfg.max_new_tokens = job.max_tokens.map_or(budget, |c| c.min(budget));
        cfg.seed = job.seed.unwrap_or(self.base_cfg.seed);
        cfg.stop_token_ids = self.stop_token_ids.clone();
        cfg.degenerate_reply_check =
            crate::chat_template::empty_reply_check(&self.model_dir, self.stop_token_ids.clone());

        let mapper = Arc::new(Mutex::new(DiffusionStreamMapper::new(
            Arc::clone(&self.tokenizer),
            self.stop_token_ids.clone(),
            self.channel_open,
            self.channel_close,
            job.enable_thinking,
            job.emit_drafts,
        )));
        {
            let mapper = Arc::clone(&mapper);
            let resp = job.resp.clone();
            cfg.step_observer = Some(Arc::new(move |ev: &crate::metal::StepProgressEvent<'_>| {
                let deltas = mapper.lock().unwrap().on_step(ev);
                for d in deltas {
                    let _ = resp.send(ServerEvent::Delta(d));
                }
            }));
        }

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
                if let Some(canonical) = canonical {
                    if let Err(err) = manager.finalize(conv_id, &canonical) {
                        // A failed finalize only costs reuse (next turn re-prefills);
                        // the reply is already correct.
                        eprintln!("serve: conversation finalize failed: {err}");
                    }
                }

                let _ = job.resp.send(ServerEvent::Done {
                    content,
                    reasoning,
                    tool_calls,
                    prompt_tokens: prompt_len,
                    completion_tokens,
                    stopped: out.stopped_on_eot,
                });
            }
            Err(err) => {
                let _ = job.resp.send(ServerEvent::Error(format!("{err}")));
            }
        }
    }
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
        emit_drafts: req.x_diffusion_drafts.unwrap_or(true),
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
