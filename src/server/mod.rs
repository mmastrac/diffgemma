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
//!   "thinking" block. Default on via serve `--think` (override with
//!   `--think=false`). Per request: `model:think` / `:think=true` / `:think=false`,
//!   or body `enable_thinking` / `chat_template_kwargs.enable_thinking`
//!   (suffix wins, then body, then server default).
//! - **`x-diffusion-draft`** — the *shimmering private messages*: as the canvas
//!   denoises, positions flip in place before they commit. Standard `content`
//!   only ever carries block-**committed** (immutable) text, so any OpenAI client
//!   sees a correct append-only reply; the speculative, still-converging draft
//!   rides entirely in this `x-` extension delta (replace-semantics), so a
//!   diffusion-aware client can render the live shimmer. On by default
//!   (per-request `x_diffusion_drafts:false` opts out).

use crate::chat_protocol::TextDecoder;
use serde::Deserialize;
use std::net::TcpListener;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

mod log;
mod wire;
pub(crate) use log::*;
pub(crate) use wire::*;

/// Reject request bodies larger than this (basic abuse guard).
const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

// ===========================================================================
// Wire protocol (request in, chunks out).
// ===========================================================================

#[derive(Deserialize)]
struct ChatRequest {
    /// Client model id; supports `:think` / `:think=true` / `:think=false` suffixes.
    #[serde(default)]
    model: Option<String>,
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
    /// Expose the private thought channel as `reasoning_content`. Precedence:
    /// `model:think…` suffix > this field / `chat_template_kwargs` > serve `--think`.
    #[serde(default)]
    enable_thinking: Option<bool>,
    #[serde(default)]
    chat_template_kwargs: Option<ChatTemplateKwargs>,
    /// Stream speculative pre-commit canvas drafts as `x-diffusion-draft`.
    /// Default: on.
    #[serde(default)]
    x_diffusion_drafts: Option<bool>,
}

#[derive(Deserialize, Default)]
struct ChatTemplateKwargs {
    #[serde(default)]
    enable_thinking: Option<bool>,
}

/// Parse a trailing `:think` / `:think=true` / `:think=false` from a model id.
pub(crate) fn parse_model_think_suffix(model: &str) -> Option<bool> {
    let (_, suffix) = model.rsplit_once(':')?;
    let s = suffix.trim();
    let lower = s.to_ascii_lowercase();
    if lower == "think" {
        return Some(true);
    }
    if let Some(v) = lower.strip_prefix("think=") {
        return match v {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        };
    }
    None
}

/// Resolve thinking for one request: model suffix > body fields > server default.
pub(crate) fn resolve_enable_thinking(
    model: Option<&str>,
    enable_thinking: Option<bool>,
    chat_template_kwargs: Option<bool>,
    server_default: bool,
) -> bool {
    if let Some(model) = model
        && let Some(v) = parse_model_think_suffix(model)
    {
        return v;
    }
    enable_thinking
        .or(chat_template_kwargs)
        .unwrap_or(server_default)
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
pub(crate) enum ServerEvent {
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

// ===========================================================================
// GPU worker: owns the session, drains the job queue one at a time.
// ===========================================================================

struct Worker {
    model_dir: std::path::PathBuf,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
    stop_token_ids: Vec<u32>,
    channel_open: Option<u32>,
    channel_close: Option<u32>,
    /// The `<|"|>` string-quote special id (mapper quote-parity: quoted
    /// channel/stop ids are literal tool-arg content).
    quote_tok: Option<u32>,
    max_seq: usize,
    steps: usize,
    no_early_stop: bool,
    base_cfg: crate::metal::StepGenerateConfig,
    tool_compact: Option<ToolCompactCfg>,
    /// `--tool-repair`: model-guided repair — error tool-response, corrected
    /// call, corrupt exchange rewound out of KV.
    tool_repair: bool,
    /// `--tool-validate`: rewind + regenerate on malformed tool-call grammar.
    tool_validate: bool,
    /// When set, each completed request writes one `{seq}.jsonl` here.
    log_dir: Option<std::path::PathBuf>,
    turn_seq: AtomicU64,
    /// Last observed prefill rate (tok/s; 0 = not yet measured), feeding the
    /// dry-start status's expected-duration estimate.
    prefill_rate_tok_s: AtomicU64,
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

mod worker;

type ServeMapper = Arc<Mutex<DiffusionStreamMapper<Arc<crate::tokenizer::Tokenizer>>>>;

/// Tokens of un-reused prompt delta below which the prefill status line is
/// not worth the noise (a sub-second prefill).
pub(crate) const PREFILL_STATUS_MIN_TOKENS: usize = 1024;

/// Fallback prefill rate before the first in-process measurement (tok/s).
/// Conservative — underestimating progress delays "Almost done." rather
/// than firing it prematurely.
const PREFILL_RATE_DEFAULT_TOK_S: f64 = 250.0;

/// The dry-start status: while a large prompt delta prefills, no step
/// observer fires, so a streaming client sees dead air (OpenCode's silent
/// cold start). Staged synthetic `reasoning_content` messages: immediately
/// "Warming up, give me a second."; at 5 s "Taking a closer look."; at
/// 50% / 90% of the expected prefill duration "Getting there." / "Almost
/// done." — expected duration = delta tokens / the worker's last observed
/// prefill rate. When a later stage is already due, earlier ones are
/// skipped rather than rushed. The first denoise step sets `started` (the
/// stream observer flips it and appends a separator). Transient by
/// construction: the text never enters the mapper, so the final `Done`
/// payload and the canonical log are untouched.
pub(crate) fn spawn_prefill_status(
    resp: &mpsc::Sender<ServerEvent>,
    delta_tokens: usize,
    rate_tok_s: f64,
    started: &Arc<std::sync::atomic::AtomicBool>,
) {
    let resp = resp.clone();
    let started = Arc::clone(started);
    std::thread::spawn(move || {
        let t0 = std::time::Instant::now();
        let expected_secs = delta_tokens as f64 / rate_tok_s.max(1.0);
        let send = |msg: &str| {
            resp.send(ServerEvent::Delta(WireDelta::Reasoning(msg.to_string())))
                .is_ok()
        };
        if !send("Warming up, give me a second.") {
            return;
        }
        // Next unemitted stage: 1 = closer look (5 s), 2 = 50%, 3 = 90%.
        let mut stage = 1u32;
        while stage <= 3 {
            std::thread::sleep(std::time::Duration::from_millis(250));
            if started.load(Ordering::Relaxed) {
                return;
            }
            let elapsed = t0.elapsed().as_secs_f64();
            let due = if elapsed >= expected_secs * 0.9 {
                3
            } else if elapsed >= expected_secs * 0.5 {
                2
            } else if elapsed >= 5.0 {
                1
            } else {
                0
            };
            if due >= stage {
                let msg = match due {
                    1 => "\nTaking a closer look.",
                    2 => "\nGetting there.",
                    _ => "\nAlmost done.",
                };
                if started.load(Ordering::Relaxed) || !send(msg) {
                    return;
                }
                stage = due + 1;
            }
        }
    });
}

/// When `strip_tool_markup` is set, hide `<|tool_call>...` grammar from
/// streamed content/draft; clients get structured `delta.tool_calls` at Done.
/// When `log_progress` is set, print a condensed serve progress line every five
/// denoise steps (and on block commit).
/// `prefill_status`: the dry-start heartbeat's flag — flipped on the first
/// denoise step (ending the heartbeat), with a separator delta so the model's
/// real reasoning starts on its own line.
fn attach_stream_observer(
    cfg: &mut crate::metal::StepGenerateConfig,
    mapper: &ServeMapper,
    resp: &mpsc::Sender<ServerEvent>,
    strip_tool_markup: bool,
    log_progress: bool,
    prefill_status: Option<Arc<std::sync::atomic::AtomicBool>>,
) {
    let mapper = Arc::clone(mapper);
    // Message-layer status (tool repair / validator interventions) surfaces
    // as a synthetic thinking block, same channel as the prefill heartbeat.
    // Wire-only: the text never reaches the mapper or the canonical log.
    let status_resp = resp.clone();
    cfg.status_notify = Some(Arc::new(move |msg: &str| {
        let _ = status_resp.send(ServerEvent::Delta(WireDelta::Reasoning(format!(
            "\n\n{msg}\n\n"
        ))));
    }));
    let resp = resp.clone();
    let suppress = std::sync::atomic::AtomicBool::new(false);
    // Dead-socket cancel: the connection thread drops its receiver when the
    // client hangs up (SSE write error), so the next delta send fails — stop
    // the in-flight generate instead of denoising into the void.
    let cancel = crate::metal::CancelToken::new();
    cfg.cancel = Some(cancel.clone());
    cfg.step_observer = Some(Arc::new(move |ev: &crate::metal::StepProgressEvent<'_>| {
        if let Some(ref started) = prefill_status
            && !started.swap(true, Ordering::Relaxed)
        {
            let _ = resp.send(ServerEvent::Delta(WireDelta::Reasoning("\n\n".into())));
        }
        if log_progress
            && (ev.step_in_block == 1 || ev.step_in_block.is_multiple_of(5) || ev.block_done)
        {
            eprintln!(
                "serve: block {}/{} step {}/{} {}%",
                ev.block_idx,
                ev.max_blocks,
                ev.step_in_block,
                ev.max_steps,
                serve_progress_pct(ev.accept_count, ev.mean_entropy, ev.argmax.len()),
            );
        }
        let deltas = mapper.lock().unwrap().on_step(ev);
        for d in deltas {
            let Some(d) = filter_tool_markup_delta(d, strip_tool_markup, &suppress) else {
                continue;
            };
            if resp.send(ServerEvent::Delta(d)).is_err() {
                cancel.cancel();
            }
        }
    }));
}

/// Release any text the paced-stream mapper is still holding (a turn can end
/// without a stop token — token budget, cancel — and held text must reach the
/// client before the final frames). Fresh markup-filter state is safe here:
/// tool markup disables pacing at commit time, so held text never contains it.
fn flush_paced(mapper: &ServeMapper, resp: &mpsc::Sender<ServerEvent>, strip_tool_markup: bool) {
    let deltas = mapper.lock().unwrap().final_flush();
    let suppress = std::sync::atomic::AtomicBool::new(false);
    for d in deltas {
        if let Some(d) = filter_tool_markup_delta(d, strip_tool_markup, &suppress) {
            let _ = resp.send(ServerEvent::Delta(d));
        }
    }
}

/// Block progress % from accept density + mean entropy. Entropy carries more
/// weight: accept plateaus well below canvas length under early-stop, while
/// mean_ent drops ~2.5 -> ~0.02 as the canvas converges.
fn serve_progress_pct(accept: u32, mean_ent: f32, canvas: usize) -> u32 {
    let n = canvas.max(1) as f32;
    let a = (accept as f32 / n).clamp(0.0, 1.0);
    let e = 1.0 - (mean_ent / 4.0).clamp(0.0, 1.0);
    (100.0 * (0.4 * a + 0.6 * e)).round() as u32
}

/// One checkpoint-bracketed summarize generation: render `messages_ctx`
/// (conversation up to and including the verbose tool response + the summarize
/// instruction), generate a bounded reply with NO step observer, extract the
/// `<summarize>…</summarize>` span, and roll the KV back so nothing of the
/// pass persists. Returns None when the prompt doesn't fit, generation fails,
/// or the model produced no usable tags (caller falls back mechanically).
pub(crate) fn run_summarize_pass(
    stage: &dyn crate::pipeline::PipelineStage,
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
    let g = crate::tools::render_conversation_guarded(messages_ctx, tools, true, false);
    let prompt = tokenizer.encode_prompt(&g).0;
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
    // Internal pass: a stage intervention here must not surface a thinking
    // block into whatever client stream the base cfg was wired to.
    cfg.status_notify = None;

    // Mark → Generate → Rewind: the checkpoint/generate/rollback pattern as
    // pipeline ops, so this pass composes with any stage chain.
    use crate::pipeline::{PipelineEvent, PipelineOp};
    let mark = match stage.call(PipelineOp::Mark) {
        PipelineEvent::Marked { kv } => kv,
        ev => {
            eprintln!("serve: tool-compact: summarize mark failed: {ev:?}");
            return None;
        }
    };
    let result = stage.call(PipelineOp::Generate {
        prompt: prompt.clone(),
        cfg: Box::new(cfg),
        label: "tool-compact".into(),
    });
    // Always rewind — the verbose response, the instruction, and the summary
    // itself must never persist in the conversation's KV.
    if let PipelineEvent::Error(err) = stage.call(PipelineOp::Rewind(mark)) {
        eprintln!("serve: tool-compact: summarize KV rollback failed: {err}");
        return None;
    }
    let out = match result {
        PipelineEvent::Generated { out, .. } => *out,
        ev => {
            eprintln!("serve: tool-compact: summarize pass failed: {ev:?}");
            return None;
        }
    };
    let generated = &out.token_ids[prompt.len().min(out.token_ids.len())..];
    let text = crate::chat_template::sanitize_model_reply(
        &tokenizer.decode(&crate::sample::strip_degenerate_token_ids(generated)),
    );
    crate::toolcompact::extract_summary(&text)
}

/// Cap an expand excerpt to a token budget so retrieval can't blow the
/// context the compactor just reclaimed.
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
    tool_repair: bool,
    tool_validate: bool,
    log_dir: Option<std::path::PathBuf>,
    think: bool,
) -> Result<(), String> {
    use crate::metal::StepGenerateConfig;

    // Quiet by default: serve prints its own condensed turn log. Explicit
    // DGQ_QUIET=0 (or any non-1) was set by the user if quiet_from_env; chat
    // restores verbose via --verbose, serve has no such flag so leave their
    // choice alone when the env was present.
    if !crate::flags::quiet_set_by_user() {
        crate::flags::set_quiet(true);
    }

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

    if let Some(ref dir) = log_dir {
        std::fs::create_dir_all(dir).map_err(|e| format!("--log-dir {}: {e}", dir.display()))?;
        eprintln!("serve: turn log -> {}", dir.display());
    }

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
    let quote_tok = tokenizer.special_token_id("<|\"|>");
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
    let tool_repair = tool_repair || crate::flags::tool_repair_enabled();
    let tool_validate = tool_validate || crate::flags::tool_validate_enabled();
    let worker = Worker {
        model_dir: model_dir.to_path_buf(),
        tokenizer,
        stop_token_ids,
        channel_open,
        channel_close,
        quote_tok,
        max_seq: ctx,
        steps,
        no_early_stop: false,
        base_cfg,
        tool_compact: tool_compact.then(|| ToolCompactCfg {
            threshold: crate::flags::tool_compact_threshold(),
            max_expand_rounds: 4,
            summarize_max_new: 512,
        }),
        tool_repair,
        tool_validate,
        log_dir,
        turn_seq: AtomicU64::new(1),
        prefill_rate_tok_s: AtomicU64::new(0),
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
    eprintln!(
        "serve: listening on http://{local}  (POST /v1/chat/completions; think default={think})"
    );
    eprintln!(
        "serve: model `{}` — override with model:think / :think=false / enable_thinking",
        model_name
    );

    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let jobs = job_tx.clone();
                let name = Arc::clone(&model_name);
                std::thread::spawn(move || handle_connection(stream, jobs, name, think));
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
mod tool_compact_smoke;
