//! Interactive `chat` subcommand: the REPL. Input parsing, slash commands,
//! the ghost suggester, and conversation state live here; producing a turn —
//! generation, tool loop, forced replies, and every protocol event — is
//! `crate::chat::engine`.

use super::*;

#[cfg(target_os = "macos")]
use crate::chat::engine::{
    PendingTurn, ShellTool, TurnRunner, TurnSpec, parse_tool_spec, run_turn, tool_declaration,
};

/// One-line-per-command summary printed by `/help`.
#[cfg(target_os = "macos")]
const SLASH_HELP: &str = "\
slash commands:
  /help            show this help
  /prompt <text>   set the system prompt (kept across /clear; bare clears it)
  /clear           reset the conversation, keeping the system prompt
  /rewind          undo the last exchange (go back one step)
  /load <file>     attach a file; expands as $$fileN$$ in your next message
  /prethink <text> seed the thinking for your next message ($$last$$/$$prompt$$)
  /prethink persistent <text>  seed the thinking on every message (bare clears)
  /response <text> force the next reply to <text> (no generation; bare clears)
  /thought <text>  inject a fully-formed thought, kept in the KV (bare clears)
  /show-thinking [on|off]  stream the model's reasoning live (bare toggles)
  /show-output [on|off]    stream the answer as it forms (bare toggles)
  /private         wipe recall history up to now; recording continues after
  /exit, /quit     end the session (Ctrl-D also exits)";

/// Instruction appended as a user turn so the model, replying as itself, ghost-
/// writes the user's likely next message. Phrased as a concrete, output-only
/// content request (not a standing directive) so the model drafts the message
/// rather than acknowledging it. Latency is bounded by the denoise-step cap
/// below — NOT by asking for a "quick"/"short" answer, which just collapses the
/// draft to a generic filler ("Merci !", "Thanks!") regardless of context.
#[cfg(target_os = "macos")]
const SUGGEST_INSTRUCTION: &str = "Based on our conversation, write the single message I am most \
likely to send you next. Make it specific to what we are actually discussing, not a generic \
pleasantry. Output only that message, in my language and style, as if I typed it — no preamble, \
no quotation marks, no explanation.";

/// rustyline editor carrying our ghost-suggestion helper and file-backed history.
#[cfg(target_os = "macos")]
type ChatEditor = rustyline::Editor<SuggestHelper, rustyline::history::DefaultHistory>;

/// Shared slot for the current ghost suggestion. The async suggester thread
/// fills it when a draft is ready (while the user sits at the prompt); the
/// rustyline `Hinter` reads it on each refresh.
#[cfg(target_os = "macos")]
type SuggestionCell = std::sync::Arc<std::sync::Mutex<Option<String>>>;

/// Shared, `Send` handle to rustyline's cross-thread external printer (a trait,
/// so the platform-specific concrete type is boxed).
#[cfg(target_os = "macos")]
type ExternalPrinterHandle =
    std::sync::Arc<std::sync::Mutex<Box<dyn rustyline::ExternalPrinter + Send>>>;

/// rustyline helper that shows a single whole-message ghost suggestion on an
/// empty line — dimmed, and accepted with Tab (or →). The text lives behind a
/// shared cell so the background suggester can populate it asynchronously.
#[cfg(target_os = "macos")]
struct SuggestHelper {
    suggestion: SuggestionCell,
}

#[cfg(target_os = "macos")]
impl rustyline::completion::Completer for SuggestHelper {
    type Candidate = String;
}

#[cfg(target_os = "macos")]
impl rustyline::validate::Validator for SuggestHelper {}

#[cfg(target_os = "macos")]
impl rustyline::hint::Hinter for SuggestHelper {
    type Hint = String;
    fn hint(&self, line: &str, pos: usize, _ctx: &rustyline::Context<'_>) -> Option<String> {
        // Offer the drafted next message only on a pristine, empty line; once the
        // user starts typing their own text, the ghost steps aside.
        if line.is_empty() && pos == 0 {
            self.suggestion.lock().ok().and_then(|s| s.clone())
        } else {
            None
        }
    }
}

#[cfg(target_os = "macos")]
impl rustyline::highlight::Highlighter for SuggestHelper {
    fn highlight_hint<'h>(&self, hint: &'h str) -> std::borrow::Cow<'h, str> {
        // Faded grey ghost text (ANSI bright-black), reset at the end.
        std::borrow::Cow::Owned(format!("\x1b[90m{hint}\x1b[0m"))
    }
}

#[cfg(target_os = "macos")]
impl rustyline::Helper for SuggestHelper {}

/// Tab binding that accepts the ghost suggestion by reading the shared cell
/// LIVE at key-press. rustyline only recomputes the inline hint on a keypress,
/// so a suggestion that arrives asynchronously (revealed by the external
/// printer while the user sits idle) isn't in `self.hint` yet — `CompleteHint`
/// would beep. This reads the cell directly instead, so the first Tab fills it.
#[cfg(target_os = "macos")]
struct TabAcceptHandler {
    suggestion: SuggestionCell,
}

#[cfg(target_os = "macos")]
impl rustyline::ConditionalEventHandler for TabAcceptHandler {
    fn handle(
        &self,
        _evt: &rustyline::Event,
        _n: rustyline::RepeatCount,
        _positive: bool,
        ctx: &rustyline::EventContext<'_>,
    ) -> Option<rustyline::Cmd> {
        let cell = self.suggestion.lock().ok().and_then(|s| s.clone());
        tab_fill_text(ctx.line().is_empty(), cell.as_deref())
            .map(|text| rustyline::Cmd::Insert(1, text))
    }
}

/// Decide what Tab fills: the draft, but only on a pristine empty line (the
/// whole-message slot) and only when a non-empty suggestion is present. Once the
/// user has typed, Tab falls through to its default (no-op) behaviour. Extracted
/// from [`TabAcceptHandler::handle`] because `EventContext` can't be built
/// outside rustyline, so this holds the logic under test.
#[cfg(target_os = "macos")]
fn tab_fill_text(line_empty: bool, suggestion: Option<&str>) -> Option<String> {
    match suggestion {
        Some(text) if line_empty && !text.is_empty() => Some(text.to_string()),
        _ => None,
    }
}

/// Best-effort "next message" draft: as a low-priority follow-up to the reply
/// just written, ask the model to ghost-write the user's likely next line. Runs
/// as a single `Generate` under the pipeline lock (the single GPU queue),
/// cancellable mid-flight. No Mark/Rewind — the probe's tokens simply linger in
/// the KV until the next real turn's prefix-match rewinds them, which is exactly
/// what normal cross-turn reuse already does. Returns `None` on cancel, tight
/// context, any hiccup, or an empty draft.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn generate_suggestion(
    pipeline: &std::sync::Mutex<crate::pipeline::Pipeline>,
    tokenizer: &tokenizer::Tokenizer,
    history: &[chat_template::ChatTurn],
    layers: usize,
    max_seq: usize,
    steps: usize,
    no_early_stop: bool,
    stop_token_ids: &[u32],
    seed: u64,
    cancel: &metal::CancelToken,
) -> Option<String> {
    use crate::pipeline::{PipelineEvent, PipelineOp};

    // Append the instruction as a user turn and let the model reply — its reply
    // is the drafted next message. Keeps the real conversation as the KV prefix
    // (cross-turn reuse survives), unlike a fresh-context re-encode.
    let mut probe = history.to_vec();
    probe.push(chat_template::ChatTurn::user(SUGGEST_INSTRUCTION));
    let prompt = chat_template::format_chat_token_ids(
        tokenizer,
        &probe,
        &chat_template::ChatFormatOptions::default(),
    )
    .ok()?;

    // Terse and cheap; skip when the fixed arena can't fit even a short draft.
    const SUGGEST_CAP: usize = 48;
    // A ghost hint is disposable, so cap the denoiser below a real turn's budget
    // (the dominant latency lever) — but not so hard it can't develop a
    // context-specific draft before settling. Thinking is already off (the chat
    // template seeds an empty thought channel).
    const SUGGEST_MAX_STEPS: usize = 16;
    if prompt.len() + metal::CANVAS + SUGGEST_CAP >= max_seq {
        return None;
    }

    let sampler = sample::sampler_for_steps(steps.min(SUGGEST_MAX_STEPS), no_early_stop);
    let mut cfg = metal::StepGenerateConfig::from_generate(
        seed,
        SUGGEST_CAP,
        max_seq,
        layers,
        sampler,
        no_early_stop,
    );
    cfg.stop_token_ids = stop_token_ids.to_vec();
    cfg.cancel = Some(cancel.clone());

    let result = pipeline.lock().unwrap().call(PipelineOp::Generate {
        prompt: prompt.clone(),
        cfg: Box::new(cfg),
        label: "suggest".into(),
    });
    // Preempted by a submit: discard whatever the aborted probe produced.
    if cancel.is_cancelled() {
        return None;
    }
    let out = match result {
        PipelineEvent::Generated { out, .. } => *out,
        _ => return None,
    };
    let new_ids =
        sample::strip_degenerate_token_ids(out.token_ids.get(prompt.len()..).unwrap_or(&[]));
    let reply = chat_template::sanitize_model_reply(&tokenizer.decode(&new_ids));
    // A suggested query is a single line: collapse any whitespace/newlines.
    let one_line = reply.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.is_empty() {
        None
    } else {
        Some(one_line)
    }
}

/// A ghost-suggestion draft running on its own thread while the user sits at the
/// prompt. `cancel_and_join` preempts it and waits for the worker to unwind, so
/// the pipeline is free and the KV settled before the next real turn.
#[cfg(target_os = "macos")]
struct SuggestTask {
    cancel: metal::CancelToken,
    handle: std::thread::JoinHandle<()>,
}

#[cfg(target_os = "macos")]
impl SuggestTask {
    fn cancel_and_join(self) {
        self.cancel.cancel();
        let _ = self.handle.join();
    }
}

/// Everything the background suggester needs, cloned per turn. Spawning consumes
/// it and hands back a [`SuggestTask`] the loop cancels on the next submit.
#[cfg(target_os = "macos")]
struct Suggester {
    pipeline: std::sync::Arc<std::sync::Mutex<crate::pipeline::Pipeline>>,
    tokenizer: std::sync::Arc<tokenizer::Tokenizer>,
    layers: usize,
    max_seq: usize,
    steps: usize,
    no_early_stop: bool,
    stop: Vec<u32>,
    seed: u64,
    /// Interactive ghost cell (rustyline `Hinter` reads it), if a tty editor.
    cell: Option<SuggestionCell>,
    /// Interactive reveal: prints the draft above the prompt when it lands.
    printer: Option<ExternalPrinterHandle>,
    /// Session sinks: a landed draft emits a `Suggestion` event (JSON drivers
    /// see it; the plain display ignores it).
    sinks: crate::chat::SharedSinks,
}

#[cfg(target_os = "macos")]
impl Suggester {
    fn spawn(self, history: Vec<chat_template::ChatTurn>, turn: u64) -> SuggestTask {
        let cancel = metal::CancelToken::new();
        let cancel_bg = cancel.clone();
        let handle = std::thread::spawn(move || {
            let sug = generate_suggestion(
                &self.pipeline,
                &self.tokenizer,
                &history,
                self.layers,
                self.max_seq,
                self.steps,
                self.no_early_stop,
                &self.stop,
                self.seed,
                &cancel_bg,
            );
            // Cancelled or empty: leave no ghost, emit no event.
            if cancel_bg.is_cancelled() {
                return;
            }
            let Some(text) = sug else {
                return;
            };
            if let Some(cell) = &self.cell
                && let Ok(mut c) = cell.lock()
            {
                *c = Some(text.clone());
            }
            if let Some(printer) = &self.printer
                && let Ok(mut p) = printer.lock()
            {
                let _ = p.print(format!("\x1b[90m⇥ {text}\x1b[0m"));
            }
            crate::chat::emit(
                &self.sinks,
                &crate::chat::ChatEvent::Suggestion { turn, text },
            );
        });
        SuggestTask { cancel, handle }
    }
}

/// Expand `$$fileN$$` markers in a submitted line to the contents of the
/// matching `/load` slot (1-based). Markers that reference an unloaded slot are
/// left verbatim, so a stray `$$file9$$` reaches the model as plain text rather
/// than vanishing silently.
#[cfg(target_os = "macos")]
fn expand_file_markers(line: &str, loaded_files: &[String]) -> String {
    let mut out = line.to_string();
    for (i, contents) in loaded_files.iter().enumerate() {
        out = out.replace(&format!("$$file{}$$", i + 1), contents);
    }
    out
}

/// A stdin line parsed as a JSON command object (`--json` input mode). A
/// non-`message` command is translated to its `/slash` equivalent and reuses the
/// existing handlers; `Message` submits its text as a user turn (never re-parsed
/// as a command, so message text may start with `/`); `Exit` ends the session.
#[cfg(target_os = "macos")]
enum JsonInput {
    Command(String),
    Message(String),
    Exit,
}

/// Translate a `{"type":...}` JSON command object into a [`JsonInput`], mirroring
/// the slash commands 1:1, the driver-facing dual of the `--json` event stream.
/// `None` for an unrecognized `type`. Accepted objects:
/// `message{text}`, `prompt{text}`, `clear`, `rewind`, `private`,
/// `prethink{text,persistent?}`, `thought{text}`, `response{text}`, `load{path}`,
/// `show_thinking{on?}`, `show_output{on?}`, `help`, `exit`/`quit`. A missing/empty
/// `text` (or `on`) yields a bare command, so it clears / toggles as in the REPL.
#[cfg(target_os = "macos")]
fn json_command_to_line(v: &serde_json::Value) -> Option<JsonInput> {
    let ty = v.get("type")?.as_str()?;
    let text = || {
        v.get("text")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_string()
    };
    let on = || match v.get("on").and_then(serde_json::Value::as_bool) {
        Some(true) => "on",
        Some(false) => "off",
        None => "", // bare -> toggle
    };
    // Join a slash keyword with its argument, dropping the space when empty so a
    // bare command clears / toggles exactly as the text interface does.
    let join = |kw: &str, arg: &str| {
        let arg = arg.trim_end();
        if arg.is_empty() {
            kw.to_string()
        } else {
            format!("{kw} {arg}")
        }
    };
    let cmd = |s: String| Some(JsonInput::Command(s));
    match ty {
        "message" => Some(JsonInput::Message(text())),
        "exit" | "quit" => Some(JsonInput::Exit),
        "help" => cmd("/help".to_string()),
        "clear" => cmd("/clear".to_string()),
        "rewind" => cmd("/rewind".to_string()),
        "private" => cmd("/private".to_string()),
        "prompt" => cmd(join("/prompt", &text())),
        "thought" => cmd(join("/thought", &text())),
        "response" => cmd(join("/response", &text())),
        "prethink" => {
            let persistent = v
                .get("persistent")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            let body = if persistent {
                format!("persistent {}", text())
            } else {
                text()
            };
            cmd(join("/prethink", &body))
        }
        "load" => {
            let path = v
                .get("path")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            cmd(join("/load", path))
        }
        "show_thinking" | "show-thinking" => cmd(join("/show-thinking", on())),
        "show_output" | "show-output" => cmd(join("/show-output", on())),
        _ => None,
    }
}

/// Parse a `/show-*` slash argument into a new flag value: bare toggles the
/// `current` state, `on`/`off` (and common synonyms) set it, anything else is
/// rejected (`None`, so the caller prints usage).
#[cfg(target_os = "macos")]
fn toggle_flag(arg: &str, current: bool) -> Option<bool> {
    match arg.trim().to_ascii_lowercase().as_str() {
        "" => Some(!current),
        "on" | "true" | "1" | "yes" => Some(true),
        "off" | "false" | "0" | "no" => Some(false),
        _ => None,
    }
}

/// History with the `/prompt` system prompt (if any) prepended as a leading
/// System turn, the shape both prompt builders expect for a system message.
#[cfg(target_os = "macos")]
pub(crate) fn run_chat_cmd(
    model_dir: &std::path::Path,
    initial_prompt: Option<String>,
    seed: u64,
    steps: usize,
    max_new_tokens: usize,
    max_layers: Option<usize>,
    no_early_stop: bool,
    raw_prompt: bool,
    verbose: bool,
    events_path: Option<PathBuf>,
    json: bool,
    ctx: Option<usize>,
    tools: Vec<String>,
    harness: Option<PathBuf>,
    show_thinking: bool,
    stream_output: bool,
) -> ExitCode {
    use metal::StepGenerateConfig;
    use std::io::{self, IsTerminal, Write};

    // `DGQ_RENDER_DEMO`: exercise the terminal renderer with synthetic events and
    // no model (terminal-choreography verification, e.g. under tmux).
    if let Ok(stage) = std::env::var("DGQ_RENDER_DEMO") {
        crate::chat::render_demo(&stage);
        return ExitCode::SUCCESS;
    }

    // Parse `--tool NAME[:DESC]=COMMAND` specs up front so a bad spec fails
    // before the model loads. Non-empty => every turn uses the tool path.
    let mut shell_tools: Vec<ShellTool> = match tools.iter().map(|s| parse_tool_spec(s)).collect() {
        Ok(t) => t,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };
    // A `--harness` file bundles a prompt, prethink template, file-backed
    // vars, and tools; its tools join any `--tool` definitions.
    let harness = match harness.as_deref().map(crate::chat::harness::Harness::load) {
        Some(Ok(h)) => Some(h),
        Some(Err(err)) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
        None => None,
    };
    if let Some(h) = &harness {
        shell_tools.extend(h.shell_tools());
    }
    let tools_json: Vec<serde_json::Value> = shell_tools.iter().map(tool_declaration).collect();

    // Quiet by default: chat is a UI, not a log. `--verbose` restores the
    // step/prefill/session logs; `--json` routes the event stream to stdout so
    // no human chrome may pollute it.
    if !verbose && !flags::quiet_set_by_user() {
        flags::set_quiet(true);
    }
    // `--json`: JSONL to stdout, all human output suppressed. Otherwise the
    // spinner/streaming renderer runs on an interactive tty (not under
    // --verbose). `--events <path>` tees the JSONL to a file either way.
    let interactive = !json && !verbose && io::stdout().is_terminal();
    // The session's event sinks: every turn event flows through this set; the
    // interactive renderer is the one extra view on the same stream.
    let mut sink_set = crate::chat::SinkSet::new();
    if json {
        sink_set.push(Box::new(crate::chat::JsonlSink(io::stdout())));
    }
    if let Some(p) = &events_path {
        match std::fs::File::create(p) {
            Ok(f) => sink_set.push(Box::new(crate::chat::JsonlSink(f))),
            Err(err) => {
                eprintln!("error: cannot open --events {}: {err}", p.display());
                return ExitCode::FAILURE;
            }
        }
    }
    if !interactive && !json {
        // Renderless human display (piped stdin / --verbose): settled lines.
        sink_set.push(Box::new(crate::chat::PlainSink));
    }
    let sinks = sink_set.into_shared();
    let want_json = json || events_path.is_some();

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: chat requires a .dgq directory (-m /path/to/quantized-weights)");
        return ExitCode::FAILURE;
    }

    let layers = match resolve_model_layers(model_dir, max_layers) {
        Ok(n) => n,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    // Full-message chat: generate until the model emits its end-of-turn token,
    // limited only by the context window (no per-turn token cap — long replies
    // stream freely). The KV arena is sized once at session open and fixed
    // thereafter, so pick a roomy `max_seq` up front. `--ctx N` overrides for
    // long-context sessions (ring-buffer sliding KV keeps this cheap:
    // ~20 KB/token + ~410 MiB fixed; 131072 ≈ 3 GiB). `--max-new-tokens`, if
    // the caller raises it above the default, becomes an optional hard ceiling.
    #[allow(non_snake_case)]
    let CHAT_MAX_SEQ: usize = ctx.unwrap_or(8192);
    // Fail-fast --ctx budget guard: refuse a context whose weights+KV would
    // swap / fail to allocate, before loading the model (see check_ctx_budget).
    if let Err(msg) = check_ctx_budget(CHAT_MAX_SEQ) {
        eprintln!("error: {msg}");
        return ExitCode::FAILURE;
    }
    let explicit_cap = if max_new_tokens > 256 {
        Some(max_new_tokens)
    } else {
        None
    };

    let stop_token_ids = config::load_generation_stop_tokens(model_dir);
    if verbose {
        eprintln!("chat: full-message stop tokens = {stop_token_ids:?}, cap = {explicit_cap:?}");
    }

    let sampler = sample::sampler_for_steps(steps, no_early_stop);
    let mut step_cfg = StepGenerateConfig::from_generate(
        seed,
        CHAT_MAX_SEQ,
        CHAT_MAX_SEQ,
        layers,
        sampler,
        no_early_stop,
    );
    step_cfg.degenerate_reply_check =
        chat_template::empty_reply_check(model_dir, stop_token_ids.clone());
    // Kept aside for the ghost-suggestion probe (step_cfg takes ownership next).
    let suggest_stop = stop_token_ids.clone();
    step_cfg.stop_token_ids = stop_token_ids;

    if interactive {
        print!("loading model… ");
        let _ = io::stdout().flush();
    }
    let open_started = std::time::Instant::now();
    // Chat is a token-pipeline client: the session lives on the pipeline
    // thread; each turn is a Generate op (stage chains insert here later).
    // Behind an `Arc<Mutex<>>` so the async suggester can drive the same single
    // GPU queue from its own thread — each `call` is one atomic lock, preserving
    // op→event pairing and the one-op-at-a-time invariant.
    let pipeline = std::sync::Arc::new(std::sync::Mutex::new(crate::pipeline::Pipeline::spawn(
        model_dir.to_path_buf(),
        CHAT_MAX_SEQ,
        steps,
    )));
    match pipeline
        .lock()
        .unwrap()
        .call(crate::pipeline::PipelineOp::Ping)
    {
        crate::pipeline::PipelineEvent::Pong => {
            if interactive {
                println!("ready ({:.1}s)", open_started.elapsed().as_secs_f64());
            } else if verbose {
                eprintln!(
                    "chat: pipeline ready ({:.2?}, layers={layers})",
                    open_started.elapsed()
                );
            }
        }
        ev => {
            eprintln!("error: pipeline open failed: {ev:?}");
            return ExitCode::FAILURE;
        }
    }

    let tok_path = model_dir.join("tokenizer.json");
    let tokenizer = match tokenizer::Tokenizer::load(&tok_path) {
        Ok(t) => std::sync::Arc::new(t),
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    // Tool turns generate the `call:{...}` grammar. Honor the model's stop rather
    // than forcing continuation past an open call -- a malformed call never
    // rescues and floods blocks (see server::worker); `tool_loop` feeds an error
    // tool response and regenerates instead. `DGQ_CONTINUE_PAST_STOP` restores the
    // old bet. Mark the `<|"|>` quote so arg strings survive under it.
    if !tools_json.is_empty() {
        step_cfg.continue_incomplete_tool_calls = crate::flags::continue_past_stop_enabled();
        step_cfg.quote_token_id = tokenizer.special_token_id("<|\"|>");
    }

    // (No warm-up: COLD-START-1 — the first fresh-session generation returning an
    // empty/EOS reply — is fixed at the root by the deterministic first-step SC
    // seed. The throwaway warm-up generation is no longer needed.)

    let mut history: Vec<chat_template::ChatTurn> = Vec::new();
    let mut turn_idx = 0u64;
    // System prompt set via `/prompt`; prepended as a leading System turn on
    // every build (survives `/clear`). Passed per-call, not captured, so
    // `/prompt` can still mutate it between turns.
    let mut system_prompt: Option<String> = harness.as_ref().and_then(|h| h.prompt.clone());

    let mut runner = TurnRunner {
        pipeline: &pipeline,
        tokenizer: &tokenizer,
        step_cfg: &mut step_cfg,
        tools_json: &tools_json,
        shell_tools: &shell_tools,
        sinks: std::sync::Arc::clone(&sinks),
        raw_prompt,
        interactive,
        json,
        max_seq: CHAT_MAX_SEQ,
        explicit_cap,
        seed,
        show_thinking,
        stream_output,
        vars: std::collections::HashMap::new(),
        file_vars: harness
            .as_ref()
            .map(|h| {
                h.vars
                    .iter()
                    .map(|(name, path)| (name.clone(), std::path::PathBuf::from(path)))
                    .collect()
            })
            .unwrap_or_default(),
    };

    // Set after any completed turn: the next idle moment should draft a fresh
    // "your likely next message" ghost suggestion (see the loop head).
    let mut suggestion_due = false;
    if let Some(first) = initial_prompt {
        let first = first.trim();
        if !first.is_empty() {
            history.push(chat_template::ChatTurn::user(first));
            let spec = TurnSpec::resolve(PendingTurn::default(), !tools_json.is_empty());
            if let Err(err) = run_turn(
                &mut runner,
                &mut history,
                &mut turn_idx,
                &spec,
                system_prompt.as_deref(),
            ) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            suggestion_due = true;
        }
    }

    if !json {
        println!(
            "chat ready (type '/help' for commands; Tab accepts a suggested reply; Ctrl-D also exits)"
        );
    }
    // Files attached via `/load <file>`. Each load appends its contents here and
    // hands back a `$$fileN$$` marker (1-based, matching the slot index). The
    // marker is expanded to the file's text only when a query is submitted, so a
    // turn can weave several files into a single prompt.
    let mut loaded_files: Vec<String> = Vec::new();

    // Thought seeds from `/prethink`. `pending_prethink` is one-shot (consumed by
    // the next message); `persistent_prethink` (`/prethink persistent <text>`)
    // seeds every message until cleared and survives `/clear`. A one-shot seed,
    // if set, wins for that single turn.
    let mut pending_prethink: Option<String> = None;
    let mut persistent_prethink: Option<String> = harness.as_ref().and_then(|h| h.prethink.clone());

    // One-shot forced reply from `/response <text>`: the next message skips
    // generation entirely: the text becomes the model turn and the KV is
    // prefilled from it (via `AlignTo`), as if the model had produced it.
    let mut pending_response: Option<String> = None;

    // One-shot fully-formed thought from `/thought <text>`: written as a closed,
    // persistent (unstripped) thought channel. With `/response` both are forced;
    // alone, the model answers after the thought closes.
    let mut pending_thought: Option<String> = None;

    // Interactive line editing via rustyline (a pure-Rust readline): arrow-key
    // history, Ctrl-A/E/K/etc., and recall persisted across sessions. Only when
    // stdin is a real terminal and we're not emitting JSON — piped and --json
    // sessions keep the plain line reader so scripted input behaves as before.
    let history_path = std::env::var_os("HOME").map(|h| {
        let mut p = PathBuf::from(h);
        p.push(".diffgemma_history");
        p
    });
    // Shared ghost-suggestion slot: the async suggester writes it when a draft
    // lands; the editor's `Hinter` reads it on each refresh.
    let suggestion_cell: SuggestionCell = std::sync::Arc::new(std::sync::Mutex::new(None));
    let mut editor: Option<ChatEditor> = if io::stdin().is_terminal() && !json {
        match ChatEditor::new() {
            Ok(mut ed) => {
                ed.set_helper(Some(SuggestHelper {
                    suggestion: std::sync::Arc::clone(&suggestion_cell),
                }));
                // Tab accepts the ghost suggestion by reading the shared cell at
                // press-time — robust even when it arrived while the user sat idle
                // and isn't in rustyline's cached inline hint yet.
                ed.bind_sequence(
                    rustyline::KeyEvent::from('\t'),
                    rustyline::EventHandler::Conditional(Box::new(TabAcceptHandler {
                        suggestion: std::sync::Arc::clone(&suggestion_cell),
                    })),
                );
                if let Some(p) = &history_path {
                    let _ = ed.load_history(p);
                }
                Some(ed)
            }
            Err(err) => {
                if verbose {
                    eprintln!("chat: readline unavailable ({err}); using plain input");
                }
                None
            }
        }
    } else {
        None
    };
    // Cross-thread printer that reveals a ready suggestion above the prompt while
    // the user sits idle — rustyline only recomputes inline hints on a keypress,
    // so without this the ghost would wait for the next interaction.
    let printer: Option<ExternalPrinterHandle> = editor
        .as_mut()
        .and_then(|ed| ed.create_external_printer().ok())
        .map(|p| {
            std::sync::Arc::new(std::sync::Mutex::new(
                Box::new(p) as Box<dyn rustyline::ExternalPrinter + Send>
            ))
        });

    let stdin = io::stdin();
    // The in-flight background suggester, if any. Cancelled + joined on the next
    // submit (or at exit) so it never overlaps a real turn on the GPU queue.
    let mut task: Option<SuggestTask> = None;
    loop {
        // A turn just finished: kick off the async next-message draft. It runs on
        // the shared pipeline while the user reads/thinks/types; the next submit
        // cancels it (below). Interactive sessions get a ghost hint plus an idle
        // reveal; JSON drivers get a `Suggestion` event if it lands before they
        // submit. Only when there's somewhere for it to go.
        if suggestion_due {
            suggestion_due = false;
            let have_editor = editor.is_some();
            if have_editor || want_json {
                if let Ok(mut c) = suggestion_cell.lock() {
                    *c = None; // drop last turn's ghost before the new draft
                }
                let suggester = Suggester {
                    pipeline: std::sync::Arc::clone(&pipeline),
                    tokenizer: std::sync::Arc::clone(&tokenizer),
                    layers,
                    max_seq: CHAT_MAX_SEQ,
                    steps,
                    no_early_stop,
                    stop: suggest_stop.clone(),
                    seed: seed.wrapping_add(turn_idx).wrapping_add(0x5_1793),
                    cell: have_editor.then(|| std::sync::Arc::clone(&suggestion_cell)),
                    printer: printer.clone(),
                    sinks: std::sync::Arc::clone(&sinks),
                };
                task = Some(suggester.spawn(history.clone(), turn_idx));
            }
        }
        // Read the next line: rustyline prints and edits its own prompt when
        // available; otherwise a plain prompt + read_line for piped/--json input.
        let raw = match editor.as_mut() {
            Some(ed) => match ed.readline("you> ") {
                Ok(l) => l,
                // Ctrl-C discards the current line and reprompts (readline norm);
                // Ctrl-D on an empty line ends the session.
                Err(rustyline::error::ReadlineError::Interrupted) => continue,
                Err(rustyline::error::ReadlineError::Eof) => break,
                Err(err) => {
                    eprintln!("error: {err}");
                    return ExitCode::FAILURE;
                }
            },
            None => {
                if !json {
                    print!("you> ");
                    let _ = io::stdout().flush();
                }
                let mut l = String::new();
                match stdin.read_line(&mut l) {
                    Ok(0) => break,
                    Ok(_) => {}
                    Err(err) => {
                        eprintln!("error: {err}");
                        return ExitCode::FAILURE;
                    }
                }
                l
            }
        };
        // A line arrived (submit or command). Preempt any in-flight suggester now
        // so the pipeline is free and its probe tokens have settled before we run
        // the next turn. (Ctrl-C `continue`s above without reaching here, leaving
        // the draft running while the user is still at a fresh prompt.)
        if let Some(t) = task.take() {
            t.cancel_and_join();
        }
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        // Every non-empty entry joins the recall history, commands included.
        // `/private` below clears everything recorded up to that point.
        if let Some(ed) = editor.as_mut() {
            let _ = ed.add_history_entry(line);
        }
        // `--json` input mode: a stdin line may be a JSON command object (the dual
        // of the JSON event stream). A recognized non-`message` command becomes
        // its `/slash` line and flows through the handlers below; a `message`
        // submits directly (`is_message`, skipping command parsing so its text may
        // start with `/`); a non-JSON / non-object line falls back to a plain user
        // message. Text sessions are unaffected.
        let mut is_message = false;
        let cmd_line: String = if json {
            match serde_json::from_str::<serde_json::Value>(line) {
                Ok(v) if v.is_object() => match json_command_to_line(&v) {
                    Some(JsonInput::Command(s)) => s,
                    Some(JsonInput::Message(t)) => {
                        is_message = true;
                        t
                    }
                    Some(JsonInput::Exit) => break,
                    None => {
                        eprintln!("chat: ignoring unknown json command: {line}");
                        continue;
                    }
                },
                _ => {
                    is_message = true;
                    line.to_string()
                }
            }
        } else {
            line.to_string()
        };
        let line = cmd_line.as_str();
        // An empty JSON `message` has nothing to submit (the emptiness check above
        // ran on the JSON envelope, not the decoded text).
        if is_message && line.trim().is_empty() {
            continue;
        }
        // Command parsing is skipped for a JSON `message` (its text is user
        // content, never a command). `is_message` is only ever set in json mode.
        if !is_message {
            if line == "/exit" || line == "/quit" {
                break;
            }
            if line == "/help" {
                if !json {
                    println!("{SLASH_HELP}");
                }
                continue;
            }
            // `/prompt <text>`: set the system prompt (bare clears it). It prepends a
            // system turn on every build; setting it mid-conversation applies from
            // the next message, so it is most useful before the first message or
            // right after `/clear`.
            if line == "/prompt" || line.starts_with("/prompt ") {
                let text = line["/prompt".len()..].trim();
                if text.is_empty() {
                    system_prompt = None;
                    if !json {
                        println!("system prompt cleared");
                    }
                } else {
                    system_prompt = Some(text.to_string());
                    if !json {
                        let note = if history.is_empty() {
                            ""
                        } else {
                            " (applies from your next message; /clear to restart)"
                        };
                        println!("system prompt set{note}");
                    }
                }
                continue;
            }
            // `/clear`: reset the conversation (history, turn seed, one-shot
            // prethink) but keep the system prompt and any persistent prethink.
            if line == "/clear" {
                history.clear();
                turn_idx = 0;
                pending_prethink = None;
                if !json {
                    match &system_prompt {
                        Some(_) => println!("conversation cleared (system prompt kept)"),
                        None => println!("conversation cleared"),
                    }
                }
                continue;
            }
            // `/rewind`: undo the last exchange (the last model reply and the user
            // message that prompted it), so you can re-ask or continue from before.
            if line == "/rewind" {
                let before = history.len();
                let role = |h: &[chat_template::ChatTurn]| h.last().map(|t| t.role);
                if role(&history) == Some(chat_template::ChatRole::Model) {
                    history.pop();
                }
                if role(&history) == Some(chat_template::ChatRole::User) {
                    history.pop();
                }
                let removed = history.len() < before;
                if removed {
                    turn_idx = turn_idx.saturating_sub(1);
                }
                if !json {
                    if removed {
                        println!("rewound one step ({} turns left)", history.len());
                    } else {
                        println!("nothing to rewind");
                    }
                }
                continue;
            }
            // `/show-thinking [on|off]`: stream the model's reasoning live (grey
            // `think>`) on every turn; bare toggles. Off (default) hides it behind a
            // spinner. Applies from the next message.
            if line == "/show-thinking" || line.starts_with("/show-thinking ") {
                let arg = line["/show-thinking".len()..].trim();
                match toggle_flag(arg, runner.show_thinking) {
                    Some(v) => {
                        runner.show_thinking = v;
                        if !json {
                            println!("show-thinking {}", if v { "on" } else { "off" });
                        }
                    }
                    None => {
                        if !json {
                            println!("usage: /show-thinking [on|off]");
                        }
                    }
                }
                continue;
            }
            // `/show-output [on|off]`: stream the answer as it forms (default), or
            // withhold it until complete when off; bare toggles.
            if line == "/show-output" || line.starts_with("/show-output ") {
                let arg = line["/show-output".len()..].trim();
                match toggle_flag(arg, runner.stream_output) {
                    Some(v) => {
                        runner.stream_output = v;
                        if !json {
                            println!("show-output {}", if v { "on" } else { "off" });
                        }
                    }
                    None => {
                        if !json {
                            println!("usage: /show-output [on|off]");
                        }
                    }
                }
                continue;
            }
            // `/private`: blow away all recall history up to this point (in-memory
            // and on disk), then keep recording forward. Call it again to wipe again
            // — each invocation is a fresh privacy checkpoint.
            if line == "/private" {
                if let Some(ed) = editor.as_mut() {
                    let _ = ed.clear_history();
                    if let Some(p) = &history_path {
                        let _ = ed.save_history(p);
                    }
                }
                if !json {
                    println!("history cleared — recording continues from here");
                }
                continue;
            }
            // `/prethink [persistent] <text>`: one-shot seed (next message), or a
            // persistent seed (every message) with the `persistent` keyword. Bare
            // `/prethink` clears both; bare `/prethink persistent` clears only the
            // persistent seed.
            if line == "/prethink" || line.starts_with("/prethink ") {
                let arg = line["/prethink".len()..].trim();
                if arg == "persistent" || arg.starts_with("persistent ") {
                    let text = arg["persistent".len()..].trim();
                    if text.is_empty() {
                        persistent_prethink = None;
                        if !json {
                            println!("persistent prethink cleared");
                        }
                    } else {
                        persistent_prethink = Some(text.to_string());
                        if !json {
                            println!("persistent prethink set; applies to every message");
                        }
                    }
                } else if arg.is_empty() {
                    pending_prethink = None;
                    persistent_prethink = None;
                    if !json {
                        println!("prethink cleared");
                    }
                } else {
                    pending_prethink = Some(arg.to_string());
                    if !json {
                        println!(
                            "prethink seeded; send your message ($$last$$/$$prompt$$ expand at send)"
                        );
                    }
                }
                continue;
            }
            // `/response <text>`: force the next reply to <text> without generating.
            // The text becomes the model turn and the KV is prefilled from it. Bare
            // clears a pending forced reply.
            if line == "/response" || line.starts_with("/response ") {
                let text = line["/response".len()..].trim();
                if text.is_empty() {
                    pending_response = None;
                    if !json {
                        println!("forced response cleared");
                    }
                } else {
                    pending_response = Some(text.to_string());
                    if !json {
                        println!("next reply forced; send your message");
                    }
                }
                continue;
            }
            // `/thought <text>`: inject <text> as a fully-formed thought on the next
            // turn, persisted in the KV (unstripped). With /response both are forced;
            // alone, the model answers after it. Bare clears. ($$last$$/$$prompt$$)
            if line == "/thought" || line.starts_with("/thought ") {
                let text = line["/thought".len()..].trim();
                if text.is_empty() {
                    pending_thought = None;
                    if !json {
                        println!("thought cleared");
                    }
                } else {
                    pending_thought = Some(text.to_string());
                    if !json {
                        println!(
                            "thought set; send your message (or /response to force the reply)"
                        );
                    }
                }
                continue;
            }
            // `/load <file>`: read the file now, park its contents in a slot, and
            // echo the marker to weave into a later query. Errors are reported and
            // the turn is skipped — no partial slot is created.
            if line == "/load" || line.starts_with("/load ") {
                let path = line["/load".len()..].trim();
                if path.is_empty() {
                    eprintln!("usage: /load <filename>");
                    continue;
                }
                match std::fs::read_to_string(path) {
                    Ok(contents) => {
                        loaded_files.push(contents);
                        let marker = loaded_files.len();
                        let bytes = loaded_files[marker - 1].len();
                        if !json {
                            println!("loaded {path} ({bytes} bytes) as $$file{marker}$$");
                        }
                    }
                    Err(err) => eprintln!("error: cannot load {path}: {err}"),
                }
                continue;
            }
        } // end `if !is_message`

        let submitted = expand_file_markers(line, &loaded_files);
        history.push(chat_template::ChatTurn::user(&submitted));
        // One-shot state is consumed here; `TurnSpec::resolve` owns the
        // precedence between /response, /thought, /prethink and --tool.
        let spec = TurnSpec::resolve(
            PendingTurn {
                response: pending_response.take(),
                thought: pending_thought.take(),
                prethink: pending_prethink.take(),
                persistent_prethink: persistent_prethink.clone(),
            },
            !tools_json.is_empty(),
        );
        if let Err(err) = run_turn(
            &mut runner,
            &mut history,
            &mut turn_idx,
            &spec,
            system_prompt.as_deref(),
        ) {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
        suggestion_due = true;
    }

    // Stop any suggester still drafting as we leave.
    if let Some(t) = task.take() {
        t.cancel_and_join();
    }
    // Persist the recall history for the next session (best-effort).
    if let (Some(ed), Some(p)) = (editor.as_mut(), history_path.as_ref()) {
        let _ = ed.save_history(p);
    }

    ExitCode::SUCCESS
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::{JsonInput, expand_file_markers, json_command_to_line, tab_fill_text};

    /// The `/slash` line a JSON command translates to (or a marker for the
    /// non-command variants), so the mapping is asserted as plain strings.
    fn json_line(json: &str) -> String {
        let v: serde_json::Value = serde_json::from_str(json).unwrap();
        match json_command_to_line(&v) {
            Some(JsonInput::Command(s)) => s,
            Some(JsonInput::Message(t)) => format!("<message>{t}"),
            Some(JsonInput::Exit) => "<exit>".to_string(),
            None => "<none>".to_string(),
        }
    }

    #[test]
    fn json_commands_map_to_slash_lines() {
        // A message carries its text verbatim (submitted, never re-parsed).
        assert_eq!(
            json_line(r#"{"type":"message","text":"/not a command"}"#),
            "<message>/not a command"
        );
        // Multi-line content survives via JSON string escaping.
        assert_eq!(
            json_line(r#"{"type":"thought","text":"a\nb"}"#),
            "/thought a\nb"
        );
        // Bare/empty text -> bare command (clears), matching the REPL.
        assert_eq!(json_line(r#"{"type":"response","text":""}"#), "/response");
        assert_eq!(json_line(r#"{"type":"prompt"}"#), "/prompt");
        // Persistent prethink.
        assert_eq!(
            json_line(r#"{"type":"prethink","text":"reason","persistent":true}"#),
            "/prethink persistent reason"
        );
        assert_eq!(json_line(r#"{"type":"prethink"}"#), "/prethink");
        // Display toggles: on/off set, absent toggles.
        assert_eq!(
            json_line(r#"{"type":"show_thinking","on":true}"#),
            "/show-thinking on"
        );
        assert_eq!(
            json_line(r#"{"type":"show_output","on":false}"#),
            "/show-output off"
        );
        assert_eq!(json_line(r#"{"type":"show_thinking"}"#), "/show-thinking");
        // Control verbs and lifecycle.
        assert_eq!(json_line(r#"{"type":"clear"}"#), "/clear");
        assert_eq!(
            json_line(r#"{"type":"load","path":"/tmp/x"}"#),
            "/load /tmp/x"
        );
        assert_eq!(json_line(r#"{"type":"exit"}"#), "<exit>");
        // Unknown type is rejected (caller ignores it).
        assert_eq!(json_line(r#"{"type":"frobnicate"}"#), "<none>");
    }

    #[test]
    fn tab_fills_only_on_empty_line_with_a_suggestion() {
        // Empty line + a suggestion → fill it.
        assert_eq!(
            tab_fill_text(true, Some("What can you do?")),
            Some("What can you do?".to_string())
        );
        // User has typed → Tab declines (falls through to default).
        assert_eq!(tab_fill_text(false, Some("What can you do?")), None);
        // No suggestion ready → nothing to fill.
        assert_eq!(tab_fill_text(true, None), None);
        // Empty suggestion is not offered.
        assert_eq!(tab_fill_text(true, Some("")), None);
    }

    #[test]
    fn expands_loaded_markers_and_leaves_unloaded_verbatim() {
        let files = vec!["ALPHA".to_string(), "BETA".to_string()];
        assert_eq!(
            expand_file_markers("summarize $$file1$$ then $$file2$$", &files),
            "summarize ALPHA then BETA"
        );
        // Same marker used twice expands each time.
        assert_eq!(
            expand_file_markers("$$file1$$ $$file1$$", &files),
            "ALPHA ALPHA"
        );
        // An unloaded slot survives as plain text rather than disappearing.
        assert_eq!(
            expand_file_markers("see $$file9$$", &files),
            "see $$file9$$"
        );
        // No markers, no files: passthrough.
        assert_eq!(expand_file_markers("plain question", &[]), "plain question");
    }
}
