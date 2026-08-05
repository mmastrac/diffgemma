//! Interactive `chat` subcommand.

use super::*;

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
    /// JSON driver sink: emits a `Suggestion` event when the draft lands.
    json_sink: Option<Box<dyn std::io::Write + Send>>,
}

#[cfg(target_os = "macos")]
impl Suggester {
    fn spawn(self, history: Vec<chat_template::ChatTurn>, turn: u64) -> SuggestTask {
        use std::io::Write as _;
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
            if let Some(mut w) = self.json_sink {
                let ev = crate::chat_protocol::ChatEvent::Suggestion { turn, text };
                if serde_json::to_writer(&mut w, &ev).is_ok() {
                    let _ = w.write_all(b"\n");
                    let _ = w.flush();
                }
            }
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
fn prepend_system(
    system: Option<&str>,
    history: &[chat_template::ChatTurn],
) -> Vec<chat_template::ChatTurn> {
    let mut turns = Vec::with_capacity(history.len() + 1);
    if let Some(s) = system.filter(|s| !s.trim().is_empty()) {
        turns.push(chat_template::ChatTurn::system(s));
    }
    turns.extend_from_slice(history);
    turns
}

/// Prefill the session KV to match the full conversation (no generation) via
/// `AlignTo`, used by `/response` to inject a forced reply as if the model had
/// produced it. Builds the target with no generation prompt so the next turn
/// reuses this prefix. Mirrors the render the next turn will use (tool-aware
/// when tools are active) so the KV prefix is reused rather than re-prefilled.
#[cfg(target_os = "macos")]
fn prefill_conversation(
    pipeline: &std::sync::Mutex<crate::pipeline::Pipeline>,
    tokenizer: &tokenizer::Tokenizer,
    history: &[chat_template::ChatTurn],
    system: Option<&str>,
    tools_json: &[serde_json::Value],
) -> Result<(), crate::Error> {
    let turns = prepend_system(system, history);
    let target = if tools_json.is_empty() {
        chat_template::format_chat_token_ids(
            tokenizer,
            &turns,
            &chat_template::ChatFormatOptions {
                add_generation_prompt: false,
                enable_thinking: true,
            },
        )?
    } else {
        let messages = chat_turns_to_messages(&turns);
        let guarded =
            crate::tools::render_conversation_guarded(&messages, tools_json, false, false);
        tokenizer.encode_prompt(&guarded).0
    };
    match pipeline
        .lock()
        .unwrap()
        .call(crate::pipeline::PipelineOp::AlignTo { target })
    {
        crate::pipeline::PipelineEvent::Aligned { .. } => Ok(()),
        crate::pipeline::PipelineEvent::Error(err) => Err(crate::Error::Pipeline(err)),
        ev => Err(crate::Error::Pipeline(format!("unexpected event {ev:?}"))),
    }
}

/// A `--tool NAME[:DESC]=COMMAND` definition: the model calls `NAME` with one
/// free-form string argument `input`, which is passed to `COMMAND` (run under
/// `sh -c`) as the `$input` environment variable; the script's stdout is fed
/// back as the tool result.
#[cfg(target_os = "macos")]
struct ShellTool {
    name: String,
    description: String,
    command: String,
}

/// Parse a `--tool` spec `NAME[:DESC]=COMMAND` (NAME and COMMAND required).
#[cfg(target_os = "macos")]
fn parse_tool_spec(spec: &str) -> Result<ShellTool, String> {
    let (meta, command) = spec
        .split_once('=')
        .ok_or_else(|| format!("--tool '{spec}': expected NAME[:DESC]=COMMAND"))?;
    let command = command.trim();
    if command.is_empty() {
        return Err(format!("--tool '{spec}': empty command"));
    }
    let (name, desc) = meta.split_once(':').unwrap_or((meta, ""));
    let (name, desc) = (name.trim(), desc.trim());
    if name.is_empty() {
        return Err(format!("--tool '{spec}': empty name"));
    }
    let description = if desc.is_empty() {
        format!("The {name} tool. Takes one string argument `input`.")
    } else {
        desc.to_string()
    };
    Ok(ShellTool {
        name: name.to_string(),
        description,
        command: command.to_string(),
    })
}

/// OpenAI-shape declaration for the tool renderer: one required string arg.
#[cfg(target_os = "macos")]
fn tool_declaration(t: &ShellTool) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": {
                "type": "object",
                "properties": { "input": {
                    "type": "string",
                    "description": "The input passed to the tool.",
                }},
                "required": ["input"],
            },
        },
    })
}

/// Run a tool's shell command, exposing each call argument as an env var (so
/// model-supplied values are never re-parsed as shell syntax). Returns stdout,
/// or a bracketed diagnostic on non-zero exit / spawn failure. Output is capped
/// so a runaway script can't blow the context window.
#[cfg(target_os = "macos")]
fn run_shell_tool(t: &ShellTool, arguments: &serde_json::Value) -> String {
    const MAX_OUTPUT: usize = 4096;
    let mut cmd = std::process::Command::new("sh");
    cmd.arg("-c").arg(&t.command);
    if let Some(obj) = arguments.as_object() {
        for (k, v) in obj {
            let val = match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            cmd.env(k, val);
        }
    }
    match cmd.output() {
        Ok(o) => {
            let mut s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if !o.status.success() {
                let code = o.status.code().map_or("?".to_string(), |c| c.to_string());
                let err = String::from_utf8_lossy(&o.stderr);
                s = format!("[exit {code}] {}{}", s, err.trim());
            }
            if s.len() > MAX_OUTPUT {
                s.truncate(MAX_OUTPUT);
                s.push_str("... [truncated]");
            }
            s
        }
        Err(e) => format!("[tool failed to run: {e}]"),
    }
}

/// `ChatTurn` history as OpenAI-shape JSON messages for the tool renderer.
#[cfg(target_os = "macos")]
fn chat_turns_to_messages(turns: &[chat_template::ChatTurn]) -> Vec<serde_json::Value> {
    turns
        .iter()
        .map(|t| {
            let role = match t.role {
                chat_template::ChatRole::System => "system",
                chat_template::ChatRole::User => "user",
                chat_template::ChatRole::Model => "assistant",
            };
            serde_json::json!({ "role": role, "content": t.content })
        })
        .collect()
}

/// First line of `s`, clipped to `n` chars with an ellipsis marker, for a
/// one-line preview of a tool argument or result.
#[cfg(target_os = "macos")]
fn preview(s: &str, n: usize) -> String {
    let line = s.lines().next().unwrap_or("");
    if line.chars().count() > n {
        format!("{}...", line.chars().take(n).collect::<String>())
    } else {
        line.to_string()
    }
}

/// One `PipelineOp::Generate`, unwrapped to the output or a pipeline error.
#[cfg(target_os = "macos")]
fn pipeline_generate(
    pipeline: &std::sync::Mutex<crate::pipeline::Pipeline>,
    step_cfg: &metal::StepGenerateConfig,
    prompt: Vec<u32>,
) -> Result<crate::generate::GenerateOutput, crate::Error> {
    match pipeline
        .lock()
        .unwrap()
        .call(crate::pipeline::PipelineOp::Generate {
            prompt,
            cfg: Box::new(step_cfg.clone()),
            label: "chat-tool".into(),
        }) {
        crate::pipeline::PipelineEvent::Generated { out, .. } => Ok(*out),
        crate::pipeline::PipelineEvent::Error(err) => Err(crate::Error::Pipeline(err)),
        ev => Err(crate::Error::Pipeline(format!("unexpected event {ev:?}"))),
    }
}

/// Per-session machinery a turn borrows: the GPU pipeline, tokenizer, the
/// mutable generation config, the tool set, and the flags that decide display.
/// Each [`TurnStrategy`] arm runs against this shared context.
#[cfg(target_os = "macos")]
struct TurnRunner<'a> {
    pipeline: &'a std::sync::Arc<std::sync::Mutex<crate::pipeline::Pipeline>>,
    tokenizer: &'a std::sync::Arc<tokenizer::Tokenizer>,
    step_cfg: &'a mut metal::StepGenerateConfig,
    model_dir: &'a std::path::Path,
    tools_json: &'a [serde_json::Value],
    shell_tools: &'a [ShellTool],
    events_file: &'a Option<std::fs::File>,
    raw_prompt: bool,
    interactive: bool,
    json: bool,
    want_json: bool,
    max_seq: usize,
    explicit_cap: Option<usize>,
    seed: u64,
    /// Live display toggles (`--show-thinking` / `--no-stream`, or the
    /// `/show-thinking` / `/show-output` slash commands). Mutated between turns.
    show_thinking: bool,
    stream_output: bool,
}

/// How a turn's thinking channel is seeded when the prompt is assembled: the
/// one lifecycle point `/prethink` modifies. `Off` is the default closed empty
/// thought (no reasoning). `Seed` carries a template expanded against the turn
/// (see [`expand_thinking`]); the model completes it, then answers. A modifier
/// orthogonal to execution: it applies to `Generate` and `Tools`, not `Forced`.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
enum Thinking<'a> {
    Off,
    /// Open seed: the model completes the reasoning, then answers (`/prethink`).
    Seed(&'a str),
    /// A closed, fully-formed thought: the model writes only the answer after
    /// it, and the thought persists in the KV, unstripped (`/thought`).
    Prefilled(&'a str),
}

/// What produces a turn's reply.
#[cfg(target_os = "macos")]
enum Execution<'a> {
    /// One `Generate` (streamed).
    Generate,
    /// The tool-calling loop. `reseed_each_round` re-seeds the thought every
    /// round (persistent prethink) rather than just round 0.
    Tools { reseed_each_round: bool },
    /// Force the reply with no generation (prefill only). `thought`, when set
    /// (`/thought` alongside `/response`), is written as a persistent thought
    /// before the reply.
    Forced {
        thought: Option<&'a str>,
        reply: &'a str,
    },
}

/// A resolved chat turn: an executor plus the thinking-channel modifier. The
/// dispatch lives in one place; a new behavior is a new `Execution` arm or a new
/// modifier, not a branch smeared through the loop.
#[cfg(target_os = "macos")]
struct Turn<'a> {
    execution: Execution<'a>,
    thinking: Thinking<'a>,
}

/// Expand a `/prethink` seed template against the turn: `$$last$$` -> the user's
/// most recent message, `$$prompt$$` -> the system prompt (both empty if
/// absent). Runs when the thinking channel is assembled, so the seed is
/// computed from live turn state rather than fixed at command time.
#[cfg(target_os = "macos")]
fn expand_thinking(
    template: &str,
    history: &[chat_template::ChatTurn],
    system: Option<&str>,
) -> String {
    let last = history
        .iter()
        .rev()
        .find(|t| t.role == chat_template::ChatRole::User)
        .map(|t| t.content.as_str())
        .unwrap_or("");
    template
        .replace("$$last$$", last)
        .replace("$$prompt$$", system.unwrap_or(""))
}

/// Split a seeded-thought generation at the model's `<channel|>`: returns the
/// completed thought (`seed` + what the model wrote before the close) and the
/// answer text after it. No close -> all thought, empty answer.
#[cfg(target_os = "macos")]
fn split_thought(seed: &str, raw: &str) -> (String, String) {
    let (completion, answer) = raw.split_once("<channel|>").unwrap_or((raw, ""));
    (format!("{seed}{completion}"), answer.to_string())
}

/// Recover an answer when a prethink thought never closed: the model sometimes
/// states its answer inside the reasoning (a thought degeneracy) and stops
/// without emitting `<channel|>`. Take the last blank-line-separated block (the
/// model usually sets the answer off after a blank line), or the whole
/// completion when there is no separator.
#[cfg(target_os = "macos")]
fn salvage_answer(completion: &str) -> String {
    let t = completion.trim();
    t.rsplit("\n\n").next().unwrap_or(t).trim().to_string()
}

#[cfg(target_os = "macos")]
impl Turn<'_> {
    /// Produce this turn's `(reply, persisted_thought)`. `history` holds the
    /// just-submitted user turn but not the reply; each arm handles its own
    /// display. The caller stores the reply (with the thought, if any, as an
    /// unstripped thought channel) and advances `turn_idx`.
    fn produce(
        &self,
        run: &mut TurnRunner,
        history: &[chat_template::ChatTurn],
        system: Option<&str>,
        turn_idx: u64,
    ) -> Result<(String, Option<String>), crate::Error> {
        match self.execution {
            Execution::Generate => run.generate_turn(history, system, self.thinking, turn_idx),
            Execution::Tools { reseed_each_round } => run
                .tool_turn(history, system, self.thinking, reseed_each_round, turn_idx)
                .map(|reply| (reply, None)),
            Execution::Forced { thought, reply } => {
                run.forced_turn(history, system, thought, reply)
            }
        }
    }
}

#[cfg(target_os = "macos")]
impl TurnRunner<'_> {
    /// Per-turn JSON sink: stdout for `--json`, a cloned events-file handle
    /// otherwise, `None` when neither is wanted.
    fn sink(&self) -> Option<Box<dyn std::io::Write + Send>> {
        if !self.want_json {
            return None;
        }
        if self.json {
            Some(Box::new(std::io::stdout()))
        } else {
            self.events_file
                .as_ref()
                .and_then(|f| f.try_clone().ok())
                .map(|f| Box::new(f) as Box<dyn std::io::Write + Send>)
        }
    }

    /// Normal / prethink turn: one `Generate` with the live streaming UI (the
    /// plain answer, or the dimmed thinking when seeded), then the answer split
    /// from the thought.
    fn generate_turn(
        &mut self,
        history: &[chat_template::ChatTurn],
        system: Option<&str>,
        thinking: Thinking,
        turn_idx: u64,
    ) -> Result<(String, Option<String>), crate::Error> {
        // Resolve the thinking seed now (`$$last$$`/`$$prompt$$` expand against
        // this turn). `close` = a fully-formed `/thought` (closed channel, model
        // answers after it, thought persisted); open = `/prethink`. Raw-prompt
        // sessions have no thought scaffold, so seeding is inert there.
        let (seed, close) = match thinking {
            Thinking::Seed(t) if !self.raw_prompt => {
                (Some(expand_thinking(t, history, system)), false)
            }
            Thinking::Prefilled(t) if !self.raw_prompt => {
                (Some(expand_thinking(t, history, system)), true)
            }
            _ => (None, false),
        };
        let turns = prepend_system(system, history);
        let prompt = match &seed {
            Some(seed) => build_chat_prompt_tokens_prethink(self.model_dir, &turns, seed, close)?,
            None => build_chat_prompt_tokens(self.model_dir, &turns, self.raw_prompt, true)?,
        };
        let prompt_len = prompt.len();
        // KV arena is fixed at session open; reserve one CANVAS block so the
        // `set_kv_len` overflow assert stays unreachable near a full context.
        let budget = self.max_seq.saturating_sub(prompt_len + metal::CANVAS);
        if budget == 0 {
            if !self.json {
                println!(
                    "model> (this turn's prompt is {prompt_len} tokens, which leaves no room \
                     for a reply within the {}-token context; cannot generate)\n\
                     \x20 to fix: restart chat with a larger window, e.g. `--ctx {}`; or send a \
                     shorter message, `/load` smaller files, or `/exit` and start a fresh session \
                     to drop the accumulated history.",
                    self.max_seq,
                    (prompt_len + metal::CANVAS)
                        .next_power_of_two()
                        .max(self.max_seq * 2)
                );
            }
            return Ok((String::new(), None));
        }
        self.step_cfg.max_new_tokens = self.explicit_cap.map_or(budget, |c| c.min(budget));
        self.step_cfg.seed = self.seed.wrapping_add(turn_idx);

        // A closed `/thought` streams only the answer (the thought is settled in
        // the prompt), so show the known thought up front; an open `/prethink`
        // streams as dimmed thinking and is split from the answer afterward.
        if close
            && !self.json
            && let Some(s) = &seed
        {
            println!("\x1b[90mthink> {}\x1b[0m", s.trim());
        }
        let started = std::time::Instant::now();
        // `/prethink` (open seed) streams the reasoning implicitly; a plain turn
        // does so only under `--show-thinking`. A closed `/thought` never enables
        // thinking display: its generation is answer-only with no `<channel|>` to
        // split on, and the thought is already printed above.
        let open_seed = seed
            .as_ref()
            .filter(|_| !close)
            .map(|s| s.trim().to_string());
        let stream = chat_ui::ChatStream::start(
            std::sync::Arc::clone(self.tokenizer),
            self.step_cfg.stop_token_ids.clone(),
            self.interactive,
            chat_ui::StreamDisplay {
                show_thinking: self.show_thinking && !close,
                prethink_seed: open_seed,
                stream_output: self.stream_output,
            },
            self.sink(),
            turn_idx,
            prompt_len,
        );
        self.step_cfg.step_observer = Some(stream.observer());
        let out = match self
            .pipeline
            .lock()
            .unwrap()
            .call(crate::pipeline::PipelineOp::Generate {
                prompt: prompt.clone(),
                cfg: Box::new(self.step_cfg.clone()),
                label: "chat".into(),
            }) {
            crate::pipeline::PipelineEvent::Generated { out, .. } => *out,
            crate::pipeline::PipelineEvent::Error(err) => return Err(crate::Error::Pipeline(err)),
            ev => return Err(crate::Error::Pipeline(format!("unexpected event {ev:?}"))),
        };
        self.step_cfg.step_observer = None;
        let elapsed = started.elapsed();

        let new_ids =
            sample::strip_degenerate_token_ids(out.token_ids.get(prompt_len..).unwrap_or(&[]));
        let raw = self.tokenizer.decode(&new_ids);
        // Open seed: the generation begins inside the thought, so split `raw` at
        // the model's `<channel|>` (before = thought, after = answer). Closed
        // `/thought` and plain turns: `raw` is the answer.
        let (reply, split_thought_display) = match &seed {
            Some(seed) if !close => {
                let (thought, answer) = split_thought(seed.trim(), &raw);
                // If the model never closed the thought, salvage the answer from
                // its tail rather than reporting none (see `salvage_answer`).
                let answer = if raw.contains("<channel|>") {
                    answer
                } else {
                    salvage_answer(&raw)
                };
                (chat_template::sanitize_model_reply(&answer), Some(thought))
            }
            _ => (chat_template::sanitize_model_reply(&raw), None),
        };
        let new_tokens = out.token_ids.len().saturating_sub(prompt_len);
        let secs = elapsed.as_secs_f64();
        stream.finish(
            &reply,
            new_tokens,
            out.denoise_steps_run,
            secs,
            out.stopped_on_eot,
        );

        // The interactive renderer (`stream.finish`) owns the on-screen thought
        // and answer. Only the non-interactive path (piped / `--verbose`) prints
        // here, since the terminal renderer was disabled.
        if !self.interactive && !self.json {
            if let Some(thought) = &split_thought_display {
                println!("\x1b[90mthink> {}\x1b[0m", thought.trim());
            }
            if reply.is_empty() {
                if split_thought_display.is_some() {
                    println!("model> (no answer; the model never closed the thought)");
                } else {
                    println!("model> (empty response)");
                }
            } else {
                println!("model> {reply}");
            }
        }
        if !self.json {
            let cap_note = if out.stopped_on_eot {
                ""
            } else {
                " · hit context limit"
            };
            println!(
                "  ({new_tokens} tok · {} steps · {secs:.1}s · {:.1} tok/s{cap_note})",
                out.denoise_steps_run,
                new_tokens as f64 / secs.max(1e-9),
            );
        }
        // A closed `/thought` persists its (expanded) thought with the reply.
        let persisted = if close { seed } else { None };
        Ok((reply, persisted))
    }

    /// Tool turn: run the render/generate/run-script loop, then the timing line.
    fn tool_turn(
        &mut self,
        history: &[chat_template::ChatTurn],
        system: Option<&str>,
        thinking: Thinking,
        reseed: bool,
        turn_idx: u64,
    ) -> Result<String, crate::Error> {
        let turns = prepend_system(system, history);
        let messages = chat_turns_to_messages(&turns);
        // Expand the thinking seed once for the turn (`$$last$$`/`$$prompt$$`).
        // The tool path seeds an open thought only; `/thought` (Prefilled) routes
        // to the plain generate path, never here.
        let seed = match thinking {
            Thinking::Seed(t) => Some(expand_thinking(t, history, system)),
            Thinking::Off | Thinking::Prefilled(_) => None,
        };
        let started = std::time::Instant::now();
        let answer = self.tool_loop(messages, seed.as_deref(), reseed, turn_idx)?;
        if !self.json {
            println!("  ({:.1}s)", started.elapsed().as_secs_f64());
        }
        Ok(answer)
    }

    /// The tool-calling loop: render with tool declarations, generate, run each
    /// backing script, feed the results back for another round. Tool
    /// calls/results live only inside this turn; only the final answer persists.
    /// `seed` (already expanded) seeds round 0, every round if `reseed`.
    fn tool_loop(
        &mut self,
        mut messages: Vec<serde_json::Value>,
        seed: Option<&str>,
        reseed: bool,
        turn_idx: u64,
    ) -> Result<String, crate::Error> {
        const MAX_ROUNDS: usize = 8;
        for round in 0..MAX_ROUNDS {
            let think_seed = if round == 0 || reseed {
                seed.map(str::trim)
            } else {
                None
            };
            let mut guarded = crate::tools::render_conversation_guarded(
                &messages,
                self.tools_json,
                true,
                think_seed.is_some(),
            );
            if let Some(s) = think_seed {
                guarded.push_str("<|channel>thought\n");
                guarded.push_str(s);
            }
            let prompt = self.tokenizer.encode_prompt(&guarded).0;
            let prompt_len = prompt.len();
            let budget = self.max_seq.saturating_sub(prompt_len + metal::CANVAS);
            if budget == 0 {
                if !self.json {
                    println!("model> (context full; cannot continue the tool conversation)");
                }
                return Ok(String::new());
            }
            self.step_cfg.max_new_tokens = self.explicit_cap.map_or(budget, |c| c.min(budget));
            self.step_cfg.seed = self.seed.wrapping_add(turn_idx).wrapping_add(round as u64);

            // Stream the thinking live when seeded; otherwise generate quietly.
            let out = if let Some(s) = think_seed {
                let stream = chat_ui::ChatStream::start(
                    std::sync::Arc::clone(self.tokenizer),
                    self.step_cfg.stop_token_ids.clone(),
                    self.interactive,
                    chat_ui::StreamDisplay {
                        show_thinking: false,
                        prethink_seed: Some(s.to_string()),
                        stream_output: self.stream_output,
                    },
                    None,
                    turn_idx,
                    prompt_len,
                );
                self.step_cfg.step_observer = Some(stream.observer());
                let out = pipeline_generate(self.pipeline, self.step_cfg, prompt)?;
                self.step_cfg.step_observer = None;
                stream.finish("", 0, out.denoise_steps_run, 0.0, out.stopped_on_eot);
                out
            } else {
                self.step_cfg.step_observer = None;
                pipeline_generate(self.pipeline, self.step_cfg, prompt)?
            };

            let new_ids =
                sample::strip_degenerate_token_ids(out.token_ids.get(prompt_len..).unwrap_or(&[]));
            let raw = self.tokenizer.decode(&new_ids);
            // With a seeded thought the generation opens inside it: split at the
            // model's `<channel|>`. Before is the completed thought (reprinted
            // dimmed), after is the answer / tool calls.
            let content = if let Some(s) = think_seed {
                let (thought, rest) = split_thought(s, &raw);
                if !self.json {
                    println!("\x1b[90mthink> {}\x1b[0m", thought.trim());
                }
                rest
            } else {
                raw.clone()
            };

            let calls = crate::tools::parse_tool_calls(&content);
            if calls.is_empty() {
                let answer = chat_template::sanitize_model_reply(&content);
                if !self.json {
                    if answer.is_empty() {
                        println!("model> (empty response)");
                    } else {
                        println!("model> {answer}");
                    }
                }
                return Ok(answer);
            }
            // The model asked to call tools: show any preamble, run each backing
            // script, and append the call + results for another round.
            let preamble = chat_template::sanitize_model_reply(
                &crate::tools::content_before_tool_calls(&content),
            );
            if !self.json && !preamble.trim().is_empty() {
                println!("model> {}", preamble.trim());
            }
            let oai = crate::tools::to_openai_tool_calls(&calls);
            messages.push(serde_json::json!({
                "role": "assistant", "content": "", "tool_calls": oai,
            }));
            for (i, call) in calls.iter().enumerate() {
                let input = match call.arguments.get("input") {
                    Some(serde_json::Value::String(s)) => s.clone(),
                    Some(v) => v.to_string(),
                    None => String::new(),
                };
                let output = match self.shell_tools.iter().find(|t| t.name == call.name) {
                    Some(t) => run_shell_tool(t, &call.arguments),
                    None => format!("error: no tool named '{}'", call.name),
                };
                if !self.json {
                    println!(
                        "tool> {}({}) -> {}",
                        call.name,
                        preview(&input, 48),
                        preview(&output, 72)
                    );
                }
                let mut msg =
                    serde_json::json!({ "role": "tool", "name": call.name, "content": output });
                if let Some(id) = oai.get(i).and_then(|c| c.get("id")).cloned() {
                    msg["tool_call_id"] = id;
                }
                messages.push(msg);
            }
        }
        if !self.json {
            println!("model> (stopped after {MAX_ROUNDS} tool rounds without a final answer)");
        }
        Ok(String::new())
    }

    /// Forced reply (`/response`, optionally with a `/thought`): no generation.
    /// The given text (preceded by the thought, if any) becomes the model turn
    /// and the KV is prefilled from the full conversation, as if the model had
    /// produced it. Returns `(reply, persisted_thought)`.
    fn forced_turn(
        &self,
        history: &[chat_template::ChatTurn],
        system: Option<&str>,
        thought: Option<&str>,
        reply: &str,
    ) -> Result<(String, Option<String>), crate::Error> {
        let thought = thought.map(|t| expand_thinking(t, history, system));
        if !self.json {
            if let Some(t) = &thought {
                println!("\x1b[90mthink> {}\x1b[0m", t.trim());
            }
            println!("model> {reply}");
        }
        // Prefill the KV to the conversation including this forced turn (thought
        // rendered as a real channel). Raw-prompt sessions have no template to
        // prefill against; history alone carries it (lazy prefill next turn).
        if !self.raw_prompt {
            let mut turns = history.to_vec();
            turns.push(match &thought {
                Some(t) => chat_template::ChatTurn::model_with_thought(t.clone(), reply),
                None => chat_template::ChatTurn::model(reply),
            });
            if let Err(err) = prefill_conversation(
                self.pipeline,
                self.tokenizer,
                &turns,
                system,
                self.tools_json,
            ) {
                eprintln!("error: prefill failed: {err}");
            }
        }
        Ok((reply.to_string(), thought))
    }
}

/// Run one chat turn: `turn` produces the reply, which is stored as the model
/// turn; `turn_idx` advances. `history` already holds the user turn.
#[cfg(target_os = "macos")]
fn run_turn(
    runner: &mut TurnRunner,
    history: &mut Vec<chat_template::ChatTurn>,
    turn_idx: &mut u64,
    turn: &Turn,
    system: Option<&str>,
) -> Result<(), crate::Error> {
    let (reply, thought) = turn.produce(runner, history, system, *turn_idx)?;
    history.push(match thought {
        Some(t) => chat_template::ChatTurn::model_with_thought(t, reply),
        None => chat_template::ChatTurn::model(reply),
    });
    *turn_idx = turn_idx.wrapping_add(1);
    Ok(())
}

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
    show_thinking: bool,
    stream_output: bool,
) -> ExitCode {
    use metal::StepGenerateConfig;
    use std::io::{self, IsTerminal, Write};

    // `DGQ_RENDER_DEMO`: exercise the terminal renderer with synthetic events and
    // no model (terminal-choreography verification, e.g. under tmux).
    if let Ok(stage) = std::env::var("DGQ_RENDER_DEMO") {
        chat_ui::render_demo(&stage);
        return ExitCode::SUCCESS;
    }

    // Parse `--tool NAME[:DESC]=COMMAND` specs up front so a bad spec fails
    // before the model loads. Non-empty => every turn uses the tool path.
    let shell_tools: Vec<ShellTool> = match tools.iter().map(|s| parse_tool_spec(s)).collect() {
        Ok(t) => t,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };
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
    let events_file = match &events_path {
        Some(p) => match std::fs::File::create(p) {
            Ok(f) => Some(f),
            Err(err) => {
                eprintln!("error: cannot open --events {}: {err}", p.display());
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };
    // Per-turn JSON sink factory: stdout for --json, a cloned handle to the
    // events file otherwise. `None` when neither is requested.
    let make_json_sink = |turn_active: bool| -> Option<Box<dyn Write + Send>> {
        if !turn_active {
            return None;
        }
        if json {
            Some(Box::new(io::stdout()))
        } else {
            events_file
                .as_ref()
                .and_then(|f| f.try_clone().ok())
                .map(|f| Box::new(f) as Box<dyn Write + Send>)
        }
    };
    let want_json = json || events_file.is_some();

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

    // Tool turns generate the `call:{...}` grammar: don't let a stop token inside
    // an unclosed call end the turn, and mark the `<|"|>` quote so arg strings
    // survive. (Harmless when unused; only tool turns emit calls.)
    if !tools_json.is_empty() {
        step_cfg.continue_incomplete_tool_calls = true;
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
    let mut system_prompt: Option<String> = None;

    let mut runner = TurnRunner {
        pipeline: &pipeline,
        tokenizer: &tokenizer,
        step_cfg: &mut step_cfg,
        model_dir,
        tools_json: &tools_json,
        shell_tools: &shell_tools,
        events_file: &events_file,
        raw_prompt,
        interactive,
        json,
        want_json,
        max_seq: CHAT_MAX_SEQ,
        explicit_cap,
        seed,
        show_thinking,
        stream_output,
    };

    // Set after any completed turn: the next idle moment should draft a fresh
    // "your likely next message" ghost suggestion (see the loop head).
    let mut suggestion_due = false;
    if let Some(first) = initial_prompt {
        let first = first.trim();
        if !first.is_empty() {
            history.push(chat_template::ChatTurn::user(first));
            let turn = Turn {
                execution: if tools_json.is_empty() {
                    Execution::Generate
                } else {
                    Execution::Tools {
                        reseed_each_round: false,
                    }
                },
                thinking: Thinking::Off,
            };
            if let Err(err) = run_turn(
                &mut runner,
                &mut history,
                &mut turn_idx,
                &turn,
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
    let mut persistent_prethink: Option<String> = None;

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
                    json_sink: if want_json {
                        make_json_sink(true)
                    } else {
                        None
                    },
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
        // Resolve the turn from the pending /thought, /response, /prethink and
        // --tool state. `/response` forces the reply (writing the /thought too, if
        // set); `/thought` alone prefills the closed thought and the model answers
        // after it; else a prethink seed, then tools/normal. Owned strings bound
        // here so the borrowed `Turn` outlives the call.
        let thought = pending_thought.take();
        let response = pending_response.take();
        let (seed_owned, reseed) = match pending_prethink.take() {
            Some(s) => (Some(s), false),
            None => (persistent_prethink.clone(), true),
        };
        let prethink = seed_owned.as_deref().map_or(Thinking::Off, Thinking::Seed);
        let turn = if let Some(r) = &response {
            Turn {
                execution: Execution::Forced {
                    thought: thought.as_deref(),
                    reply: r,
                },
                thinking: Thinking::Off,
            }
        } else if let Some(t) = &thought {
            Turn {
                execution: Execution::Generate,
                thinking: Thinking::Prefilled(t),
            }
        } else if !tools_json.is_empty() {
            Turn {
                execution: Execution::Tools {
                    reseed_each_round: reseed,
                },
                thinking: prethink,
            }
        } else {
            Turn {
                execution: Execution::Generate,
                thinking: prethink,
            }
        };
        if let Err(err) = run_turn(
            &mut runner,
            &mut history,
            &mut turn_idx,
            &turn,
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
    use super::{
        JsonInput, expand_file_markers, expand_thinking, json_command_to_line, parse_tool_spec,
        run_shell_tool, salvage_answer, split_thought, tab_fill_text, tool_declaration,
    };
    use crate::chat_template::ChatTurn;

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
    fn salvage_answer_takes_the_trailing_block() {
        // The model reasoned, then stated the answer after a blank line.
        assert_eq!(
            salvage_answer("The project 'Bluebird' ships in March.\n\nMarch."),
            "March."
        );
        // No separator: the whole completion is the recovered answer.
        assert_eq!(
            salvage_answer("just reasoning, ending in 56"),
            "just reasoning, ending in 56"
        );
        // Empty / whitespace: nothing to salvage.
        assert_eq!(salvage_answer("  \n\n  "), "");
    }

    #[test]
    fn expand_thinking_substitutes_last_and_prompt() {
        let history = [
            ChatTurn::user("first question"),
            ChatTurn::model("an answer"),
            ChatTurn::user("what about New York?"),
        ];
        let out = expand_thinking(
            "The user asked: $$last$$. Persona: $$prompt$$.",
            &history,
            Some("be terse"),
        );
        assert_eq!(
            out,
            "The user asked: what about New York?. Persona: be terse."
        );
        // Missing system / no user turn -> empty substitutions, no panic.
        assert_eq!(expand_thinking("[$$prompt$$]", &history, None), "[]");
        assert_eq!(expand_thinking("$$last$$", &[], None), "");
        // A template with no placeholders passes through.
        assert_eq!(expand_thinking("plain seed", &history, None), "plain seed");
    }

    #[test]
    fn split_thought_splits_at_channel_close() {
        // seed + completion before the close; answer after.
        assert_eq!(
            split_thought("Let me think.", " Step 1.<channel|>The answer is 42."),
            (
                "Let me think. Step 1.".to_string(),
                "The answer is 42.".to_string()
            )
        );
        // Never closed: all thought, empty answer.
        assert_eq!(
            split_thought("seed", " unfinished"),
            ("seed unfinished".to_string(), String::new())
        );
    }

    #[test]
    fn tool_spec_parses_name_desc_and_command() {
        let t = parse_tool_spec("weather:Get the weather=curl -s wttr.in/$input").unwrap();
        assert_eq!(t.name, "weather");
        assert_eq!(t.description, "Get the weather");
        assert_eq!(t.command, "curl -s wttr.in/$input");
        // No description: a generic one is synthesized.
        let d = parse_tool_spec("echo=cat").unwrap();
        assert_eq!(d.name, "echo");
        assert!(d.description.contains("echo"));
        // Malformed specs are rejected.
        assert!(parse_tool_spec("noequals").is_err());
        assert!(parse_tool_spec("=cmd").is_err());
        assert!(parse_tool_spec("name=").is_err());
    }

    #[test]
    fn shell_tool_runs_with_input_env_var() {
        let t = parse_tool_spec("shout=printf '%s' \"$input\" | tr a-z A-Z").unwrap();
        let out = run_shell_tool(&t, &serde_json::json!({ "input": "hello" }));
        assert_eq!(out, "HELLO");
        // A model-supplied value with shell metacharacters is data, not syntax.
        let t = parse_tool_spec("id=printf '%s' \"$input\"").unwrap();
        let out = run_shell_tool(&t, &serde_json::json!({ "input": "; echo pwned" }));
        assert_eq!(out, "; echo pwned");
    }

    #[test]
    fn tool_declaration_has_one_required_string_input() {
        let t = parse_tool_spec("weather:desc=cmd").unwrap();
        let d = tool_declaration(&t);
        assert_eq!(d["function"]["name"], "weather");
        assert_eq!(d["function"]["parameters"]["required"][0], "input");
        assert_eq!(
            d["function"]["parameters"]["properties"]["input"]["type"],
            "string"
        );
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
