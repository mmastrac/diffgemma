//! Interactive `chat` subcommand.

use super::*;

/// One-line-per-command summary printed by `/help`.
#[cfg(target_os = "macos")]
const SLASH_HELP: &str = "\
slash commands:
  /help            show this help
  /prompt <text>   set the system prompt (bare clears it)
  /load <file>     attach a file; expands as $$fileN$$ in your next message
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
) -> ExitCode {
    use metal::StepGenerateConfig;
    use std::io::{self, IsTerminal, Write};

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
    let pipeline = std::sync::Arc::new(std::sync::Mutex::new(
        crate::pipeline::Pipeline::spawn(model_dir.to_path_buf(), CHAT_MAX_SEQ, steps),
    ));
    match pipeline.lock().unwrap().call(crate::pipeline::PipelineOp::Ping) {
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

    // (No warm-up: COLD-START-1 — the first fresh-session generation returning an
    // empty/EOS reply — is fixed at the root by the deterministic first-step SC
    // seed. The throwaway warm-up generation is no longer needed.)

    let mut history: Vec<chat_template::ChatTurn> = Vec::new();
    let mut turn_idx = 0u64;
    // System prompt set via `/prompt`; prepended as a leading System turn on
    // every build. Passed per-call, not captured, so `/prompt` can still mutate
    // it between turns.
    let mut system_prompt: Option<String> = None;

    let mut run_turn = |history: &mut Vec<chat_template::ChatTurn>,
                        turn_idx: &mut u64,
                        system: Option<&str>|
     -> Result<(), crate::Error> {
        let turns = prepend_system(system, history);
        let prompt = build_chat_prompt_tokens(model_dir, &turns, raw_prompt)?;
        let prompt_len = prompt.len();
        // KV arena is fixed at session open (CHAT_MAX_SEQ); the reply may use
        // the whole remaining context (no artificial cap), or the caller's
        // explicit ceiling if one was given. Reserve one CANVAS block: each
        // denoise block writes a CANVAS-wide canvas at [kv_len..kv_len+CANVAS],
        // so kv_len + CANVAS must stay within max_seq — this keeps the
        // `set_kv_len` overflow assert unreachable from a near-full context.
        let budget = CHAT_MAX_SEQ.saturating_sub(prompt_len + metal::CANVAS);
        if budget == 0 {
            if !json {
                println!(
                    "model> (this turn's prompt is {prompt_len} tokens, which leaves no room \
                     for a reply within the {CHAT_MAX_SEQ}-token context; cannot generate)\n\
                     \x20 to fix: restart chat with a larger window, e.g. `--ctx {}`; or send a \
                     shorter message, `/load` smaller files, or `/exit` and start a fresh session \
                     to drop the accumulated history.",
                    (prompt_len + metal::CANVAS).next_power_of_two().max(CHAT_MAX_SEQ * 2)
                );
            }
            history.push(chat_template::ChatTurn::model(String::new()));
            *turn_idx = turn_idx.wrapping_add(1);
            return Ok(());
        }
        step_cfg.max_new_tokens = explicit_cap.map_or(budget, |c| c.min(budget));
        step_cfg.seed = seed.wrapping_add(*turn_idx);
        let this_turn = *turn_idx;
        *turn_idx = turn_idx.wrapping_add(1);

        let started = std::time::Instant::now();
        let stream = chat_ui::ChatStream::start(
            std::sync::Arc::clone(&tokenizer),
            step_cfg.stop_token_ids.clone(),
            interactive,
            make_json_sink(want_json),
            this_turn,
            prompt_len,
        );
        step_cfg.step_observer = Some(stream.observer());
        let out = match pipeline.lock().unwrap().call(crate::pipeline::PipelineOp::Generate {
            prompt: prompt.clone(),
            cfg: Box::new(step_cfg.clone()),
            label: "chat".into(),
        }) {
            crate::pipeline::PipelineEvent::Generated { out, .. } => *out,
            crate::pipeline::PipelineEvent::Error(err) => {
                return Err(crate::Error::Pipeline(err));
            }
            ev => return Err(crate::Error::Pipeline(format!("unexpected event {ev:?}"))),
        };
        step_cfg.step_observer = None;
        let elapsed = started.elapsed();

        let new_ids =
            sample::strip_degenerate_token_ids(out.token_ids.get(prompt_len..).unwrap_or(&[]));
        let reply = chat_template::sanitize_model_reply(&tokenizer.decode(&new_ids));
        let new_tokens = out.token_ids.len().saturating_sub(prompt_len);
        let secs = elapsed.as_secs_f64();
        stream.finish(
            &reply,
            new_tokens,
            out.denoise_steps_run,
            secs,
            out.stopped_on_eot,
        );

        if !interactive && !json {
            // Non-interactive human output (piped / --verbose): print the reply
            // once, since the terminal renderer was disabled.
            if reply.is_empty() {
                println!("model> (empty response)");
            } else {
                println!("model> {reply}");
            }
        }
        if !json {
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
        history.push(chat_template::ChatTurn::model(reply));
        Ok(())
    };

    // Set after any completed turn: the next idle moment should draft a fresh
    // "your likely next message" ghost suggestion (see the loop head).
    let mut suggestion_due = false;
    if let Some(first) = initial_prompt {
        let first = first.trim();
        if !first.is_empty() {
            history.push(chat_template::ChatTurn::user(first));
            if let Err(err) = run_turn(&mut history, &mut turn_idx, system_prompt.as_deref()) {
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
                Box::new(p) as Box<dyn rustyline::ExternalPrinter + Send>,
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
                    json_sink: if want_json { make_json_sink(true) } else { None },
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
        // the next message, so it is most useful before the first message.
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
                        " (applies from your next message)"
                    };
                    println!("system prompt set{note}");
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

        let submitted = expand_file_markers(line, &loaded_files);
        history.push(chat_template::ChatTurn::user(&submitted));
        if let Err(err) = run_turn(&mut history, &mut turn_idx, system_prompt.as_deref()) {
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
    use super::{expand_file_markers, tab_fill_text};

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
        assert_eq!(expand_file_markers("$$file1$$ $$file1$$", &files), "ALPHA ALPHA");
        // An unloaded slot survives as plain text rather than disappearing.
        assert_eq!(expand_file_markers("see $$file9$$", &files), "see $$file9$$");
        // No markers, no files: passthrough.
        assert_eq!(expand_file_markers("plain question", &[]), "plain question");
    }
}
