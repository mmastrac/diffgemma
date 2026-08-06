//! The turn engine: everything that produces one chat turn's event stream.
//!
//! A [`TurnSpec`] pairs a [`ReplySource`] (generate / tool loop / forced
//! reply) with a [`ThoughtPlan`] (the model's own thinking, a `/prethink`
//! seed, or an injected `/thought`); [`TurnSpec::resolve`] owns the
//! precedence between the REPL's pending commands. [`TurnRunner`] holds the
//! per-session machinery (the GPU pipeline, tokenizer, generation config,
//! tool set) and runs the turn, emitting every protocol event through the
//! session sinks. The REPL in `commands::chat` only parses input and owns
//! conversation state.

use super::{
    ChatEvent, ChatStream, SharedSinks, StreamDisplay,
    sink::{preview, print_think},
};
use crate::decoder::ChannelIds;
use crate::{chat_template, metal, sample, tokenizer};

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
        // enable_thinking must match the tool loop's render (always on), or
        // this prefix would never be reused by the next turn's prompt.
        let messages = chat_turns_to_messages(&turns);
        let guarded = crate::tools::render_conversation_guarded(&messages, tools_json, false, true);
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

/// One declared tool parameter (always a string on the wire).
pub(crate) struct ToolParam {
    pub name: String,
    pub description: String,
    pub required: bool,
}

/// A shell-backed tool definition: the model calls `name` with the declared
/// string `params`, which reach `command` (run under `sh -c`) as environment
/// variables; the script's stdout is fed back as the tool result.
pub(crate) struct ShellTool {
    name: String,
    description: String,
    command: String,
    params: Vec<ToolParam>,
}

impl ShellTool {
    pub fn new(name: String, description: String, command: String, params: Vec<ToolParam>) -> Self {
        Self {
            name,
            description,
            command,
            params,
        }
    }
}

/// Parse a `--tool` spec `NAME[:DESC]=COMMAND` (NAME and COMMAND required):
/// one free-form required string argument `input`.
pub(crate) fn parse_tool_spec(spec: &str) -> Result<ShellTool, String> {
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
    Ok(ShellTool::new(
        name.to_string(),
        description,
        command.to_string(),
        vec![ToolParam {
            name: "input".to_string(),
            description: "The input passed to the tool.".to_string(),
            required: true,
        }],
    ))
}

/// OpenAI-shape declaration for the tool renderer, one string property per
/// declared param (none is valid: a no-argument tool).
pub(crate) fn tool_declaration(t: &ShellTool) -> serde_json::Value {
    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();
    for p in &t.params {
        properties.insert(
            p.name.clone(),
            serde_json::json!({ "type": "string", "description": p.description }),
        );
        if p.required {
            required.push(serde_json::Value::String(p.name.clone()));
        }
    }
    serde_json::json!({
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    })
}

/// Run a tool's shell command. Each call argument exports as an env var of
/// its own name, and every session variable as `HARNESS_<name>` — model-
/// supplied values are never re-parsed as shell syntax, and the prefix keeps
/// state vars from colliding with params, so a tool can read and carry its
/// own state (`::set counter=$((HARNESS_counter + 1))`-style). Returns
/// stdout, or a bracketed diagnostic on non-zero exit / spawn failure.
/// Output is capped so a runaway script can't blow the context window.
fn run_shell_tool(
    t: &ShellTool,
    arguments: &serde_json::Value,
    vars: &std::collections::HashMap<String, String>,
) -> String {
    const MAX_OUTPUT: usize = 4096;
    let mut cmd = std::process::Command::new("sh");
    cmd.arg("-c").arg(&t.command);
    for (k, v) in vars {
        cmd.env(format!("HARNESS_{k}"), v);
    }
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

/// One-line rendering of a call's arguments for events and `tool>` lines: the
/// bare value for the classic single `input` arg, `k=v` pairs otherwise.
fn call_input_summary(arguments: &serde_json::Value) -> String {
    let text = |v: &serde_json::Value| match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    match arguments.as_object() {
        Some(obj) if obj.len() == 1 && obj.contains_key("input") => text(&obj["input"]),
        Some(obj) => obj
            .iter()
            .map(|(k, v)| format!("{k}={}", text(v)))
            .collect::<Vec<_>>()
            .join(", "),
        None => String::new(),
    }
}

/// `ChatTurn` history as OpenAI-shape JSON messages for the tool renderer.
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

/// A terminal `Done` for a turn that produced nothing (context full, tool
/// rounds exhausted): the protocol's invariant is that every `TurnStart` is
/// closed by exactly one `Done`.
fn done_empty() -> ChatEvent {
    ChatEvent::Done {
        tokens: 0,
        steps: 0,
        secs: 0.0,
        stopped: false,
        text: String::new(),
        thought: None,
    }
}

/// The dim per-turn stats line: tokens, steps, wall time, rate.
fn stats_line(tokens: usize, steps: usize, secs: f64, stopped: bool) -> String {
    let cap_note = if stopped { "" } else { " · hit context limit" };
    format!(
        "({tokens} tok · {steps} steps · {secs:.1}s · {:.1} tok/s{cap_note})",
        tokens as f64 / secs.max(1e-9),
    )
}

/// One `PipelineOp::Generate`, unwrapped to the output or a pipeline error.
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
pub(crate) struct TurnRunner<'a> {
    pub pipeline: &'a std::sync::Arc<std::sync::Mutex<crate::pipeline::Pipeline>>,
    pub tokenizer: &'a std::sync::Arc<tokenizer::Tokenizer>,
    pub step_cfg: &'a mut metal::StepGenerateConfig,
    pub tools_json: &'a [serde_json::Value],
    pub shell_tools: &'a [ShellTool],
    /// The session's protocol + plain-display consumers. Every turn event
    /// flows through here; the interactive renderer is the one extra view.
    pub sinks: SharedSinks,
    pub raw_prompt: bool,
    pub interactive: bool,
    pub json: bool,
    pub max_seq: usize,
    pub explicit_cap: Option<usize>,
    pub seed: u64,
    /// Live display toggles (`--show-thinking` / `--no-stream`, or the
    /// `/show-thinking` / `/show-output` slash commands). Mutated between turns.
    pub show_thinking: bool,
    pub stream_output: bool,
    /// Session variables set by a tool's `::set NAME=VALUE` output, referenced in
    /// a `/prethink` seed as `$$NAME$$`. Persist across turns.
    pub vars: std::collections::HashMap<String, String>,
    /// Harness-declared file-backed variables: `name` reads from its file at
    /// every use (empty until something writes it), so a tool can buffer
    /// multi-line state in the file where a `::set` value is one line. A
    /// file-backed name shadows a `::set` of the same name.
    pub file_vars: std::collections::HashMap<String, std::path::PathBuf>,
    /// Tool rounds per turn before the loop gives up.
    pub max_rounds: usize,
}

/// Tool rounds per turn unless the harness raises it.
pub(crate) const DEFAULT_MAX_ROUNDS: usize = 8;

/// What a turn's reply comes from.
#[derive(Debug, PartialEq)]
pub(crate) enum ReplySource {
    /// One streamed `Generate`.
    Generate,
    /// The tool-calling loop.
    Tools,
    /// A forced reply (`/response`): no generation, the text becomes the model
    /// turn and the KV is prefilled to match.
    Forced(String),
}

/// What the thought channel holds this turn. Templates expand against live
/// turn state (see [`expand_thinking`]) when the prompt is assembled.
#[derive(Debug, PartialEq)]
pub(crate) enum ThoughtPlan {
    /// The model's own thinking, per the session defaults.
    Model,
    /// `/prethink`: an open seed the model completes before answering. A
    /// `persistent` seed re-applies on every tool round, a one-shot on round 0
    /// only (a plain generate has one round, so the flag is inert there).
    Seeded { template: String, persistent: bool },
    /// `/thought`: a closed, fully-formed thought the reply follows; persisted
    /// in the KV, unstripped. Composes with `Generate` (the model answers
    /// after it) and `Forced` (both injected); [`TurnSpec::resolve`] never
    /// pairs it with `Tools`.
    Injected(String),
}

/// A resolved chat turn: the reply source plus the thought-channel plan. The
/// dispatch lives in one place; a new behavior is a new arm, not a branch
/// smeared through the loop.
#[derive(Debug, PartialEq)]
pub(crate) struct TurnSpec {
    pub source: ReplySource,
    pub thought: ThoughtPlan,
}

/// The REPL's pending command state at message submit; the REPL moves its
/// one-shot slots in and [`TurnSpec::resolve`] consumes them.
#[derive(Default)]
pub(crate) struct PendingTurn {
    /// `/response`: force the next reply.
    pub response: Option<String>,
    /// `/thought`: inject a closed thought.
    pub thought: Option<String>,
    /// `/prethink`: one-shot seed for the next message.
    pub prethink: Option<String>,
    /// `/prethink persistent`: seed for every message.
    pub persistent_prethink: Option<String>,
}

impl TurnSpec {
    /// Resolve pending REPL state into the turn to run. Precedence:
    /// `/response` forces the reply, carrying a pending `/thought` with it;
    /// `/thought` alone injects the closed thought before a plain generate
    /// (bypassing tools — an injected thought pins what the reply follows,
    /// and a tool round would generate past it); otherwise a prethink seed
    /// (one-shot over persistent) modifies the tool loop or a plain generate.
    pub fn resolve(pending: PendingTurn, has_tools: bool) -> TurnSpec {
        let PendingTurn {
            response,
            thought,
            prethink,
            persistent_prethink,
        } = pending;
        if let Some(reply) = response {
            return TurnSpec {
                source: ReplySource::Forced(reply),
                thought: thought.map_or(ThoughtPlan::Model, ThoughtPlan::Injected),
            };
        }
        if let Some(t) = thought {
            return TurnSpec {
                source: ReplySource::Generate,
                thought: ThoughtPlan::Injected(t),
            };
        }
        let thought = match (prethink, persistent_prethink) {
            (Some(template), _) => ThoughtPlan::Seeded {
                template,
                persistent: false,
            },
            (None, Some(template)) => ThoughtPlan::Seeded {
                template,
                persistent: true,
            },
            (None, None) => ThoughtPlan::Model,
        };
        TurnSpec {
            source: if has_tools {
                ReplySource::Tools
            } else {
                ReplySource::Generate
            },
            thought,
        }
    }
}

/// What a turn produced: the reply, and the thought that persists with it in
/// history (a closed `/thought` turn keeps its thought in the KV).
pub(crate) struct TurnOutcome {
    pub reply: String,
    pub thought: Option<String>,
}

/// Expand a `/prethink` seed template against the turn: `$$last$$` -> the user's
/// most recent message, `$$prompt$$` -> the system prompt, `$$NAME$$` -> a
/// session variable (all empty if absent). Runs when the thinking channel is
/// assembled, so the seed is computed from live turn state rather than fixed
/// at command time.
fn expand_thinking(
    template: &str,
    history: &[chat_template::ChatTurn],
    system: Option<&str>,
    vars: &std::collections::HashMap<String, String>,
) -> String {
    let last = history
        .iter()
        .rev()
        .find(|t| t.role == chat_template::ChatRole::User)
        .map(|t| t.content.as_str())
        .unwrap_or("");
    let mut out = template
        .replace("$$last$$", last)
        .replace("$$prompt$$", system.unwrap_or(""));
    for (name, value) in vars {
        out = out.replace(&format!("$${name}$$"), value);
    }
    out
}

/// How many valid tool calls to run before the first corruption: the valid
/// attempts preceding the first invalid one (`first_bad`), or all valid attempts
/// when the sequence is clean. Runs the good prefix rather than aborting every
/// call, which makes the model re-plan from scratch.
fn valid_prefix_len(attempts: &[(String, bool)], first_bad: Option<usize>) -> usize {
    let upto = first_bad.unwrap_or(attempts.len());
    attempts[..upto].iter().filter(|(_, valid)| *valid).count()
}

/// A control directive a tool emits through its output, out of band from the
/// text the model sees. One per trimmed line: `::end [reply]` ends the turn —
/// with the rest of the line as the final reply, or, when bare, the tool's
/// remaining output (the multi-line form: compose the reply, then `::end`) —
/// and `::set NAME=VALUE` sets a session variable.
enum ToolDirective {
    End(String),
    Set(String, String),
}

/// Split a tool's output into the model-facing text (directive lines removed) and
/// the directives it carried.
fn parse_tool_directives(output: &str) -> (String, Vec<ToolDirective>) {
    let mut content: Vec<&str> = Vec::new();
    let mut directives = Vec::new();
    for line in output.lines() {
        let t = line.trim();
        if t == "::end" {
            directives.push(ToolDirective::End(String::new()));
        } else if let Some(reply) = t.strip_prefix("::end ") {
            directives.push(ToolDirective::End(reply.trim().to_string()));
        } else if let Some(rest) = t.strip_prefix("::set ")
            && let Some((name, value)) = rest.split_once('=')
        {
            directives.push(ToolDirective::Set(
                name.trim().to_string(),
                value.trim().to_string(),
            ));
        } else {
            content.push(line);
        }
    }
    (content.join("\n"), directives)
}

impl TurnSpec {
    /// Produce this turn's outcome. `history` holds the just-submitted user
    /// turn but not the reply; each arm handles its own display. The caller
    /// stores the reply (with the thought, if any, as an unstripped thought
    /// channel) and advances `turn_idx`.
    fn produce(
        &self,
        run: &mut TurnRunner,
        history: &[chat_template::ChatTurn],
        system: Option<&str>,
        turn_idx: u64,
    ) -> Result<TurnOutcome, crate::Error> {
        match &self.source {
            ReplySource::Generate => run.generate_turn(history, system, &self.thought, turn_idx),
            ReplySource::Tools => {
                let (seed, reseed) = match &self.thought {
                    ThoughtPlan::Seeded {
                        template,
                        persistent,
                    } => (Some(template.as_str()), *persistent),
                    _ => (None, false),
                };
                run.tool_turn(history, system, seed, reseed, turn_idx)
                    .map(|reply| TurnOutcome {
                        reply,
                        thought: None,
                    })
            }
            ReplySource::Forced(reply) => {
                let thought = match &self.thought {
                    ThoughtPlan::Injected(t) => Some(t.as_str()),
                    _ => None,
                };
                run.forced_turn(history, system, thought, reply)
            }
        }
    }
}

impl TurnRunner<'_> {
    /// Emit one protocol event through the session sinks.
    fn emit(&self, ev: &ChatEvent) {
        super::emit(&self.sinks, ev);
    }

    /// The session variables as seen right now: `::set` values overlaid by
    /// file-backed variables read fresh (mid-turn writes are visible to the
    /// next expansion or tool run; an unreadable file reads empty).
    fn live_vars(&self) -> std::collections::HashMap<String, String> {
        let mut vars = self.vars.clone();
        for (name, path) in &self.file_vars {
            vars.insert(
                name.clone(),
                std::fs::read_to_string(path).unwrap_or_default(),
            );
        }
        vars
    }

    /// Normal / prethink turn: one `Generate` with the live streaming UI (the
    /// plain answer, or the dimmed thinking when seeded), then the answer split
    /// from the thought.
    fn generate_turn(
        &mut self,
        history: &[chat_template::ChatTurn],
        system: Option<&str>,
        thought: &ThoughtPlan,
        turn_idx: u64,
    ) -> Result<TurnOutcome, crate::Error> {
        // Expand the thought template now (`$$last$$`/`$$prompt$$` bind to this
        // turn). `close` = an injected `/thought` (closed channel, model
        // answers after it, thought persisted); open = a `/prethink` seed.
        // Raw-prompt sessions have no thought scaffold, so both are inert there.
        let (seed, close) = match thought {
            ThoughtPlan::Seeded { template, .. } if !self.raw_prompt => (
                Some(expand_thinking(
                    template,
                    history,
                    system,
                    &self.live_vars(),
                )),
                false,
            ),
            ThoughtPlan::Injected(template) if !self.raw_prompt => (
                Some(expand_thinking(
                    template,
                    history,
                    system,
                    &self.live_vars(),
                )),
                true,
            ),
            _ => (None, false),
        };
        let turns = prepend_system(system, history);
        let prompt = match &seed {
            Some(seed) => {
                chat_template::format_chat_token_ids_prethink(self.tokenizer, &turns, seed, close)?
            }
            None if self.raw_prompt => {
                let text = turns.last().map(|t| t.content.as_str()).unwrap_or("");
                self.tokenizer.encode(text, false)
            }
            None => chat_template::format_chat_token_ids(
                self.tokenizer,
                &turns,
                &chat_template::ChatFormatOptions {
                    add_generation_prompt: true,
                    enable_thinking: true,
                },
            )?,
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
            self.emit(&done_empty());
            return Ok(TurnOutcome {
                reply: String::new(),
                thought: None,
            });
        }
        self.step_cfg.max_new_tokens = self.explicit_cap.map_or(budget, |c| c.min(budget));
        self.step_cfg.seed = self.seed.wrapping_add(turn_idx);

        // A closed `/thought` streams only the answer (the thought is settled in
        // the prompt), so show the known thought up front; an open `/prethink`
        // streams as dimmed thinking and is split from the answer afterward.
        // (Interactive only: the renderless display prints it settled, from
        // `Done.thought`.)
        if close
            && self.interactive
            && let Some(s) = &seed
        {
            print_think(s.trim());
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
        let stream = ChatStream::start(
            std::sync::Arc::clone(self.tokenizer),
            self.step_cfg.stop_token_ids.clone(),
            self.interactive,
            StreamDisplay {
                show_thinking: self.show_thinking && !close,
                prethink_seed: open_seed,
                start_in_thought: false,
                stream_output: self.stream_output,
            },
            std::sync::Arc::clone(&self.sinks),
            0,
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
        // Open seed: the generation begins inside the thought, so split it at
        // the model's `<channel|>` (before = thought, after = answer) with the
        // token-level walk. Closed `/thought` and plain turns: the whole
        // generation is the answer.
        let (reply, split_thought_display) =
            match &seed {
                Some(seed) if !close => {
                    let (thought, answer) = ChannelIds::from_tokenizer(self.tokenizer)
                        .settle_prethink(self.tokenizer, seed.trim(), &new_ids);
                    (chat_template::sanitize_model_reply(&answer), Some(thought))
                }
                _ => (
                    chat_template::sanitize_model_reply(&self.tokenizer.decode(&new_ids)),
                    None,
                ),
            };
        let new_tokens = out.token_ids.len().saturating_sub(prompt_len);
        let secs = elapsed.as_secs_f64();
        let stats = stats_line(new_tokens, out.denoise_steps_run, secs, out.stopped_on_eot);
        stream.finish(Some(&reply), Some(&stats));

        // A closed `/thought` persists its (expanded) thought with the reply;
        // the surfaced `Done.thought` is that or the split streamed reasoning.
        let persisted = if close { seed } else { None };
        self.emit(&ChatEvent::Done {
            tokens: new_tokens,
            steps: out.denoise_steps_run,
            secs,
            stopped: out.stopped_on_eot,
            text: reply.clone(),
            thought: persisted
                .clone()
                .or_else(|| split_thought_display.map(|t| t.trim().to_string())),
        });
        Ok(TurnOutcome {
            reply,
            thought: persisted,
        })
    }

    /// Tool turn: run the render/generate/run-script loop. `seed` is the raw
    /// `/prethink` template; it expands once against this turn.
    fn tool_turn(
        &mut self,
        history: &[chat_template::ChatTurn],
        system: Option<&str>,
        seed: Option<&str>,
        reseed: bool,
        turn_idx: u64,
    ) -> Result<String, crate::Error> {
        let turns = prepend_system(system, history);
        let messages = chat_turns_to_messages(&turns);
        let base =
            crate::tools::render_conversation_guarded(&messages, self.tools_json, true, true);
        let seed = seed.map(|t| expand_thinking(t, history, system, &self.live_vars()));
        self.tool_loop(base, seed.as_deref(), reseed, turn_idx)
    }

    /// The tool-calling loop: generate, run each backing script, feed the
    /// results back for another round. Round 0 prompts with the rendered
    /// conversation; every later round EXTENDS the model's own output
    /// verbatim — its reasoning and calls stay in context (a re-render
    /// dropped them and the model re-planned from scratch every round), and
    /// each round is a pure KV extension instead of a rewind at the stripped
    /// thought. Tool calls/results still live only inside this turn: the next
    /// user turn renders thought-free from history as always. `seed` (already
    /// expanded) seeds round 0, every round if `reseed`.
    fn tool_loop(
        &mut self,
        base: String,
        seed: Option<&str>,
        reseed: bool,
        turn_idx: u64,
    ) -> Result<String, crate::Error> {
        let max_rounds = self.max_rounds;
        let started = std::time::Instant::now();
        // Turn totals for the terminal `Done` (per-round numbers ride the
        // `RoundStart`/`RoundEnd` events).
        let (mut total_tokens, mut total_steps) = (0usize, 0usize);
        let channels = ChannelIds::from_tokenizer(self.tokenizer);
        // Force the thought OPEN in the prompt: left to open the channel
        // itself, the model often narrates its plan into the visible answer
        // instead. Beginning in-thought pins the reasoning to the thought
        // block; the `/prethink` seed, when set, continues there.
        let mut prompt = {
            let mut text = base;
            text.push_str(crate::tools::OPEN_THOUGHT);
            if let Some(s) = seed {
                text.push_str(s.trim());
            }
            self.tokenizer.encode_prompt(&text).0
        };
        for round in 0..max_rounds {
            let think_seed = if round == 0 || reseed {
                seed.map(str::trim)
            } else {
                None
            };
            // The thought channel is always open — the model plans its calls,
            // exactly as a plain turn always reasons. Displaying the thought
            // stays behind `--show-thinking` / a `/prethink` seed.
            let display_thinking = self.show_thinking || think_seed.is_some();
            let prompt_len = prompt.len();
            let budget = self.max_seq.saturating_sub(prompt_len + metal::CANVAS);
            if budget == 0 {
                if !self.json {
                    println!("model> (context full; cannot continue the tool conversation)");
                }
                self.emit(&done_empty());
                return Ok(String::new());
            }
            self.step_cfg.max_new_tokens = self.explicit_cap.map_or(budget, |c| c.min(budget));
            self.step_cfg.seed = self.seed.wrapping_add(turn_idx).wrapping_add(round as u64);

            // Every round streams into the sinks; an interactive tty always
            // gets the live view (spinner, streamed output, thought when
            // surfaced) — never dead air while a round generates.
            let stream = ChatStream::start(
                std::sync::Arc::clone(self.tokenizer),
                self.step_cfg.stop_token_ids.clone(),
                self.interactive,
                StreamDisplay {
                    show_thinking: self.show_thinking,
                    prethink_seed: think_seed.map(str::to_string),
                    start_in_thought: true,
                    stream_output: self.stream_output,
                },
                std::sync::Arc::clone(&self.sinks),
                round,
                prompt_len,
            );
            self.step_cfg.step_observer = Some(stream.observer());
            let out = pipeline_generate(self.pipeline, self.step_cfg, prompt.clone())?;
            self.step_cfg.step_observer = None;
            let new_tokens = out.token_ids.len().saturating_sub(prompt_len);
            total_tokens += new_tokens;
            total_steps += out.denoise_steps_run;

            let new_ids =
                sample::strip_degenerate_token_ids(out.token_ids.get(prompt_len..).unwrap_or(&[]));
            let (settled_thought, content) =
                channels.settle_tool_reply(self.tokenizer, think_seed.unwrap_or(""), &new_ids);
            // `None` = the model never closed its thought (the round began
            // inside one). The thought rides events (and the display) only
            // when surfaced — hidden reasoning stays hidden, as on a plain
            // turn.
            let closed_thought = settled_thought.is_some();
            let thought = settled_thought
                .as_deref()
                .map(str::trim)
                .filter(|t| display_thinking && !t.is_empty())
                .map(str::to_string);

            // No `call:` attempt at all: this is the final answer. A round
            // whose thought never closed stated it inside the reasoning —
            // salvage the answer from the tail rather than replying with the
            // whole monologue.
            let attempts = crate::tools::scan_call_attempts(&content);
            if attempts.is_empty() {
                let answer = if closed_thought {
                    chat_template::sanitize_model_reply(&content)
                } else {
                    chat_template::sanitize_model_reply(&crate::decoder::salvage_answer(&content))
                };
                let secs = started.elapsed().as_secs_f64();
                let stats = stats_line(total_tokens, total_steps, secs, out.stopped_on_eot);
                stream.finish(Some(&answer), Some(&stats));
                self.emit(&ChatEvent::Done {
                    tokens: total_tokens,
                    steps: total_steps,
                    secs,
                    stopped: out.stopped_on_eot,
                    text: answer.clone(),
                    thought,
                });
                return Ok(answer);
            }
            // Partial recovery: run the valid calls up to the first corrupted one,
            // then (if anything is corrupt) feed an error for it and let the model
            // reissue from there. Aborting every call makes it re-plan from scratch
            // and misattribute the failure. `corrupt` also covers a dangling
            // `<|tool_call>` opener that `scan_call_attempts` does not list.
            let calls = crate::tools::parse_tool_calls(&content);
            let first_bad = attempts.iter().position(|(_, valid)| !valid);
            let corrupt = crate::tools::has_incomplete_tool_call(&content);
            let run_count = valid_prefix_len(&attempts, first_bad);
            let to_run = &calls[..run_count.min(calls.len())];

            let preamble = chat_template::sanitize_model_reply(
                &crate::tools::content_before_tool_calls(&content),
            );
            let preamble_line = (!preamble.trim().is_empty()).then(|| preamble.trim());
            stream.finish(preamble_line, None);
            // The turn continues into tool execution; `Done` comes later.
            self.emit(&ChatEvent::RoundEnd {
                round,
                text: preamble.trim().to_string(),
                thought: thought.clone(),
            });
            let mut ended: Option<String> = None;
            let mut responses: Vec<(String, String)> = Vec::new();
            for call in to_run.iter() {
                let input = call_input_summary(&call.arguments);
                self.emit(&ChatEvent::ToolCall {
                    round,
                    name: call.name.clone(),
                    input: input.clone(),
                });
                let raw_output = match self.shell_tools.iter().find(|t| t.name == call.name) {
                    Some(t) => run_shell_tool(t, &call.arguments, &self.live_vars()),
                    None => format!("error: no tool named '{}'", call.name),
                };
                // A tool steers the session out of band via `::` directive lines;
                // the rest is the result the model sees.
                let (output, directives) = parse_tool_directives(&raw_output);
                for d in directives {
                    match d {
                        ToolDirective::Set(name, value) => {
                            if self.interactive {
                                println!("  · set {name} = {}", preview(&value, 60));
                            }
                            self.emit(&ChatEvent::VarSet {
                                name: name.clone(),
                                value: value.clone(),
                            });
                            self.vars.insert(name, value);
                        }
                        ToolDirective::End(reply) => {
                            // Bare `::end`: the tool's remaining output IS the
                            // reply (the multi-line form).
                            ended = Some(if reply.is_empty() {
                                output.clone()
                            } else {
                                reply
                            });
                        }
                    }
                }
                self.emit(&ChatEvent::ToolResult {
                    round,
                    name: call.name.clone(),
                    input: input.clone(),
                    output: output.clone(),
                });
                if self.interactive {
                    let head = format!("tool> {}({})", call.name, preview(&input, 48));
                    match output.trim_end() {
                        "" => println!("{head} -> (no output)"),
                        body if body.contains('\n') => {
                            // Multi-line output: header, then the body indented.
                            println!("{head} ->");
                            for line in body.lines() {
                                println!("  {line}");
                            }
                        }
                        body => println!("{head} -> {body}"),
                    }
                }
                responses.push((call.name.clone(), output));
            }
            // A tool ended the turn: use its reply as the final answer instead of
            // generating another round.
            if let Some(reply) = ended {
                if self.interactive && !reply.is_empty() {
                    println!("model> {reply}");
                }
                self.emit(&ChatEvent::Done {
                    tokens: total_tokens,
                    steps: total_steps,
                    secs: started.elapsed().as_secs_f64(),
                    stopped: true,
                    text: reply.clone(),
                    thought: None,
                });
                return Ok(reply);
            }
            // Continuation: the model's own tokens, the tool responses in call
            // order, and a reopened thought for the next round. A corrupted
            // call stays visible in the model's output with an error response
            // answering it, so it can see what it did wrong and reissue.
            let mut tail = String::new();
            for (name, output) in &responses {
                tail.push_str(&crate::tools::render_tool_response_guarded(
                    name,
                    &serde_json::json!({ "content": output }),
                ));
            }
            if corrupt {
                let name = first_bad
                    .map(|k| attempts[k].0.clone())
                    .filter(|n| !n.is_empty())
                    .unwrap_or_else(|| "tool".to_string());
                if self.interactive {
                    println!("tool> {name}(malformed) -> asking the model to reissue");
                }
                self.emit(&ChatEvent::ToolRetry {
                    round,
                    name: name.clone(),
                });
                tail.push_str(&crate::tools::render_tool_response_guarded(
                    &name,
                    &serde_json::json!({ "content": "error: this tool call was malformed and was \
                        not run; the calls before it ran. Reissue this call and any that should \
                        follow it, corrected." }),
                ));
            }
            tail.push_str(crate::tools::OPEN_THOUGHT);
            if reseed && let Some(s) = seed {
                tail.push_str(s.trim());
            }
            prompt = out.token_ids;
            prompt.extend(self.tokenizer.encode_prompt(&tail).0);
        }
        if self.interactive {
            println!("model> (stopped after {max_rounds} tool rounds without a final answer)");
        }
        self.emit(&done_empty());
        Ok(String::new())
    }

    /// Forced reply (`/response`, optionally with a `/thought`): no generation.
    /// The given text (preceded by the thought, if any) becomes the model turn
    /// and the KV is prefilled from the full conversation, as if the model had
    /// produced it.
    fn forced_turn(
        &self,
        history: &[chat_template::ChatTurn],
        system: Option<&str>,
        thought: Option<&str>,
        reply: &str,
    ) -> Result<TurnOutcome, crate::Error> {
        let thought = thought.map(|t| expand_thinking(t, history, system, &self.live_vars()));
        // No generation ran, so no renderer exists: echo directly on an
        // interactive tty; plain sessions display from the events below.
        if self.interactive {
            if let Some(t) = &thought {
                print_think(t.trim());
            }
            println!("model> {reply}");
        }
        if let Some(t) = &thought {
            self.emit(&ChatEvent::Thought {
                text: t.trim().to_string(),
            });
        }
        self.emit(&ChatEvent::Text {
            committed: reply.len(),
            text: reply.to_string(),
        });
        self.emit(&ChatEvent::Done {
            tokens: 0,
            steps: 0,
            secs: 0.0,
            stopped: true,
            text: reply.to_string(),
            thought: thought.clone(),
        });
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
        Ok(TurnOutcome {
            reply: reply.to_string(),
            thought,
        })
    }
}

/// Run one chat turn: `spec` produces the reply, which is stored as the model
/// turn; `turn_idx` advances. `history` already holds the user turn.
pub(crate) fn run_turn(
    runner: &mut TurnRunner,
    history: &mut Vec<chat_template::ChatTurn>,
    turn_idx: &mut u64,
    spec: &TurnSpec,
    system: Option<&str>,
) -> Result<(), crate::Error> {
    runner.emit(&ChatEvent::TurnStart { turn: *turn_idx });
    let out = spec.produce(runner, history, system, *turn_idx)?;
    history.push(match out.thought {
        Some(t) => chat_template::ChatTurn::model_with_thought(t, out.reply),
        None => chat_template::ChatTurn::model(out.reply),
    });
    *turn_idx = turn_idx.wrapping_add(1);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        PendingTurn, ReplySource, ThoughtPlan, ToolDirective, TurnSpec, expand_thinking,
        parse_tool_directives, parse_tool_spec, run_shell_tool, tool_declaration, valid_prefix_len,
    };
    use crate::chat_template::ChatTurn;

    fn pending(
        response: Option<&str>,
        thought: Option<&str>,
        prethink: Option<&str>,
        persistent: Option<&str>,
    ) -> PendingTurn {
        let own = |v: Option<&str>| v.map(str::to_string);
        PendingTurn {
            response: own(response),
            thought: own(thought),
            prethink: own(prethink),
            persistent_prethink: own(persistent),
        }
    }

    #[test]
    fn resolve_response_wins_and_carries_the_thought() {
        // `/response` + `/thought`: both forced, prethink ignored.
        let spec = TurnSpec::resolve(
            pending(Some("canned"), Some("staged"), Some("seed"), None),
            true,
        );
        assert_eq!(spec.source, ReplySource::Forced("canned".into()));
        assert_eq!(spec.thought, ThoughtPlan::Injected("staged".into()));
        // `/response` alone: no thought.
        let spec = TurnSpec::resolve(pending(Some("canned"), None, None, None), false);
        assert_eq!(spec.thought, ThoughtPlan::Model);
    }

    #[test]
    fn resolve_injected_thought_bypasses_tools() {
        let spec = TurnSpec::resolve(pending(None, Some("staged"), None, None), true);
        assert_eq!(spec.source, ReplySource::Generate);
        assert_eq!(spec.thought, ThoughtPlan::Injected("staged".into()));
    }

    #[test]
    fn resolve_one_shot_prethink_beats_persistent() {
        let spec = TurnSpec::resolve(pending(None, None, Some("once"), Some("always")), false);
        assert_eq!(
            spec.thought,
            ThoughtPlan::Seeded {
                template: "once".into(),
                persistent: false,
            }
        );
        let spec = TurnSpec::resolve(pending(None, None, None, Some("always")), true);
        assert_eq!(spec.source, ReplySource::Tools);
        assert_eq!(
            spec.thought,
            ThoughtPlan::Seeded {
                template: "always".into(),
                persistent: true,
            }
        );
    }

    #[test]
    fn resolve_defaults_to_the_session_shape() {
        let spec = TurnSpec::resolve(PendingTurn::default(), false);
        assert_eq!(spec.source, ReplySource::Generate);
        assert_eq!(spec.thought, ThoughtPlan::Model);
        let spec = TurnSpec::resolve(PendingTurn::default(), true);
        assert_eq!(spec.source, ReplySource::Tools);
    }

    #[test]
    fn expand_thinking_substitutes_last_and_prompt() {
        let history = [
            ChatTurn::user("first question"),
            ChatTurn::model("an answer"),
            ChatTurn::user("what about New York?"),
        ];
        let mut vars = std::collections::HashMap::new();
        vars.insert("mood".to_string(), "tense".to_string());
        let out = expand_thinking(
            "The user asked: $$last$$. Persona: $$prompt$$. Mood: $$mood$$.",
            &history,
            Some("be terse"),
            &vars,
        );
        assert_eq!(
            out,
            "The user asked: what about New York?. Persona: be terse. Mood: tense."
        );
        // Missing system / no user turn -> empty substitutions, no panic.
        let empty = std::collections::HashMap::new();
        assert_eq!(
            expand_thinking("[$$prompt$$]", &history, None, &empty),
            "[]"
        );
        assert_eq!(expand_thinking("$$last$$", &[], None, &empty), "");
        // A template with no placeholders passes through.
        assert_eq!(
            expand_thinking("plain seed", &history, None, &empty),
            "plain seed"
        );
    }

    #[test]
    fn valid_prefix_runs_up_to_first_corruption() {
        let ok = |n: &str| (n.to_string(), true);
        let bad = |n: &str| (n.to_string(), false);
        // Clean: run every valid call.
        let clean = [ok("a"), ok("b"), ok("c")];
        assert_eq!(valid_prefix_len(&clean, None), 3);
        // Corruption at index 2: the two valid calls before it run.
        let mixed = [ok("a"), ok("b"), bad("c"), ok("d")];
        assert_eq!(valid_prefix_len(&mixed, Some(2)), 2);
        // First call corrupt: nothing runs.
        let lead = [bad("a"), ok("b")];
        assert_eq!(valid_prefix_len(&lead, Some(0)), 0);
    }

    #[test]
    fn parse_tool_directives_splits_control_from_content() {
        // A pure directive: no model-facing content.
        let (content, dirs) = parse_tool_directives("::end Ready.");
        assert_eq!(content, "");
        assert!(matches!(&dirs[..], [ToolDirective::End(r)] if r == "Ready."));
        // Mixed: the scene text survives, the set is consumed.
        let (content, dirs) = parse_tool_directives("A dim bistro.\n::set scene=A dim bistro.");
        assert_eq!(content, "A dim bistro.");
        assert!(
            matches!(&dirs[..], [ToolDirective::Set(n, v)] if n == "scene" && v == "A dim bistro.")
        );
        // Bare end, and a non-directive `::` line stays content.
        let (content, dirs) = parse_tool_directives("::end\n::not a directive");
        assert_eq!(content, "::not a directive");
        assert!(matches!(&dirs[..], [ToolDirective::End(r)] if r.is_empty()));
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
        let vars = std::collections::HashMap::new();
        let out = run_shell_tool(&t, &serde_json::json!({ "input": "hello" }), &vars);
        assert_eq!(out, "HELLO");
        // A model-supplied value with shell metacharacters is data, not syntax.
        let t = parse_tool_spec("id=printf '%s' \"$input\"").unwrap();
        let out = run_shell_tool(&t, &serde_json::json!({ "input": "; echo pwned" }), &vars);
        assert_eq!(out, "; echo pwned");
    }

    #[test]
    fn shell_tool_sees_state_vars_under_harness_prefix() {
        let t =
            parse_tool_spec("bump=printf '::set n=%s\\nwas %s' \"$input\" \"$HARNESS_n\"").unwrap();
        let vars = std::collections::HashMap::from([("n".to_string(), "41".to_string())]);
        let out = run_shell_tool(&t, &serde_json::json!({ "input": "42" }), &vars);
        // The state var arrives prefixed; the directive line is control, the
        // rest is the model-facing result.
        let (content, directives) = super::parse_tool_directives(&out);
        assert_eq!(content, "was 41");
        assert!(
            matches!(&directives[..], [super::ToolDirective::Set(n, v)] if n == "n" && v == "42")
        );
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
}
