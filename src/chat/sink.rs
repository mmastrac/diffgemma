//! Event sinks: the consumers of the [`ChatEvent`] stream.
//!
//! The stream is the single source of truth for a turn; a sink only observes
//! it. [`JsonlSink`] serializes every event (the `--json` / `--events`
//! protocol surface); [`PlainSink`] is the human display for sessions without
//! a live terminal renderer (piped stdin, `--verbose`) — settled `think>` /
//! `model>` / `tool>` lines, never streaming. The interactive renderer
//! (`chat::render`) consumes the same stream through its own machinery
//! (viewport + ticker), so no display path computes semantics of its own.

use super::ChatEvent;
use std::io::Write;
use std::sync::{Arc, Mutex};

/// A consumer of the chat event stream.
pub trait EventSink: Send {
    fn emit(&mut self, ev: &ChatEvent);
}

/// The session's sink fan-out. Behind `Arc<Mutex>` so the generation-thread
/// observer, the background suggester, and the REPL thread all feed the same
/// consumers.
#[derive(Default)]
pub struct SinkSet {
    sinks: Vec<Box<dyn EventSink>>,
}

/// Shared handle to the session's sinks.
pub type SharedSinks = Arc<Mutex<SinkSet>>;

impl SinkSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, sink: Box<dyn EventSink>) {
        self.sinks.push(sink);
    }

    pub fn emit(&mut self, ev: &ChatEvent) {
        for s in &mut self.sinks {
            s.emit(ev);
        }
    }

    /// Wrap into the shared handle every producer holds.
    pub fn into_shared(self) -> SharedSinks {
        Arc::new(Mutex::new(self))
    }
}

/// Emit one event through a shared sink set (convenience for producers).
pub fn emit(sinks: &SharedSinks, ev: &ChatEvent) {
    if let Ok(mut s) = sinks.lock() {
        s.emit(ev);
    }
}

/// One JSON object per line per event, flushed immediately so a driver (or a
/// tail on the `--events` file) sees events as they happen.
pub struct JsonlSink<W: Write + Send>(pub W);

impl<W: Write + Send> EventSink for JsonlSink<W> {
    fn emit(&mut self, ev: &ChatEvent) {
        if serde_json::to_writer(&mut self.0, ev).is_ok() {
            let _ = self.0.write_all(b"\n");
            let _ = self.0.flush();
        }
    }
}

/// Human display for renderless sessions (piped stdin, `--verbose`): prints
/// only settled output — the streaming events pass through untouched.
pub struct PlainSink;

const GREY: &str = "\x1b[90m";
const RESET: &str = "\x1b[0m";

/// First line of `s`, clipped to `n` chars with an ellipsis marker, for a
/// one-line preview of a tool argument or value.
pub(crate) fn preview(s: &str, n: usize) -> String {
    let line = s.lines().next().unwrap_or("");
    if line.chars().count() > n {
        format!("{}...", line.chars().take(n).collect::<String>())
    } else {
        line.to_string()
    }
}

impl PlainSink {
    fn settled(&self, text: &str, thought: Option<&str>) {
        if let Some(t) = thought.map(str::trim).filter(|t| !t.is_empty()) {
            println!("{GREY}think> {t}{RESET}");
        }
        if text.is_empty() {
            if thought.is_some() {
                println!("model> (no answer; the model never closed the thought)");
            } else {
                println!("model> (empty response)");
            }
        } else {
            println!("model> {text}");
        }
    }
}

impl EventSink for PlainSink {
    fn emit(&mut self, ev: &ChatEvent) {
        match ev {
            // A generation settled but the turn continues (tool calls follow):
            // show what the model said so far. An empty preamble prints
            // nothing — the tool lines that follow carry the round.
            ChatEvent::RoundEnd { text, thought, .. } => {
                let t = thought.as_deref().map(str::trim).filter(|t| !t.is_empty());
                if let Some(t) = t {
                    println!("{GREY}think> {t}{RESET}");
                }
                if !text.trim().is_empty() {
                    println!("model> {}", text.trim());
                }
            }
            ChatEvent::ToolResult {
                name,
                input,
                output,
                ..
            } => {
                let head = format!("tool> {}({})", name, preview(input, 48));
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
            ChatEvent::ToolRetry { name, .. } => {
                println!("tool> {name}(malformed) -> discarded; asking the model to reissue");
            }
            ChatEvent::VarSet { name, value } => {
                println!("  · set {name} = {}", preview(value, 60));
            }
            ChatEvent::Done {
                tokens,
                steps,
                secs,
                stopped,
                text,
                thought,
            } => {
                self.settled(text, thought.as_deref());
                if *tokens > 0 {
                    let cap_note = if *stopped { "" } else { " · hit context limit" };
                    println!(
                        "  ({tokens} tok · {steps} steps · {secs:.1}s · {:.1} tok/s{cap_note})",
                        *tokens as f64 / secs.max(1e-9),
                    );
                } else if *secs > 0.0 {
                    println!("  ({secs:.1}s)");
                }
            }
            _ => {}
        }
    }
}
