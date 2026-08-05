//! The chat event stream — one turn, one stream of semantic events.
//!
//! Everything a turn does flows through [`ChatEvent`]: streamed thought and
//! answer text, block commits, tool rounds, forced injections, and the final
//! authoritative reply. Producers (the [`crate::decoder`] stream decoder for
//! the per-step token flow, [`engine`] for turn orchestration) emit into sinks.
//! Consumers (the terminal renderer, the JSONL writer) only observe the stream.
//! No consumer computes semantics, no producer renders.
//!
//! - [`event`] gives the `ChatEvent` protocol (JSONL-serializable).
//! - [`engine`] runs turn production: generation, the tool loop, forced replies.
//! - [`harness`] handles the `--harness` JSON format, a session as a small
//!   tool-driven application.
//! - [`render`] is the interactive terminal renderer (viewport pane, spinner),
//!   a `ChatEvent` consumer.

pub(crate) mod engine;
mod event;
pub(crate) mod harness;
mod render;
mod sink;

pub use event::ChatEvent;
pub use render::{ChatStream, StreamDisplay, render_demo};
pub use sink::{JsonlSink, PlainSink, SharedSinks, SinkSet, emit};
