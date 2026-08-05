//! The chat event stream — one turn, one stream of semantic events.
//!
//! Everything a turn does flows through [`ChatEvent`]: streamed thought and
//! answer text, block commits, tool rounds, forced injections, and the final
//! authoritative reply. Producers ([`StreamDecoder`] for the per-step token
//! flow, [`engine`] for turn orchestration) emit into sinks; consumers (the
//! terminal renderer, the JSONL writer) only ever observe the stream. No
//! consumer computes semantics, no producer renders.
//!
//! - [`event`] — the `ChatEvent` protocol (JSONL-serializable).
//! - [`decode`] — the pure decoding core: stable-prefix stabilization over
//!   per-step canvas snapshots and the token-level thought/answer split.
//! - [`engine`] — turn production: generation, the tool loop, forced replies.
//! - [`harness`] — the `--harness` JSON format: a session as a small
//!   tool-driven application.
//! - [`render`] — the interactive terminal renderer (viewport pane, spinner),
//!   a `ChatEvent` consumer.

mod decode;
pub(crate) mod engine;
mod event;
pub(crate) mod harness;
mod render;
mod sink;

pub use decode::{StreamDecoder, TextDecoder};
pub use event::ChatEvent;
pub use render::{ChatStream, StreamDisplay, render_demo};
pub use sink::{JsonlSink, PlainSink, SharedSinks, SinkSet, emit};

pub(crate) use decode::{ChannelIds, Stabilizer, salvage_answer};
