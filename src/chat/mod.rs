//! The chat event stream — one turn, one stream of semantic events.
//!
//! Everything a turn does flows through [`ChatEvent`]: streamed thought and
//! answer text, block commits, tool rounds, forced injections, and the final
//! authoritative reply. Producers ([`StreamDecoder`] for the per-step token
//! flow, the turn drivers in `commands::chat` for orchestration) emit into
//! [`EventSink`]s; consumers (the terminal renderer, the JSONL writer) only
//! ever observe the stream. No consumer computes semantics, no producer
//! renders.
//!
//! - [`event`] — the `ChatEvent` protocol (JSONL-serializable, versioned by
//!   additivity).
//! - [`decode`] — the pure decoding core: stable-prefix stabilization over
//!   per-step canvas snapshots and the token-level thought/answer split.
//! - [`render`] — the interactive terminal renderer (viewport pane, spinner),
//!   a `ChatEvent` consumer.

mod decode;
mod event;
mod render;

pub use decode::{StreamDecoder, TextDecoder};
pub use event::ChatEvent;
pub use render::{ChatStream, StreamDisplay, render_demo};

pub(crate) use decode::strip_thought_open;
