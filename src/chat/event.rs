//! The `ChatEvent` protocol: a JSONL-serializable stream of what a chat turn
//! is doing — status, streamed text, block commits, rewinds — observable with
//! `--events <file>` or on stdout with `--json`.

use serde::Serialize;

/// One event in the chat stream. Serialized as a JSON object tagged by `type`.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatEvent {
    /// A new user turn began.
    TurnStart { turn: u64, prompt_tokens: usize },
    /// The denoiser started a new canvas block.
    BlockStart { block: usize },
    /// Per denoise step: convergence telemetry. `locked` = length of the
    /// stable stop-cut prefix (tokens the sampler is confident about).
    Status {
        block: usize,
        step: u32,
        max_steps: usize,
        accepted: u32,
        canvas: usize,
        locked: usize,
    },
    /// The full visible answer text so far. Bytes `[0, committed)` are FINAL
    /// (block-committed, immutable); `[committed, len)` is the speculative
    /// stable-prefix draft and may still change or rewind. `committed` is a
    /// char boundary.
    Text { committed: usize, text: String },
    /// The visible draft diverged from what was previously emitted before its
    /// end: everything after byte `common` changed (a "stable" position was
    /// revised). Always immediately followed by a `Text` with the new content.
    /// `common` is a char boundary.
    Rewind { common: usize },
    /// Streamed reasoning (thought-channel) text so far. Rendered as grey
    /// `think>` lines and never part of the committed answer. Emitted only when
    /// a turn surfaces its reasoning live (`--show-thinking` / `/prethink`); a
    /// plain turn keeps its thought hidden and this event is absent.
    Thought { text: String },
    /// A canvas block's tokens became final (immutable).
    BlockCommit {
        block: usize,
        committed_tokens: usize,
    },
    /// The turn finished. `text` is the authoritative reply.
    Done {
        tokens: usize,
        steps: usize,
        secs: f64,
        stopped: bool,
        text: String,
    },
    /// A low-priority, model-drafted guess at the user's *next* message,
    /// produced after the reply. `turn` is the upcoming turn it would seed.
    /// Advisory: a driver may submit it verbatim, edit it, or ignore it.
    Suggestion { turn: u64, text: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suggestion_event_serializes_tagged() {
        let ev = ChatEvent::Suggestion {
            turn: 3,
            text: "Sounds good — ship it.".to_string(),
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert_eq!(
            json,
            r#"{"type":"suggestion","turn":3,"text":"Sounds good — ship it."}"#
        );
    }
}
