//! The `ChatEvent` protocol: a JSONL-serializable stream of what a chat turn
//! is doing — status, streamed text, block commits, rewinds — observable with
//! `--events <file>` or on stdout with `--json`.

use serde::Serialize;

/// One event in the chat stream. Serialized as a JSON object tagged by `type`.
///
/// Turn shape: `TurnStart`, then per generation a `RoundStart` (round 0 is the
/// first; tool turns run several) carrying the streamed
/// `BlockStart`/`Status`/`Thought`/`Text`/`Rewind`/`BlockCommit` events, then
/// either a `RoundEnd` (the turn continues — tool events follow) or the single
/// terminal `Done`. A forced reply (`/response`) skips rounds entirely:
/// `TurnStart` → `Thought?` → `Text` → `Done`.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatEvent {
    /// A new user turn began.
    TurnStart { turn: u64 },
    /// A generation is starting: round 0 is a turn's first; a tool turn runs
    /// one round per model↔tool exchange. `prompt_tokens` is this round's
    /// rendered prompt size. Text/Thought streams restart from empty at each
    /// round.
    RoundStart { round: usize, prompt_tokens: usize },
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
    /// A round's generation settled but the turn CONTINUES — the model issued
    /// tool calls and another round follows. `text` is the settled narration
    /// before the calls (may be empty); `thought` the settled reasoning when
    /// the round surfaced it. The turn's terminal event is always `Done`.
    RoundEnd {
        round: usize,
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        thought: Option<String>,
    },
    /// The model called a tool; execution is starting.
    ToolCall {
        round: usize,
        name: String,
        input: String,
    },
    /// A tool finished; `output` is what the model will see next round.
    ToolResult {
        round: usize,
        name: String,
        input: String,
        output: String,
    },
    /// A malformed tool call was discarded; the model is asked to reissue it.
    ToolRetry { round: usize, name: String },
    /// A tool set a session variable via a `::set NAME=VALUE` directive.
    VarSet { name: String, value: String },
    /// The turn finished. `text` is the authoritative reply; `thought` the
    /// settled reasoning that goes with it (a surfaced or injected thought).
    /// `tokens`/`steps`/`secs` are totals across the turn's rounds.
    Done {
        tokens: usize,
        steps: usize,
        secs: f64,
        stopped: bool,
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        thought: Option<String>,
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
