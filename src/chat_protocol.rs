//! Chat event protocol.
//!
//! A JSONL-serializable stream of what the denoiser is doing — status,
//! streamed text, block commits, and rewinds — plus a pure `StreamDecoder`
//! that turns raw per-step canvas snapshots (`StepProgressEvent`) into those
//! events. Decoupling the semantic events from rendering makes the stream
//! observable (emit it to a file with `--events`, or stdout with `--json`) and
//! makes the fragile stable-prefix logic unit-testable.

use crate::metal::StepProgressEvent;
use serde::Serialize;

/// Consecutive-step repeats before a canvas position is treated as stable
/// enough to stream. 2 proved too eager on fast-converging replies.
const STABLE_STREAK: u32 = 2;

/// Thought-channel markers as they decode to text (byte-identical to the ids).
const CHANNEL_OPEN: &str = "<|channel>";
const CHANNEL_CLOSE: &str = "<channel|>";

/// Strip a leading thought-open marker (`<|channel>thought\n`) from decoded
/// reasoning so the surfaced thought is clean text. A normal always-thinking
/// turn generates the open marker itself; a `/prethink` continuation begins
/// inside the thought (the marker was in the prompt) and is returned unchanged.
/// While the role line is still forming (`<|channel>thou`), yields "" until the
/// newline arrives rather than flashing the partial marker.
pub(crate) fn strip_thought_open(s: &str) -> &str {
    let Some(rest) = s.strip_prefix(CHANNEL_OPEN) else {
        return s; // no open marker: a prethink continuation
    };
    match rest.split_once('\n') {
        Some((_role, body)) => body,
        None => "", // role line ("thought\n") not complete yet
    }
}

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

/// Token-id → text decoding, abstracted so the decoder is testable without the
/// real 32 MB tokenizer.
pub trait TextDecoder {
    fn decode(&self, ids: &[u32]) -> String;
    fn decode_append(&self, out: &mut String, ids: &[u32]);
}

impl TextDecoder for std::sync::Arc<crate::tokenizer::Tokenizer> {
    fn decode(&self, ids: &[u32]) -> String {
        (**self).decode(ids)
    }
    fn decode_append(&self, out: &mut String, ids: &[u32]) {
        (**self).decode_append(out, ids);
    }
}

/// Longest common prefix of `a` and `b` in bytes, always a char boundary.
fn common_prefix_len(a: &str, b: &str) -> usize {
    let mut n = 0;
    let (mut ai, mut bi) = (a.char_indices(), b.char_indices());
    loop {
        match (ai.next(), bi.next()) {
            (Some((i, ca)), Some((_, cb))) if ca == cb => n = i + ca.len_utf8(),
            _ => break,
        }
    }
    n
}

/// Converts the raw per-step canvas argmax stream into `ChatEvent`s.
///
/// Stable-prefix rule: a position's argmax must hold for `STABLE_STREAK`
/// consecutive steps before it streams; the visible prefix is cut at the first
/// stop token. On `block_done` the whole canvas is committed (immutable). The
/// committed text is rebuilt once per block (not per step), so per-step work is
/// O(canvas), not O(total reply).
pub struct StreamDecoder<D: TextDecoder> {
    decoder: D,
    stops: Vec<u32>,
    block_idx: usize,
    last_argmax: Vec<u32>,
    streak: Vec<u32>,
    committed_ids: Vec<u32>,
    committed_text: String,
    ended: bool,
    last_text: String,
    /// Surface the reasoning live: emit `Thought` events for the thought-channel
    /// content and only treat the post-`<channel|>` text as the answer. Set by
    /// `--show-thinking` and implied by a `/prethink` seed. When false the
    /// thought is stripped silently and only the answer streams (the default).
    thinking_display: bool,
    /// Last surfaced thought text, for `Thought`-event dedup.
    last_thought: String,
    /// Set for a `/prethink` turn: generation runs inside an already-open
    /// thought channel, so the surfaced reasoning is `seed + thought-so-far`
    /// (up to the model's `<channel|>`). Absent for a plain `--show-thinking`
    /// turn, whose generation opens its own thought channel.
    prethink_seed: Option<String>,
}

impl<D: TextDecoder> StreamDecoder<D> {
    pub fn new(decoder: D, stops: Vec<u32>) -> Self {
        Self {
            decoder,
            stops,
            block_idx: 0,
            last_argmax: Vec::new(),
            streak: Vec::new(),
            committed_ids: Vec::new(),
            committed_text: String::new(),
            ended: false,
            last_text: String::new(),
            thinking_display: false,
            last_thought: String::new(),
            prethink_seed: None,
        }
    }

    /// Surface reasoning live: emit `Thought` events and split the answer at the
    /// model's `<channel|>`. `seed` is the `/prethink` continuation prefix (the
    /// reasoning the prompt already opened with); `None` for a plain
    /// `--show-thinking` turn that opens its own thought channel.
    pub fn with_thinking(mut self, show_thinking: bool, seed: Option<String>) -> Self {
        self.thinking_display = show_thinking || seed.is_some();
        self.prethink_seed = seed;
        self
    }

    /// The immutable committed reply text so far.
    #[allow(dead_code)]
    pub fn committed_text(&self) -> &str {
        &self.committed_text
    }

    pub fn on_step(&mut self, ev: &StepProgressEvent<'_>) -> Vec<ChatEvent> {
        let mut out = Vec::new();
        if self.ended {
            return out;
        }

        if ev.block_idx != self.block_idx {
            self.block_idx = ev.block_idx;
            self.last_argmax.clear();
            self.streak.clear();
            out.push(ChatEvent::BlockStart {
                block: ev.block_idx,
            });
        }

        // Per-position stable-streak update.
        if self.last_argmax.len() != ev.argmax.len() {
            self.last_argmax = ev.argmax.to_vec();
            self.streak = vec![0; ev.argmax.len()];
        } else {
            for i in 0..ev.argmax.len() {
                if ev.argmax[i] == self.last_argmax[i] {
                    self.streak[i] = self.streak[i].saturating_add(1);
                } else {
                    self.streak[i] = 0;
                    self.last_argmax[i] = ev.argmax[i];
                }
            }
        }

        // Stable prefix (whole canvas on commit), cut at the first stop token.
        let prefix_end = if ev.block_done {
            ev.argmax.len()
        } else {
            self.streak
                .iter()
                .position(|&k| k < STABLE_STREAK)
                .unwrap_or(ev.argmax.len())
        };
        let mut block_ids = Vec::with_capacity(prefix_end);
        let mut hit_stop = false;
        for &id in &ev.argmax[..prefix_end] {
            if self.stops.contains(&id) {
                hit_stop = true;
                break;
            }
            block_ids.push(id);
        }

        out.push(ChatEvent::Status {
            block: ev.block_idx,
            step: ev.step_in_block,
            max_steps: ev.max_steps,
            accepted: ev.accept_count,
            canvas: ev.argmax.len(),
            locked: block_ids.len(),
        });

        // Thinking display (`--show-thinking` / `/prethink`): re-decode the whole
        // generation each step and split it at the model's `<channel|>`. The
        // reasoning before it streams as `Thought`, the answer after it as the
        // normal `Text`. The full re-decode splits a thought that spans several
        // blocks correctly, where the append-then-sanitize path below needs the
        // open marker in the live block.
        if self.thinking_display {
            if ev.block_done {
                self.committed_ids.extend_from_slice(&block_ids);
                self.last_argmax.clear();
                self.streak.clear();
                if hit_stop {
                    self.ended = true;
                }
            }
            let seed = self.prethink_seed.as_deref().unwrap_or("");
            // Split a decoded generation at the thought close: reasoning before,
            // answer after (empty until the thought closes).
            let split = |raw: &str| -> (String, String) {
                match raw.split_once(CHANNEL_CLOSE) {
                    Some((b, a)) => (
                        format!("{seed}{}", strip_thought_open(b)),
                        crate::chat_template::sanitize_model_reply(a),
                    ),
                    None => (format!("{seed}{}", strip_thought_open(raw)), String::new()),
                }
            };
            let strip = crate::sample::strip_degenerate_token_ids;
            // Committed view = finalized blocks only; its answer is the immutable
            // prefix. Full view = committed + the live draft. Deriving `committed`
            // from the committed view keeps it monotonic across blocks, so the
            // renderer never resets its print cursor mid-answer and re-emits the
            // whole reply on each block boundary.
            let committed_raw = self.decoder.decode(&strip(&self.committed_ids));
            let (_, committed_answer) = split(&committed_raw);
            let full_raw = if ev.block_done {
                committed_raw
            } else {
                let mut ids = self.committed_ids.clone();
                ids.extend_from_slice(&block_ids);
                self.decoder.decode(&strip(&ids))
            };
            let (thought, answer) = split(&full_raw);

            if thought != self.last_thought {
                out.push(ChatEvent::Thought {
                    text: thought.clone(),
                });
                self.last_thought = thought;
            }
            if answer != self.last_text {
                if !answer.starts_with(&self.last_text) {
                    out.push(ChatEvent::Rewind {
                        common: common_prefix_len(&self.last_text, &answer),
                    });
                }
                let committed = common_prefix_len(&committed_answer, &answer);
                out.push(ChatEvent::Text {
                    committed,
                    text: answer.clone(),
                });
                self.last_text = answer;
            }
            return out;
        }

        if ev.block_done {
            self.committed_ids.extend_from_slice(&block_ids);
            self.last_argmax.clear();
            self.streak.clear();
            let raw = self
                .decoder
                .decode(&crate::sample::strip_degenerate_token_ids(
                    &self.committed_ids,
                ));
            self.committed_text = crate::chat_template::sanitize_model_reply(&raw);
            out.push(ChatEvent::BlockCommit {
                block: ev.block_idx,
                committed_tokens: block_ids.len(),
            });
            if hit_stop {
                self.ended = true;
            }
        }

        // Full visible text = committed + this block's speculative draft.
        let mut text = self.committed_text.clone();
        if !ev.block_done {
            let suffix = crate::sample::strip_degenerate_token_ids(&block_ids);
            if !suffix.is_empty() {
                self.decoder.decode_append(&mut text, &suffix);
                text = crate::chat_template::sanitize_model_reply(&text);
            }
        }
        let committed = common_prefix_len(&self.committed_text, &text);

        if text != self.last_text {
            if !text.starts_with(&self.last_text) {
                out.push(ChatEvent::Rewind {
                    common: common_prefix_len(&self.last_text, &text),
                });
            }
            out.push(ChatEvent::Text {
                committed,
                text: text.clone(),
            });
            self.last_text = text;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial decoder: token id maps to its char; ids outside printable ASCII
    /// decode to nothing. Good enough to exercise the id-level stable-prefix
    /// logic without the real tokenizer or sanitizer markers.
    struct FakeDecoder;
    impl TextDecoder for FakeDecoder {
        fn decode(&self, ids: &[u32]) -> String {
            let mut s = String::new();
            self.decode_append(&mut s, ids);
            s
        }
        fn decode_append(&self, out: &mut String, ids: &[u32]) {
            for &id in ids {
                if (32..127).contains(&id) {
                    out.push(id as u8 as char);
                }
            }
        }
    }

    fn step<'a>(block: usize, step: u32, argmax: &'a [u32], done: bool) -> StepProgressEvent<'a> {
        StepProgressEvent {
            block_idx: block,
            max_blocks: 4,
            step_in_block: step,
            max_steps: 48,
            argmax,
            accept_count: 0,
            mean_entropy: 0.0,
            block_done: done,
        }
    }

    fn texts(events: &[ChatEvent]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|e| match e {
                ChatEvent::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn thoughts(events: &[ChatEvent]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|e| match e {
                ChatEvent::Thought { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

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

    #[test]
    fn stable_prefix_requires_two_sightings() {
        let mut d = StreamDecoder::new(FakeDecoder, vec![]);
        let hi = [b'H' as u32, b'i' as u32];
        // Step 1: fresh, streak 0 → nothing stable yet.
        let e1 = d.on_step(&step(1, 1, &hi, false));
        assert_eq!(texts(&e1), Vec::<&str>::new());
        // Step 2: streak 1 → still not stable.
        let e2 = d.on_step(&step(1, 2, &hi, false));
        assert_eq!(texts(&e2), Vec::<&str>::new());
        // Step 3: streak 2 → both positions stream.
        let e3 = d.on_step(&step(1, 3, &hi, false));
        assert_eq!(texts(&e3), vec!["Hi"]);
    }

    #[test]
    fn prethink_streams_seed_then_thought() {
        let mut d =
            StreamDecoder::new(FakeDecoder, vec![]).with_thinking(false, Some("SEED ".into()));
        let hi = [b'H' as u32, b'i' as u32];
        // The reasoning surfaces as `Thought` (never `Text`): the injected seed
        // shows immediately, before any generated thought stabilizes.
        let e1 = d.on_step(&step(1, 1, &hi, false));
        assert_eq!(thoughts(&e1), vec!["SEED "]);
        assert_eq!(texts(&e1), Vec::<&str>::new());
        assert_eq!(
            thoughts(&d.on_step(&step(1, 2, &hi, false))),
            Vec::<&str>::new()
        );
        // Then the thought streams in, still prefixed by the seed.
        assert_eq!(
            thoughts(&d.on_step(&step(1, 3, &hi, false))),
            vec!["SEED Hi"]
        );
    }

    #[test]
    fn strip_thought_open_drops_only_the_open_marker() {
        // A normal turn opens its own thought: the marker + role line go.
        assert_eq!(
            strip_thought_open("<|channel>thought\nreasoning"),
            "reasoning"
        );
        // A `/prethink` continuation has no open marker: unchanged.
        assert_eq!(
            strip_thought_open("continuing the seed"),
            "continuing the seed"
        );
        // Role line still forming: nothing to show yet.
        assert_eq!(strip_thought_open("<|channel>thou"), "");
    }

    /// Decoder that also renders two sentinel ids as the thought-channel markers,
    /// so a normal-turn `--show-thinking` flow can be exercised end to end.
    struct MarkerDecoder;
    impl TextDecoder for MarkerDecoder {
        fn decode(&self, ids: &[u32]) -> String {
            let mut s = String::new();
            self.decode_append(&mut s, ids);
            s
        }
        fn decode_append(&self, out: &mut String, ids: &[u32]) {
            for &id in ids {
                match id {
                    200 => out.push_str("<|channel>thought\n"),
                    201 => out.push_str("<channel|>"),
                    32..=127 => out.push(id as u8 as char),
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn show_thinking_splits_reasoning_from_answer() {
        // Normal turn (no seed) with show_thinking: the model opens its own
        // thought, reasons, closes, then answers.
        let mut d = StreamDecoder::new(MarkerDecoder, vec![]).with_thinking(true, None);
        // Reasoning canvas: <|channel>thought\n 'H' 'i', stabilizes by step 3.
        let reasoning = [200u32, b'H' as u32, b'i' as u32];
        let _ = d.on_step(&step(1, 1, &reasoning, false));
        let _ = d.on_step(&step(1, 2, &reasoning, false));
        let e = d.on_step(&step(1, 3, &reasoning, false));
        // The reasoning surfaced as Thought, never as answer Text.
        assert_eq!(thoughts(&e), vec!["Hi"]);
        assert_eq!(texts(&e), Vec::<&str>::new());
        // Now the thought closes and the answer appears (committed on block_done).
        let full = [
            200u32,
            b'H' as u32,
            b'i' as u32,
            201,
            b'O' as u32,
            b'k' as u32,
        ];
        let done = d.on_step(&step(1, 4, &full, true));
        assert_eq!(texts(&done), vec!["Ok"]);
    }

    /// Regression: a multi-block answer's committed prefix must grow monotonically
    /// (a decrease reset the renderer's print cursor and re-emitted the reply).
    #[test]
    fn thinking_answer_committed_is_monotonic_across_blocks() {
        let pairs = |events: &[ChatEvent]| -> Vec<(usize, String)> {
            events
                .iter()
                .filter_map(|e| match e {
                    ChatEvent::Text { committed, text } => Some((*committed, text.clone())),
                    _ => None,
                })
                .collect()
        };
        let mut d = StreamDecoder::new(MarkerDecoder, vec![]).with_thinking(true, None);
        // Block 1 completes: thought "Hi" closes, answer "Ab" is committed (len 2).
        let b1 = [
            200u32,
            b'H' as u32,
            b'i' as u32,
            201,
            b'A' as u32,
            b'b' as u32,
        ];
        assert_eq!(
            pairs(&d.on_step(&step(1, 1, &b1, true))),
            vec![(2, "Ab".into())]
        );
        // Block 2 extends the answer with a still-speculative "cd": the committed
        // prefix stays at the finalized length 2, never dropping to 0.
        let b2 = [b'c' as u32, b'd' as u32];
        let _ = d.on_step(&step(2, 1, &b2, false));
        let _ = d.on_step(&step(2, 2, &b2, false));
        assert_eq!(
            pairs(&d.on_step(&step(2, 3, &b2, false))),
            vec![(2, "Abcd".into())]
        );
    }

    #[test]
    fn stop_token_cuts_the_visible_prefix() {
        let mut d = StreamDecoder::new(FakeDecoder, vec![99]);
        let canvas = [b'O' as u32, b'k' as u32, 99, b'X' as u32];
        for s in 1..=3 {
            let _ = d.on_step(&step(1, s, &canvas, false));
        }
        let e = d.on_step(&step(1, 4, &canvas, false));
        // Everything after the stop token (99) is dropped, even though "X"
        // is also stable.
        assert_eq!(texts(&e), Vec::<&str>::new()); // unchanged: step 3 already showed "Ok"
        assert_eq!(d.committed_text(), "");
        let e3 = {
            let mut d2 = StreamDecoder::new(FakeDecoder, vec![99]);
            for s in 1..=2 {
                let _ = d2.on_step(&step(1, s, &canvas, false));
            }
            d2.on_step(&step(1, 3, &canvas, false))
        };
        assert_eq!(texts(&e3), vec!["Ok"]);
    }

    #[test]
    fn block_done_commits_full_canvas_and_ends_on_stop() {
        let mut d = StreamDecoder::new(FakeDecoder, vec![99]);
        let canvas = [b'H' as u32, b'i' as u32, 99];
        let e = d.on_step(&step(1, 5, &canvas, true));
        assert_eq!(d.committed_text(), "Hi");
        assert!(
            e.iter()
                .any(|ev| matches!(ev, ChatEvent::BlockCommit { .. }))
        );
        // Ended: a further step yields nothing.
        assert!(d.on_step(&step(1, 6, &canvas, true)).is_empty());
    }

    #[test]
    fn rewind_emitted_when_a_stable_position_revises() {
        let mut d = StreamDecoder::new(FakeDecoder, vec![]);
        let cat = [b'C' as u32, b'a' as u32, b't' as u32];
        for s in 1..=3 {
            let _ = d.on_step(&step(1, s, &cat, false));
        }
        // Now the middle position flips: the stable "Cat" draft rewinds to "C"
        // on the flip step, then re-stabilizes to "Cot".
        let cot = [b'C' as u32, b'o' as u32, b't' as u32];
        let mut all = Vec::new();
        all.extend(d.on_step(&step(1, 4, &cot, false)));
        all.extend(d.on_step(&step(1, 5, &cot, false)));
        let last = d.on_step(&step(1, 6, &cot, false));
        all.extend(last.clone());
        assert!(
            all.iter()
                .any(|ev| matches!(ev, ChatEvent::Rewind { common: 1 }))
        );
        assert_eq!(texts(&last), vec!["Cot"]);
    }
}
