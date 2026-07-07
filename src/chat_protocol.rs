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
        }
    }

    /// The immutable committed reply text so far.
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
            step_in_block: step,
            max_steps: 48,
            argmax,
            accept_count: 0,
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
