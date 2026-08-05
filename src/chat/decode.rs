//! The pure decoding core shared by every stream consumer: a [`Stabilizer`]
//! that turns raw per-step canvas snapshots (`StepProgressEvent`) into stable
//! stop-cut id prefixes, a token-level [`ChannelIds::split`] that separates
//! thought-channel ids from answer ids (quote-aware: channel markup inside a
//! quoted tool arg is data, never grammar), and the [`StreamDecoder`] that
//! composes them into semantic [`ChatEvent`]s. The serve wire mapper builds on
//! the same two primitives, so chat and serve cannot drift on the fragile
//! stable-prefix and channel-split rules.

use super::ChatEvent;
use crate::metal::StepProgressEvent;

/// Consecutive-step repeats before a canvas position is treated as stable
/// enough to stream. 2 proved too eager on fast-converging replies.
pub(crate) const STABLE_STREAK: u32 = 2;

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

/// The special-token ids that structure a generation: the thought channel
/// markers and the `<|"|>` string quote. `None` entries simply never match,
/// so a model (or test decoder) without a marker degrades gracefully.
#[derive(Clone, Copy, Default)]
pub(crate) struct ChannelIds {
    pub open: Option<u32>,
    pub close: Option<u32>,
    pub quote: Option<u32>,
}

/// Reasoning (thought-channel) and answer ids split out of a generation.
pub(crate) struct SplitIds {
    pub reasoning: Vec<u32>,
    pub content: Vec<u32>,
    /// A thought span actually closed (a `<channel|>` seen outside quotes).
    /// False when the model never closed its thought — the settled-output
    /// fallbacks key off this.
    pub closed: bool,
}

impl ChannelIds {
    pub fn from_tokenizer(tok: &crate::tokenizer::Tokenizer) -> Self {
        Self {
            open: tok.special_token_id("<|channel>"),
            close: tok.special_token_id("<channel|>"),
            quote: tok.special_token_id("<|\"|>"),
        }
    }

    /// Split `ids` into thought-channel and answer ids at the token level.
    ///
    /// Reasoning is ONLY what sits inside an explicit `<|channel>…<channel|>`
    /// span the model actually emitted — classification follows emission, not
    /// a mode flag (the old string rule "everything is reasoning until a close
    /// appears" silently swallowed a whole turn when the model skipped the
    /// thought ceremony and answered with a bare tool call). An unclosed span
    /// runs to the end, so mid-thought streaming works. Channel ids inside an
    /// open `<|"|>` quote run are literal arg content: they neither toggle the
    /// thought state nor get filtered — a tool call writing a file that
    /// contains chat markup round-trips it byte-exact.
    ///
    /// `start_in_thought` = the generation began inside an already-open
    /// thought (a `/prethink` continuation whose open marker was in the
    /// prompt); the walk starts in-span with the role line already consumed.
    pub fn split<D: TextDecoder>(
        &self,
        decoder: &D,
        ids: &[u32],
        start_in_thought: bool,
    ) -> SplitIds {
        let mut reasoning: Vec<u32> = Vec::new();
        let mut content: Vec<u32> = Vec::new();
        let mut in_thought = start_in_thought;
        let mut in_quote = false;
        let mut closed = false;
        // A `<|channel>` takes its NAME token (+ trailing newline) with it: the
        // open special is dropped here, so a mid-span re-open would otherwise
        // leak a bare "thought" line into the surfaced reasoning.
        let mut skip_name = false;
        for &id in ids {
            if Some(id) == self.quote {
                in_quote = !in_quote;
                // The quote id is grammar the tool parser needs — keep it.
                if in_thought {
                    reasoning.push(id);
                } else {
                    content.push(id);
                }
            } else if in_quote {
                // Literal quoted content: channel ids included, verbatim.
                if in_thought {
                    reasoning.push(id);
                } else {
                    content.push(id);
                }
            } else if Some(id) == self.open {
                in_thought = true;
                skip_name = true;
            } else if Some(id) == self.close {
                in_thought = false;
                closed = true;
            } else if in_thought {
                if skip_name {
                    let tok = decoder.decode(&[id]);
                    if tok == "thought" {
                        continue;
                    }
                    skip_name = false;
                    if tok == "\n" {
                        continue;
                    }
                }
                reasoning.push(id);
            } else {
                content.push(id);
            }
        }
        SplitIds {
            reasoning,
            content,
            closed,
        }
    }

    /// Settle a finished `/prethink` generation (began inside an open thought)
    /// into `(thought, answer)` text. The thought is `seed` + the model's
    /// completion; the answer is what follows the model's `<channel|>`,
    /// unsanitized. When the thought never closed the model usually stated
    /// its answer inside the reasoning (a thought degeneracy) — salvage it
    /// from the completion's tail rather than reporting none.
    pub fn settle_prethink<D: TextDecoder>(
        &self,
        decoder: &D,
        seed: &str,
        ids: &[u32],
    ) -> (String, String) {
        let s = self.split(decoder, ids, true);
        let completion = decoder.decode(&s.reasoning);
        let answer = if s.closed {
            decoder.decode(&s.content)
        } else {
            salvage_answer(&completion)
        };
        (format!("{seed}{completion}"), answer)
    }

    /// Settle a finished tool-mode generation into `(reasoning, rest)` text.
    /// A closed thought separates the surfaced reasoning (prefixed by the
    /// `/prethink` `seed`, if any) from the answer / tool calls. Without a
    /// close the model went straight to output — often tool calls — so there
    /// is no separate reasoning and `rest` is the whole decoded text, keeping
    /// the calls parseable.
    pub fn settle_tool_reply<D: TextDecoder>(
        &self,
        decoder: &D,
        thinking: bool,
        seed: &str,
        ids: &[u32],
    ) -> (Option<String>, String) {
        if !thinking {
            return (None, decoder.decode(ids));
        }
        let s = self.split(decoder, ids, !seed.is_empty());
        if s.closed {
            let thought = format!("{seed}{}", decoder.decode(&s.reasoning))
                .trim()
                .to_string();
            (Some(thought), decoder.decode(&s.content))
        } else {
            (None, decoder.decode(ids))
        }
    }
}

/// Recover an answer from a thought that never closed: take the last
/// blank-line-separated block (the model usually sets the answer off after a
/// blank line), or the whole completion when there is no separator.
pub(crate) fn salvage_answer(completion: &str) -> String {
    let t = completion.trim();
    t.rsplit("\n\n").next().unwrap_or(t).trim().to_string()
}

/// The stable, stop-cut prefix a single denoise step yields.
pub(crate) struct StablePrefix {
    /// This step opened a new canvas block.
    pub new_block: bool,
    /// The stable prefix ids (the whole canvas on block commit), cut at the
    /// first effective stop token.
    pub ids: Vec<u32>,
    /// A stop token ended the prefix.
    pub hit_stop: bool,
}

/// Per-position stable-streak tracking + stop-token cut over raw per-step
/// canvas snapshots — the shared front half of every stream decoder (terminal
/// chat and the serve wire). A position's argmax must hold for
/// [`STABLE_STREAK`] consecutive steps before it streams; on `block_done` the
/// whole canvas is stable by definition. With `stop_skip_quoted`, a stop id
/// inside an open `<|"|>` quote run is literal content, not a stop — this must
/// mirror the engine's `continue_incomplete_tool_calls` scan, or the stream
/// would cut text the engine kept (and vice versa).
pub(crate) struct Stabilizer {
    stops: Vec<u32>,
    quote: Option<u32>,
    stop_skip_quoted: bool,
    block_idx: usize,
    last_argmax: Vec<u32>,
    streak: Vec<u32>,
    /// Quote parity of everything committed so far, carried across blocks.
    committed_quote_open: bool,
}

impl Stabilizer {
    pub fn new(stops: Vec<u32>, quote: Option<u32>, stop_skip_quoted: bool) -> Self {
        Self {
            stops,
            quote,
            stop_skip_quoted,
            block_idx: 0,
            last_argmax: Vec::new(),
            streak: Vec::new(),
            committed_quote_open: false,
        }
    }

    pub fn on_step(&mut self, ev: &StepProgressEvent<'_>) -> StablePrefix {
        let new_block = ev.block_idx != self.block_idx;
        if new_block {
            self.block_idx = ev.block_idx;
            self.last_argmax.clear();
            self.streak.clear();
        }

        // Per-position stable-streak update (a canvas-length change —
        // shrink-on-retry — resets everything).
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
        let mut ids = Vec::with_capacity(prefix_end);
        let mut hit_stop = false;
        let mut in_quote = self.stop_skip_quoted && self.committed_quote_open;
        for &id in &ev.argmax[..prefix_end] {
            if self.stop_skip_quoted && Some(id) == self.quote {
                in_quote = !in_quote;
            } else if !in_quote && self.stops.contains(&id) {
                hit_stop = true;
                break;
            }
            ids.push(id);
        }

        if ev.block_done {
            self.last_argmax.clear();
            self.streak.clear();
            // The caller commits exactly `ids`; fold their quote parity so the
            // next block's stop scan starts in the right state.
            if let Some(q) = self.quote {
                let flips = ids.iter().filter(|&&id| id == q).count();
                if flips % 2 == 1 {
                    self.committed_quote_open = !self.committed_quote_open;
                }
            }
        }

        StablePrefix {
            new_block,
            ids,
            hit_stop,
        }
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

/// Converts the raw per-step canvas argmax stream into `ChatEvent`s: a
/// [`Stabilizer`] front end, a token-level [`ChannelIds::split`] for turns
/// that surface their reasoning, and text diffing into
/// `Thought`/`Text`/`Rewind` events. The committed text is rebuilt once per
/// block (not per step), so per-step work is O(canvas), not O(total reply).
pub struct StreamDecoder<D: TextDecoder> {
    decoder: D,
    channels: ChannelIds,
    stab: Stabilizer,
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
            channels: ChannelIds::default(),
            stab: Stabilizer::new(stops, None, false),
            committed_ids: Vec::new(),
            committed_text: String::new(),
            ended: false,
            last_text: String::new(),
            thinking_display: false,
            last_thought: String::new(),
            prethink_seed: None,
        }
    }

    /// Bind the model's channel/quote special ids (required for a turn that
    /// surfaces its reasoning; without them everything streams as answer).
    pub(crate) fn with_channels(mut self, channels: ChannelIds) -> Self {
        self.channels = channels;
        self
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

    /// Decode answer-side ids to sanitized visible text.
    fn decode_content(&self, ids: &[u32]) -> String {
        crate::chat_template::sanitize_model_reply(
            &self
                .decoder
                .decode(&crate::sample::strip_degenerate_token_ids(ids)),
        )
    }

    pub fn on_step(&mut self, ev: &StepProgressEvent<'_>) -> Vec<ChatEvent> {
        let mut out = Vec::new();
        if self.ended {
            return out;
        }

        let sp = self.stab.on_step(ev);
        if sp.new_block {
            out.push(ChatEvent::BlockStart {
                block: ev.block_idx,
            });
        }
        out.push(ChatEvent::Status {
            block: ev.block_idx,
            step: ev.step_in_block,
            max_steps: ev.max_steps,
            accepted: ev.accept_count,
            canvas: ev.argmax.len(),
            locked: sp.ids.len(),
        });

        // Thinking display (`--show-thinking` / `/prethink`): split the whole
        // generation at the token level each step. The reasoning streams as
        // `Thought`, the answer after the model's `<channel|>` as the normal
        // `Text`. The full re-split handles a thought that spans several blocks,
        // where an append-only path would need the open marker in the live block.
        if self.thinking_display {
            if ev.block_done {
                self.committed_ids.extend_from_slice(&sp.ids);
                if sp.hit_stop {
                    self.ended = true;
                }
            }
            let seed = self.prethink_seed.as_deref().unwrap_or("");
            let in_thought = self.prethink_seed.is_some();
            let strip = crate::sample::strip_degenerate_token_ids;
            // Committed view = finalized blocks only; its answer is the immutable
            // prefix. Full view = committed + the live draft. Deriving `committed`
            // from the committed view keeps it monotonic across blocks, so the
            // renderer never resets its print cursor mid-answer and re-emits the
            // whole reply on each block boundary.
            let committed_split =
                self.channels
                    .split(&self.decoder, &strip(&self.committed_ids), in_thought);
            let committed_answer = self.decode_content(&committed_split.content);
            let full_split = if ev.block_done {
                committed_split
            } else {
                let mut ids = self.committed_ids.clone();
                ids.extend_from_slice(&sp.ids);
                self.channels.split(&self.decoder, &strip(&ids), in_thought)
            };
            let thought = format!(
                "{seed}{}",
                self.decoder.decode(&strip(&full_split.reasoning))
            );
            let answer = self.decode_content(&full_split.content);

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
            self.committed_ids.extend_from_slice(&sp.ids);
            self.committed_text = self.decode_content(&self.committed_ids);
            out.push(ChatEvent::BlockCommit {
                block: ev.block_idx,
                committed_tokens: sp.ids.len(),
            });
            if sp.hit_stop {
                self.ended = true;
            }
        }

        // Full visible text = committed + this block's speculative draft.
        let mut text = self.committed_text.clone();
        if !ev.block_done {
            let suffix = crate::sample::strip_degenerate_token_ids(&sp.ids);
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

    /// Sentinel channel ids for exercising the token-level split: 200 opens the
    /// thought channel, 201 closes it (the decoder never renders them — the
    /// walk consumes them as toggles, exactly like the real specials).
    const OPEN: u32 = 200;
    const CLOSE: u32 = 201;

    fn channel_ids() -> ChannelIds {
        ChannelIds {
            open: Some(OPEN),
            close: Some(CLOSE),
            quote: None,
        }
    }

    #[test]
    fn split_consumes_open_marker_and_role_name() {
        // decode(ids) for FakeDecoder renders "thought" / "\n" name tokens as
        // chars; use explicit ids so the name-skip logic is exercised.
        struct NameDecoder;
        impl TextDecoder for NameDecoder {
            fn decode(&self, ids: &[u32]) -> String {
                let mut s = String::new();
                self.decode_append(&mut s, ids);
                s
            }
            fn decode_append(&self, out: &mut String, ids: &[u32]) {
                for &id in ids {
                    match id {
                        300 => out.push_str("thought"),
                        301 => out.push('\n'),
                        32..=127 => out.push(id as u8 as char),
                        _ => {}
                    }
                }
            }
        }
        let ids = [OPEN, 300, 301, b'H' as u32, b'i' as u32, CLOSE, b'O' as u32];
        let split = channel_ids().split(&NameDecoder, &ids, false);
        assert_eq!(split.reasoning, vec![b'H' as u32, b'i' as u32]);
        assert_eq!(split.content, vec![b'O' as u32]);
        // A prethink continuation starts in-thought with no marker at all.
        let split = channel_ids().split(&NameDecoder, &[b'H' as u32, CLOSE, b'O' as u32], true);
        assert_eq!(split.reasoning, vec![b'H' as u32]);
        assert_eq!(split.content, vec![b'O' as u32]);
    }

    #[test]
    fn split_keeps_quoted_channel_markup_as_data() {
        const QUOTE: u32 = 202;
        let ids = ChannelIds {
            open: Some(OPEN),
            close: Some(CLOSE),
            quote: Some(QUOTE),
        };
        // Inside an open quote run, a `<channel|>` id is literal arg content:
        // it must not close the (never-opened) thought or vanish from content.
        let stream = [b'A' as u32, QUOTE, CLOSE, OPEN, QUOTE, b'B' as u32];
        let split = ids.split(&FakeDecoder, &stream, false);
        assert_eq!(
            split.content,
            vec![b'A' as u32, QUOTE, CLOSE, OPEN, QUOTE, b'B' as u32]
        );
        assert!(split.reasoning.is_empty());
    }

    #[test]
    fn show_thinking_splits_reasoning_from_answer() {
        // Normal turn (no seed) with show_thinking: the model opens its own
        // thought, reasons, closes, then answers.
        let mut d = StreamDecoder::new(FakeDecoder, vec![])
            .with_channels(channel_ids())
            .with_thinking(true, None);
        // Reasoning canvas: <|channel> 'H' 'i', stabilizes by step 3.
        let reasoning = [OPEN, b'H' as u32, b'i' as u32];
        let _ = d.on_step(&step(1, 1, &reasoning, false));
        let _ = d.on_step(&step(1, 2, &reasoning, false));
        let e = d.on_step(&step(1, 3, &reasoning, false));
        // The reasoning surfaced as Thought, never as answer Text.
        assert_eq!(thoughts(&e), vec!["Hi"]);
        assert_eq!(texts(&e), Vec::<&str>::new());
        // Now the thought closes and the answer appears (committed on block_done).
        let full = [OPEN, b'H' as u32, b'i' as u32, CLOSE, b'O' as u32, b'k' as u32];
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
        let mut d = StreamDecoder::new(FakeDecoder, vec![])
            .with_channels(channel_ids())
            .with_thinking(true, None);
        // Block 1 completes: thought "Hi" closes, answer "Ab" is committed (len 2).
        let b1 = [OPEN, b'H' as u32, b'i' as u32, CLOSE, b'A' as u32, b'b' as u32];
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

    #[test]
    fn salvage_answer_takes_the_trailing_block() {
        // The model reasoned, then stated the answer after a blank line.
        assert_eq!(
            salvage_answer("The project 'Bluebird' ships in March.\n\nMarch."),
            "March."
        );
        // No separator: the whole completion is the recovered answer.
        assert_eq!(
            salvage_answer("just reasoning, ending in 56"),
            "just reasoning, ending in 56"
        );
        // Empty / whitespace: nothing to salvage.
        assert_eq!(salvage_answer("  \n\n  "), "");
    }

    #[test]
    fn settle_prethink_splits_at_close_and_salvages_without_one() {
        let ids = channel_ids();
        // Closed: seed + completion before the close; answer after.
        let closed = [b'S' as u32, b'1' as u32, CLOSE, b'4' as u32, b'2' as u32];
        assert_eq!(
            ids.settle_prethink(&FakeDecoder, "Think. ", &closed),
            ("Think. S1".to_string(), "42".to_string())
        );
        // Never closed: all thought; the answer is salvaged from its tail
        // (here: no blank-line separator, so the whole completion).
        let open = [b'S' as u32, b'1' as u32];
        assert_eq!(
            ids.settle_prethink(&FakeDecoder, "seed ", &open),
            ("seed S1".to_string(), "S1".to_string())
        );
    }

    #[test]
    fn settle_tool_reply_keeps_calls_without_a_channel_close() {
        let ids = channel_ids();
        let calls = [b'c' as u32, b'a' as u32, b'l' as u32, b'l' as u32];
        // A closed span separates reasoning from the calls.
        let stream = [OPEN, b'H' as u32, b'i' as u32, CLOSE, calls[0], calls[1], calls[2], calls[3]];
        let (thought, content) = ids.settle_tool_reply(&FakeDecoder, true, "", &stream);
        assert_eq!(thought.as_deref(), Some("Hi"));
        assert_eq!(content, "call");
        // No close (model went straight to calls): whole text is kept so the
        // calls still parse — the regression this guards against.
        let (thought, content) = ids.settle_tool_reply(&FakeDecoder, true, "", &calls);
        assert_eq!(thought, None);
        assert_eq!(content, "call");
        // Unclosed span: the raw decode (markers render as nothing here) is
        // returned whole rather than an empty content.
        let open_span = [OPEN, b'X' as u32, calls[0]];
        let (thought, content) = ids.settle_tool_reply(&FakeDecoder, true, "", &open_span);
        assert_eq!(thought, None);
        assert_eq!(content, "Xc");
        // Thinking off: never split.
        let (thought, content) = ids.settle_tool_reply(&FakeDecoder, false, "", &stream);
        assert_eq!(thought, None);
        assert_eq!(content, "Hicall");
    }

    #[test]
    fn stabilizer_skips_quoted_stops_and_carries_parity() {
        const QUOTE: u32 = 7;
        const STOP: u32 = 99;
        let mut st = Stabilizer::new(vec![STOP], Some(QUOTE), true);
        // Block 1 commits with an OPEN quote run (one quote id).
        let b1 = [b'A' as u32, QUOTE, b'B' as u32];
        let sp = st.on_step(&step(1, 1, &b1, true));
        assert_eq!(sp.ids, b1);
        assert!(!sp.hit_stop);
        // Block 2: still inside the quote → the stop id is literal content.
        let b2 = [STOP, QUOTE, STOP];
        let sp = st.on_step(&step(2, 1, &b2, true));
        assert!(sp.new_block);
        assert_eq!(sp.ids, vec![STOP, QUOTE]);
        assert!(sp.hit_stop); // the second STOP falls outside the closed quote
    }
}
