//! [`DiffusionStreamMapper`]: the serve stream decoder. It turns the raw
//! per-step canvas argmax stream into [`WireDelta`]s, splitting the thought
//! channel from the answer and separating block-committed text from the
//! speculative draft. No GPU, no sockets, so it is unit-tested directly. The
//! OpenAI SSE envelope framing that consumes these deltas lives in
//! `server::wire`.

use super::{ChannelIds, Stabilizer, StepProgressEvent, TextDecoder};

/// One streamed delta produced by the mapper, before OpenAI-envelope framing.
#[derive(Debug, Clone, PartialEq)]
pub enum WireDelta {
    /// Append to `reasoning_content` (the private thought channel).
    Reasoning(String),
    /// Append to `content` (block-committed, immutable answer text).
    Content(String),
    /// Replace the speculative draft, the shimmering still-converging answer.
    Draft {
        text: String,
        committed: usize,
        block: usize,
        step: u32,
    },
}

/// Options for a [`DiffusionStreamMapper`], all defaulting to off or none so a
/// caller sets only what differs from a plain content-only stream.
#[derive(Default)]
pub struct MapperConfig {
    pub stops: Vec<u32>,
    pub channel_open: Option<u32>,
    pub channel_close: Option<u32>,
    pub quote: Option<u32>,
    pub stop_skip_quoted: bool,
    pub thinking: bool,
    pub emit_drafts: bool,
    pub paced: bool,
}

pub struct DiffusionStreamMapper<D: TextDecoder> {
    decoder: D,
    /// The shared stable-prefix front end (streak tracking, stop cut, quoted-
    /// stop skipping), one implementation with the terminal chat decoder.
    stab: Stabilizer,
    /// Channel/quote special ids for the shared token-level thought split.
    channels: ChannelIds,
    thinking: bool,
    emit_drafts: bool,
    /// The generation begins inside an already-open thought channel whose open
    /// marker was in the prompt (a tool round's forced-open thought). The split
    /// must classify the leading ids as reasoning even though no open marker
    /// appears in the stream.
    start_in_thought: bool,

    /// Block-committed token ids accumulated across all committed blocks.
    committed_ids: Vec<u32>,
    /// Full committed reasoning / content text (the split of `committed_ids`).
    /// With pacing on, deltas may lag these, while [`content`](Self::content)
    /// and [`reasoning`](Self::reasoning) always return the full committed text.
    emitted_reasoning: String,
    emitted_content: String,
    /// Last draft text emitted (dedupe replace-semantics chunks).
    last_draft: String,
    ended: bool,

    /// Paced release: hold a committed block's text and dribble it out during
    /// the next block's denoise (paced by its step progress against an EMA of
    /// block step counts) so multi-block turns stream smoothly instead of
    /// bursting a block every N seconds. Everything flushes at turn end or the
    /// moment a tool call appears in committed content, since agents consume
    /// tool turns whole and must never be delayed.
    paced: bool,
    /// Byte cursors into `emitted_reasoning` / `emitted_content` already
    /// released as deltas (equal to the full lengths when pacing is off).
    released_reasoning: usize,
    released_content: usize,
    /// EMA of committed blocks' step counts, the pacing denominator.
    est_block_steps: f32,
    /// Pacing permanently off for this turn (tool call seen).
    pace_off: bool,
}

/// Reasoning (thought-channel) and answer text split out of a committed id list.
struct Split {
    reasoning: String,
    content: String,
}

impl<D: TextDecoder> DiffusionStreamMapper<D> {
    pub fn new(decoder: D, cfg: MapperConfig) -> Self {
        Self {
            decoder,
            stab: Stabilizer::new(cfg.stops, cfg.quote, cfg.stop_skip_quoted),
            channels: ChannelIds {
                open: cfg.channel_open,
                close: cfg.channel_close,
                quote: cfg.quote,
            },
            thinking: cfg.thinking,
            emit_drafts: cfg.emit_drafts,
            start_in_thought: false,
            committed_ids: Vec::new(),
            emitted_reasoning: String::new(),
            emitted_content: String::new(),
            last_draft: String::new(),
            ended: false,
            paced: cfg.paced,
            released_reasoning: 0,
            released_content: 0,
            est_block_steps: 16.0,
            pace_off: false,
        }
    }

    /// The generation begins inside an already-open thought (the open marker was
    /// in the prompt). Only meaningful with `thinking` on, since the forced
    /// opener is seeded only on a thinking request.
    pub fn starting_in_thought(mut self) -> Self {
        self.start_in_thought = true;
        self
    }

    /// Committed (immutable) answer text emitted so far.
    pub fn content(&self) -> &str {
        &self.emitted_content
    }

    /// Committed reasoning text emitted so far.
    pub fn reasoning(&self) -> &str {
        &self.emitted_reasoning
    }

    #[allow(dead_code)]
    pub fn ended(&self) -> bool {
        self.ended
    }

    /// Split committed ids into (reasoning, content) via the shared token-level
    /// walk ([`ChannelIds::split`]). With thinking off, everything is content.
    fn split(&self, ids: &[u32]) -> Split {
        if !self.thinking {
            return Split {
                reasoning: String::new(),
                content: self.decoder.decode_content(ids),
            };
        }
        let split = self
            .channels
            .split(&self.decoder, ids, self.start_in_thought);
        Split {
            reasoning: self.clean_reasoning(&split.reasoning),
            content: self.decoder.decode_content(&split.content),
        }
    }

    /// Decode a reasoning-region id slice and strip the `<|channel>thought\n`
    /// opener scaffold the model emits when thinking is enabled.
    ///
    /// Mid-thought re-open ids arrive already consumed by the `split` walk, so
    /// the string replaces below catch the text-form (BPE) markers. Display
    /// only: quoted literals in a rehearsed call may lose markers here, but
    /// repair and salvage read the raw decode rather than this.
    fn clean_reasoning(&self, ids: &[u32]) -> String {
        let raw = self
            .decoder
            .decode(&crate::sample::strip_degenerate_token_ids(ids));
        let mut s = raw
            .trim_start_matches("<|channel>")
            .trim_start()
            .trim_start_matches("thought")
            .trim()
            .to_string();
        // Text-form markers (BPE of the scaffold, or decoded specials).
        for marker in [
            crate::tools::OPEN_THOUGHT,
            "<|channel>thought",
            "<|channel>",
            "<channel|>",
        ] {
            s = s.replace(marker, "");
        }
        s.trim().to_string()
    }

    /// The current speculative answer draft: the stable-prefix answer region,
    /// including the still-converging tail, sanitized to visible text.
    fn draft_text(&self, stable_ids: &[u32]) -> String {
        let mut all = self.committed_ids.clone();
        all.extend_from_slice(stable_ids);
        self.split(&all).content
    }

    pub fn on_step(&mut self, ev: &StepProgressEvent<'_>) -> Vec<WireDelta> {
        let mut out = Vec::new();
        if self.ended {
            return out;
        }

        let sp = self.stab.on_step(ev);
        let (stable_ids, hit_stop) = (sp.ids, sp.hit_stop);

        // Draft (replace-semantics), emitted every step it changes.
        if self.emit_drafts {
            let text = self.draft_text(&stable_ids);
            if text != self.last_draft {
                out.push(WireDelta::Draft {
                    committed: self.emitted_content.len(),
                    block: ev.block_idx,
                    step: ev.step_in_block,
                    text: text.clone(),
                });
                self.last_draft = text;
            }
        }

        // Commit: fold this block's stable ids into the committed stream.
        // Unpaced (or flush conditions): emit the newly-committed text now.
        // Paced: hold it and let the per-step release below dribble it out
        // during the next block's denoise.
        if ev.block_done {
            self.committed_ids.extend_from_slice(&stable_ids);
            let split = self.split(&self.committed_ids);
            // A rare non-prefix revision invalidates the release cursors.
            if !split.reasoning.starts_with(&self.emitted_reasoning) {
                self.released_reasoning = 0;
            }
            if !split.content.starts_with(&self.emitted_content) {
                self.released_content = 0;
            }
            self.emitted_reasoning = split.reasoning;
            self.emitted_content = split.content;
            // EMA of block step counts, the pacing denominator.
            self.est_block_steps = 0.5 * self.est_block_steps + 0.5 * ev.step_in_block as f32;
            // Premature stop inside an unfinished tool turn (open call, or
            // trailing prose after a closed call): keep the mapper open.
            if hit_stop && !crate::tools::should_continue_past_stop(&self.emitted_content) {
                self.ended = true;
            }
            // Tool call in committed content: agents consume tool turns whole,
            // so flush and stop pacing for the rest of the turn.
            if self.emitted_content.contains("call:") {
                self.pace_off = true;
            }
            if !self.paced || self.pace_off || self.ended {
                self.release_up_to(usize::MAX, usize::MAX, &mut out);
            }
        } else if self.paced && !self.pace_off {
            // Paced release: fraction of the held text proportional to this
            // block's step progress against the estimated block length, so the
            // previous block's text finishes streaming roughly when this block
            // commits.
            let frac = (ev.step_in_block as f32 / self.est_block_steps.max(1.0)).clamp(0.0, 1.0);
            let target = |released: usize, full: usize| {
                released + ((full - released) as f32 * frac) as usize
            };
            self.release_up_to(
                target(self.released_reasoning, self.emitted_reasoning.len()),
                target(self.released_content, self.emitted_content.len()),
                &mut out,
            );
        }
        out
    }

    /// Emit append deltas advancing the release cursors toward the byte targets
    /// (clamped to full length, snapped back to char boundaries).
    fn release_up_to(
        &mut self,
        reasoning_target: usize,
        content_target: usize,
        out: &mut Vec<WireDelta>,
    ) {
        let cut = |s: &str, mut t: usize| -> usize {
            t = t.min(s.len());
            while t < s.len() && !s.is_char_boundary(t) {
                t -= 1;
            }
            t
        };
        let rt = cut(&self.emitted_reasoning, reasoning_target);
        if rt > self.released_reasoning {
            out.push(WireDelta::Reasoning(
                self.emitted_reasoning[self.released_reasoning..rt].to_string(),
            ));
            self.released_reasoning = rt;
        }
        let ct = cut(&self.emitted_content, content_target);
        if ct > self.released_content {
            out.push(WireDelta::Content(
                self.emitted_content[self.released_content..ct].to_string(),
            ));
            self.released_content = ct;
        }
    }

    /// Release everything still held at the end of generation, so a turn that
    /// ends without a stop token does not strand paced text.
    pub fn final_flush(&mut self) -> Vec<WireDelta> {
        let mut out = Vec::new();
        self.release_up_to(usize::MAX, usize::MAX, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{FakeDecoder, step};
    use super::*;

    fn c(ch: char) -> u32 {
        ch as u32
    }

    fn draft_of(deltas: &[WireDelta]) -> Option<String> {
        deltas.iter().find_map(|d| match d {
            WireDelta::Draft { text, .. } => Some(text.clone()),
            _ => None,
        })
    }

    /// Channels 1/2/4 with thinking on, the common setup for the split tests.
    fn thinking_cfg() -> MapperConfig {
        MapperConfig {
            channel_open: Some(1),
            channel_close: Some(2),
            thinking: true,
            ..Default::default()
        }
    }

    /// Decoder that surfaces channel specials as their literal markup (like the
    /// real SentencePiece specials), so mid-thought re-opens stay detectable.
    struct ChannelDecoder;
    impl TextDecoder for ChannelDecoder {
        fn decode(&self, ids: &[u32]) -> String {
            let mut s = String::new();
            self.decode_append(&mut s, ids);
            s
        }
        fn decode_append(&self, out: &mut String, ids: &[u32]) {
            for &id in ids {
                match id {
                    1 => out.push_str("<|channel>"),
                    2 => out.push_str("<channel|>"),
                    3 => out.push_str("thought"),
                    4 => out.push_str("<|\"|>"),
                    10 => out.push('\n'),
                    id if (32..127).contains(&id) => out.push(id as u8 as char),
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn content_only_commits_answer_text() {
        let mut m = DiffusionStreamMapper::new(FakeDecoder, MapperConfig::default());
        let hi = [c('H'), c('i')];
        // Not committed yet: no content deltas.
        assert!(m.on_step(&step(1, 1, &hi, false)).is_empty());
        // Commit the block: content delta "Hi".
        let out = m.on_step(&step(1, 2, &hi, true));
        assert_eq!(out, vec![WireDelta::Content("Hi".to_string())]);
        assert_eq!(m.content(), "Hi");
    }

    #[test]
    fn draft_streams_stable_prefix_before_commit() {
        let mut m = DiffusionStreamMapper::new(
            FakeDecoder,
            MapperConfig {
                emit_drafts: true,
                ..Default::default()
            },
        );
        let hi = [c('H'), c('i')];
        // Streak must reach STABLE_STREAK before the draft shows anything.
        assert!(draft_of(&m.on_step(&step(1, 1, &hi, false))).is_none());
        let _ = m.on_step(&step(1, 2, &hi, false));
        let d = draft_of(&m.on_step(&step(1, 3, &hi, false)));
        assert_eq!(d.as_deref(), Some("Hi"));
    }

    #[test]
    fn thinking_splits_reasoning_from_content() {
        // Canvas: <open> t h o u g h t <close> H i
        let mut m = DiffusionStreamMapper::new(FakeDecoder, thinking_cfg());
        let canvas = [
            1,
            c('t'),
            c('h'),
            c('o'),
            c('u'),
            c('g'),
            c('h'),
            c('t'),
            2,
            c('H'),
            c('i'),
        ];
        let out = m.on_step(&step(1, 2, &canvas, true));
        // "thought" opener scaffold stripped from reasoning, answer is "Hi".
        assert!(out.contains(&WireDelta::Content("Hi".to_string())));
        assert_eq!(m.content(), "Hi");
        assert!(m.reasoning().is_empty() || !m.reasoning().contains("Hi"));
    }

    #[test]
    fn thinking_drops_trailing_channel_close_from_content() {
        // Real log shape: <open> ... <close> Answer <close> (extra close before tools).
        let mut m = DiffusionStreamMapper::new(FakeDecoder, thinking_cfg());
        let canvas = [
            1,
            c('x'),
            2,
            c('O'),
            c('k'),
            2, // must not leak into content
            c('!'),
        ];
        let _ = m.on_step(&step(1, 2, &canvas, true));
        assert_eq!(m.content(), "Ok!");
    }

    #[test]
    fn thinking_strips_nested_channel_reopen_from_reasoning() {
        // OpenCode nests a Thought UI when `<|channel>thought` appears mid-reasoning.
        let thought = 3u32;
        let mut m = DiffusionStreamMapper::new(ChannelDecoder, thinking_cfg());
        // <open> thought \n Hi <open> thought \n ! <close> A
        let canvas = [
            1,
            thought,
            c('\n'),
            c('H'),
            c('i'),
            1,
            thought,
            c('\n'),
            c('!'),
            2,
            c('A'),
        ];
        let _ = m.on_step(&step(1, 2, &canvas, true));
        assert!(
            !m.reasoning().contains("<|channel>"),
            "nested open leaked: {:?}",
            m.reasoning()
        );
        assert!(
            !m.reasoning().contains("<channel|>"),
            "close leaked: {:?}",
            m.reasoning()
        );
        // The re-open's name token must go with it, or a bare "thought" line
        // leaks into the client's Thought UI.
        assert!(
            !m.reasoning().contains("thought"),
            "channel name leaked: {:?}",
            m.reasoning()
        );
        assert_eq!(m.content(), "A");
    }

    /// Thinking mode on, but the model skips the thought ceremony and answers
    /// with a bare tool call, no channel markers anywhere. Classification
    /// follows emission: with no thought span, everything is content.
    #[test]
    fn thinking_bare_reply_without_thought_span_is_content() {
        let mut m = DiffusionStreamMapper::new(FakeDecoder, thinking_cfg());
        let canvas: Vec<u32> = "call:edit{x:1}".chars().map(|ch| ch as u32).collect();
        let _ = m.on_step(&step(1, 2, &canvas, true));
        assert_eq!(m.content(), "call:edit{x:1}");
        assert_eq!(m.reasoning(), "");
    }

    /// Tokens before the first explicit thought open are content (turn-opener
    /// ceremony), classified by emission order.
    #[test]
    fn thinking_pre_open_tokens_stay_content() {
        let mut m = DiffusionStreamMapper::new(FakeDecoder, thinking_cfg());
        let canvas = [c('A'), 1, c('z'), 2, c('B')];
        let _ = m.on_step(&step(1, 2, &canvas, true));
        assert_eq!(m.content(), "AB");
        assert!(
            m.reasoning().contains('z'),
            "thought span lost: {:?}",
            m.reasoning()
        );
    }

    /// A channel-open id inside a `<|"|>` quote run is literal tool-arg content:
    /// it must stay in content verbatim (not toggle the thought state, not get
    /// filtered) so the parsed call's args round-trip byte-exact.
    #[test]
    fn quoted_channel_id_stays_in_content_verbatim() {
        let quote = 4u32;
        let mut m = DiffusionStreamMapper::new(
            ChannelDecoder,
            MapperConfig {
                quote: Some(quote),
                ..thinking_cfg()
            },
        );
        // call:e{x:<Q> <open> <Q>}, where the quoted open is data.
        let mut canvas: Vec<u32> = "call:e{x:".chars().map(|ch| ch as u32).collect();
        canvas.extend([quote, 1, quote]);
        canvas.push(c('}'));
        let _ = m.on_step(&step(1, 2, &canvas, true));
        assert_eq!(m.content(), "call:e{x:<|\"|><|channel><|\"|>}");
        assert_eq!(m.reasoning(), "");
        let calls = crate::tools::parse_tool_calls(m.content());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["x"].as_str().unwrap(), "<|channel>");
    }

    /// With stop-skip enabled (mirroring the engine's continue-past-stop
    /// policy), a stop id inside an open quote run is content rather than a
    /// stop, including when the quote opened in an earlier committed block.
    #[test]
    fn quoted_stop_id_streams_through_when_skip_enabled() {
        let quote = 4u32;
        let mk = |skip: bool| {
            DiffusionStreamMapper::new(
                ChannelDecoder,
                MapperConfig {
                    stops: vec![99],
                    quote: Some(quote),
                    stop_skip_quoted: skip,
                    ..Default::default()
                },
            )
        };
        // Block 1 leaves the quote open, block 2 carries the stop id inside it.
        let block1: Vec<u32> = vec![c('a'), quote, c('b')];
        let block2: Vec<u32> = vec![c('c'), 99, quote, c('d')];
        let mut m = mk(true);
        let _ = m.on_step(&step(1, 2, &block1, true));
        let _ = m.on_step(&step(2, 2, &block2, true));
        assert!(!m.ended(), "quoted stop must not end the stream");
        assert!(m.content().contains('d'), "content: {:?}", m.content());
        // Same stream without the skip: the stop cuts inside the quote.
        let mut m = mk(false);
        let _ = m.on_step(&step(1, 2, &block1, true));
        let _ = m.on_step(&step(2, 2, &block2, true));
        assert!(m.ended());
        assert!(!m.content().contains('d'));
    }

    #[test]
    fn stop_token_ends_and_cuts() {
        let mut m = DiffusionStreamMapper::new(
            FakeDecoder,
            MapperConfig {
                stops: vec![99],
                ..Default::default()
            },
        );
        let canvas = [c('O'), c('k'), 99, c('X')];
        let out = m.on_step(&step(1, 2, &canvas, true));
        assert_eq!(out, vec![WireDelta::Content("Ok".to_string())]);
        assert!(m.ended());
        // Further steps are inert.
        assert!(m.on_step(&step(1, 3, &canvas, true)).is_empty());
    }

    #[test]
    fn paced_stream_holds_commit_and_releases_during_next_block() {
        let paced = || {
            MapperConfig {
                paced: true,
                ..Default::default()
            }
        };
        let mut m = DiffusionStreamMapper::new(FakeDecoder, paced());
        let text: Vec<u32> = "abcdefghij".chars().map(|ch| ch as u32).collect();
        // Block 1 commits after 16 steps (seeds the EMA at (16+16)/2 = 16), and
        // paced mode holds the text, so no Content delta at commit.
        let out = m.on_step(&step(1, 16, &text, true));
        assert!(
            out.iter().all(|d| !matches!(d, WireDelta::Content(_))),
            "paced commit must hold content: {out:?}"
        );
        // content() still reports the full committed text (final-response path).
        assert_eq!(m.content(), "abcdefghij");
        // Block 2 progresses: half the estimated steps releases ~half the text.
        let tail = [c('z')];
        let mut streamed = String::new();
        for s in 1..=8 {
            for d in m.on_step(&step(2, s, &tail, false)) {
                if let WireDelta::Content(t) = d {
                    streamed.push_str(&t);
                }
            }
        }
        assert!(
            !streamed.is_empty() && streamed.len() < 10,
            "partial release expected, got {streamed:?}"
        );
        assert!("abcdefghij".starts_with(&streamed));
        // Block 2 commits with a stop, the turn ends, and everything flushes.
        let stop = [c('z'), 999];
        let mut m2 = DiffusionStreamMapper::new(
            FakeDecoder,
            MapperConfig {
                stops: vec![999],
                ..paced()
            },
        );
        let _ = m2.on_step(&step(1, 16, &text, true));
        let out = m2.on_step(&step(2, 4, &stop, true));
        let content: String = out
            .iter()
            .filter_map(|d| match d {
                WireDelta::Content(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(content, "abcdefghijz", "turn end must flush all held text");
    }

    #[test]
    fn paced_stream_final_flush_releases_held_text() {
        let mut m = DiffusionStreamMapper::new(
            FakeDecoder,
            MapperConfig {
                paced: true,
                ..Default::default()
            },
        );
        let text: Vec<u32> = "hold me".chars().map(|ch| ch as u32).collect();
        let _ = m.on_step(&step(1, 16, &text, true));
        // Turn ends without a stop token (budget): final_flush drains the rest.
        let out = m.final_flush();
        let content: String = out
            .iter()
            .filter_map(|d| match d {
                WireDelta::Content(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(content, "hold me");
        assert!(m.final_flush().is_empty(), "second flush is a no-op");
    }

    #[test]
    fn paced_stream_tool_call_disables_pacing() {
        let mut m = DiffusionStreamMapper::new(
            FakeDecoder,
            MapperConfig {
                paced: true,
                ..Default::default()
            },
        );
        let text: Vec<u32> = "call:read{".chars().map(|ch| ch as u32).collect();
        // A committed block containing a tool call flushes immediately.
        let out = m.on_step(&step(1, 8, &text, true));
        let content: String = out
            .iter()
            .filter_map(|d| match d {
                WireDelta::Content(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(content, "call:read{");
        // Pacing stays off: the next block's commit also emits directly.
        let more: Vec<u32> = "}".chars().map(|ch| ch as u32).collect();
        let out = m.on_step(&step(2, 4, &more, true));
        assert!(
            out.contains(&WireDelta::Content("}".to_string())),
            "pace_off must persist: {out:?}"
        );
    }

    /// A tool round whose prompt seeds the thought open (`start_in_thought`):
    /// the leading ids are reasoning even with no open marker in the stream, and
    /// content starts after the model's `<channel|>`.
    #[test]
    fn start_in_thought_classifies_leading_ids_as_reasoning() {
        let mut m =
            DiffusionStreamMapper::new(FakeDecoder, thinking_cfg()).starting_in_thought();
        let canvas = [c('p'), c('l'), c('a'), c('n'), 2, c('O'), c('k')];
        let _ = m.on_step(&step(1, 2, &canvas, true));
        assert_eq!(m.reasoning(), "plan");
        assert_eq!(m.content(), "Ok");
    }
}
