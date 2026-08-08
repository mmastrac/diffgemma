//! [`StreamDecoder`]: the per-step canvas stream to [`ChatEvent`] mapper that
//! drives the interactive terminal renderer.

use super::{ChannelIds, Stabilizer, StepProgressEvent, TextDecoder};
use crate::chat::ChatEvent;

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

/// Converts the raw per-step canvas argmax stream into `ChatEvent`s through a
/// [`Stabilizer`] front end, a token-level [`ChannelIds::split`] for turns that
/// surface their reasoning, and text diffing into `Thought`/`Text`/`Rewind`
/// events. The committed text is rebuilt once per block, keeping per-step work
/// at O(canvas) rather than O(total reply).
pub struct StreamDecoder<D: TextDecoder> {
    decoder: D,
    channels: ChannelIds,
    stab: Stabilizer,
    committed_ids: Vec<u32>,
    committed_text: String,
    ended: bool,
    last_text: String,
    /// Surface the reasoning live: emit `Thought` events for the thought-channel
    /// content and treat only the post-`<channel|>` text as the answer. Set by
    /// `--show-thinking` and implied by a `/prethink` seed. Off by default, when
    /// the thought is stripped silently and only the answer streams.
    thinking_display: bool,
    /// Last surfaced thought text, for `Thought`-event dedup.
    last_thought: String,
    /// Present on a `/prethink` turn, whose generation runs inside an
    /// already-open thought channel, so the surfaced reasoning is
    /// `seed + thought-so-far` up to the model's `<channel|>`. A plain
    /// `--show-thinking` turn opens its own thought channel and leaves this
    /// `None`.
    prethink_seed: Option<String>,
    /// Generation begins inside an already-open thought (a `/prethink`
    /// continuation or a tool round's forced-open thought). The split classifies
    /// the leading ids as reasoning regardless of whether display is on.
    start_in_thought: bool,
    /// Paced release: hold a finished block's committed answer and dribble it out
    /// over the next block's denoise (a fraction of the remaining length each
    /// step, against an EMA of block step counts) so a multi-block answer types
    /// out smoothly instead of a whole block landing every couple of seconds. The
    /// serve mapper does this for the wire; this is its terminal-decoder twin
    /// (see [`super::serve`]). Off by default, enabled per turn via
    /// [`paced`](Self::paced). The final block has no successor to pace against,
    /// so it lands whole at `finish`.
    paced: bool,
    /// Bytes of the committed answer already revealed as `Text` (equal to the
    /// committed length when pacing is off).
    released_content: usize,
    /// EMA of committed blocks' step counts, the pacing denominator.
    est_block_steps: f32,
    /// Pacing off for the rest of this turn: a tool call was committed, and an
    /// agent consumes a tool round whole, so it must never wait on a dribble.
    pace_off: bool,
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
            start_in_thought: false,
            paced: false,
            released_content: 0,
            est_block_steps: 16.0,
            pace_off: false,
        }
    }

    /// Bind the model's channel/quote special ids, required for a turn that
    /// surfaces its reasoning. Without them everything streams as answer.
    pub(crate) fn with_channels(mut self, channels: ChannelIds) -> Self {
        self.channels = channels;
        self
    }

    /// Surface reasoning live: emit `Thought` events and split the answer at the
    /// model's `<channel|>`. `seed` is the `/prethink` continuation prefix (the
    /// reasoning the prompt already opened with), or `None` for a plain
    /// `--show-thinking` turn that opens its own thought channel.
    pub fn with_thinking(mut self, show_thinking: bool, seed: Option<String>) -> Self {
        self.thinking_display = show_thinking || seed.is_some();
        self.start_in_thought = self.start_in_thought || seed.is_some();
        self.prethink_seed = seed;
        self
    }

    /// The generation begins inside an already-open thought (the open marker was
    /// in the prompt). Reasoning is split out from the first id, and silently
    /// discarded unless display is on.
    pub fn starting_in_thought(mut self) -> Self {
        self.start_in_thought = true;
        self
    }

    /// Dribble each committed block's answer out over the next block's denoise
    /// rather than surfacing it whole, so the answer types out smoothly. Mirrors
    /// the serve mapper's paced release ([`super::serve`]); the final block still
    /// lands at `finish`, having no successor to pace against.
    pub fn paced(mut self, paced: bool) -> Self {
        self.paced = paced;
        self
    }

    /// The immutable committed reply text so far.
    #[allow(dead_code)]
    pub fn committed_text(&self) -> &str {
        &self.committed_text
    }

    /// Cap the visible answer to the paced release cursor and advance it. Unpaced
    /// (or once a tool call is committed) this reveals the whole draft, exactly as
    /// before. Paced, it holds a finished block and releases a step-fraction of
    /// the remaining committed length each following step, so the previous block
    /// finishes streaming roughly as this one commits. Returns the
    /// `(committed_prefix_len, visible_text)` for the `Text` event.
    fn paced_reveal(
        &mut self,
        committed_answer: &str,
        draft_answer: &str,
        ev: &StepProgressEvent<'_>,
    ) -> (usize, String) {
        if self.paced && !self.pace_off && committed_answer.contains("call:") {
            self.pace_off = true;
        }
        if ev.block_done {
            // The block's true length is now known; fold it into the EMA that
            // sets how fast the next block releases this one's text.
            self.est_block_steps = 0.5 * self.est_block_steps + 0.5 * ev.step_in_block as f32;
        }
        if !self.paced || self.pace_off {
            self.released_content = committed_answer.len();
            return (
                common_prefix_len(committed_answer, draft_answer),
                draft_answer.to_string(),
            );
        }
        // Paced. The cursor only ever trails the committed length: a block's own
        // finishing step holds it (nothing new to pace against yet) and the
        // following block's steps carry it forward.
        self.released_content = self.released_content.min(committed_answer.len());
        if !ev.block_done {
            let frac = (ev.step_in_block as f32 / self.est_block_steps.max(1.0)).clamp(0.0, 1.0);
            let remaining = committed_answer.len() - self.released_content;
            self.released_content += (remaining as f32 * frac) as usize;
        }
        // Snap back to a char boundary so the slice never splits a code point.
        while self.released_content < committed_answer.len()
            && !committed_answer.is_char_boundary(self.released_content)
        {
            self.released_content -= 1;
        }
        (
            self.released_content,
            committed_answer[..self.released_content].to_string(),
        )
    }

    /// Emit a `Text` (preceded by a `Rewind` when the new text revises rather
    /// than extends what streamed), and remember it for the next diff.
    fn emit_text(&mut self, out: &mut Vec<ChatEvent>, committed: usize, text: String) {
        if text == self.last_text {
            return;
        }
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

        // Reasoning-separating path (`--show-thinking`, `/prethink`, or
        // start-in-thought): re-split the whole generation each step so a thought
        // spanning several blocks still classifies. Reasoning streams as
        // `Thought` (when displayed), the answer after `<channel|>` as `Text`.
        if self.thinking_display || self.start_in_thought {
            if ev.block_done {
                self.committed_ids.extend_from_slice(&sp.ids);
                out.push(ChatEvent::BlockCommit {
                    block: ev.block_idx,
                    committed_tokens: sp.ids.len(),
                });
                if sp.hit_stop {
                    self.ended = true;
                }
            }
            let seed = self.prethink_seed.as_deref().unwrap_or("");
            let in_thought = self.start_in_thought;
            let strip = crate::sample::strip_degenerate_token_ids;
            // Committed view: finalized blocks only, whose answer is the
            // immutable prefix. Full view: committed plus the live draft.
            // Deriving `committed` from the committed view keeps it monotonic
            // across blocks, so the renderer holds its print cursor mid-answer
            // instead of re-emitting the whole reply on each block boundary.
            let committed_split =
                self.channels
                    .split(&self.decoder, &strip(&self.committed_ids), in_thought);
            let committed_answer = self.decoder.decode_content(&committed_split.content);
            let full_split = if ev.block_done {
                committed_split
            } else {
                let mut ids = self.committed_ids.clone();
                ids.extend_from_slice(&sp.ids);
                self.channels.split(&self.decoder, &strip(&ids), in_thought)
            };
            let thought = format!(
                "{seed}{}",
                self.decoder.decode_reasoning(&full_split.reasoning)
            );
            let answer = self.decoder.decode_content(&full_split.content);

            if self.thinking_display && thought != self.last_thought {
                out.push(ChatEvent::Thought {
                    text: thought.clone(),
                });
                self.last_thought = thought;
            }
            let (committed, text) = self.paced_reveal(&committed_answer, &answer, ev);
            self.emit_text(&mut out, committed, text);
            return out;
        }

        if ev.block_done {
            self.committed_ids.extend_from_slice(&sp.ids);
            self.committed_text = self.decoder.decode_content(&self.committed_ids);
            out.push(ChatEvent::BlockCommit {
                block: ev.block_idx,
                committed_tokens: sp.ids.len(),
            });
            if sp.hit_stop {
                self.ended = true;
            }
        }

        // Full visible text: the committed answer plus this block's speculative
        // draft. Cloned up front so the paced-reveal borrow does not collide with
        // the committed field.
        let committed_answer = self.committed_text.clone();
        let mut draft = committed_answer.clone();
        if !ev.block_done {
            let suffix = crate::sample::strip_degenerate_token_ids(&sp.ids);
            if !suffix.is_empty() {
                self.decoder.decode_append(&mut draft, &suffix);
                draft = crate::chat_template::sanitize_model_reply(&draft);
            }
        }
        let (committed, text) = self.paced_reveal(&committed_answer, &draft, ev);
        self.emit_text(&mut out, committed, text);
        out
    }
}
