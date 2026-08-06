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
            self.committed_text = self.decoder.decode_content(&self.committed_ids);
            out.push(ChatEvent::BlockCommit {
                block: ev.block_idx,
                committed_tokens: sp.ids.len(),
            });
            if sp.hit_stop {
                self.ended = true;
            }
        }

        // Full visible text: committed plus this block's speculative draft.
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
