//! [`StreamDecoder`]: the chat adapter over the canonical [`DiffusionStream`].
//! It turns the stream's committed/draft [`StreamEvent`]s into the renderer's
//! [`ChatEvent`]s. Only the committed answer is paced (through a [`PaceClock`])
//! so it types out; reasoning surfaces whole as `Thought`. The stream's
//! `text[..committed]` never-taken-back guarantee is what the renderer flushes
//! into scrollback.

use super::{
    Channel, ChannelIds, DiffusionStream, PaceClock, StepProgressEvent, StreamConfig, StreamEvent,
    TextDecoder,
};
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

pub struct StreamDecoder<D: TextDecoder> {
    stream: DiffusionStream<D>,
    /// Paces the committed answer's reveal so it types out. Reasoning is not
    /// paced; it surfaces whole.
    clock: PaceClock,
    paced: bool,
    /// Bytes of the committed answer already revealed as `Text`.
    released: usize,
    /// Emit `Thought` events (`--show-thinking` / a `/prethink` seed). Off keeps
    /// the reasoning split but discards it, streaming only the answer.
    thinking_display: bool,
    start_in_thought: bool,
    /// A `/prethink` seed prepended to surfaced reasoning.
    prethink_seed: Option<String>,
    last_text: String,
    last_thought: String,
    ended: bool,
}

impl<D: TextDecoder> StreamDecoder<D> {
    pub fn new(decoder: D, stops: Vec<u32>) -> Self {
        Self {
            stream: DiffusionStream::new(
                decoder,
                StreamConfig {
                    stops,
                    ..Default::default()
                },
            ),
            clock: PaceClock::new(),
            paced: false,
            released: 0,
            thinking_display: false,
            start_in_thought: false,
            prethink_seed: None,
            last_text: String::new(),
            last_thought: String::new(),
            ended: false,
        }
    }

    /// Bind the model's channel/quote special ids, required for a turn that
    /// surfaces its reasoning. Without them everything streams as answer.
    pub(crate) fn with_channels(mut self, channels: ChannelIds) -> Self {
        self.stream.set_channels(channels);
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
        self.sync_split();
        self
    }

    /// The generation begins inside an already-open thought (the open marker was
    /// in the prompt). Reasoning is split out from the first id, and silently
    /// discarded unless display is on.
    pub fn starting_in_thought(mut self) -> Self {
        self.start_in_thought = true;
        self.sync_split();
        self
    }

    /// Dribble each committed block's answer out over the next block's denoise
    /// rather than surfacing it whole, so the answer types out smoothly. The
    /// final block has no successor to pace against, so it lands whole at
    /// `finish`.
    pub fn paced(mut self, paced: bool) -> Self {
        self.paced = paced;
        self
    }

    /// The immutable committed reply text so far.
    #[allow(dead_code)]
    pub fn committed_text(&self) -> &str {
        self.stream.content()
    }

    /// The stream splits reasoning out whenever the answer is displayed or the
    /// turn began in a thought (so hidden reasoning is still separated).
    fn sync_split(&mut self) {
        self.stream
            .set_thinking(self.thinking_display || self.start_in_thought);
        self.stream.set_start_in_thought(self.start_in_thought);
    }

    /// Reveal the committed answer as the renderer's `Text`. Paced advances the
    /// release cursor a step-fraction each call (so the answer types out even on
    /// a step the model text did not change); unpaced reveals the whole draft,
    /// and only when new content arrived. `full` is the latest content draft, if
    /// one was emitted this step.
    fn reveal_content(
        &mut self,
        out: &mut Vec<ChatEvent>,
        full: Option<String>,
        ev: &StepProgressEvent<'_>,
    ) {
        let committed = self.stream.content().to_string();
        let (out_committed, out_text) = if self.paced && !self.clock.pace_off() {
            self.released = self.clock.target(
                true,
                ev.block_done,
                ev.step_in_block,
                self.released,
                &committed,
            );
            (self.released, committed[..self.released].to_string())
        } else {
            self.released = committed.len();
            match full {
                Some(full) => (common_prefix_len(&committed, &full), full),
                None => return,
            }
        };
        if out_text == self.last_text {
            return;
        }
        if !out_text.starts_with(&self.last_text) {
            out.push(ChatEvent::Rewind {
                common: common_prefix_len(&self.last_text, &out_text),
            });
        }
        out.push(ChatEvent::Text {
            committed: out_committed,
            text: out_text.clone(),
        });
        self.last_text = out_text;
    }

    pub fn on_step(&mut self, ev: &StepProgressEvent<'_>) -> Vec<ChatEvent> {
        let mut out = Vec::new();
        if self.ended {
            return out;
        }
        let mut full_content = None;
        for e in self.stream.on_step(ev) {
            match e {
                StreamEvent::Block { block } => out.push(ChatEvent::BlockStart { block }),
                StreamEvent::Progress {
                    block,
                    step,
                    max_steps,
                    canvas,
                    locked,
                    accepted,
                } => out.push(ChatEvent::Status {
                    block,
                    step,
                    max_steps,
                    accepted,
                    canvas,
                    locked,
                }),
                StreamEvent::Commit { block, tokens } => {
                    out.push(ChatEvent::BlockCommit {
                        block,
                        committed_tokens: tokens,
                    });
                    self.clock
                        .on_commit(ev.step_in_block, self.stream.content());
                }
                StreamEvent::Text {
                    channel: Channel::Reasoning,
                    text,
                    ..
                } => {
                    if self.thinking_display {
                        let thought =
                            format!("{}{}", self.prethink_seed.as_deref().unwrap_or(""), text);
                        if thought != self.last_thought {
                            out.push(ChatEvent::Thought {
                                text: thought.clone(),
                            });
                            self.last_thought = thought;
                        }
                    }
                }
                StreamEvent::Text {
                    channel: Channel::Content,
                    text,
                    ..
                } => full_content = Some(text),
                StreamEvent::End { .. } => self.ended = true,
            }
        }
        self.reveal_content(&mut out, full_content, ev);
        out
    }
}
