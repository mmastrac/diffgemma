//! The one canonical diffusion stream both the terminal and the wire consume.
//!
//! [`DiffusionStream`] turns the raw per-step canvas ([`StepProgressEvent`])
//! into [`StreamEvent`]s: the [`Stabilizer`] stop-cut front end and the
//! token-level [`ChannelIds::split`] live here once, so chat and serve stay in
//! step on those fragile rules by consuming the same events rather than
//! re-deriving them.
//!
//! Each channel's text is a two-region model. `text[..committed]` is the
//! model-locked prefix: immutable across events but for a rare re-denoise
//! revision (which a consumer sees as `text[..committed]` changing, and which
//! [`WireDelta`](super::serve)/chat handle by resetting their release). Beyond
//! it, `text[committed..]` is the still-converging draft. The stream carries the
//! model's truth and paces nothing; how fast a consumer reveals the committed
//! region is a display policy it applies with [`PacedRelease`](super::pace).

use super::{ChannelIds, Stabilizer, StepProgressEvent, TextDecoder};

/// Which side of the thought split a [`StreamEvent::Text`] carries.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Channel {
    /// The answer (post-`<channel|>` text, or everything when thinking is off).
    Content,
    /// The reasoning (inside a `<|channel>...<channel|>` thought span).
    Reasoning,
}

/// One event in the canonical stream.
#[derive(Clone, PartialEq, Debug)]
pub enum StreamEvent {
    /// A new canvas block began denoising.
    Block { block: usize },
    /// Per-step denoise progress (the diffusion status line).
    Progress {
        block: usize,
        step: u32,
        max_steps: usize,
        canvas: usize,
        locked: usize,
        accepted: u32,
    },
    /// A channel's text this step. `text[..committed]` is the model-locked
    /// prefix (see the module note on the rare revision); `text[committed..]` is
    /// the draft. Emitted whenever `(committed, text)` changes.
    Text {
        channel: Channel,
        committed: usize,
        text: String,
    },
    /// A block's stable ids locked into the committed stream.
    Commit { block: usize, tokens: usize },
    /// Generation ended (a stop token fell outside a tool call).
    End { stopped: bool },
}

/// Special ids and thought-split behavior for a [`DiffusionStream`].
#[derive(Default)]
pub struct StreamConfig {
    pub stops: Vec<u32>,
    pub channel_open: Option<u32>,
    pub channel_close: Option<u32>,
    pub quote: Option<u32>,
    pub stop_skip_quoted: bool,
    /// Split reasoning from content. Off routes every id to `Content`.
    pub thinking: bool,
    /// Generation begins inside an already-open thought (a tool round's
    /// forced-open thought, or a `/prethink` continuation).
    pub start_in_thought: bool,
}

/// One channel's committed text and the change-detection cursor for its `Text`.
#[derive(Default)]
struct Chan {
    committed: String,
    last_committed: usize,
    last_text: String,
    seen: bool,
}

pub struct DiffusionStream<D: TextDecoder> {
    decoder: D,
    stab: Stabilizer,
    channels: ChannelIds,
    thinking: bool,
    start_in_thought: bool,
    committed_ids: Vec<u32>,
    content: Chan,
    reasoning: Chan,
    ended: bool,
}

impl<D: TextDecoder> DiffusionStream<D> {
    pub fn new(decoder: D, cfg: StreamConfig) -> Self {
        Self {
            decoder,
            stab: Stabilizer::new(cfg.stops, cfg.quote, cfg.stop_skip_quoted),
            channels: ChannelIds {
                open: cfg.channel_open,
                close: cfg.channel_close,
                quote: cfg.quote,
            },
            thinking: cfg.thinking,
            start_in_thought: cfg.start_in_thought,
            committed_ids: Vec::new(),
            content: Chan::default(),
            reasoning: Chan::default(),
            ended: false,
        }
    }

    /// Post-construction config, for consumers that assemble it through a
    /// builder chain (the chat decoder). All only affect the split, which runs
    /// from the first step, so setting them before any `on_step` is enough.
    pub fn set_channels(&mut self, channels: ChannelIds) {
        self.channels = channels;
    }
    pub fn set_thinking(&mut self, thinking: bool) {
        self.thinking = thinking;
    }
    pub fn set_start_in_thought(&mut self, start_in_thought: bool) {
        self.start_in_thought = start_in_thought;
    }

    /// The model-locked answer text so far.
    pub fn content(&self) -> &str {
        &self.content.committed
    }

    /// The model-locked reasoning text so far.
    pub fn reasoning(&self) -> &str {
        &self.reasoning.committed
    }

    /// Split ids into `(reasoning, content)` text. Thinking off is all content.
    fn split(&self, ids: &[u32]) -> (String, String) {
        if !self.thinking {
            return (String::new(), self.decoder.decode_content(ids));
        }
        let s = self
            .channels
            .split(&self.decoder, ids, self.start_in_thought);
        (
            self.decoder.decode_reasoning(&s.reasoning),
            self.decoder.decode_content(&s.content),
        )
    }

    pub fn on_step(&mut self, ev: &StepProgressEvent<'_>) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        if self.ended {
            return out;
        }
        let sp = self.stab.on_step(ev);
        if sp.new_block {
            out.push(StreamEvent::Block {
                block: ev.block_idx,
            });
        }
        out.push(StreamEvent::Progress {
            block: ev.block_idx,
            step: ev.step_in_block,
            max_steps: ev.max_steps,
            canvas: ev.argmax.len(),
            locked: sp.ids.len(),
            accepted: ev.accept_count,
        });

        // Committed + this step's speculative draft, the full per-channel text.
        let mut all = self.committed_ids.clone();
        all.extend_from_slice(&sp.ids);
        let (full_reasoning, full_content) = self.split(&all);

        if ev.block_done {
            self.committed_ids.extend_from_slice(&sp.ids);
            let (committed_reasoning, committed_content) = self.split(&self.committed_ids);
            self.content.committed = committed_content;
            self.reasoning.committed = committed_reasoning;
            if sp.hit_stop && !crate::tools::should_continue_past_stop(&self.content.committed) {
                self.ended = true;
            }
            out.push(StreamEvent::Commit {
                block: ev.block_idx,
                tokens: sp.ids.len(),
            });
        }

        // Reasoning before content, the order both consumers expect.
        if self.thinking {
            emit_text(
                &mut out,
                Channel::Reasoning,
                &mut self.reasoning,
                &full_reasoning,
            );
        }
        emit_text(&mut out, Channel::Content, &mut self.content, &full_content);
        if self.ended {
            out.push(StreamEvent::End { stopped: true });
        }
        out
    }
}

/// Push a `Text` for `chan` when `(committed_len, text)` changed. `committed` is
/// clamped into `text` and snapped to a char boundary so a slice never splits a
/// code point.
fn emit_text(out: &mut Vec<StreamEvent>, channel: Channel, chan: &mut Chan, text: &str) {
    let mut committed = chan.committed.len().min(text.len());
    while committed > 0 && !text.is_char_boundary(committed) {
        committed -= 1;
    }
    if chan.seen && committed == chan.last_committed && text == chan.last_text {
        return;
    }
    chan.seen = true;
    chan.last_committed = committed;
    chan.last_text = text.to_string();
    out.push(StreamEvent::Text {
        channel,
        committed,
        text: text.to_string(),
    });
}
