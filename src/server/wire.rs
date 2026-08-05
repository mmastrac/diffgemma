use serde::Serialize;

use crate::chat::{ChannelIds, Stabilizer, TextDecoder};

/// One streamed delta produced by the mapper, before OpenAI-envelope framing.
#[derive(Debug, Clone, PartialEq)]
pub enum WireDelta {
    /// Append to `reasoning_content` (the private thought channel).
    Reasoning(String),
    /// Append to `content` (block-committed, immutable answer text).
    Content(String),
    /// Replace the speculative draft — the shimmering, still-converging answer.
    Draft {
        text: String,
        committed: usize,
        block: usize,
        step: u32,
    },
}

// ===========================================================================
// DiffusionStreamMapper — the pure core (no GPU, no sockets; unit-tested).
//
// Turns the raw per-step canvas argmax stream into `WireDelta`s, splitting the
// harmony thought channel from the answer and separating block-committed text
// from the speculative draft.
// ===========================================================================

pub struct DiffusionStreamMapper<D: TextDecoder> {
    decoder: D,
    /// The shared stable-prefix front end (streak tracking, stop cut, quoted-
    /// stop skipping — one implementation with the terminal chat decoder).
    stab: Stabilizer,
    /// Channel/quote special ids for the shared token-level thought split.
    channels: ChannelIds,
    thinking: bool,
    emit_drafts: bool,

    /// Block-committed token ids accumulated across all committed blocks.
    committed_ids: Vec<u32>,
    /// Full committed reasoning / content text (the split of `committed_ids`).
    /// With pacing on, deltas may lag these; [`content`](Self::content) /
    /// [`reasoning`](Self::reasoning) always return the full committed text.
    emitted_reasoning: String,
    emitted_content: String,
    /// Last draft text emitted (dedupe replace-semantics chunks).
    last_draft: String,
    ended: bool,

    /// Paced release: hold a committed block's text and dribble it out during
    /// the NEXT block's denoise (paced by its step progress against an EMA of
    /// block step counts) so multi-block turns stream smoothly instead of
    /// bursting a block every N seconds. Everything flushes at turn end or
    /// the moment a tool call appears in committed content (agents consume
    /// tool turns whole — never delay them).
    paced: bool,
    /// Byte cursors into `emitted_reasoning` / `emitted_content` already
    /// RELEASED as deltas (== full lengths when pacing is off).
    released_reasoning: usize,
    released_content: usize,
    /// EMA of committed blocks' step counts — the pacing denominator.
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        decoder: D,
        stops: Vec<u32>,
        channel_open: Option<u32>,
        channel_close: Option<u32>,
        quote: Option<u32>,
        stop_skip_quoted: bool,
        thinking: bool,
        emit_drafts: bool,
        paced: bool,
    ) -> Self {
        Self {
            decoder,
            stab: Stabilizer::new(stops, quote, stop_skip_quoted),
            channels: ChannelIds {
                open: channel_open,
                close: channel_close,
                quote,
            },
            thinking,
            emit_drafts,
            committed_ids: Vec::new(),
            emitted_reasoning: String::new(),
            emitted_content: String::new(),
            last_draft: String::new(),
            ended: false,
            paced,
            released_reasoning: 0,
            released_content: 0,
            est_block_steps: 16.0,
            pace_off: false,
        }
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

    /// Split committed ids into (reasoning, content) via the shared
    /// token-level walk (`chat::ChannelIds::split`). With thinking off,
    /// everything is content.
    fn split(&self, ids: &[u32]) -> Split {
        if !self.thinking {
            return Split {
                reasoning: String::new(),
                content: self.decode_content(ids),
            };
        }
        let split = self.channels.split(&self.decoder, ids, false);
        Split {
            reasoning: self.clean_reasoning(&split.reasoning),
            content: self.decode_content(&split.content),
        }
    }

    /// Decode answer-side ids and sanitize the text. Stray channel specials
    /// (the model may re-emit `<channel|>` after the answer) are dropped by
    /// the `split` walk before ids get here; text-form leftovers die in the
    /// masked `sanitize_model_reply` — while QUOTED channel markup (literal
    /// tool-arg content) survives both.
    fn decode_content(&self, ids: &[u32]) -> String {
        crate::chat_template::sanitize_model_reply(
            &self
                .decoder
                .decode(&crate::sample::strip_degenerate_token_ids(ids)),
        )
    }

    /// Decode a reasoning-region id slice and strip the `<|channel>thought\n`
    /// opener scaffold the model emits when thinking is enabled.
    ///
    /// Mid-thought re-open ids never reach here (the `split` walk consumes
    /// them as toggles); the string replaces below catch TEXT-FORM (BPE)
    /// markers. Display-only: quoted literals in a rehearsed call may lose
    /// markers here, but repair/salvage read the raw decode, not this.
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
            "<|channel>thought\n",
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

    pub fn on_step(&mut self, ev: &crate::metal::StepProgressEvent<'_>) -> Vec<WireDelta> {
        let mut out = Vec::new();
        if self.ended {
            return out;
        }

        // Shared stable-prefix front end: streak tracking, stop cut, and
        // cross-block quote parity for quoted-stop skipping.
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
            // EMA of block step counts — the pacing denominator.
            self.est_block_steps = 0.5 * self.est_block_steps + 0.5 * ev.step_in_block as f32;
            // Premature stop inside an unfinished tool turn (open call, or
            // trailing prose after a closed call): keep the mapper open.
            if hit_stop && !crate::tools::should_continue_past_stop(&self.emitted_content) {
                self.ended = true;
            }
            // Tool call in committed content: agents consume tool turns whole
            // — flush and stop pacing for the rest of the turn.
            if self.emitted_content.contains("call:") {
                self.pace_off = true;
            }
            if !self.paced || self.pace_off || self.ended {
                self.release_up_to(usize::MAX, usize::MAX, &mut out);
            }
        } else if self.paced && !self.pace_off {
            // Paced release: fraction of the HELD text proportional to this
            // block's step progress against the estimated block length, so
            // the previous block's text finishes streaming roughly when this
            // block commits.
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

    /// Emit append deltas advancing the release cursors toward the byte
    /// targets (clamped to full length, snapped back to char boundaries).
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

    /// Release everything still held (end of generation — a turn that ends
    /// without a stop token must not strand paced text).
    pub fn final_flush(&mut self) -> Vec<WireDelta> {
        let mut out = Vec::new();
        self.release_up_to(usize::MAX, usize::MAX, &mut out);
        out
    }
}

/// If `next` extends `prev` (or diverges), return the suffix to append. Returns
/// `None` when nothing new. On a rare non-prefix revision, re-emit the whole new
/// text (append-only sinks can't retract; committed text almost never revises).
#[cfg(test)]
pub(crate) fn append_delta(prev: &str, next: &str) -> Option<String> {
    if next == prev {
        None
    } else if let Some(suffix) = next.strip_prefix(prev) {
        (!suffix.is_empty()).then(|| suffix.to_string())
    } else {
        Some(next.to_string())
    }
}

// ===========================================================================
// OpenAI SSE / JSON envelope framing.
// ===========================================================================

#[derive(Serialize)]
pub(crate) struct DeltaBody {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning_content: Option<String>,
    #[serde(rename = "x-diffusion-draft", skip_serializing_if = "Option::is_none")]
    pub(crate) draft: Option<DraftBody>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_calls: Option<Vec<serde_json::Value>>,
}

#[derive(Serialize)]
pub(crate) struct DraftBody {
    pub(crate) text: String,
    pub(crate) committed: usize,
    pub(crate) block: usize,
    pub(crate) step: u32,
}

#[derive(Serialize)]
pub(crate) struct ChunkChoice {
    pub(crate) index: u32,
    pub(crate) delta: DeltaBody,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
pub(crate) struct Chunk {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created: u64,
    pub(crate) model: String,
    pub(crate) choices: Vec<ChunkChoice>,
}

pub(crate) fn empty_delta() -> DeltaBody {
    DeltaBody {
        role: None,
        content: None,
        reasoning_content: None,
        draft: None,
        tool_calls: None,
    }
}

impl WireDelta {
    pub(crate) fn into_delta_body(self) -> DeltaBody {
        match self {
            WireDelta::Reasoning(s) => DeltaBody {
                reasoning_content: Some(s),
                ..empty_delta()
            },
            WireDelta::Content(s) => DeltaBody {
                content: Some(s),
                ..empty_delta()
            },
            WireDelta::Draft {
                text,
                committed,
                block,
                step,
            } => DeltaBody {
                draft: Some(DraftBody {
                    text,
                    committed,
                    block,
                    step,
                }),
                ..empty_delta()
            },
        }
    }
}

pub(crate) fn finish_reason_for(tool_calls: &[serde_json::Value], stopped: bool) -> &'static str {
    if !tool_calls.is_empty() {
        "tool_calls"
    } else if stopped {
        "stop"
    } else {
        "length"
    }
}

/// Truncate / suppress wire deltas that would leak native tool-call markup into
/// OpenAI `content`. Returns `None` when the delta should not be sent.
pub(crate) fn filter_tool_markup_delta(
    d: WireDelta,
    strip: bool,
    suppress: &std::sync::atomic::AtomicBool,
) -> Option<WireDelta> {
    use std::sync::atomic::Ordering;
    if !strip {
        return Some(d);
    }
    match d {
        WireDelta::Content(s) => {
            if suppress.load(Ordering::Relaxed) {
                return None;
            }
            match s.find("<|tool_call>") {
                Some(i) => {
                    suppress.store(true, Ordering::Relaxed);
                    let keep = s[..i].trim_end().to_string();
                    (!keep.is_empty()).then_some(WireDelta::Content(keep))
                }
                None => Some(WireDelta::Content(s)),
            }
        }
        WireDelta::Draft {
            mut text,
            committed,
            block,
            step,
        } => {
            if let Some(i) = text.find("<|tool_call>") {
                text.truncate(i);
                while text.ends_with(char::is_whitespace) {
                    text.pop();
                }
            }
            Some(WireDelta::Draft {
                text,
                committed,
                block,
                step,
            })
        }
        other => Some(other),
    }
}
