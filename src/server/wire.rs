use serde::Serialize;

use crate::chat_protocol::TextDecoder;

/// Consecutive-step repeats before a canvas position is treated as stable enough
/// to stream (matches the terminal chat renderer's `STABLE_STREAK`).
const STABLE_STREAK: u32 = 2;

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
    stops: Vec<u32>,
    channel_open: Option<u32>,
    channel_close: Option<u32>,
    thinking: bool,
    emit_drafts: bool,

    block_idx: usize,
    last_argmax: Vec<u32>,
    streak: Vec<u32>,
    /// Block-committed token ids accumulated across all committed blocks.
    committed_ids: Vec<u32>,
    /// Reasoning / content text already emitted as deltas (append cursors).
    emitted_reasoning: String,
    emitted_content: String,
    /// Last draft text emitted (dedupe replace-semantics chunks).
    last_draft: String,
    ended: bool,
}

/// Reasoning (thought-channel) and answer text split out of a committed id list.
struct Split {
    reasoning: String,
    content: String,
}

impl<D: TextDecoder> DiffusionStreamMapper<D> {
    pub fn new(
        decoder: D,
        stops: Vec<u32>,
        channel_open: Option<u32>,
        channel_close: Option<u32>,
        thinking: bool,
        emit_drafts: bool,
    ) -> Self {
        Self {
            decoder,
            stops,
            channel_open,
            channel_close,
            thinking,
            emit_drafts,
            block_idx: 0,
            last_argmax: Vec::new(),
            streak: Vec::new(),
            committed_ids: Vec::new(),
            emitted_reasoning: String::new(),
            emitted_content: String::new(),
            last_draft: String::new(),
            ended: false,
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

    /// Split committed ids into (reasoning, content). With thinking off,
    /// everything is content. With thinking on, reasoning is ONLY what sits
    /// inside an explicit `<|channel>…<channel|>` span the model actually
    /// emitted — classification follows emission, not the mode flag. The old
    /// rule ("everything is reasoning until a close appears") silently
    /// swallowed a whole turn when the model skipped the thought ceremony
    /// and answered with a bare tool call (field incident 2026-07-17: a
    /// well-formed edit call streamed as reasoning_content, client got an
    /// empty message, and the repair stage — which judged the call visible
    /// and valid — was rightly silent). An unclosed span still runs to the
    /// end of the committed ids, so mid-thought streaming is unchanged.
    fn split(&self, ids: &[u32]) -> Split {
        if !self.thinking {
            return Split {
                reasoning: String::new(),
                content: self.decode_content(ids),
            };
        }
        let mut reasoning_ids: Vec<u32> = Vec::new();
        let mut content_ids: Vec<u32> = Vec::new();
        let mut in_thought = false;
        for &id in ids {
            if Some(id) == self.channel_open {
                in_thought = true;
            } else if Some(id) == self.channel_close {
                in_thought = false;
            } else if in_thought {
                reasoning_ids.push(id);
            } else {
                content_ids.push(id);
            }
        }
        Split {
            reasoning: self.clean_reasoning(&reasoning_ids),
            content: self.decode_content(&content_ids),
        }
    }

    /// Decode answer-side ids: drop further channel specials (the model may
    /// re-emit `<channel|>` after the answer), then sanitize the text.
    fn decode_content(&self, ids: &[u32]) -> String {
        let filtered: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|&id| Some(id) != self.channel_open && Some(id) != self.channel_close)
            .collect();
        crate::chat_template::sanitize_model_reply(
            &self
                .decoder
                .decode(&crate::sample::strip_degenerate_token_ids(&filtered)),
        )
    }

    /// Decode a reasoning-region id slice and strip the `<|channel>thought\n`
    /// opener scaffold the model emits when thinking is enabled.
    fn clean_reasoning(&self, ids: &[u32]) -> String {
        // Drop channel specials entirely — mid-thought re-opens (or stale-canvas
        // leftovers) must not survive as nested Thought UI markup.
        let filtered: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|&id| Some(id) != self.channel_open && Some(id) != self.channel_close)
            .collect();
        let raw = self
            .decoder
            .decode(&crate::sample::strip_degenerate_token_ids(&filtered));
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

        if ev.block_idx != self.block_idx {
            self.block_idx = ev.block_idx;
            self.last_argmax.clear();
            self.streak.clear();
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
        let mut stable_ids = Vec::with_capacity(prefix_end);
        let mut hit_stop = false;
        for &id in &ev.argmax[..prefix_end] {
            if self.stops.contains(&id) {
                hit_stop = true;
                break;
            }
            stable_ids.push(id);
        }

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

        // Commit: fold this block's stable ids into the committed stream and
        // emit any newly-committed reasoning / content as append deltas.
        if ev.block_done {
            self.committed_ids.extend_from_slice(&stable_ids);
            let split = self.split(&self.committed_ids);
            if let Some(delta) = append_delta(&self.emitted_reasoning, &split.reasoning) {
                out.push(WireDelta::Reasoning(delta));
                self.emitted_reasoning = split.reasoning;
            }
            if let Some(delta) = append_delta(&self.emitted_content, &split.content) {
                out.push(WireDelta::Content(delta));
                self.emitted_content = split.content;
            }
            self.last_argmax.clear();
            self.streak.clear();
            // Premature stop inside an unfinished tool turn (open call, or
            // trailing prose after a closed call): keep the mapper open.
            if hit_stop && !crate::tools::should_continue_past_stop(&self.emitted_content) {
                self.ended = true;
            }
        }
        out
    }
}

/// If `next` extends `prev` (or diverges), return the suffix to append. Returns
/// `None` when nothing new. On a rare non-prefix revision, re-emit the whole new
/// text (append-only sinks can't retract; committed text almost never revises).
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
