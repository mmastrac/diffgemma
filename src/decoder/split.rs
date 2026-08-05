//! The token-level thought/answer split and the settled-output fallbacks that
//! recover an answer when the model never closes its thought.

use super::TextDecoder;

/// The special-token ids that structure a generation: the thought channel
/// markers and the `<|"|>` string quote. A `None` entry never matches, so a
/// model (or test decoder) without a marker degrades gracefully.
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
    /// A thought span closed: a `<channel|>` was seen outside quotes. The
    /// settled-output fallbacks key off this to recover an answer when the
    /// model runs its thought to the end.
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
    /// Reasoning is only what sits inside an explicit `<|channel>...<channel|>`
    /// span the model actually emitted. Classification follows emission rather
    /// than a mode flag, so a turn that skips the thought ceremony and answers
    /// with a bare tool call keeps its whole output visible. An unclosed span
    /// runs to the end, so mid-thought streaming works. Channel ids inside an
    /// open `<|"|>` quote run are literal arg content: they stay put and keep
    /// their thought state, so a tool call writing a file that contains chat
    /// markup round-trips it byte-exact.
    ///
    /// `start_in_thought` marks a generation that began inside an already-open
    /// thought (a `/prethink` continuation whose open marker was in the
    /// prompt). The walk starts in-span with the role line already consumed.
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
        // A `<|channel>` takes its name token (plus trailing newline) with it.
        // The open special is dropped here, so a mid-span re-open would
        // otherwise leak a bare "thought" line into the surfaced reasoning.
        let mut skip_name = false;
        for &id in ids {
            if Some(id) == self.quote {
                in_quote = !in_quote;
                // The quote id is grammar the tool parser needs, so keep it.
                if in_thought {
                    reasoning.push(id);
                } else {
                    content.push(id);
                }
            } else if in_quote {
                // Literal quoted content, channel ids included, verbatim.
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
    /// into `(thought, answer)` text. The thought is `seed` plus the model's
    /// completion. The answer is what follows the model's `<channel|>`,
    /// unsanitized. A thought that never closes usually states its answer
    /// inside the reasoning (a thought degeneracy), so salvage it from the
    /// completion's tail rather than reporting none.
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
    /// The round began inside an open thought (the prompt seeds the marker), so
    /// the model's `<channel|>` separates the reasoning (prefixed by the
    /// `/prethink` `seed`, if any) from the answer and tool calls. A generation
    /// with no close has no separate reasoning, so `rest` is the whole decoded
    /// text, keeping calls the model left inside its thought parseable.
    pub fn settle_tool_reply<D: TextDecoder>(
        &self,
        decoder: &D,
        seed: &str,
        ids: &[u32],
    ) -> (Option<String>, String) {
        let s = self.split(decoder, ids, true);
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
