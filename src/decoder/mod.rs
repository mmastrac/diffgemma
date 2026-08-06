//! The decoding core shared by every stream consumer. A [`Stabilizer`] turns
//! raw per-step canvas snapshots (`StepProgressEvent`) into stable stop-cut id
//! prefixes. A token-level [`ChannelIds::split`] separates thought-channel ids
//! from answer ids, staying quote-aware so channel markup inside a quoted tool
//! arg reads as data rather than grammar. [`StreamDecoder`] composes the two
//! into semantic [`ChatEvent`]s. The serve wire mapper builds on the same
//! primitives, keeping chat and serve in step on the fragile stable-prefix and
//! channel-split rules.

mod serve;
mod split;
mod stabilize;
mod stream;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod testutil;

pub use serve::{DiffusionStreamMapper, MapperConfig, WireDelta};
pub use stream::StreamDecoder;

pub(crate) use split::{ChannelIds, salvage_answer};
pub(crate) use stabilize::{Stabilizer, first_unquoted_stop};

/// Per-denoise-step progress snapshot, the contract between the generator and
/// every stream decoder. `argmax` is the active canvas slice only, dropping
/// stale rows past `active_canvas` after shrink-on-retry. Under the MLX-exact
/// sampler the block's final commit is that slice, so a stable prefix is a
/// faithful preview.
pub struct StepProgressEvent<'a> {
    /// 1-based committed-block index this step belongs to.
    pub block_idx: usize,
    /// Total blocks budgeted for this generate (`max_new_tokens / CANVAS`).
    pub max_blocks: usize,
    /// 1-based denoise step within the block.
    pub step_in_block: u32,
    pub max_steps: usize,
    /// Active-canvas argmax of this step (`active_canvas` entries).
    pub argmax: &'a [u32],
    /// Positions accepted by the entropy-bound rule this step.
    pub accept_count: u32,
    /// Mean per-position entropy (nats) this step.
    pub mean_entropy: f32,
    /// True once this block's argmax is finalized and past retry. A mid-attempt
    /// "would-stop" step stays false so stream mappers never commit a discarded
    /// empty or degenerate canvas, which once spliced a second
    /// `<|channel>thought` into `reasoning_content`.
    pub block_done: bool,
}

/// Token-id to text decoding, abstracted so the decoder is testable without the
/// real 32 MB tokenizer.
pub trait TextDecoder {
    fn decode(&self, ids: &[u32]) -> String;
    fn decode_append(&self, out: &mut String, ids: &[u32]);

    /// Decode answer-side ids to sanitized visible text. Degenerate ids are
    /// dropped and the text-form markers die in the masked sanitizer, while
    /// quoted tool-arg markup survives both.
    fn decode_content(&self, ids: &[u32]) -> String {
        crate::chat_template::sanitize_model_reply(
            &self.decode(&crate::sample::strip_degenerate_token_ids(ids)),
        )
    }
}

impl TextDecoder for std::sync::Arc<crate::tokenizer::Tokenizer> {
    fn decode(&self, ids: &[u32]) -> String {
        (**self).decode(ids)
    }
    fn decode_append(&self, out: &mut String, ids: &[u32]) {
        (**self).decode_append(out, ids);
    }
}
