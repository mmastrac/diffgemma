//! Shared test fixtures for the decoder unit tests.

use super::{StepProgressEvent, TextDecoder};

/// Trivial decoder: a token id maps to its char, and ids outside printable
/// ASCII decode to nothing. Enough to exercise the id-level logic without the
/// real tokenizer or sanitizer markers.
pub(crate) struct FakeDecoder;

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

pub(crate) fn step<'a>(
    block: usize,
    step: u32,
    argmax: &'a [u32],
    done: bool,
) -> StepProgressEvent<'a> {
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
