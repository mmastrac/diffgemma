//! Per-position stable-streak tracking and stop-token cut, the shared front
//! half of every stream decoder (terminal chat and the serve wire).

use super::StepProgressEvent;

/// Consecutive-step repeats before a canvas position streams as stable.
pub(crate) const STABLE_STREAK: u32 = 2;

/// The stable, stop-cut prefix a single denoise step yields.
pub(crate) struct StablePrefix {
    /// This step opened a new canvas block.
    pub new_block: bool,
    /// The stable prefix ids (the whole canvas on block commit), cut at the
    /// first effective stop token.
    pub ids: Vec<u32>,
    /// A stop token ended the prefix.
    pub hit_stop: bool,
}

/// Per-position stable-streak tracking plus a stop-token cut over raw per-step
/// canvas snapshots. A position's argmax must hold for [`STABLE_STREAK`]
/// consecutive steps before it streams. On `block_done` the whole canvas is
/// stable by definition. With `stop_skip_quoted`, a stop id inside an open
/// `<|"|>` quote run is literal content. That must match the engine's
/// `continue_incomplete_tool_calls` scan so the stream and the engine cut the
/// same text.
pub(crate) struct Stabilizer {
    stops: Vec<u32>,
    quote: Option<u32>,
    stop_skip_quoted: bool,
    block_idx: usize,
    last_argmax: Vec<u32>,
    streak: Vec<u32>,
    /// Quote parity of everything committed so far, carried across blocks.
    committed_quote_open: bool,
}

impl Stabilizer {
    pub fn new(stops: Vec<u32>, quote: Option<u32>, stop_skip_quoted: bool) -> Self {
        Self {
            stops,
            quote,
            stop_skip_quoted,
            block_idx: 0,
            last_argmax: Vec::new(),
            streak: Vec::new(),
            committed_quote_open: false,
        }
    }

    pub fn on_step(&mut self, ev: &StepProgressEvent<'_>) -> StablePrefix {
        let new_block = ev.block_idx != self.block_idx;
        if new_block {
            self.block_idx = ev.block_idx;
            self.last_argmax.clear();
            self.streak.clear();
        }

        // Per-position stable-streak update. A canvas-length change
        // (shrink-on-retry) resets everything.
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
        let mut ids = Vec::with_capacity(prefix_end);
        let mut hit_stop = false;
        let mut in_quote = self.stop_skip_quoted && self.committed_quote_open;
        for &id in &ev.argmax[..prefix_end] {
            if self.stop_skip_quoted && Some(id) == self.quote {
                in_quote = !in_quote;
            } else if !in_quote && self.stops.contains(&id) {
                hit_stop = true;
                break;
            }
            ids.push(id);
        }

        if ev.block_done {
            self.last_argmax.clear();
            self.streak.clear();
            // The caller commits exactly `ids`, so fold their quote parity to
            // start the next block's stop scan in the right state.
            if let Some(q) = self.quote {
                let flips = ids.iter().filter(|&&id| id == q).count();
                if flips % 2 == 1 {
                    self.committed_quote_open = !self.committed_quote_open;
                }
            }
        }

        StablePrefix {
            new_block,
            ids,
            hit_stop,
        }
    }
}
