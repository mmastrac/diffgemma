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

/// Index of the first stop id in `ids` not inside an open `<|"|>` quote run,
/// with quote parity seeded by `start_in_quote`. `quote = None` is the plain
/// scan. Shared by the stream [`Stabilizer`] and the engine's generation-side
/// stop-cut so the two cut identical text.
pub(crate) fn first_unquoted_stop(
    ids: &[u32],
    stops: &[u32],
    quote: Option<u32>,
    start_in_quote: bool,
) -> Option<usize> {
    let Some(q) = quote else {
        return ids.iter().position(|id| stops.contains(id));
    };
    let mut in_quote = start_in_quote;
    for (i, &id) in ids.iter().enumerate() {
        if id == q {
            in_quote = !in_quote;
        } else if !in_quote && stops.contains(&id) {
            return Some(i);
        }
    }
    None
}

/// Per-position stable-streak tracking plus a stop-token cut over raw per-step
/// canvas snapshots. A position's argmax must hold for [`STABLE_STREAK`]
/// consecutive steps before it streams. On `block_done` the whole canvas is
/// stable by definition. The stop cut runs the shared [`first_unquoted_stop`],
/// so a `stop_skip_quoted` run cuts exactly where the engine's generation-side
/// stop does.
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
        let skip_quote = self.stop_skip_quoted.then_some(self.quote).flatten();
        let prefix = &ev.argmax[..prefix_end];
        let cut = first_unquoted_stop(prefix, &self.stops, skip_quote, self.committed_quote_open);
        let hit_stop = cut.is_some();
        let ids = prefix[..cut.unwrap_or(prefix_end)].to_vec();

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
