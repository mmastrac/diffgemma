//! Paced release, shared by the wire mapper and the terminal decoder.
//!
//! The canonical stream ([`super::core`]) carries the model-committed text
//! whole. A consumer that wants it to arrive smoothly rather than a block at a
//! time reveals it through a [`PaceClock`]: a finished block's committed bytes
//! dribble out over the next block's denoise, a fraction of the remainder each
//! step against an EMA of block step counts. One clock per decoder (the block
//! length estimate and the tool-call gate are shared); one release cursor per
//! channel, since reasoning and content advance independently.

/// The per-decoder pacing state: the block-length EMA and the tool-call gate.
pub struct PaceClock {
    est_block_steps: f32,
    pace_off: bool,
}

impl PaceClock {
    pub fn new() -> Self {
        Self {
            est_block_steps: 16.0,
            pace_off: false,
        }
    }

    /// A block committed: fold its length into the EMA that sets how fast the
    /// next block releases this one's text, and stop pacing for good once a tool
    /// call is in the committed answer (an agent consumes a tool round whole, so
    /// it must never wait on a dribble).
    pub fn on_commit(&mut self, step_in_block: u32, committed_content: &str) {
        self.est_block_steps = 0.5 * self.est_block_steps + 0.5 * step_in_block as f32;
        if committed_content.contains("call:") {
            self.pace_off = true;
        }
    }

    pub fn pace_off(&self) -> bool {
        self.pace_off
    }

    /// Where a cursor at `released` should reach this step. Unpaced or gated,
    /// the whole committed length; a block's own finishing step holds the cursor
    /// (nothing new to pace against yet); otherwise a step-fraction of the
    /// remainder. Clamped into `committed` at a char boundary, never retreating.
    pub fn target(
        &self,
        paced: bool,
        block_done: bool,
        step_in_block: u32,
        released: usize,
        committed: &str,
    ) -> usize {
        let full = committed.len();
        let released = released.min(full);
        let mut t = if !paced || self.pace_off {
            full
        } else if block_done {
            released
        } else {
            let frac = (step_in_block as f32 / self.est_block_steps.max(1.0)).clamp(0.0, 1.0);
            released + ((full - released) as f32 * frac) as usize
        };
        t = t.min(full).max(released);
        while t < full && !committed.is_char_boundary(t) {
            t -= 1;
        }
        t
    }
}

impl Default for PaceClock {
    fn default() -> Self {
        Self::new()
    }
}
