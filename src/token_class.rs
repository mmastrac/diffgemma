//! Output-token classification: a linear probe over canvas hidden states.
//!
//! A denoise step already computes a hidden vector for every canvas position at
//! every layer. Classifying "what mode is this position being emitted in"
//! (prose / code / tool-call) is then one dot product per position — ~720K MACs
//! for a 256-row canvas at hidden 2816, against a step measured in hundreds of
//! milliseconds. The result is 256 floats, small enough to ride the readback
//! the hot path already performs.
//!
//! The probe is a DIRECTION plus a midpoint, both fitted offline from
//! `step-layer-probe` dumps (difference-of-class-means; see PLAN). Scoring is
//! `(h/|h| - mid) · w`, sign gives the class. Vectors are L2-normalised first
//! so the probe cannot be won by activation magnitude, which varies by layer
//! and position for reasons unrelated to mode.
//!
//! What the evidence supports, and what it does not:
//! - MEASURED: mode follows the most recently COMMITTED content, readable at
//!   LOO accuracy 1.00 (L9/L29) when that content sits in the KV cache — the
//!   cross-block case.
//! - NOT MEASURED: reading a position whose OWN token holds the content
//!   (intra-block). Enabling this and looking at the colours in chat IS that
//!   experiment; treat early labels as a hypothesis under test, not ground
//!   truth.
//!
//! Off unless `DGQ_TOKEN_CLASS=<probe.json>` names a probe. Pure
//! instrumentation: it reads hidden state and never steers, so generation stays
//! byte-identical and `golden` is unaffected.

use crate::Error;
use std::path::Path;

/// A fitted linear probe over one layer's hidden state.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct TokenClassProbe {
    /// Layer whose post-layer hidden state this was fitted on. `layers` means
    /// the final norm output.
    pub layer: usize,
    /// Class names, indexed by the value `classify` returns. Exactly 2 for a
    /// sign-thresholded direction: index 0 is the NEGATIVE side.
    pub class_names: Vec<String>,
    /// Direction (`mean(class1) - mean(class0)`), length = hidden dim.
    pub w: Vec<f32>,
    /// Midpoint between the class means, subtracted before projecting.
    pub mid: Vec<f32>,
    /// Free-text provenance: which dumps this was fitted on, and its held-out
    /// score. Carried so a probe file is self-describing when it turns up in
    /// six months with a surprising label on screen.
    #[serde(default)]
    pub provenance: String,
}

impl TokenClassProbe {
    pub fn load(path: &Path) -> Result<Self, Error> {
        let text = std::fs::read_to_string(path).map_err(Error::Io)?;
        let p: Self = serde_json::from_str(&text).map_err(Error::Json)?;
        p.validate()?;
        Ok(p)
    }

    fn validate(&self) -> Result<(), Error> {
        if self.w.len() != self.mid.len() {
            return Err(Error::Runtime("token-class probe: w and mid differ in length"));
        }
        if self.w.is_empty() {
            return Err(Error::Runtime("token-class probe: empty direction"));
        }
        if self.class_names.len() != 2 {
            return Err(Error::Runtime("token-class probe: expected exactly 2 class names"));
        }
        Ok(())
    }

    pub fn hidden_dim(&self) -> usize {
        self.w.len()
    }

    /// Signed score for one position. Positive => `class_names[1]`.
    ///
    /// `hidden` is the raw hidden row; it is L2-normalised here rather than by
    /// the caller so every call site agrees on the convention.
    pub fn score(&self, hidden: &[f32]) -> f32 {
        debug_assert_eq!(hidden.len(), self.w.len());
        let n = hidden.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n <= 0.0 {
            return 0.0;
        }
        let inv = 1.0 / n;
        hidden
            .iter()
            .zip(&self.mid)
            .zip(&self.w)
            .map(|((h, m), w)| (h * inv - m) * w)
            .sum()
    }

    /// Class index for one position: 0 = negative side, 1 = positive side.
    #[cfg(test)]
    pub fn classify(&self, hidden: &[f32]) -> u8 {
        u8::from(self.score(hidden) > 0.0)
    }

    /// Classify a whole canvas plane laid out row-major as `rows x dim`.
    /// Returns one label per row, and the raw scores so a caller can render
    /// confidence (|score| near 0 means the probe is undecided, which is the
    /// interesting case at a mode BOUNDARY).
    pub fn classify_rows(&self, plane: &[f32], rows: usize) -> (Vec<u8>, Vec<f32>) {
        let d = self.w.len();
        let mut labels = Vec::with_capacity(rows);
        let mut scores = Vec::with_capacity(rows);
        for r in 0..rows {
            let Some(row) = plane.get(r * d..(r + 1) * d) else {
                break;
            };
            let s = self.score(row);
            scores.push(s);
            labels.push(u8::from(s > 0.0));
        }
        (labels, scores)
    }

    pub fn class_name(&self, label: u8) -> &str {
        self.class_names
            .get(label as usize)
            .map_or("?", String::as_str)
    }
}

/// Collector for FITTING a probe in the deployment regime.
///
/// The cold-fitted direction was measured NOT to transfer to committed
/// positions (every score landed on one side of zero — a distribution shift,
/// since a noise position and a position holding its own resolved token have
/// different hidden means). So the probe has to be fitted on hidden state
/// captured where it will be USED: at block commit, during real generation.
///
/// The engine cannot know a sample's class label, so the flow is inverted: a
/// command enables collection, runs one generation, drains the rows, and
/// attaches the label it already knows. Off unless explicitly enabled, and
/// the push site is behind that check, so normal generation is untouched.
pub mod collect {
    use std::sync::Mutex;

    static SINK: Mutex<Option<Vec<Vec<f32>>>> = Mutex::new(None);

    /// Start collecting. Any previously collected rows are discarded.
    pub fn enable() {
        *SINK.lock().unwrap() = Some(Vec::new());
    }
    /// Stop collecting and discard anything buffered.
    pub fn disable() {
        *SINK.lock().unwrap() = None;
    }
    pub fn is_enabled() -> bool {
        SINK.lock().unwrap().is_some()
    }
    /// Append committed answer-region rows. No-op when disabled.
    pub fn push_rows(rows: impl Iterator<Item = Vec<f32>>) {
        if let Some(sink) = SINK.lock().unwrap().as_mut() {
            sink.extend(rows);
        }
    }
    /// Take everything collected so far, leaving collection enabled.
    pub fn take() -> Vec<Vec<f32>> {
        SINK.lock()
            .unwrap()
            .as_mut()
            .map(std::mem::take)
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe() -> TokenClassProbe {
        // 3-d toy: the direction is +x, the midpoint sits at the origin.
        TokenClassProbe {
            layer: 29,
            class_names: vec!["prose".into(), "code".into()],
            w: vec![1.0, 0.0, 0.0],
            mid: vec![0.0, 0.0, 0.0],
            provenance: "unit test".into(),
        }
    }

    #[test]
    fn sign_selects_the_class() {
        let p = probe();
        assert_eq!(p.classify(&[1.0, 0.0, 0.0]), 1);
        assert_eq!(p.class_name(p.classify(&[1.0, 0.0, 0.0])), "code");
        assert_eq!(p.classify(&[-1.0, 0.0, 0.0]), 0);
        assert_eq!(p.class_name(p.classify(&[-1.0, 0.0, 0.0])), "prose");
    }

    /// Magnitude must not decide the class: a probe that could be won by
    /// activation norm would track layer/position artefacts rather than mode.
    #[test]
    fn scoring_is_scale_invariant() {
        let p = probe();
        let a = p.score(&[3.0, 4.0, 0.0]);
        let b = p.score(&[30.0, 40.0, 0.0]);
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }

    /// A zero row (an untouched canvas slot) must not be silently classified
    /// as the positive class.
    #[test]
    fn zero_row_scores_zero_and_reads_negative() {
        let p = probe();
        assert_eq!(p.score(&[0.0, 0.0, 0.0]), 0.0);
        assert_eq!(p.classify(&[0.0, 0.0, 0.0]), 0);
    }

    #[test]
    fn classify_rows_walks_the_plane_and_stops_at_the_end() {
        let p = probe();
        // 3 rows: +x, -x, +x
        let plane = vec![1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let (labels, scores) = p.classify_rows(&plane, 3);
        assert_eq!(labels, vec![1, 0, 1]);
        assert_eq!(scores.len(), 3);
        // Asking for more rows than the plane holds truncates rather than
        // reading past the end.
        let (labels, _) = p.classify_rows(&plane, 99);
        assert_eq!(labels.len(), 3);
    }

    #[test]
    fn malformed_probes_are_rejected() {
        let mut p = probe();
        p.w.push(1.0);
        assert!(p.validate().is_err(), "w/mid length mismatch");
        let mut p = probe();
        p.class_names.push("tool".into());
        assert!(p.validate().is_err(), "sign threshold implies exactly 2 classes");
    }
}
