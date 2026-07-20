//! `soft` battery — INDIRECT retrieval. Non-blocking by design.
//!
//! `longctx` asks questions whose wording sits next to the answer in the
//! document, so a reply can be produced by matching surface tokens. This
//! battery removes that crutch: the planted fact shares no tokens with the
//! question, so answering requires resolving an attribute, a synonym, an
//! ordinal, a unit, or a chain of two hops.
//!
//! The softness is in the QUERY, not the judging. Scoring stays keyword-based
//! — cheap, deterministic, no judge model — but `expect_any` is any-of rather
//! than require-all, because soft answers legitimately vary in phrasing and
//! an all-of rule would re-import exactly the brittleness this exists to
//! escape.
//!
//! **Authoring rule, enforced mechanically by [`authoring_violations`]:** if
//! an expected answer also appears in the question, the probe silently
//! degenerates into an ordinary `longctx` probe and measures nothing. That
//! failure is invisible in the results — the probe still passes, it just
//! stops testing indirection — so it is checked before any GPU time is spent
//! rather than left to review.
//!
//! ABSENCE probes invert the metric: they ask about something the document
//! never says, and a correct reply declines to answer. They measure
//! hallucination, which nothing else here covers, so they are reported as
//! their own rate instead of being averaged into the retrieval rate.
//!
//! Non-blocking: the rates are REPORTED and excluded from pass/fail. A probe
//! counts as passed if it ran at all. Anyone who wants a floor opts in with
//! `--gate 'soft_pct>=baseline'`.

/// One indirect-retrieval probe.
#[derive(serde::Deserialize)]
pub(crate) struct SoftProbe {
    pub(crate) id: String,
    /// Fixture document path, relative to the spec file's directory.
    pub(crate) doc: String,
    /// Truncate the document to this many tokens first; 0 (the default) uses
    /// the whole document. Present so soft probes can later be laddered by
    /// depth the way `longctx` is, without a fixture migration.
    #[serde(default)]
    pub(crate) doc_tokens: usize,
    pub(crate) question: String,
    /// ANY of these in the reply counts as found (whole-word run match).
    pub(crate) expect_any: Vec<String>,
    /// Which kind of indirection this exercises. Reporting only — it makes a
    /// result table say WHICH reasoning shape failed, not just how many did.
    pub(crate) class: String,
    /// Inverts the metric: the document does NOT contain this, and a correct
    /// reply says so.
    #[serde(default)]
    pub(crate) absence: bool,
    pub(crate) max_steps: usize,
}

/// Retrieval and hallucination tallies. Kept apart because ABSENCE inverts:
/// averaging them would let confident hallucination cancel out good recall.
#[derive(Default, Clone, Debug)]
pub(crate) struct SoftCounts {
    pub(crate) found: u64,
    pub(crate) total: u64,
    pub(crate) absence_ok: u64,
    pub(crate) absence_total: u64,
}

impl SoftCounts {
    pub(crate) fn merge(&mut self, o: &Self) {
        self.found += o.found;
        self.total += o.total;
        self.absence_ok += o.absence_ok;
        self.absence_total += o.absence_total;
    }
}

/// Did the reply contain any accepted answer?
pub(crate) fn matched(reply: &str, probe: &SoftProbe) -> bool {
    probe
        .expect_any
        .iter()
        .any(|e| super::smoketest::smoke_answer_matches(reply, e, &[]))
}

/// Probes whose expected answer leaks into their own question. Any hit here
/// means the probe is not measuring indirection, so this is a hard error at
/// startup, not a warning.
pub(crate) fn authoring_violations(probes: &[SoftProbe]) -> Vec<String> {
    let mut bad = Vec::new();
    for p in probes {
        for e in &p.expect_any {
            if super::smoketest::smoke_answer_matches(&p.question, e, &[]) {
                bad.push(format!(
                    "soft probe {:?}: expected answer {:?} also appears in its own question \
                     — the probe degenerates into a plain longctx lookup",
                    p.id, e
                ));
            }
        }
    }
    bad
}

/// The question as put to the model. The "if the document does not say"
/// clause is deliberate and applies to EVERY probe, not just absence ones:
/// without it, a refusal is unavailable to the model and the absence probes
/// would measure whether it invents permission to decline rather than whether
/// it hallucinates. Uniform wording keeps the two classes comparable.
pub(crate) fn prompt_for(excerpt: &str, question: &str) -> String {
    format!(
        "{excerpt}\n\n[end of document]\nAnswer using only the document above. \
         If the document does not contain the answer, say that it is not mentioned. \
         Question: {question}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(question: &str, expect: &[&str]) -> SoftProbe {
        SoftProbe {
            id: "t".into(),
            doc: "d".into(),
            doc_tokens: 0,
            question: question.into(),
            expect_any: expect.iter().map(|s| (*s).to_string()).collect(),
            class: "test".into(),
            absence: false,
            max_steps: 60,
        }
    }

    #[test]
    fn any_of_not_all_of() {
        let p = probe("how often?", &["fortnight", "two weeks", "14 days"]);
        assert!(matched("Every fortnight.", &p));
        assert!(matched("About two weeks apart.", &p), "any-of, not all-of");
        assert!(!matched("Every month.", &p));
    }

    /// The authoring rule is the whole point of the battery, so it is checked
    /// mechanically rather than trusted to review.
    #[test]
    fn answer_leaking_into_the_question_is_rejected() {
        let leaky = probe("Was the collar mauve or green?", &["mauve"]);
        assert_eq!(authoring_violations(std::slice::from_ref(&leaky)).len(), 1);

        let sound = probe("An animal wore a collar. What colour was it?", &["mauve"]);
        assert!(authoring_violations(std::slice::from_ref(&sound)).is_empty());
    }

    /// The shipped fixture must obey its own rule — a regression here would
    /// silently turn soft probes back into longctx probes.
    #[test]
    fn shipped_soft_probes_obey_the_authoring_rule() {
        let text = std::fs::read_to_string("fixtures/smoketest/prompts.json")
            .expect("fixture readable from the crate root");
        let spec: serde_json::Value = serde_json::from_str(&text).unwrap();
        let probes: Vec<SoftProbe> =
            serde_json::from_value(spec["soft"].clone()).expect("soft probes parse");
        assert!(probes.len() >= 8, "expected a class-covering set");
        let bad = authoring_violations(&probes);
        assert!(bad.is_empty(), "{bad:#?}");
        // Every probe should name its indirection class, and at least one
        // must be an ABSENCE probe or the hallucination rate is vacuous.
        assert!(probes.iter().all(|p| !p.class.is_empty()));
        assert!(probes.iter().any(|p| p.absence));
    }
}
