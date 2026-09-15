//! `content` battery — scores WHAT a free-form reply says. Non-blocking.
//!
//! The commit gate's `convergence` probes ask long questions and check only
//! that the canvas converged within budget; the reply itself is never read.
//! `adherence` reads the reply but only on prompts with a one-word answer.
//! Nothing measured long-answer quality, so a port (or a sampler change) that
//! converged on fluent nonsense scored the same as one that answered.
//!
//! Judging stays keyword-based, deterministic and judge-model-free, in the
//! house style. A probe carries a RUBRIC: groups of alternatives, ANY member
//! of a group counts as that group hit, and the rate is groups hit over
//! groups. Any-of inside a group because free-form answers vary in wording
//! ("scattering" / "scattered" / "Rayleigh"); all groups required for a FULL
//! verdict because leaving out condensation is not a summary of the water
//! cycle. `forbid` names things a correct reply must not say (the wrong
//! primary colours). Structure checks (`lines`, `sentences`, word bounds)
//! cover the prompts whose instruction IS the form: a haiku has three lines
//! whatever it says.
//!
//! **Authoring rule, enforced by [`authoring_violations`]:** no rubric
//! alternative may appear in its own prompt. "Why is the sky blue?" scored on
//! "blue" is satisfied by a reply that repeats the question. Same reasoning as
//! the `soft` battery, same mechanical check, same test pinning the fixture.
//!
//! Non-blocking: long answers are trajectory-sensitive at bf16 precision
//! (the port's two arena arms differ from each other as much as either does
//! from the engine), so a single-seed pass/fail here would be arbitrary. The
//! rates are reported and a floor is opt-in: `--gate 'content_pct>=baseline'`.

/// One content probe.
#[derive(serde::Deserialize)]
pub(crate) struct ContentProbe {
    pub(crate) id: String,
    /// Which answer shape this exercises (explain, list, form, summary,
    /// compare). Reporting only.
    pub(crate) class: String,
    pub(crate) prompt: String,
    /// Groups of alternatives. A group is hit when ANY alternative appears in
    /// the reply as a whole word run; the rate is groups hit / groups.
    pub(crate) rubric: Vec<Vec<String>>,
    /// Whole-word runs that must NOT appear. Each hit is a wrong-content
    /// finding and blocks the FULL verdict.
    #[serde(default)]
    pub(crate) forbid: Vec<String>,
    /// Non-empty lines the reply must have exactly (a haiku is 3).
    #[serde(default)]
    pub(crate) lines: Option<usize>,
    /// Sentences the reply must have exactly ("in two sentences").
    #[serde(default)]
    pub(crate) sentences: Option<usize>,
    #[serde(default)]
    pub(crate) min_words: usize,
    /// 0 = unbounded.
    #[serde(default)]
    pub(crate) max_words: usize,
    /// Runaway guard and one FULL criterion; not a ratchet.
    pub(crate) max_steps: usize,
}

/// Battery tallies. `rubric_*` is the headline rate; `full` counts probes
/// that met every criterion at once (rubric, forbid, structure, budget),
/// which is the number a quality claim about long answers should quote.
#[derive(Default, Clone, Debug)]
pub(crate) struct ContentCounts {
    pub(crate) rubric_hit: u64,
    pub(crate) rubric_total: u64,
    pub(crate) forbid_hits: u64,
    pub(crate) structure_ok: u64,
    pub(crate) structure_total: u64,
    pub(crate) full: u64,
    pub(crate) probes: u64,
}

/// What one reply scored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Verdict {
    /// Per rubric group.
    pub(crate) hit: Vec<bool>,
    /// The `forbid` entries that appeared.
    pub(crate) forbidden: Vec<String>,
    pub(crate) words: usize,
    pub(crate) lines: usize,
    pub(crate) sentences: usize,
    pub(crate) structure_ok: bool,
    pub(crate) converged: bool,
}

impl Verdict {
    pub(crate) fn hits(&self) -> usize {
        self.hit.iter().filter(|&&h| h).count()
    }
    /// Every criterion met at once.
    pub(crate) fn full(&self) -> bool {
        self.hit.iter().all(|&h| h)
            && self.forbidden.is_empty()
            && self.structure_ok
            && self.converged
    }
}

/// Non-empty lines.
pub(crate) fn count_lines(reply: &str) -> usize {
    reply.lines().filter(|l| !l.trim().is_empty()).count()
}

/// Runs of `.`, `!` or `?` that end the text or are followed by whitespace.
/// "3.5" does not count; "e.g. this" does, which is an accepted cost on the
/// prompts this battery asks.
pub(crate) fn count_sentences(reply: &str) -> usize {
    let cs: Vec<char> = reply.chars().collect();
    let mut n = 0;
    let mut i = 0;
    while i < cs.len() {
        if matches!(cs[i], '.' | '!' | '?') {
            let mut j = i;
            while j < cs.len() && matches!(cs[j], '.' | '!' | '?') {
                j += 1;
            }
            // Closing quotes or brackets may sit between the terminator and
            // the whitespace.
            let mut k = j;
            while k < cs.len() && matches!(cs[k], '"' | '\'' | ')' | ']' | '*' | '_') {
                k += 1;
            }
            if k >= cs.len() || cs[k].is_whitespace() {
                n += 1;
            }
            i = j;
        } else {
            i += 1;
        }
    }
    n
}

pub(crate) fn count_words(reply: &str) -> usize {
    reply.split_whitespace().count()
}

/// Score one reply. `steps` is the committed denoise steps the reply took.
pub(crate) fn judge(reply: &str, probe: &ContentProbe, steps: usize) -> Verdict {
    let matches = |term: &str| super::smoketest::smoke_answer_matches(reply, term, &[]);
    let hit = probe
        .rubric
        .iter()
        .map(|group| group.iter().any(|alt| matches(alt)))
        .collect();
    let forbidden = probe
        .forbid
        .iter()
        .filter(|f| matches(f))
        .cloned()
        .collect();
    let words = count_words(reply);
    let lines = count_lines(reply);
    let sentences = count_sentences(reply);
    let structure_ok = probe.lines.is_none_or(|n| n == lines)
        && probe.sentences.is_none_or(|n| n == sentences)
        && words >= probe.min_words
        && (probe.max_words == 0 || words <= probe.max_words);
    Verdict {
        hit,
        forbidden,
        words,
        lines,
        sentences,
        structure_ok,
        converged: steps <= probe.max_steps,
    }
}

/// Probes that cannot measure what they claim. A rubric term echoed from the
/// prompt scores a reply for repeating the question; an empty rubric scores
/// nothing; inverted word bounds can never pass.
pub(crate) fn authoring_violations(probes: &[ContentProbe]) -> Vec<String> {
    let mut bad = Vec::new();
    for p in probes {
        if p.rubric.is_empty() || p.rubric.iter().any(Vec::is_empty) {
            bad.push(format!("content probe {:?}: empty rubric group", p.id));
        }
        for alt in p.rubric.iter().flatten() {
            if super::smoketest::smoke_answer_matches(&p.prompt, alt, &[]) {
                bad.push(format!(
                    "content probe {:?}: rubric term {:?} also appears in its own prompt \
                     — a reply that repeats the question would score it",
                    p.id, alt
                ));
            }
        }
        if p.max_words != 0 && p.min_words > p.max_words {
            bad.push(format!(
                "content probe {:?}: min_words {} > max_words {}",
                p.id, p.min_words, p.max_words
            ));
        }
    }
    bad
}

/// What a battery run produced, beyond the tallies: the (passed, total,
/// failures) triple the battery loop keeps, and every full reply so a run can
/// be re-judged after a rubric edit without a model (`--replies-out`).
pub(crate) struct RunResult {
    pub(crate) counts: ContentCounts,
    pub(crate) passed: usize,
    pub(crate) total: usize,
    pub(crate) failures: Vec<String>,
    pub(crate) replies: std::collections::BTreeMap<String, ReplyRecord>,
}

/// Run every probe through `run_one` and print one line each. A probe that
/// RAN counts as passed, the rates carry the quality, and only a probe that
/// could not run is a failure. `run_one` may be the live session or a lookup
/// into pre-generated replies (`smoketest --replies`); the judge does not
/// know which.
pub(crate) fn run_probes(
    probes: &[ContentProbe],
    run_one: &mut dyn FnMut(&str, &str) -> Result<(usize, String), crate::Error>,
) -> RunResult {
    let mut counts = ContentCounts::default();
    let (mut passed, mut total) = (0usize, 0usize);
    let mut failures = Vec::new();
    let mut replies = std::collections::BTreeMap::new();
    for p in probes {
        total += 1;
        let (st, reply) = match run_one(&p.id, &p.prompt) {
            Ok(v) => v,
            Err(err) => {
                println!("  {:<22} ERROR  {err}", p.id);
                failures.push(p.id.clone());
                continue;
            }
        };
        passed += 1;
        replies.insert(
            p.id.clone(),
            ReplyRecord {
                reply: reply.clone(),
                steps: st,
            },
        );
        let v = judge(&reply, p, st);
        counts.probes += 1;
        counts.rubric_hit += v.hits() as u64;
        counts.rubric_total += p.rubric.len() as u64;
        counts.forbid_hits += v.forbidden.len() as u64;
        counts.structure_total += 1;
        counts.structure_ok += u64::from(v.structure_ok);
        counts.full += u64::from(v.full());
        let missing: Vec<&str> = p
            .rubric
            .iter()
            .zip(&v.hit)
            .filter(|(_, h)| !**h)
            .map(|(g, _)| g[0].as_str())
            .collect();
        let mut notes = Vec::new();
        if !missing.is_empty() {
            notes.push(format!("missing {missing:?}"));
        }
        if !v.forbidden.is_empty() {
            notes.push(format!("FORBIDDEN {:?}", v.forbidden));
        }
        if !v.structure_ok {
            let mut want = Vec::new();
            if let Some(n) = p.lines {
                want.push(format!("lines {}/{n}", v.lines));
            }
            if let Some(n) = p.sentences {
                want.push(format!("sentences {}/{n}", v.sentences));
            }
            if p.min_words > 0 || p.max_words > 0 {
                want.push(format!("words {} not in {}..{}", v.words, p.min_words, p.max_words));
            }
            notes.push(format!("structure [{}]", want.join(", ")));
        }
        if !v.converged {
            notes.push("over budget".to_string());
        }
        let prev = reply
            .chars()
            .take(48)
            .collect::<String>()
            .replace('\n', " ");
        println!(
            "  {id:<22} {mark:<4} {cls:<8} rubric {h}/{t} steps {st:>3}/{max:<3} {notes} | {prev}",
            id = p.id,
            mark = if v.full() { "FULL" } else { "part" },
            cls = p.class,
            h = v.hits(),
            t = p.rubric.len(),
            max = p.max_steps,
            notes = notes.join("; "),
        );
    }
    RunResult {
        counts,
        passed,
        total,
        failures,
        replies,
    }
}

/// Print the battery's rate line.
pub(crate) fn report(c: &ContentCounts) {
    let pct = |a: u64, b: u64| if b == 0 { 0.0 } else { 100.0 * a as f64 / b as f64 };
    println!(
        "content: rubric {}/{} ({:.1}%)  full {}/{} ({:.1}%)  structure {}/{}  forbidden hits {}  [rates only — not pass/fail]",
        c.rubric_hit,
        c.rubric_total,
        pct(c.rubric_hit, c.rubric_total),
        c.full,
        c.probes,
        pct(c.full, c.probes),
        c.structure_ok,
        c.structure_total,
        c.forbid_hits,
    );
}

/// Pre-generated replies for `smoketest --replies FILE`, and what
/// `--replies-out FILE` writes: `{ "<probe id>": {"reply": "...", "steps":
/// N}, ... }`. `steps` defaults to 0, which counts as converged; a scorer for
/// another engine's output supplies it when it has it.
#[derive(serde::Deserialize, serde::Serialize)]
pub(crate) struct ReplyRecord {
    pub(crate) reply: String,
    #[serde(default)]
    pub(crate) steps: usize,
}

pub(crate) fn load_replies(
    path: &std::path::Path,
) -> Result<std::collections::BTreeMap<String, ReplyRecord>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))
}

/// Write the replies a run produced, in the shape `load_replies` reads.
pub(crate) fn save_replies(
    path: &std::path::Path,
    replies: &std::collections::BTreeMap<String, ReplyRecord>,
) {
    match serde_json::to_string_pretty(replies) {
        Ok(text) => {
            if let Err(e) = std::fs::write(path, text) {
                eprintln!("content: cannot write {}: {e}", path.display());
            }
        }
        Err(e) => eprintln!("content: cannot encode replies: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(prompt: &str, rubric: &[&[&str]]) -> ContentProbe {
        ContentProbe {
            id: "t".into(),
            class: "test".into(),
            prompt: prompt.into(),
            rubric: rubric
                .iter()
                .map(|g| g.iter().map(|s| (*s).to_string()).collect())
                .collect(),
            forbid: vec![],
            lines: None,
            sentences: None,
            min_words: 0,
            max_words: 0,
            max_steps: 60,
        }
    }

    #[test]
    fn groups_are_any_of_and_full_needs_every_group() {
        let p = probe(
            "Why?",
            &[&["rayleigh", "scattering"], &["wavelength", "wavelengths"]],
        );
        let v = judge("Rayleigh scattering favours short wavelengths.", &p, 3);
        assert_eq!(v.hit, vec![true, true]);
        assert!(v.full());
        let v = judge("Because of scattering.", &p, 3);
        assert_eq!(v.hit, vec![true, false]);
        assert_eq!(v.hits(), 1);
        assert!(!v.full());
    }

    #[test]
    fn forbid_and_budget_block_full_but_not_the_rubric_rate() {
        let mut p = probe("Which?", &[&["red"], &["blue"]]);
        p.forbid = vec!["purple".into()];
        let v = judge("Red, blue and purple.", &p, 3);
        assert_eq!(v.hits(), 2);
        assert_eq!(v.forbidden, vec!["purple".to_string()]);
        assert!(!v.full());
        let v = judge("Red and blue.", &p, 61);
        assert_eq!(v.hits(), 2);
        assert!(!v.converged);
        assert!(!v.full());
    }

    #[test]
    fn structure_counts_lines_sentences_and_words() {
        assert_eq!(count_lines("a\n\n  \nb\nc\n"), 3);
        assert_eq!(count_sentences("One. Two! Three?"), 3);
        assert_eq!(count_sentences("Pi is 3.14 here. Done..."), 2);
        assert_eq!(count_sentences("He said \"stop.\" Then left."), 2);
        assert_eq!(count_sentences("no terminator"), 0);
        assert_eq!(count_words("  a  b\nc "), 3);

        let mut p = probe("Haiku?", &[&["sea"]]);
        p.lines = Some(3);
        p.max_words = 20;
        let ok = judge("Sea under the moon\nwaves fold into silver foam\nthe tide keeps its time", &p, 2);
        assert!(ok.structure_ok);
        let two = judge("Sea under the moon\nwaves fold into silver foam", &p, 2);
        assert!(!two.structure_ok);
        assert!(!two.full());

        let mut p = probe("Summary?", &[&["evaporation"]]);
        p.sentences = Some(2);
        assert!(judge("Evaporation lifts it. Rain returns it.", &p, 2).structure_ok);
        assert!(!judge("Evaporation lifts it. Rain returns it. Again.", &p, 2).structure_ok);
    }

    #[test]
    fn rubric_terms_echoed_from_the_prompt_are_rejected() {
        let leaky = probe("Why is the sky blue?", &[&["blue"]]);
        assert_eq!(authoring_violations(std::slice::from_ref(&leaky)).len(), 1);
        let sound = probe("Why is the sky blue?", &[&["scattering"]]);
        assert!(authoring_violations(std::slice::from_ref(&sound)).is_empty());
        let empty = probe("Why?", &[]);
        assert_eq!(authoring_violations(std::slice::from_ref(&empty)).len(), 1);
        let mut inverted = probe("Why?", &[&["x"]]);
        inverted.min_words = 5;
        inverted.max_words = 2;
        assert_eq!(authoring_violations(std::slice::from_ref(&inverted)).len(), 1);
    }

    /// A probe that cannot run is the only failure; a probe that ran counts as
    /// passed whatever it said, and the tallies carry the quality.
    #[test]
    fn ran_is_passed_and_tallies_carry_the_rates() {
        let mut a = probe("A?", &[&["alpha"], &["beta"]]);
        a.id = "a".into();
        let mut b = probe("B?", &[&["gamma"]]);
        b.id = "b".into();
        let mut c = probe("C?", &[&["delta"]]);
        c.id = "c".into();
        let mut run = |id: &str, _prompt: &str| -> Result<(usize, String), crate::Error> {
            match id {
                "a" => Ok((3, "alpha only".into())),
                "b" => Ok((99, "gamma".into())),
                _ => Err(crate::Error::Config("no reply".into())),
            }
        };
        let r = run_probes(&[a, b, c], &mut run);
        assert_eq!((r.passed, r.total), (2, 3));
        assert_eq!(r.failures, vec!["c".to_string()]);
        assert_eq!(r.counts.probes, 2);
        assert_eq!((r.counts.rubric_hit, r.counts.rubric_total), (2, 3));
        // b hit its rubric but blew the budget, so nothing is FULL.
        assert_eq!(r.counts.full, 0);
        assert_eq!(r.replies.len(), 2);
        assert_eq!(r.replies["b"].steps, 99);
    }

    /// The shipped fixture must obey its own rule, and cover every answer
    /// shape the module doc promises.
    #[test]
    fn shipped_content_probes_obey_the_authoring_rule() {
        let text = std::fs::read_to_string("fixtures/smoketest/prompts.json")
            .expect("fixture readable from the crate root");
        let spec: serde_json::Value = serde_json::from_str(&text).unwrap();
        let probes: Vec<ContentProbe> =
            serde_json::from_value(spec["content"].clone()).expect("content probes parse");
        assert!(probes.len() >= 8, "expected a shape-covering set");
        let bad = authoring_violations(&probes);
        assert!(bad.is_empty(), "{bad:#?}");
        for shape in ["explain", "list", "form", "summary", "compare"] {
            assert!(probes.iter().any(|p| p.class == shape), "no {shape} probe");
        }
        // Form probes must actually check a form.
        assert!(probes
            .iter()
            .filter(|p| p.class == "form")
            .all(|p| p.lines.is_some() || p.sentences.is_some()));
        let mut ids: Vec<&str> = probes.iter().map(|p| p.id.as_str()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), probes.len(), "duplicate probe id");
    }
}
