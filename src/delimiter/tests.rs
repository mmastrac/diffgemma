use super::*;
use std::sync::{Mutex, MutexGuard};

/// The counters are process-global, so any test that reads a DELTA races every
/// other test running in parallel. All checking in this module goes through
/// this lock; the counter test holds it across its whole body.
static COUNTERS: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    COUNTERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// One checked block, serialized against the counter test.
fn check(text: &str, block_start: usize, mode: BlockMode, term: bool) -> Report {
    let _g = lock();
    check_block(text, block_start, mode, term)
}

/// Check a standalone reply (no prior blocks) that ended on a stop token.
fn terminated(text: &str, mode: BlockMode) -> Report {
    check(text, 0, mode, true)
}

/// ...and one that ran to the 256-token canvas edge.
fn at_edge(text: &str, mode: BlockMode) -> Report {
    check(text, 0, mode, false)
}

fn checks(r: &Report) -> Vec<&'static str> {
    let mut v: Vec<_> = r.findings.iter().map(|f| f.check).collect();
    v.sort_unstable();
    v.dedup();
    v
}

// ---------------------------------------------------------------------------
// The measured defect class (ARCHITECTURE Part I): convention blends.
// ---------------------------------------------------------------------------

/// The two specimens recorded in ARCHITECTURE, both single stray `"` in
/// otherwise-correct bash. Neither is detectable by confidence (they commit at
/// p_max ~1.0); both leave an odd `"` count on their line.
#[test]
fn convention_blends_are_caught_by_line_parity() {
    for line in [
        "if [ $#\" -lt 1 ]; then",
        "if [[ $f == *$SEARCH\"* ]]; then",
    ] {
        let text = format!("```bash\n#!/bin/bash\n{line}\n  echo hi\nfi\n```");
        let r = terminated(&text, BlockMode::Code);
        assert!(r.failed(), "missed the blend in: {line}");
        assert!(checks(&r).contains(&"quote_line"), "{:?}", r.findings);
    }
}

/// Correct bash must be silent — including the forms the blends were blending
/// BETWEEN, which is the comparison that matters.
#[test]
fn the_valid_forms_the_blend_mixes_are_silent() {
    for line in [
        "if [ $# -lt 1 ]; then",
        "if [ \"$#\" -ne 1 ]; then",
        "if [[ $f == *$SEARCH* ]]; then",
        "if [[ $f == *\"$SEARCH\"* ]]; then",
    ] {
        let text = format!("```bash\n#!/bin/bash\n{line}\n  echo hi\nfi\n```");
        let r = terminated(&text, BlockMode::Code);
        assert!(!r.failed(), "false positive on {line}: {:?}", r.findings);
    }
}

/// MEASURED false positive (`programmatic`, 3 blocks, 6 findings):
/// the cross-quoting idiom makes BOTH per-line quote counts odd in perfectly
/// correct python. This is why `quote_line` and `quote_region` are reported as
/// separate checks — on the same run `quote_region` scored 2/2 precision and
/// caught both real blends, while `quote_line` scored 2/8. The sensitive check
/// is pinned here as KNOWN-NOISY rather than quietly fixed: what it buys over
/// the state machine is a blend whose stray quote gets re-paired later, and
/// that did not occur once in 54 judgeable blocks.
#[test]
fn cross_quoting_is_a_known_quote_line_false_positive() {
    let text = "import sys\nprint(sys.argv[1].replace(\"'\", '\"'))\n";
    let r = terminated(text, BlockMode::Code);
    let by = |c: &str| r.findings.iter().filter(|f| f.check == c).count();
    assert_eq!(
        by("quote_line"),
        2,
        "the noisy check fires: {:?}",
        r.findings
    );
    assert_eq!(
        by("quote_region"),
        0,
        "the state machine must NOT: {:?}",
        r.findings
    );
    assert_eq!(by("bracket"), 0, "{:?}", r.findings);
}

/// The two ARCHITECTURE convention-blend specimens as they were ACTUALLY
/// emitted by the model during the observational run — bare, unfenced bash.
/// `quote_region` caught both; this pins that it still does.
#[test]
fn measured_live_blends_are_caught_by_the_specific_check() {
    for text in [
        "#!/bin/bash\nif [ $#\" -lt 1 ]; then\n  echo usage\n  exit 1\nfi\n",
        "#!/bin/bash\nwhile read -r line; do\nif [[ \"$line\" == *$SEARCH\"* ]]; then\necho hit\nfi\ndone\n",
    ] {
        let r = terminated(text, BlockMode::Code);
        assert!(
            r.findings.iter().any(|f| f.check == "quote_region"),
            "quote_region missed a measured live blend: {:?}",
            r.findings
        );
    }
}

// ---------------------------------------------------------------------------
// Gate 1: block mode. Prose is the dominant false-positive source.
// ---------------------------------------------------------------------------

/// Prose that trips every naive check — apostrophes, a smiley, a lone paren —
/// must produce nothing, because the content checks never run.
#[test]
fn prose_blocks_skip_content_checks_entirely() {
    let text =
        "Don't worry :) it's fine — see the note (below.\nIt's a \"quote that never closes\n";
    let r = terminated(text, BlockMode::Prose);
    assert!(!r.failed(), "{:?}", r.findings);
    assert!(r.prose_skipped);
}

/// The same text WITHOUT the gate is what the gate is buying: if this stopped
/// flagging, the prose gate would be measuring nothing.
#[test]
fn the_same_prose_does_trip_when_ungated() {
    let text = "```python\nDon't worry (a lot\n```";
    let r = terminated(text, BlockMode::Code);
    assert!(r.failed(), "the gate would have nothing to suppress");
}

/// A fenced code block is checked even when the surrounding block is code —
/// but an UNKNOWN mode (no probe loaded) must still check fences, so a run
/// without `DGQ_TOKEN_CLASS` measures the ungated rate rather than nothing.
#[test]
fn unknown_mode_still_checks_fences() {
    let text = "Here you go:\n```bash\necho \"unclosed\n```\n";
    let r = terminated(text, BlockMode::Unknown);
    assert!(r.failed(), "{:?}", r.findings);
    assert!(!r.prose_skipped);
}

/// Unknown mode must not scan UNSNIFFABLE bare text as code: that is the prose
/// false-positive class arriving through the back door.
#[test]
fn unknown_mode_does_not_scan_unsniffable_bare_text() {
    let r = terminated("Don't do that (yet.\n", BlockMode::Unknown);
    assert!(!r.failed(), "{:?}", r.findings);
    assert!(r.vacuous(), "nothing was scannable");
}

/// ...but it DOES scan a span the sniffer is sure about. `#!/bin/bash` is code
/// on its own evidence; requiring a probe to say so would leave the checker
/// useless in the default configuration, where no probe is loaded.
#[test]
fn unknown_mode_scans_confidently_sniffed_bare_code() {
    let text = "#!/bin/bash\nif [ $#\" -lt 1 ]; then\n  echo usage\nfi\n";
    let r = terminated(text, BlockMode::Unknown);
    assert!(r.failed(), "{:?}", r.findings);
    assert_eq!(r.findings[0].lang, "bash");
}

// ---------------------------------------------------------------------------
// Gate 2: the block-edge rule.
// ---------------------------------------------------------------------------

/// A block that ran out of canvas legitimately leaves brackets open for the
/// next block; only a terminated block may be judged on trailing balance.
#[test]
fn trailing_imbalance_is_judged_only_on_terminated_blocks() {
    let text = "def f():\n    return foo(bar\n";
    assert!(terminated(text, BlockMode::Code).failed());
    assert!(!at_edge(text, BlockMode::Code).failed());
}

/// An UNDERFLOW is edge-independent: a closer with no opener is wrong however
/// the block ended.
#[test]
fn underflow_is_judged_at_the_canvas_edge_too() {
    let text = "    y = x))\n    z = 1\n";
    let r = at_edge(text, BlockMode::Code);
    assert!(r.failed(), "{:?}", r.findings);
    assert_eq!(r.findings[0].kind, Kind::Underflow);
}

/// A CLOSED fence is a complete construct, so its interior is judgeable on
/// trailing balance even when the block itself hit the canvas edge.
#[test]
fn closed_fence_interiors_are_complete_constructs() {
    let text = "```python\nx = foo(1\n```\nand then";
    let r = at_edge(text, BlockMode::Code);
    assert!(r.failed(), "{:?}", r.findings);
}

/// An unclosed fence at the canvas edge is a fence that continues next block.
#[test]
fn unclosed_fence_is_only_a_finding_when_terminated() {
    let text = "Here:\n```bash\necho hi\n";
    assert!(!at_edge(text, BlockMode::Code).failed());
    let r = terminated(text, BlockMode::Code);
    assert!(checks(&r).contains(&"fence"), "{:?}", r.findings);
}

/// One stray closer must not cascade into a finding per subsequent closer.
#[test]
fn a_stray_closer_reports_once() {
    let text = "```python\na = )\nb = (1)\nc = (2)\nd = (3)\n```";
    let r = terminated(text, BlockMode::Code);
    assert_eq!(
        r.findings.iter().filter(|f| f.check == "bracket").count(),
        1,
        "{:?}",
        r.findings
    );
}

// ---------------------------------------------------------------------------
// Multi-line constructs: handled, not skipped.
// ---------------------------------------------------------------------------

/// A bash heredoc's body is arbitrary text and routinely holds unbalanced
/// quotes and braces. Consumed opaquely.
#[test]
fn bash_heredocs_do_not_break_parity() {
    let text = "```bash\ncat <<EOF\nit's got a \" and a } in it\nEOF\necho done\n```";
    let r = terminated(text, BlockMode::Code);
    assert!(!r.failed(), "{:?}", r.findings);
}

#[test]
fn quoted_and_dash_heredoc_tags_are_recognised() {
    let text = "```bash\ncat <<-'EOF'\nunbalanced \" here\n\tEOF\necho ok\n```";
    assert!(!terminated(text, BlockMode::Code).failed());
}

/// `$#` and `${x#y}` are not comments; a real `#` comment is.
#[test]
fn bash_hash_is_a_comment_only_at_a_word_start() {
    let ok = "```bash\nn=$#\necho ${p#foo}\n# it's a comment with \" in it\necho \"$n\"\n```";
    assert!(!terminated(ok, BlockMode::Code).failed());
}

/// Python triple-quoted strings hold anything, including odd quote counts.
#[test]
fn python_triple_quotes_are_opaque() {
    let text = "```python\ndef f():\n    \"\"\"Don't (worry \" about it.\"\"\"\n    return 1\n```";
    let r = terminated(text, BlockMode::Code);
    assert!(!r.failed(), "{:?}", r.findings);
}

/// A rust lifetime is an unpaired `'` in CORRECT code — the language-level
/// version of the prose apostrophe, and the reason `'` is excluded for rust.
#[test]
fn rust_lifetimes_are_not_quote_imbalances() {
    let text = "```rust\nfn f<'a>(s: &'a str) -> &'a str { s }\n```";
    let r = terminated(text, BlockMode::Code);
    assert!(!r.failed(), "{:?}", r.findings);
}

#[test]
fn rust_raw_strings_are_opaque() {
    let text = "```rust\nfn main() { let s = r#\"a \" and a ( here\"#; println!(\"{}\", s); }\n```";
    let r = terminated(text, BlockMode::Code);
    assert!(!r.failed(), "{:?}", r.findings);
}

/// JS template literals span lines.
#[test]
fn js_template_literals_are_opaque() {
    let text = "```js\nconst s = `line one \"\nline two (`;\nconsole.log(s);\n```";
    assert!(!terminated(text, BlockMode::Code).failed());
}

/// A language with no spec gets brackets only — an explicit, recorded skip
/// rather than a stream of false positives from unmodelled syntax.
#[test]
fn unknown_languages_skip_quote_checks_but_keep_brackets() {
    let quotes = "```haskell\nmain = putStrLn \"unbalanced\n```";
    assert!(!terminated(quotes, BlockMode::Code).failed());
    let brackets = "```haskell\nmain = f (1\n```";
    let r = terminated(brackets, BlockMode::Code);
    assert!(r.failed());
    assert_eq!(r.findings[0].lang, "haskell");
}

/// A single-quoted bash string makes an inner `"` literal, so the region
/// state machine stays quiet where naive parity would not.
#[test]
fn single_quoted_bash_hides_inner_double_quotes() {
    let text = "```bash\necho 'say \"hi\" twice'\necho done\n```";
    let r = terminated(text, BlockMode::Code);
    assert!(!r.failed(), "{:?}", r.findings);
}

/// Escaped quotes do not count toward parity.
#[test]
fn escaped_quotes_are_not_counted() {
    let text = "```python\nprint(\"a \\\" b\")\n```";
    assert!(!terminated(text, BlockMode::Code).failed());
}

// ---------------------------------------------------------------------------
// Reuse: the tool grammar is checked by the tools module, not re-implemented.
// ---------------------------------------------------------------------------

/// A COMPLETE call's arg body is a masked range precisely because
/// `read_braced` balanced it, so the bracket scan must not re-examine it —
/// and must not be fooled by braces and quotes inside a written file body.
#[test]
fn complete_tool_call_args_are_not_rescanned() {
    let text = "<|tool_call>call:write{path:<|\"|>a.sh<|\"|>,content:<|\"|>if [ $x ]; then }\n<|\"|>}<tool_call|>";
    let r = terminated(text, BlockMode::Code);
    assert!(!r.failed(), "{:?}", r.findings);
}

/// An unbalanced call IS a structural failure, and it comes from the existing
/// validator rather than a second bracket counter.
#[test]
fn incomplete_tool_call_is_a_finding_when_terminated() {
    let text = "<|tool_call>call:write{path:<|\"|>a.sh<|\"|>,content:<|\"|>oops";
    let r = terminated(text, BlockMode::Unknown);
    assert!(checks(&r).contains(&"tool_grammar"), "{:?}", r.findings);
    // ...but at the canvas edge the call simply continues into the next block.
    assert!(!at_edge(text, BlockMode::Unknown).failed());
}

/// The tool grammar is PROTOCOL, so it is checked even in a prose block.
#[test]
fn tool_grammar_is_checked_even_in_prose_blocks() {
    let text = "<|tool_call>call:write{path:<|\"|>a.sh<|\"|>,content:<|\"|>oops";
    let r = terminated(text, BlockMode::Prose);
    assert!(r.prose_skipped);
    assert!(checks(&r).contains(&"tool_grammar"), "{:?}", r.findings);
}

/// Markers quoted inside a code fence are DATA. Scanning them as protocol is
/// the failure `masked_ranges` exists to prevent: this reply demonstrates the
/// grammar, it does not speak it.
#[test]
fn markers_quoted_in_a_fence_are_data() {
    let text = "Example of the grammar:\n```\ncall:write{path:<|\"|>x<|\"|>}\n```\nThat's it.";
    let r = terminated(text, BlockMode::Unknown);
    assert!(!checks(&r).contains(&"tool_grammar"), "{:?}", r.findings);
    assert!(!r.failed(), "{:?}", r.findings);
}

/// An UNLABELLED fence still gets the bracket scan. Recorded as a decision,
/// not an accident: brackets nest in essentially every language, so the check
/// is language-agnostic in a way quote parity is not. But unlabelled fences
/// also hold log output, ASCII tables and pseudocode, so this is the most
/// likely source of false positives in the measurement — if the observational
/// run shows it dominating, this is the test to change.
#[test]
fn unlabelled_fences_are_bracket_checked() {
    let text = "```\nfoo(bar\n```";
    let r = terminated(text, BlockMode::Unknown);
    assert!(r.failed(), "{:?}", r.findings);
    assert_eq!(r.findings[0].check, "bracket");
}

// ---------------------------------------------------------------------------
// Attribution: a finding belongs to the block that introduced it.
// ---------------------------------------------------------------------------

/// A defect committed in an earlier block must not be re-reported by every
/// later block — otherwise one bad line inflates the rate by the reply length.
#[test]
fn findings_are_attributed_to_the_block_that_introduced_them() {
    let prior = "```python\nx = foo(1\n```\n";
    let full = format!("{prior}```python\ny = 2\n```\n");
    let first = check(prior, 0, BlockMode::Code, true);
    assert!(first.failed());
    let second = check(&full, prior.len(), BlockMode::Code, true);
    assert!(!second.failed(), "{:?}", second.findings);
}

/// The counters are how "did the lever actually fire?" gets answered before
/// any arm comparison is believed, so they have to move for the right reasons:
/// a prose block counts as CHECKED and GATED, not as a pass.
#[test]
fn counters_separate_gated_blocks_from_passing_ones() {
    let _g = lock();
    let c0 = counters();
    check_block("all fine here.", 0, BlockMode::Prose, true);
    let c1 = counters();
    assert_eq!(
        (
            c1.checked - c0.checked,
            c1.prose_skipped - c0.prose_skipped,
            c1.failed - c0.failed
        ),
        (1, 1, 0)
    );
    check_block("```python\nf(1\n```", 0, BlockMode::Code, true);
    let c2 = counters();
    assert_eq!(
        (
            c2.checked - c1.checked,
            c2.prose_skipped - c1.prose_skipped,
            c2.failed - c1.failed
        ),
        (1, 0, 1)
    );
}

/// A block with nothing scannable must be VACUOUS, not a pass.
///
/// This is the regression test for a measurement error that actually
/// happened: the first `programmatic` run reported a clean 0 findings across
/// 54 blocks, which looked like the checker vindicating the code path. It was
/// the checker never looking at anything — the battery emits bare unfenced
/// code, which an `unknown`-mode block does not scan. A zero denominator must
/// be visible in the output, not indistinguishable from success.
#[test]
fn a_block_with_nothing_scannable_is_vacuous_not_passing() {
    let _g = lock();
    let before = counters();
    // Unknown mode, no fence: there is nothing this checker may look at.
    let r = check_block("x = foo(1\n", 0, BlockMode::Unknown, true);
    assert_eq!(r.regions_scanned, 0);
    assert!(r.vacuous(), "no region scanned must read as vacuous");
    assert!(!r.failed());
    assert_eq!(counters().vacuous - before.vacuous, 1);

    // The SAME text in code mode is scanned, and is not vacuous.
    let r = check_block("x = foo(1\n", 0, BlockMode::Code, true);
    assert!(r.regions_scanned > 0);
    assert!(!r.vacuous());
    assert!(r.failed(), "{:?}", r.findings);
}

// ---------------------------------------------------------------------------
// Bare (unfenced) code: the common case, not a corner.
// ---------------------------------------------------------------------------

/// The convention blend the whole exercise is about, emitted WITHOUT a fence —
/// which is how the `programmatic` battery actually emits code (0/18 probes
/// fenced). Reaching it needs the language sniffed from content.
#[test]
fn bare_bash_convention_blend_is_caught() {
    let text = "#!/bin/bash\nif [ $#\" -lt 1 ]; then\n  echo usage\nfi\n";
    let r = terminated(text, BlockMode::Code);
    assert!(r.failed(), "{:?}", r.findings);
    let f = r.findings.iter().find(|f| f.check == "quote_line").unwrap();
    assert_eq!(f.lang, "bash");
}

/// Correct bare bash stays silent — the sniffer must not manufacture findings.
#[test]
fn correct_bare_bash_is_silent() {
    let text = "#!/bin/bash\nfor f in *.txt; do\n  echo \"$f\"\ndone\n";
    assert!(!terminated(text, BlockMode::Code).failed());
}

#[test]
fn bare_python_is_sniffed_and_checked() {
    let ok = "import sys\n\ndef main():\n    print(\"hi\")\n";
    assert!(!terminated(ok, BlockMode::Code).failed());
    let bad = "import sys\n\ndef main():\n    print(\"hi)\n";
    let r = terminated(bad, BlockMode::Code);
    assert!(r.failed(), "{:?}", r.findings);
}

/// Ambiguous or absent markers fall back to GENERIC (brackets only) rather
/// than guessing a language whose quote rules would invent findings.
#[test]
fn an_ambiguous_region_falls_back_to_generic() {
    // Both bash and python markers: not conclusive.
    let mixed = "#!/bin/bash\ndef f():\n  echo \"unbalanced\n";
    let r = terminated(mixed, BlockMode::Code);
    assert!(
        !r.findings.iter().any(|f| f.check.starts_with("quote")),
        "an ambiguous sniff must not run quote rules: {:?}",
        r.findings
    );
    // No markers at all: also generic.
    assert_eq!(super::sniff_language("a = 1\nb = 2\n"), "");
}

// ---------------------------------------------------------------------------
// Robustness: instrumentation must never panic on model output.
// ---------------------------------------------------------------------------

/// Multi-byte UTF-8 at every offset, unterminated everything, empty input —
/// the checker rides the hot commit path, so a panic here is a broken
/// generation. (`split_top_level` already ate this bug once on `naïve`.)
#[test]
fn pathological_inputs_do_not_panic() {
    for text in [
        "",
        "```",
        "```bash",
        "```bash\n",
        "```rust\nlet s = r#\"",
        "```bash\ncat <<",
        "```bash\ncat <<EOF\n",
        "naïve — 日本語 (\" '\n```python\nnaïve(\"日本語\n```",
        "<|tool_call>call:",
        "<|tool_call>call:x{",
        "```python\n\"\"\"",
        "))))((((",
    ] {
        for mode in [BlockMode::Prose, BlockMode::Code, BlockMode::Unknown] {
            for term in [true, false] {
                check(text, 0, mode, term);
            }
        }
    }
}
