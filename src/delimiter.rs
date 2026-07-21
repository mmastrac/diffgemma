//! Structural delimiter/parity checking of committed blocks.
//!
//! A convention blend (ARCHITECTURE Part I) is a pair of positions that are
//! each individually plausible and jointly wrong — `== *$SEARCH"*` where both
//! `*$SEARCH*` and `*"$SEARCH"*` are valid. No confidence threshold sees it
//! (they commit at p_max ~1.0), but the STRUCTURE does: the emitted line has an
//! odd number of `"`. This module is that structural read.
//!
//! **Observational only.** Nothing here steers generation: it decodes committed
//! text, counts delimiters, and reports. Generation stays byte-identical and
//! `golden` is unaffected. Repair is a separate, later decision that this
//! measurement is supposed to inform — see PLAN "Delimiter parity check".
//!
//! Three deliberate constraints, each of which exists because ignoring it
//! produces a useless number:
//!
//! 1. **Every marker scan goes through `masked_ranges`.** The reply carries
//!    protocol markers AND literal data that looks like them; the tools module
//!    already computes which byte ranges are data (complete `call:NAME{…}` arg
//!    bodies, closed ``` fences). Raw `find` would count delimiters inside a
//!    quoted file body as if they were the model's own code.
//! 2. **Gated by BLOCK MODE.** In prose, `don't`, `:)` and a lone paren are
//!    normal — that is the dominant false-positive source, and the block-level
//!    classifier ([`crate::token_class`]) exists to suppress it. A prose block
//!    is not checked for content structure at all.
//! 3. **The block-edge rule.** A block that ran to the 256-token canvas edge
//!    legitimately leaves delimiters open for the next block; only an
//!    eos/stop-terminated block can be judged on TRAILING balance. An
//!    UNDERFLOW (a closer with no opener) is a real error either way, so it is
//!    judged unconditionally. Same rule the confidence trim encodes via
//!    `region_end`.
//!
//! Findings carry the check that produced them so the measurement can tell a
//! sensitive check from a specific one rather than collapsing both into one
//! rate. `quote_line` (per-line parity) and `quote_region` (a string-state
//! machine) deliberately disagree: the first catches `== *$SEARCH"*` on the
//! line where it happened, the second only notices if nothing later re-pairs
//! the quote. Which one is worth keeping is a measurement, not a guess.

use std::sync::atomic::{AtomicU64, Ordering};

/// Output mode of the block being checked, from the block-level probe
/// (`DGQ_TOKEN_CLASS`). `Unknown` when no probe is loaded — the checks still
/// run, so a run without a probe measures the UNGATED rate that gating is
/// supposed to improve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockMode {
    Prose,
    Code,
    Unknown,
}

impl BlockMode {
    /// Map a probe label through its class name. The probe is a signed
    /// direction with `class_names[1]` on the positive side; the fitted spec
    /// names them `prose`/`code`, but read the NAME rather than assuming the
    /// sign so a re-fit with swapped classes cannot silently invert the gate.
    pub fn from_class_name(name: &str) -> Self {
        match name {
            "code" => Self::Code,
            "prose" => Self::Prose,
            _ => Self::Unknown,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Prose => "prose",
            Self::Code => "code",
            Self::Unknown => "unknown",
        }
    }
}

/// Whether a finding is edge-sensitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A closer with no opener. Wrong regardless of where the block ended.
    Underflow,
    /// Something still open at the end of the checked region. Only meaningful
    /// when the region is a COMPLETE construct (a closed fence) or the block
    /// was eos/stop-terminated.
    Trailing,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Underflow => "underflow",
            Self::Trailing => "trailing",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// Which check fired: `tool_grammar`, `fence`, `bracket`, `quote_line`,
    /// `quote_region`. Reported separately so a noisy check can be identified
    /// and dropped instead of poisoning the aggregate rate.
    pub check: &'static str,
    pub kind: Kind,
    /// Byte offset in the checked text — used to attribute a finding to the
    /// block that introduced it rather than re-reporting it every block.
    pub at: usize,
    /// Language the region was scanned as (`""` outside a fence).
    pub lang: String,
    pub detail: String,
}

/// One block's verdict.
#[derive(Debug, Clone, Default)]
pub struct Report {
    pub findings: Vec<Finding>,
    /// True when content checks were SKIPPED because the block is prose. The
    /// tool-grammar check runs regardless (it is protocol, not content).
    pub prose_skipped: bool,
    /// How many content regions were actually scanned.
    ///
    /// This is the difference between "0 failures" and "0 OPPORTUNITIES", and
    /// it exists because the distinction was already missed once: the first
    /// `programmatic` run reported a clean 0/54 that turned out to be a
    /// checker with nothing to look at (the battery emits BARE code, and an
    /// `unknown`-mode block does not scan bare text). A rate whose denominator
    /// is zero must not read as a pass.
    pub regions_scanned: usize,
}

impl Report {
    pub fn failed(&self) -> bool {
        !self.findings.is_empty()
    }

    /// The block was checked but nothing was scannable — no verdict either way.
    pub fn vacuous(&self) -> bool {
        self.regions_scanned == 0 && self.findings.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Counters — so "did the lever fire?" is answerable without parsing logs.
// ---------------------------------------------------------------------------

static BLOCKS_CHECKED: AtomicU64 = AtomicU64::new(0);
static BLOCKS_PROSE_SKIPPED: AtomicU64 = AtomicU64::new(0);
static BLOCKS_VACUOUS: AtomicU64 = AtomicU64::new(0);
static BLOCKS_FAILED: AtomicU64 = AtomicU64::new(0);
static FINDINGS: AtomicU64 = AtomicU64::new(0);

/// Totals since process start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Counters {
    pub checked: u64,
    pub prose_skipped: u64,
    /// Checked but with nothing scannable — the vacuity guard.
    pub vacuous: u64,
    pub failed: u64,
    pub findings: u64,
}

pub fn counters() -> Counters {
    Counters {
        checked: BLOCKS_CHECKED.load(Ordering::Relaxed),
        prose_skipped: BLOCKS_PROSE_SKIPPED.load(Ordering::Relaxed),
        vacuous: BLOCKS_VACUOUS.load(Ordering::Relaxed),
        failed: BLOCKS_FAILED.load(Ordering::Relaxed),
        findings: FINDINGS.load(Ordering::Relaxed),
    }
}

fn record(report: &Report) {
    BLOCKS_CHECKED.fetch_add(1, Ordering::Relaxed);
    if report.prose_skipped {
        BLOCKS_PROSE_SKIPPED.fetch_add(1, Ordering::Relaxed);
    }
    if report.vacuous() {
        BLOCKS_VACUOUS.fetch_add(1, Ordering::Relaxed);
    }
    if report.failed() {
        BLOCKS_FAILED.fetch_add(1, Ordering::Relaxed);
        FINDINGS.fetch_add(report.findings.len() as u64, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Check `text` (the reply so far, INCLUDING the block just committed) and
/// return the findings introduced at/after `block_start` — the byte offset
/// where this block's text begins.
///
/// `terminated` is the block-edge rule: true only when the block ended on an
/// eos/stop token rather than running out of canvas.
pub fn check_block(text: &str, block_start: usize, mode: BlockMode, terminated: bool) -> Report {
    let masks = crate::tools::masked_ranges(text);
    let mut findings = Vec::new();

    // --- Check 1: tool grammar -------------------------------------------
    //
    // Reused wholesale from the tools module: `validate_tool_reply` is already
    // the strict, masked, unit-tested verdict on the `<|"|>` / `{}` / `[]` arg
    // grammar, and `read_braced` is what MADE a complete arg body a masked
    // range in the first place. So a masked arg body is balanced by
    // construction, and re-scanning it for brackets would be rebuilding a
    // check that already passed. Every unbalanced case is the UNMASKED one,
    // which this catches.
    //
    // Attribution without offsets: a finding belongs to this block iff the
    // reply was valid before it and is not now.
    if let Err(e) = crate::tools::validate_tool_reply(text) {
        let before_ok = crate::tools::validate_tool_reply(&text[..block_start]).is_ok();
        // An open `call:NAME{…}` at the canvas edge is a call that continues
        // into the next block, not a defect.
        let edge_excuse = !terminated && e == "incomplete tool call";
        if before_ok && !edge_excuse {
            findings.push(Finding {
                check: "tool_grammar",
                kind: if e == "incomplete tool call" {
                    Kind::Trailing
                } else {
                    Kind::Underflow
                },
                at: block_start,
                lang: String::new(),
                detail: e.to_string(),
            });
        }
    }

    // --- Content checks: prose blocks stop here ---------------------------
    if mode == BlockMode::Prose {
        let report = Report {
            findings: findings
                .into_iter()
                .filter(|f| f.at >= block_start)
                .collect(),
            prose_skipped: true,
            regions_scanned: 0,
        };
        record(&report);
        return report;
    }

    // --- Check 2: unclosed fence -----------------------------------------
    //
    // `masked_ranges` consumed every CLOSED fence pair, so leftover unmasked
    // ``` markers are candidates for an opener with no close — but PARITY is
    // the test, not first-occurrence. `masked_ranges` deliberately declines to
    // mask a fence pair that straddles a complete `call:NAME{…}` (masking it
    // would hide a real call), so a reply demonstrating the tool grammar
    // inside a fence leaves BOTH of its markers unmasked. Those pair up; only
    // an odd count is an actual unclosed fence.
    //
    // Edge-sensitive by nature: a fence opened in this block and closed in the
    // next is normal.
    let fences = unmasked_occurrences(text, &masks, "```");
    if terminated
        && fences.len() % 2 == 1
        && let Some(&at) = fences.last()
    {
        findings.push(Finding {
            check: "fence",
            kind: Kind::Trailing,
            at,
            lang: String::new(),
            detail: "unclosed ``` fence".into(),
        });
    }

    // --- Checks 3-5: per-region structure ---------------------------------
    //
    // Two region classes, with DIFFERENT edge rules:
    //  - A closed fence is a COMPLETE construct. Its interior is judgeable on
    //    trailing balance no matter where the block ended.
    //  - Unmasked text is bare (unfenced) code, only scanned when the probe
    //    says the block is code, and only judged on trailing balance when the
    //    block terminated.
    let mut regions_scanned = 0usize;
    for &(s, e) in &masks {
        if !text[s..e].starts_with("```") {
            continue; // a tool arg body — already covered by check 1
        }
        let Some((lang, body_start)) = fence_header(text, s, e) else {
            continue;
        };
        let body_end = e.saturating_sub(3).max(body_start);
        regions_scanned += 1;
        scan_region(
            &text[body_start..body_end],
            body_start,
            lang_spec(&lang),
            &lang,
            true,
            &mut findings,
        );
    }

    // Bare code outside any fence — the COMMON case, not a corner: the
    // `programmatic` battery emits unfenced code in 0/18 probes with a fence.
    // Checking only fences would leave the checker structurally blind exactly
    // where the measured convention blends live.
    //
    // The language is sniffed from the content, and only on an unambiguous
    // signal (`#!/bin/bash`, `def f(`, `fn main`). That is the twenty-line
    // token heuristic PLAN's "WAY OUT THERE" section names as the bar any
    // internal-state probe would have to beat; here it is the cheap thing that
    // works, and it is deliberately conservative — no signal means GENERIC
    // (brackets only), never a guessed language whose quote rules would
    // manufacture findings.
    //
    // Two admission rules, because the mode gate and the sniffer are
    // INDEPENDENT evidence:
    //  - `Code`: scan every bare span, sniffed or not. The probe said code.
    //  - `Unknown` (no probe loaded): scan only spans the sniffer is confident
    //    about. `#!/bin/bash` is code on its own evidence and needs no probe to
    //    say so — but an unsniffable span might be prose, and prose is the
    //    dominant false-positive source, so it is left alone.
    // `Prose` never reaches here.
    for (s, e) in unmasked_spans(text, &masks) {
        let src = &text[s..e];
        let lang = sniff_language(src);
        if mode == BlockMode::Unknown && lang.is_empty() {
            continue;
        }
        regions_scanned += 1;
        scan_region(src, s, lang_spec(lang), lang, terminated, &mut findings);
    }

    let report = Report {
        findings: findings
            .into_iter()
            .filter(|f| f.at >= block_start)
            .collect(),
        prose_skipped: false,
        regions_scanned,
    };
    record(&report);
    report
}

/// Guess the language of a BARE (unfenced) code region from distinctive
/// surface markers, or `""` when nothing is conclusive.
///
/// Deliberately conservative and deliberately dumb. It exists to unlock the
/// per-language quote rules on unfenced code, and every finding records the
/// language it was scanned as, so a wrong guess shows up as an attributable
/// cluster in the trace rather than as an unexplained rate. Ambiguity (markers
/// for two languages, or none) resolves to GENERIC rather than to a coin flip.
fn sniff_language(src: &str) -> &'static str {
    let head: String = src.chars().take(4000).collect();
    let bash = head.contains("#!/bin/bash")
        || head.contains("#!/bin/sh")
        || head.contains("#!/usr/bin/env bash")
        || head.contains("\nfi\n")
        || head.contains("\ndone\n")
        || head.contains("$(")
        || head.contains("echo \"");
    let python = head.contains("#!/usr/bin/env python")
        || head.contains("\ndef ")
        || head.starts_with("def ")
        || head.contains("\nimport ")
        || head.starts_with("import ")
        || head.contains("\nif __name__");
    let rust = head.contains("fn main(")
        || head.contains("\npub fn ")
        || head.contains("\nlet mut ")
        || head.contains("println!");
    match (bash, python, rust) {
        (true, false, false) => "bash",
        (false, true, false) => "python",
        (false, false, true) => "rust",
        // Two languages' markers, or none: not conclusive.
        _ => "",
    }
}

/// Split the fence at `[s, e)` into its info string (lowercased language) and
/// the byte offset where the body starts.
fn fence_header(text: &str, s: usize, e: usize) -> Option<(String, usize)> {
    let after = s + 3;
    let rel = text[after..e].find('\n')?;
    let info = text[after..after + rel].trim().to_ascii_lowercase();
    // ```bash {.numberLines} — take the first word only.
    let lang = info.split_whitespace().next().unwrap_or("").to_string();
    Some((lang, after + rel + 1))
}

/// Every unmasked occurrence of `needle`, ascending.
fn unmasked_occurrences(text: &str, masks: &[(usize, usize)], needle: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while let Some(pos) = crate::tools::find_unmasked(text, masks, needle, at) {
        out.push(pos);
        at = pos + needle.len();
    }
    out
}

/// Complement of `masks` within `text`, as ascending byte ranges.
fn unmasked_spans(text: &str, masks: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut at = 0usize;
    for &(s, e) in masks {
        if s > at {
            out.push((at, s));
        }
        at = at.max(e);
    }
    if at < text.len() {
        out.push((at, text.len()));
    }
    out
}

// ---------------------------------------------------------------------------
// Language specs
// ---------------------------------------------------------------------------

/// What a language's surface syntax does to delimiter counting.
///
/// The fields exist to neutralise the things that legitimately break naive
/// parity — comments, heredocs, multi-line strings, raw strings, lifetimes —
/// so that what is left over is a real imbalance. A language whose constructs
/// are NOT modelled here is scanned with [`GENERIC`] (brackets only): an
/// explicit skip, recorded as `lang` in the output, rather than a silent
/// stream of false positives.
struct LangSpec {
    line_comment: &'static [&'static str],
    block_comment: Option<(&'static str, &'static str)>,
    /// String delimiters that may span lines; consumed opaquely.
    multiline: &'static [(&'static str, &'static str)],
    /// String delimiters that may NOT span lines. Drive both the region state
    /// machine and per-line parity.
    line_strings: &'static [char],
    /// Bash `<<EOF` / `<<-'EOF'` heredocs.
    heredoc: bool,
    /// Rust `r#"…"#`.
    raw_string: bool,
}

/// Unknown language: brackets only. Quote parity is NOT run — without knowing
/// the comment and string syntax it flags apostrophes, lifetimes and `<<` and
/// measures nothing but its own naivety.
static GENERIC: LangSpec = LangSpec {
    line_comment: &[],
    block_comment: None,
    multiline: &[],
    line_strings: &[],
    heredoc: false,
    raw_string: false,
};

/// Bash: `"` only. `'` is deliberately excluded — `$'…'`, and apostrophes in
/// the `#` comments this spec already strips, make it unreliable, and the
/// MEASURED convention-blend specimens (`if [ $#" -lt 1 ]`, `== *$SEARCH"*`)
/// are both `"`.
static BASH: LangSpec = LangSpec {
    line_comment: &["#"],
    block_comment: None,
    multiline: &[("'", "'")],
    line_strings: &['"'],
    heredoc: true,
    raw_string: false,
};

/// Python: triple quotes are consumed first, so the remaining `"`/`'` cannot
/// span a line and per-line parity is exact.
static PYTHON: LangSpec = LangSpec {
    line_comment: &["#"],
    block_comment: None,
    multiline: &[("\"\"\"", "\"\"\""), ("'''", "'''")],
    line_strings: &['"', '\''],
    heredoc: false,
    raw_string: false,
};

/// Rust: `'` is excluded because a lifetime (`&'a str`) is an unpaired quote
/// in correct code — exactly the false positive the mode gate exists to avoid,
/// reappearing at the language level.
static RUST: LangSpec = LangSpec {
    line_comment: &["//"],
    block_comment: Some(("/*", "*/")),
    multiline: &[],
    line_strings: &['"'],
    heredoc: false,
    raw_string: true,
};

/// C-family (js/ts/go/java/c/c++): backticked template literals span lines and
/// are consumed opaquely; `'` is a valid string delimiter in js but a char
/// literal in c/go, so only `"` is counted.
static CLIKE: LangSpec = LangSpec {
    line_comment: &["//"],
    block_comment: Some(("/*", "*/")),
    multiline: &[("`", "`")],
    line_strings: &['"'],
    heredoc: false,
    raw_string: false,
};

fn lang_spec(lang: &str) -> &'static LangSpec {
    match lang {
        "bash" | "sh" | "shell" | "zsh" => &BASH,
        "python" | "py" => &PYTHON,
        "rust" | "rs" => &RUST,
        "js" | "javascript" | "ts" | "typescript" | "go" | "java" | "c" | "cpp" | "c++" => &CLIKE,
        _ => &GENERIC,
    }
}

// ---------------------------------------------------------------------------
// The scanner
// ---------------------------------------------------------------------------

/// Scan one region, appending findings.
///
/// Three passes, because the checks need different amounts of blanking:
/// 1. Blank comments, heredocs, raw strings and multi-line strings → `s1`.
///    These are the constructs that make line-local parity meaningless.
/// 2. Run the string state machine over `s1`, blanking single-line strings →
///    `s2`, emitting `quote_region` for anything left open.
/// 3. Brackets on `s2` (so delimiters inside strings do not count);
///    `quote_line` parity on `s1` (which still has its quotes).
///
/// Blanking replaces bytes with spaces in place, preserving newlines AND byte
/// offsets, so every finding's `at` still points into the original text.
fn scan_region(
    src: &str,
    base: usize,
    spec: &LangSpec,
    lang: &str,
    judge_trailing: bool,
    out: &mut Vec<Finding>,
) {
    let s1 = blank_opaque(src, spec);
    let s2 = blank_line_strings(&s1, spec, base, lang, judge_trailing, out);
    brackets(&s2, base, lang, judge_trailing, out);
    quote_line_parity(&s1, spec, base, lang, judge_trailing, out);
}

/// Replace a byte range with spaces, keeping newlines so line numbering and
/// per-line parity survive.
fn blank(buf: &mut [u8], range: std::ops::Range<usize>) {
    for b in &mut buf[range] {
        if *b != b'\n' {
            *b = b' ';
        }
    }
}

/// Pass 1: comments, heredocs, raw strings, multi-line strings → spaces.
fn blank_opaque(src: &str, spec: &LangSpec) -> String {
    let mut buf = src.as_bytes().to_vec();
    let b = src.as_bytes();
    let mut i = 0usize;
    let mut at_line_start = true;
    while i < b.len() {
        // Heredoc: `<<`, optional `-`, optional quote, identifier. Runs to a
        // line whose trimmed content is the identifier.
        if spec.heredoc
            && b[i] == b'<'
            && i + 1 < b.len()
            && b[i + 1] == b'<'
            && b.get(i + 2) != Some(&b'<')
            && let Some((tag, after)) = heredoc_tag(b, i + 2)
        {
            let end = heredoc_end(b, after, &tag);
            blank(&mut buf, i..end);
            i = end;
            continue;
        }
        // Rust raw string: r#"…"#
        if spec.raw_string
            && b[i] == b'r'
            && let Some((hashes, after)) = raw_string_open(b, i + 1)
        {
            let end = raw_string_end(b, after, hashes);
            blank(&mut buf, i..end);
            i = end;
            continue;
        }
        // Line comment. Bash's `#` only counts at the start of a word, so
        // `$#` and `${x#y}` are not comments.
        if let Some(tok) = spec
            .line_comment
            .iter()
            .find(|t| b[i..].starts_with(t.as_bytes()))
            && (*tok != "#" || at_line_start || b[i - 1].is_ascii_whitespace())
        {
            let end = src[i..].find('\n').map_or(b.len(), |r| i + r);
            blank(&mut buf, i..end);
            i = end;
            continue;
        }
        // Block comment.
        if let Some((open, close)) = spec.block_comment
            && b[i..].starts_with(open.as_bytes())
        {
            let end = src[i + open.len()..]
                .find(close)
                .map_or(b.len(), |r| i + open.len() + r + close.len());
            blank(&mut buf, i..end);
            i = end;
            continue;
        }
        // Multi-line string.
        if let Some(&(open, close)) = spec
            .multiline
            .iter()
            .find(|(o, _)| b[i..].starts_with(o.as_bytes()))
        {
            let end = src[i + open.len()..]
                .find(close)
                .map_or(b.len(), |r| i + open.len() + r + close.len());
            blank(&mut buf, i..end);
            i = end;
            continue;
        }
        at_line_start = b[i] == b'\n';
        i += 1;
    }
    String::from_utf8_lossy(&buf).into_owned()
}

/// A heredoc tag at `b[i..]`, plus the offset just past it.
fn heredoc_tag(b: &[u8], mut i: usize) -> Option<(Vec<u8>, usize)> {
    if b.get(i) == Some(&b'-') {
        i += 1;
    }
    while b.get(i) == Some(&b' ') {
        i += 1;
    }
    let quote = matches!(b.get(i), Some(&b'\'') | Some(&b'"')).then(|| b[i]);
    if quote.is_some() {
        i += 1;
    }
    let start = i;
    while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
        i += 1;
    }
    if i == start {
        return None;
    }
    let tag = b[start..i].to_vec();
    if quote.is_some() {
        i += 1;
    }
    Some((tag, i))
}

/// End of a heredoc body: just past the line whose trimmed content is `tag`.
fn heredoc_end(b: &[u8], from: usize, tag: &[u8]) -> usize {
    let mut line_start = match b[from..].iter().position(|&c| c == b'\n') {
        Some(r) => from + r + 1,
        None => return b.len(),
    };
    while line_start < b.len() {
        let line_end = b[line_start..]
            .iter()
            .position(|&c| c == b'\n')
            .map_or(b.len(), |r| line_start + r);
        let line = &b[line_start..line_end];
        let trimmed: &[u8] = {
            let s = line
                .iter()
                .position(|c| !c.is_ascii_whitespace())
                .unwrap_or(line.len());
            let e = line
                .iter()
                .rposition(|c| !c.is_ascii_whitespace())
                .map_or(s, |p| p + 1);
            &line[s..e]
        };
        if trimmed == tag {
            return (line_end + 1).min(b.len());
        }
        if line_end >= b.len() {
            return b.len();
        }
        line_start = line_end + 1;
    }
    b.len()
}

/// `r###"` opener: returns (hash count, offset past the quote).
fn raw_string_open(b: &[u8], mut i: usize) -> Option<(usize, usize)> {
    let start = i;
    while b.get(i) == Some(&b'#') {
        i += 1;
    }
    (b.get(i) == Some(&b'"')).then(|| (i - start, i + 1))
}

fn raw_string_end(b: &[u8], from: usize, hashes: usize) -> usize {
    let mut i = from;
    while i < b.len() {
        if b[i] == b'"'
            && b[i + 1..]
                .iter()
                .take(hashes)
                .filter(|&&c| c == b'#')
                .count()
                == hashes
        {
            return i + 1 + hashes;
        }
        i += 1;
    }
    b.len()
}

/// Pass 2: single-line strings → spaces, emitting `quote_region` for one left
/// open at the end of the region.
///
/// A real state machine rather than a count: inside `'…'` a `"` is literal and
/// vice versa, which is what stops `echo 'say "hi'` from reading as an
/// imbalance. The cost is specificity — if a later stray quote re-pairs the
/// one that blended, this check stays silent and only `quote_line` fires.
fn blank_line_strings(
    src: &str,
    spec: &LangSpec,
    base: usize,
    lang: &str,
    judge_trailing: bool,
    out: &mut Vec<Finding>,
) -> String {
    let mut buf = src.as_bytes().to_vec();
    let b = src.as_bytes();
    let mut i = 0usize;
    while i < b.len() {
        let c = b[i] as char;
        if spec.line_strings.contains(&c) {
            let open = i;
            let mut j = i + 1;
            let mut closed = false;
            while j < b.len() {
                if b[j] == b'\\' {
                    j += 2;
                    continue;
                }
                if b[j] == b'\n' {
                    break; // unterminated on this line
                }
                if b[j] as char == c {
                    closed = true;
                    break;
                }
                j += 1;
            }
            if closed {
                blank(&mut buf, open..j + 1);
                i = j + 1;
                continue;
            }
            // Unterminated. Only a finding when the region may be judged on
            // trailing balance; otherwise the string plausibly continues into
            // the next block.
            if judge_trailing {
                out.push(Finding {
                    check: "quote_region",
                    kind: Kind::Trailing,
                    at: base + open,
                    lang: lang.to_string(),
                    detail: format!("unterminated {c} at line {}", line_of(src, open)),
                });
            }
            blank(&mut buf, open..j.min(b.len()));
            i = j;
            continue;
        }
        i += 1;
    }
    String::from_utf8_lossy(&buf).into_owned()
}

/// Bracket balance over string-blanked text.
///
/// Underflow is reported unconditionally: a `)` with no opener is wrong no
/// matter where the block ended. Trailing depth is edge-gated.
fn brackets(src: &str, base: usize, lang: &str, judge_trailing: bool, out: &mut Vec<Finding>) {
    let mut stack: Vec<(char, usize)> = Vec::new();
    for (i, c) in src.char_indices() {
        match c {
            '{' | '[' | '(' => stack.push((c, i)),
            '}' | ']' | ')' => {
                let want = match c {
                    '}' => '{',
                    ']' => '[',
                    _ => '(',
                };
                match stack.last() {
                    Some(&(open, _)) if open == want => {
                        stack.pop();
                    }
                    _ => {
                        out.push(Finding {
                            check: "bracket",
                            kind: Kind::Underflow,
                            at: base + i,
                            lang: lang.to_string(),
                            detail: format!("`{c}` with no opener at line {}", line_of(src, i)),
                        });
                        // Do not unwind further: one stray closer would
                        // otherwise cascade into a finding for every
                        // subsequent closer in the region.
                        stack.clear();
                    }
                }
            }
            _ => {}
        }
    }
    if judge_trailing && let Some(&(open, at)) = stack.first() {
        out.push(Finding {
            check: "bracket",
            kind: Kind::Trailing,
            at: base + at,
            lang: lang.to_string(),
            detail: format!(
                "`{open}` never closed at line {} ({} open)",
                line_of(src, at),
                stack.len()
            ),
        });
    }
}

/// Per-line quote parity over comment/heredoc/multiline-blanked text.
///
/// The sensitive check: `if [[ "$f" == *$SEARCH"* ]]` has three `"` on one
/// line, which is a defect even though the file's total may stay even.
fn quote_line_parity(
    src: &str,
    spec: &LangSpec,
    base: usize,
    lang: &str,
    judge_trailing: bool,
    out: &mut Vec<Finding>,
) {
    if spec.line_strings.is_empty() {
        return;
    }
    let last_line = src.lines().count();
    let mut off = 0usize;
    for (n, line) in src.lines().enumerate() {
        // A block that ran to the canvas edge has a genuinely half-written
        // final line; nothing about it is judgeable.
        if !judge_trailing && n + 1 == last_line {
            break;
        }
        for &q in spec.line_strings {
            if count_unescaped(line, q) % 2 == 1 {
                out.push(Finding {
                    check: "quote_line",
                    kind: Kind::Trailing,
                    at: base + off,
                    lang: lang.to_string(),
                    detail: format!("odd `{q}` count on line {}", n + 1),
                });
            }
        }
        off += line.len() + 1;
    }
}

/// Occurrences of `q` not preceded by an odd run of backslashes.
fn count_unescaped(line: &str, q: char) -> usize {
    let mut n = 0usize;
    let mut backslashes = 0usize;
    for c in line.chars() {
        if c == '\\' {
            backslashes += 1;
            continue;
        }
        if c == q && backslashes.is_multiple_of(2) {
            n += 1;
        }
        backslashes = 0;
    }
    n
}

fn line_of(src: &str, at: usize) -> usize {
    src[..at.min(src.len())].matches('\n').count() + 1
}

#[cfg(test)]
mod tests;
