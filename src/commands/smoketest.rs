//! `smoketest` gate subcommand + fixture spec types.

use super::*;
use super::programmatic::{ProgCounts, ProgProbe};

/// Which tier of the spec a run exercises. The tiers have different session
/// budgets and different judging, so exactly one runs per invocation; `census`
/// names them as batteries.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum Battery {
    /// Adherence + convergence — the 17/17 commit gate.
    #[default]
    Smoke,
    /// Long-context doc-QA ladder (E13).
    LongCtx,
    /// Generate a program, execute it, judge stdout + exit code.
    Programmatic,
}

#[cfg(target_os = "macos")]
impl Battery {
    /// Single source of truth for the battery names `census --battery` accepts.
    pub(crate) fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "smoke" => Self::Smoke,
            "longctx" => Self::LongCtx,
            "programmatic" => Self::Programmatic,
            _ => return None,
        })
    }
    pub(crate) const KNOWN: &'static str = "smoke, longctx, programmatic";
}

/// Smoketest prompt spec (`fixtures/smoketest/prompts.json`).
#[cfg(target_os = "macos")]
#[derive(serde::Deserialize)]
pub(crate) struct SmoketestSpec {
    #[serde(default)]
    convergence: Vec<SmokeConvergence>,
    #[serde(default)]
    adherence: Vec<SmokeAdherence>,
    /// Long-context doc-QA ladder (E13; `smoketest --longctx` only). Judges
    /// grounded COMPREHENSION of a real document at increasing prompt lengths
    /// — the failure class needle probes provably miss (task #64: retrieval
    /// rides a few sharp attention edges and stayed EXACT while grounded
    /// answers collapsed into fluent hallucination).
    #[serde(default)]
    longctx: Vec<SmokeLongCtx>,
    /// Executable-correctness probes (`census --battery programmatic`). Judged
    /// by RUNNING the generated program — see `commands::programmatic`.
    #[serde(default)]
    programmatic: Vec<ProgProbe>,
    /// Gate baseline seed. Trajectory-reshuffling accepted changes re-baseline
    /// the gate here (single-seed pass/fail is arbitrary for such changes; the
    /// multi-seed aggregate is the real quality metric — see working notes).
    /// An explicit CLI `--seed` (anything != 42) still overrides for sweeps.
    #[serde(default)]
    seed: Option<u64>,
}
/// Free-form prompt that must converge within `max_steps` denoise steps.
#[cfg(target_os = "macos")]
#[derive(serde::Deserialize)]
pub(crate) struct SmokeConvergence {
    id: String,
    prompt: String,
    max_steps: usize,
}
/// Long-context doc-QA probe: a fixture document truncated to `doc_tokens`
/// prompt tokens + a question about facts planted near the truncation edge
/// (in-window-unique, so a grounded answer proves the model READ that depth).
#[cfg(target_os = "macos")]
#[derive(serde::Deserialize)]
pub(crate) struct SmokeLongCtx {
    id: String,
    /// Fixture document path, relative to the spec file's directory. Frozen
    /// snapshot — do NOT regenerate from the live repo docs (answers drift).
    doc: String,
    /// Truncate the document to this many tokens before the question.
    doc_tokens: usize,
    question: String,
    /// EVERY entry must appear in the reply (normalized word-run match).
    require: Vec<String>,
    max_steps: usize,
}
/// Prompt with exactly one correct answer; gated on both answer + convergence.
#[cfg(target_os = "macos")]
#[derive(serde::Deserialize)]
pub(crate) struct SmokeAdherence {
    id: String,
    prompt: String,
    answer: String,
    /// Additional acceptable spellings (e.g. "h2o", "h₂o").
    #[serde(default)]
    accept: Vec<String>,
    max_steps: usize,
}
/// Lowercase, alphanumeric-only, single-spaced — for word-boundary matching.
/// Digit<->letter transitions also split ("1085s" -> "1085 s"), so a numeric
/// pattern matches a reply with a unit suffix. Both sides of a match are
/// normalized identically, so patterns like "h2o" keep matching ("h 2 o" on
/// both sides).
#[cfg(target_os = "macos")]
pub(crate) fn smoke_normalize(s: &str) -> String {
    #[derive(PartialEq, Clone, Copy)]
    enum Class {
        Space,
        Digit,
        Letter,
    }
    let mut out = String::new();
    let mut prev = Class::Space;
    for c in s.chars() {
        if c.is_alphanumeric() {
            let cur = if c.is_ascii_digit() {
                Class::Digit
            } else {
                Class::Letter
            };
            if prev != Class::Space && prev != cur {
                out.push(' ');
            }
            for lc in c.to_lowercase() {
                out.push(lc);
            }
            prev = cur;
        } else if prev != Class::Space {
            out.push(' ');
            prev = Class::Space;
        }
    }
    out.trim().to_string()
}
/// Does `reply` contain `answer` (or an accepted alternate) as a whole word run?
#[cfg(target_os = "macos")]
pub(crate) fn smoke_answer_matches(reply: &str, answer: &str, accept: &[String]) -> bool {
    let r = format!(" {} ", smoke_normalize(reply));
    std::iter::once(answer)
        .chain(accept.iter().map(String::as_str))
        .any(|a| {
            let an = smoke_normalize(a);
            !an.is_empty() && r.contains(&format!(" {an} "))
        })
}
/// Structured result of one battery run.
///
/// The gate's verdict used to be an `ExitCode` and nothing else, so any
/// caller wanting more (the `census` campaign runner) had to recover
/// pass/fail from a process exit status and everything else from side
/// channels. Returning the numbers directly makes them gateable.
#[derive(Debug, Clone, Default)]
pub(crate) struct SmokeOutcome {
    pub(crate) passed: usize,
    pub(crate) total: usize,
    /// Long-context keyword retrieval (`--longctx` only; 0/0 otherwise).
    pub(crate) kw_found: usize,
    pub(crate) kw_total: usize,
    /// Denoise steps of the COMMITTED blocks (what the gate ratchets on).
    pub(crate) steps_committed: usize,
    /// ALL denoise steps run, including work discarded by re-rolls — the
    /// difference is what retries actually cost.
    pub(crate) steps_total: usize,
    pub(crate) failures: Vec<String>,
    /// Executable-correctness tallies (`Battery::Programmatic` only). `None`
    /// for every other battery, so the census reports "no cases" rather than a
    /// fabricated 0/0 pass rate.
    pub(crate) prog: Option<ProgCounts>,
    /// Set when the run could not start (bad model dir, missing spec, …);
    /// distinct from "ran and failed prompts".
    pub(crate) error: Option<String>,
}

impl SmokeOutcome {
    fn err(msg: impl Into<String>) -> Self {
        Self {
            error: Some(msg.into()),
            ..Default::default()
        }
    }
    /// Ids of the prompts that failed, for a caller that reports detail.
    pub(crate) fn failed_ids(&self) -> &[String] {
        &self.failures
    }
    pub(crate) fn ok(&self) -> bool {
        self.error.is_none() && self.total > 0 && self.passed == self.total
    }
}

/// CLI adapter: run the gate, map the outcome to a process exit code.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_smoketest_cmd(
    model_dir: &std::path::Path,
    prompts_path: Option<&std::path::Path>,
    seed: Option<u64>,
    steps: usize,
    max_layers: Option<usize>,
    raw_prompt: bool,
    filter: Option<&str>,
    repeat: usize,
    longctx: bool,
) -> ExitCode {
    let out = run_smoketest(
        model_dir,
        prompts_path,
        seed,
        steps,
        max_layers,
        raw_prompt,
        filter,
        repeat,
        if longctx {
            Battery::LongCtx
        } else {
            Battery::Smoke
        },
    );
    if let Some(e) = &out.error {
        eprintln!("{e}");
    }
    if out.ok() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Convergence + adherence gate over a prompt set. Reuses the chat session path
/// so each prompt is a fresh single-turn generation; reports actual vs threshold
/// denoise steps (ratchet thresholds down in the JSON as the engine improves).
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_smoketest(
    model_dir: &std::path::Path,
    prompts_path: Option<&std::path::Path>,
    seed: Option<u64>,
    steps: usize,
    max_layers: Option<usize>,
    raw_prompt: bool,
    filter: Option<&str>,
    repeat: usize,
    battery: Battery,
) -> SmokeOutcome {
    use metal::{StepGenerateConfig, StepGenerateSession, generate_with_session};

    let longctx = battery == Battery::LongCtx;

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        return SmokeOutcome::err("error: smoketest requires a .dgq directory (-m /path/to/quantized-weights)");
    }

    let default_path = std::path::PathBuf::from("fixtures/smoketest/prompts.json");
    let path = prompts_path.unwrap_or(default_path.as_path());
    let mut spec: SmoketestSpec = match std::fs::read_to_string(path) {
        Ok(text) => match serde_json::from_str(&text) {
            Ok(s) => s,
            Err(err) => {
                return SmokeOutcome::err(format!("error: parse {}: {err}", path.display()));
            }
        },
        Err(err) => {
            return SmokeOutcome::err(format!("error: read {}: {err}", path.display()));
        }
    };

    // `--filter <pat>`: keep only prompts whose id contains <pat> (case-insensitive).
    if let Some(pat) = filter {
        let pat = pat.to_ascii_lowercase();
        spec.adherence
            .retain(|p| p.id.to_ascii_lowercase().contains(&pat));
        spec.convergence
            .retain(|p| p.id.to_ascii_lowercase().contains(&pat));
        spec.longctx
            .retain(|p| p.id.to_ascii_lowercase().contains(&pat));
        spec.programmatic
            .retain(|p| p.id.to_ascii_lowercase().contains(&pat));
        let kept = match battery {
            Battery::LongCtx => spec.longctx.len(),
            Battery::Programmatic => spec.programmatic.len(),
            Battery::Smoke => spec.adherence.len() + spec.convergence.len(),
        };
        if kept == 0 {
            return SmokeOutcome::err(format!("smoketest: no prompts match filter {pat:?}"));
        }
        eprintln!("smoketest: filter {pat:?} -> {kept} prompt(s)");
    }
    if longctx && spec.longctx.is_empty() {
        return SmokeOutcome::err("smoketest: --longctx but the spec has no longctx probes");
    }
    if battery == Battery::Programmatic {
        if spec.programmatic.is_empty() {
            return SmokeOutcome::err("smoketest: the spec has no programmatic probes");
        }
        // Toolchains are checked BEFORE the session opens: a missing `rustc`
        // must cost a second, not a campaign, and must never be recorded as
        // the model's compile failure.
        if let Err(e) = programmatic::preflight(&spec.programmatic) {
            return SmokeOutcome::err(e);
        }
    }

    // Gate baseline seed: the spec pins it (re-baselined with accepted
    // trajectory-reshuffling changes); an explicit CLI --seed overrides for
    // multi-seed sweeps. (Formerly `--seed 42` was indistinguishable from the
    // CLI default and silently ran the spec seed — sweeps that included 42
    // were double-counting the gate seed.)
    let seed = seed.or(spec.seed).unwrap_or(42);

    let layers = match resolve_model_layers(model_dir, max_layers) {
        Ok(n) => n,
        Err(err) => {
            return SmokeOutcome::err(format!("error: {err}"));
        }
    };

    // Per-step denoise progress would drown the gate report.
    if !flags::quiet_set_by_user() {
        flags::set_quiet(true);
    }

    const SMOKE_MAX_SEQ: usize = 2048;
    const SMOKE_GEN_CAP: usize = 512; // ~2 canvas blocks; bounds gate time + KV
    // Longctx tier needs headroom for the deepest ladder rung (~20.6k doc).
    const LONGCTX_MAX_SEQ: usize = 24576;
    // Programs are far longer than gate answers. At the 512-token smoke cap a
    // ~60-line Rust program was cut off mid-expression and scored
    // `compile_fail` — the battery reporting OUR ceiling as the model's
    // error. The cap must sit well clear of the longest probe, or every hard
    // probe silently measures the budget instead of the model.
    const PROG_MAX_SEQ: usize = 4096;
    const PROG_GEN_CAP: usize = 1536;
    let smoke_max_seq = match battery {
        Battery::LongCtx => LONGCTX_MAX_SEQ,
        Battery::Programmatic => PROG_MAX_SEQ,
        Battery::Smoke => SMOKE_MAX_SEQ,
    };
    let gen_cap = match battery {
        Battery::Programmatic => PROG_GEN_CAP,
        _ => SMOKE_GEN_CAP,
    };
    let stop_token_ids = config::load_generation_stop_tokens(model_dir);
    let sampler = sample::sampler_for_steps(steps, false);
    let mut step_cfg =
        StepGenerateConfig::from_generate(seed, 1024, smoke_max_seq, layers, sampler, false);
    step_cfg.degenerate_reply_check =
        chat_template::empty_reply_check(model_dir, stop_token_ids.clone());
    step_cfg.stop_token_ids = stop_token_ids;

    let mut session = match StepGenerateSession::open(model_dir, &step_cfg, None) {
        Ok((s, compile)) => {
            eprintln!("smoketest: session ready ({compile:.2?}, {layers}L, sampler cap {steps})");
            s
        }
        Err(err) => {
            return SmokeOutcome::err(format!("error: {err}"));
        }
    };

    let tok_path = model_dir.join("tokenizer.json");
    let tokenizer = match tokenizer::Tokenizer::load(&tok_path) {
        Ok(t) => t,
        Err(err) => {
            return SmokeOutcome::err(format!("error: {err}"));
        }
    };

    // Step accounting, accumulated by the closure below: committed steps are
    // what the gate ratchets on; the total additionally counts work thrown
    // away by re-rolls, so `total - committed` is the retry cost.
    let mut steps_committed_total = 0usize;
    let mut steps_run_total = 0usize;
    // (denoise_steps, reply) for one fresh single-turn generation.
    let mut run_one = |prompt_text: &str| -> Result<(usize, String), crate::Error> {
        // Each prompt is independent — drop prior KV so we re-prefill fresh
        // (chat's KV-reuse continuation would otherwise answer the first prompt).
        session.reset_kv();
        let history = vec![chat_template::ChatTurn::user(prompt_text)];
        let prompt = build_chat_prompt_tokens(model_dir, &history, raw_prompt)?;
        let prompt_len = prompt.len();
        // Bound generation (and thus time + KV) — a gate doesn't need essays.
        step_cfg.max_new_tokens = gen_cap.min(smoke_max_seq.saturating_sub(prompt_len).max(1));
        let out = generate_with_session(&mut session, &prompt, &step_cfg, "smoketest")?;
        let new_ids =
            sample::strip_degenerate_token_ids(out.token_ids.get(prompt_len..).unwrap_or(&[]));
        let reply = chat_template::sanitize_model_reply(&tokenizer.decode(&new_ids));
        // Convergence gate = steps of the COMMITTED blocks (block_steps_eff),
        // not total denoise work. Identical to denoise_steps_run unless the
        // empty-reply retry (DGQ_EMPTY_REPLY_RETRY>0) re-rolled a degenerate
        // first block — discarded re-roll steps are latency, not a convergence
        // regression, so they must not count against the per-prompt budget.
        let committed_steps: usize = out.block_steps_eff.iter().map(|&s| s as usize).sum();
        steps_committed_total += committed_steps;
        steps_run_total += out.denoise_steps_run;
        Ok((committed_steps, reply))
    };

    // (No warm-up: cold-start is fixed at the root by the deterministic first-step
    // SC seed. Verified engine 16/16 with the warm-up removed.)

    let mut passed = 0usize;
    let mut total = 0usize;
    let mut failures: Vec<String> = Vec::new();
    // Hoisted out of the longctx branch so the outcome can carry retrieval
    // (the per-tier locals below still drive the printed line).
    let mut kw_found_total = 0usize;
    let mut kw_total_total = 0usize;
    let mut prog_counts: Option<ProgCounts> = None;

    println!(
        "\nsmoketest: {} (seed {seed}, {layers}L, sampler cap {steps} steps)",
        model_dir.display()
    );

    // Longctx doc token ids, cached per fixture path (each rung re-truncates).
    let mut doc_ids_cache: std::collections::HashMap<String, Vec<u32>> = Default::default();
    let spec_dir = path.parent().unwrap_or(std::path::Path::new("."));

    for iter in 0..repeat {
        if repeat > 1 {
            println!(
                "\n===== iteration {}/{repeat} (same session, no re-warmup) =====",
                iter + 1
            );
        }
        // `--longctx` runs ONLY the doc-QA ladder (its session budget differs);
        // the default gate runs only adherence+convergence and stays fast.
        if longctx {
            println!("\n[longctx doc-QA]");
            // Keyword-level retrieval accounting (catches stochastic drift that
            // the per-doc binary pass/fail hides): count markers found vs total.
            let mut kw_found = 0usize;
            let mut kw_total = 0usize;
            for p in &spec.longctx {
                total += 1;
                if !doc_ids_cache.contains_key(&p.doc) {
                    let doc_path = spec_dir.join(&p.doc);
                    match std::fs::read_to_string(&doc_path) {
                        Ok(text) => {
                            doc_ids_cache.insert(p.doc.clone(), tokenizer.encode(&text, false));
                        }
                        Err(err) => {
                            println!("  {:<22} ERROR  read {}: {err}", p.id, doc_path.display());
                            failures.push(p.id.clone());
                            continue;
                        }
                    }
                }
                let ids = &doc_ids_cache[&p.doc];
                let n = p.doc_tokens.min(ids.len());
                let excerpt = tokenizer.decode(&ids[..n]);
                let prompt_text = format!(
                    "{excerpt}\n\n[end of document]\nAnswer from the document above: {q}",
                    q = p.question
                );
                let (st, reply) = match run_one(&prompt_text) {
                    Ok(v) => v,
                    Err(err) => {
                        println!("  {:<22} ERROR  {err}", p.id);
                        failures.push(p.id.clone());
                        continue;
                    }
                };
                let missing: Vec<&str> = p
                    .require
                    .iter()
                    .filter(|k| !smoke_answer_matches(&reply, k, &[]))
                    .map(String::as_str)
                    .collect();
                kw_total += p.require.len();
                kw_found += p.require.len() - missing.len();
                kw_total_total += p.require.len();
                kw_found_total += p.require.len() - missing.len();
                let conv_ok = st <= p.max_steps;
                let ok = missing.is_empty() && conv_ok;
                if ok {
                    passed += 1;
                } else {
                    failures.push(p.id.clone());
                }
                let mark = if ok { "PASS" } else { "FAIL" };
                let af = if missing.is_empty() {
                    "ok".to_string()
                } else {
                    format!("MISSING {missing:?}")
                };
                let max = p.max_steps;
                let prev = reply
                    .chars()
                    .take(72)
                    .collect::<String>()
                    .replace('\n', " ");
                let found = p.require.len() - missing.len();
                println!(
                    "  {id:<22} {mark:<4} doc {n:>5}tok steps {st:>3}/{max:<3} kw {found}/{tot} {af}  | {prev}",
                    id = p.id,
                    tot = p.require.len()
                );
            }
            let rate = if kw_total > 0 {
                100.0 * kw_found as f64 / kw_total as f64
            } else {
                100.0
            };
            println!(
                "longctx keyword retrieval: {kw_found}/{kw_total} ({rate:.1}%)  [drift = {:.1}%]",
                100.0 - rate
            );
            continue;
        }
        // `programmatic` likewise runs alone: its verdict is execution, not
        // text, and mixing it into the commit gate would make a `rustc`
        // hiccup fail the gate.
        if battery == Battery::Programmatic {
            println!("\n[programmatic]");
            let (counts, p, t, mut f) = programmatic::run_probes(&spec.programmatic, &mut run_one);
            passed += p;
            total += t;
            failures.append(&mut f);
            let acc = prog_counts.get_or_insert_with(ProgCounts::default);
            acc.probes += counts.probes;
            acc.cases += counts.cases;
            acc.pass += counts.pass;
            acc.compile_fail += counts.compile_fail;
            acc.wrong_output += counts.wrong_output;
            acc.fenced += counts.fenced;
            let rate = if counts.cases > 0 {
                100.0 * counts.pass as f64 / counts.cases as f64
            } else {
                0.0
            };
            println!(
                "programmatic: {}/{} cases pass ({rate:.1}%)  compile_fail {}  wrong_output {}  fenced {}/{} probes",
                counts.pass, counts.cases, counts.compile_fail, counts.wrong_output,
                counts.fenced, counts.probes,
            );
            continue;
        }
        if !spec.adherence.is_empty() {
            println!("\n[adherence]");
            for p in &spec.adherence {
                total += 1;
                let (st, reply) = match run_one(&p.prompt) {
                    Ok(v) => v,
                    Err(err) => {
                        println!("  {:<22} ERROR  {err}", p.id);
                        failures.push(p.id.clone());
                        continue;
                    }
                };
                let answer_ok = smoke_answer_matches(&reply, &p.answer, &p.accept);
                let conv_ok = st <= p.max_steps;
                let ok = answer_ok && conv_ok;
                if ok {
                    passed += 1;
                } else {
                    failures.push(p.id.clone());
                }
                let prev = reply
                    .chars()
                    .take(56)
                    .collect::<String>()
                    .replace('\n', " ");
                let mark = if ok { "PASS" } else { "FAIL" };
                let af = if answer_ok { "ok " } else { "BAD" };
                let max = p.max_steps;
                let ans = &p.answer;
                println!(
                    "  {id:<22} {mark:<4} steps {st:>3}/{max:<3} answer {af} \"{ans}\"  | {prev}",
                    id = p.id
                );
            }
        }

        if !spec.convergence.is_empty() {
            println!("\n[convergence]");
            for p in &spec.convergence {
                total += 1;
                let (st, reply) = match run_one(&p.prompt) {
                    Ok(v) => v,
                    Err(err) => {
                        println!("  {:<22} ERROR  {err}", p.id);
                        failures.push(p.id.clone());
                        continue;
                    }
                };
                let ok = st <= p.max_steps && !reply.trim().is_empty();
                if ok {
                    passed += 1;
                } else {
                    failures.push(p.id.clone());
                }
                let mark = if ok { "PASS" } else { "FAIL" };
                let max = p.max_steps;
                let prev = reply
                    .chars()
                    .take(72)
                    .collect::<String>()
                    .replace('\n', " ");
                println!(
                    "  {id:<22} {mark:<4} steps {st:>3}/{max:<3}  | {prev}",
                    id = p.id
                );
            }
        }
    } // end repeat loop

    println!("\nsmoketest: {passed}/{total} passed");
    if passed != total {
        println!("failed: {}", failures.join(", "));
    }
    SmokeOutcome {
        passed,
        total,
        kw_found: kw_found_total,
        kw_total: kw_total_total,
        steps_committed: steps_committed_total,
        steps_total: steps_run_total,
        failures,
        prog: prog_counts,
        error: None,
    }
}
