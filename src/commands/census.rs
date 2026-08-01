//! `census` — run a matrix of FLAG ARMS × BATTERIES in one process and judge
//! it against explicit gates.
//!
//! The quality levers in this engine (commit-confidence trim, prefix-exit,
//! …) are all decided the same way: run the gates under a couple of flag
//! settings, count how much contested output reaches committed KV, and
//! compare. That ritual used to be a shell loop plus a throwaway analysis
//! script per investigation. This makes it a command.
//!
//! An ARM is a name plus `DGQ_*` overrides in env form, parsed by
//! [`RuntimeConfig::from_pairs`] — the SAME helpers and the SAME validation
//! the process env gets, so a typo'd arm is rejected before the campaign
//! spends GPU hours on it. An arm states only what it overrides.
//!
//! A BATTERY is a unit of gated work (`smoke`, `longctx`, `programmatic`,
//! `soft`). Every battery yields the same wart metrics, because those come
//! from the denoise path's p_max trace rather than from anything
//! battery-specific:
//!
//!   hard  — a COMMITTED row with p_max < 0.5 (insertion/omission class)
//!   dup   — a COMMITTED row below tau that argmax-copies a neighbour
//!           (the duplication micro-stutter)
//!
//! Only rows below the block's `kept` count: a row the trim cut never
//! reached KV, which is the whole point of the lever being measured.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// One arm: a label plus the `DGQ_*` overrides that define it.
struct Arm {
    name: String,
    pairs: Vec<(String, String)>,
}

/// Flat metrics for one (arm, battery, seed) run. Flat so gates and
/// baseline diffs work uniformly across batteries.
#[derive(Default, Clone)]
struct Metrics {
    passed: bool,
    blocks: u64,
    committed_rows: u64,
    hard: u64,
    dup: u64,
    trims: u64,
    steps: u64,
    prompts_passed: u64,
    prompts_total: u64,
    kw_found: u64,
    kw_total: u64,
    /// Denoise steps of committed blocks, and ALL steps run (incl. work
    /// discarded by re-rolls). The gap is what retries cost.
    steps_committed: u64,
    steps_run: u64,
    /// How many runs contributed a `run_summary` record. 0 means the
    /// battery-level numbers below are unknown (an older trace), which is
    /// why the pass column reads "-" rather than a fabricated verdict.
    summaries: u64,
    /// `programmatic` battery: executable correctness. THREE outcome states,
    /// kept apart on purpose — `compile_fail` (not a program) and
    /// `wrong_output` (a program that computes the wrong thing) are different
    /// findings, and their sum with `prog_pass` is `cases`. `fenced` counts
    /// PROBES whose reply arrived wrapped in a markdown fence, which measures
    /// instruction-following rather than correctness.
    cases: u64,
    prog_pass: u64,
    compile_fail: u64,
    wrong_output: u64,
    fenced: u64,
    probes: u64,
    /// `soft` battery. `absence_*` is a SEPARATE rate because those probes
    /// invert the metric — a correct answer is a refusal — and averaging them
    /// into retrieval would let confident hallucination cancel good recall.
    soft_found: u64,
    soft_total: u64,
    absence_ok: u64,
    absence_total: u64,
}

impl Metrics {
    fn merge(&mut self, o: &Metrics) {
        self.passed &= o.passed;
        self.blocks += o.blocks;
        self.committed_rows += o.committed_rows;
        self.hard += o.hard;
        self.dup += o.dup;
        self.trims += o.trims;
        self.steps += o.steps;
        self.prompts_passed += o.prompts_passed;
        self.prompts_total += o.prompts_total;
        self.kw_found += o.kw_found;
        self.kw_total += o.kw_total;
        self.steps_committed += o.steps_committed;
        self.steps_run += o.steps_run;
        self.summaries += o.summaries;
        self.cases += o.cases;
        self.prog_pass += o.prog_pass;
        self.compile_fail += o.compile_fail;
        self.wrong_output += o.wrong_output;
        self.fenced += o.fenced;
        self.probes += o.probes;
        self.soft_found += o.soft_found;
        self.soft_total += o.soft_total;
        self.absence_ok += o.absence_ok;
        self.absence_total += o.absence_total;
    }

    /// Steps thrown away by re-rolls.
    fn steps_retry(&self) -> u64 {
        self.steps_run.saturating_sub(self.steps_committed)
    }

    /// Keyword retrieval across the arm's runs; 100 when unmeasured, so a
    /// retrieval gate cannot fail a battery that has no keywords.
    fn retrieval_pct(&self) -> f64 {
        if self.kw_total == 0 {
            100.0
        } else {
            100.0 * self.kw_found as f64 / self.kw_total as f64
        }
    }
    /// The decision line: contested rows that reached KV, per 1k committed.
    fn contested_per_1k(&self) -> f64 {
        if self.committed_rows == 0 {
            0.0
        } else {
            1000.0 * (self.hard + self.dup) as f64 / self.committed_rows as f64
        }
    }
    /// Share of executed cases that produced the expected stdout AND exit
    /// code. 0 when nothing ran — the OPPOSITE default from `retrieval_pct`,
    /// because "no cases executed" is a floor here, not an unmeasured
    /// dimension a gate should be blind to.
    fn prog_pass_pct(&self) -> f64 {
        if self.cases == 0 {
            0.0
        } else {
            100.0 * self.prog_pass as f64 / self.cases as f64
        }
    }
    fn compile_fail_pct(&self) -> f64 {
        if self.cases == 0 {
            0.0
        } else {
            100.0 * self.compile_fail as f64 / self.cases as f64
        }
    }
    /// Probes whose reply arrived fenced, as a percentage.
    fn fenced_pct(&self) -> f64 {
        if self.probes == 0 {
            0.0
        } else {
            100.0 * self.fenced as f64 / self.probes as f64
        }
    }
    /// Indirect-retrieval rate. 0 when nothing ran, like `prog_pass_pct`:
    /// an unmeasured quality rate must not PASS a gate that asks for a floor.
    fn soft_pct(&self) -> f64 {
        if self.soft_total == 0 {
            0.0
        } else {
            100.0 * self.soft_found as f64 / self.soft_total as f64
        }
    }
    /// Share of ABSENCE probes correctly declined — the hallucination rate,
    /// inverted. Low here means the model invents facts the document lacks.
    fn absence_pct(&self) -> f64 {
        if self.absence_total == 0 {
            0.0
        } else {
            100.0 * self.absence_ok as f64 / self.absence_total as f64
        }
    }
    fn mean_steps(&self) -> f64 {
        if self.blocks == 0 {
            0.0
        } else {
            self.steps as f64 / self.blocks as f64
        }
    }
    fn get(&self, key: &str) -> Option<f64> {
        Some(match key {
            "passed" => f64::from(u8::from(self.passed)),
            "blocks" => self.blocks as f64,
            "committed_rows" => self.committed_rows as f64,
            "hard" => self.hard as f64,
            "dup" => self.dup as f64,
            "trims" => self.trims as f64,
            "contested_per_1k" => self.contested_per_1k(),
            "mean_steps" => self.mean_steps(),
            "prompts_passed" => self.prompts_passed as f64,
            "prompts_total" => self.prompts_total as f64,
            "retrieval_pct" => self.retrieval_pct(),
            "steps_committed" => self.steps_committed as f64,
            "steps_run" => self.steps_run as f64,
            "steps_retry" => self.steps_retry() as f64,
            "cases" => self.cases as f64,
            "prog_pass" => self.prog_pass as f64,
            "compile_fail" => self.compile_fail as f64,
            "wrong_output" => self.wrong_output as f64,
            "fenced" => self.fenced as f64,
            "probes" => self.probes as f64,
            "prog_pass_pct" => self.prog_pass_pct(),
            "compile_fail_pct" => self.compile_fail_pct(),
            "fenced_pct" => self.fenced_pct(),
            "soft_found" => self.soft_found as f64,
            "soft_total" => self.soft_total as f64,
            "soft_pct" => self.soft_pct(),
            "absence_ok" => self.absence_ok as f64,
            "absence_total" => self.absence_total as f64,
            "absence_pct" => self.absence_pct(),
            _ => return None,
        })
    }
}

/// `NAME:KEY=VAL,KEY=VAL` (the override list may be empty: `base:`).
fn parse_arm(spec: &str) -> Result<Arm, String> {
    let (name, rest) = spec.split_once(':').ok_or_else(|| {
        format!("arm {spec:?} must be NAME:KEY=VAL[,KEY=VAL...] (use NAME: for no overrides)")
    })?;
    if name.is_empty() {
        return Err(format!("arm {spec:?} has an empty name"));
    }
    let mut pairs = Vec::new();
    for kv in rest.split(',').filter(|s| !s.trim().is_empty()) {
        let (k, v) = kv
            .split_once('=')
            .ok_or_else(|| format!("arm {name}: override {kv:?} must be KEY=VALUE"))?;
        pairs.push((k.trim().to_string(), v.trim().to_string()));
    }
    Ok(Arm {
        name: name.to_string(),
        pairs,
    })
}

/// `metric<op>value`, where value is a number or `baseline[*factor]`.
struct Gate {
    metric: String,
    op: String,
    value: f64,
    vs_baseline: bool,
}

fn parse_gate(spec: &str) -> Result<Gate, String> {
    for op in ["<=", ">=", "==", "<", ">"] {
        if let Some((m, v)) = spec.split_once(op) {
            let (m, v) = (m.trim(), v.trim());
            let (vs_baseline, value) = if let Some(f) = v.strip_prefix("baseline") {
                let f = f.trim();
                let factor = if let Some(mul) = f.strip_prefix('*') {
                    mul.trim()
                        .parse::<f64>()
                        .map_err(|_| format!("gate {spec:?}: bad factor"))?
                } else if f.is_empty() {
                    1.0
                } else {
                    return Err(format!(
                        "gate {spec:?}: expected `baseline` or `baseline*N`"
                    ));
                };
                (true, factor)
            } else {
                (
                    false,
                    v.parse::<f64>()
                        .map_err(|_| format!("gate {spec:?}: {v:?} is not a number"))?,
                )
            };
            return Ok(Gate {
                metric: m.to_string(),
                op: op.to_string(),
                value,
                vs_baseline,
            });
        }
    }
    Err(format!(
        "gate {spec:?} must be METRIC<OP>VALUE (<=, >=, ==, <, >)"
    ))
}

fn cmp_ok(op: &str, lhs: f64, rhs: f64) -> bool {
    match op {
        "<=" => lhs <= rhs + f64::EPSILON,
        ">=" => lhs + f64::EPSILON >= rhs,
        "==" => (lhs - rhs).abs() < 1e-9,
        "<" => lhs < rhs,
        ">" => lhs > rhs,
        _ => false,
    }
}

/// Append the battery's own verdict to the trace, so the file carries
/// everything a later analysis needs.
#[cfg(target_os = "macos")]
fn append_run_summary(path: &Path, out: &super::smoketest::SmokeOutcome) {
    use std::io::Write;
    let mut rec = serde_json::json!({
        "kind": "run_summary",
        "passed": out.ok(),
        "prompts_passed": out.passed,
        "prompts_total": out.total,
        "kw_found": out.kw_found,
        "kw_total": out.kw_total,
        "steps_committed": out.steps_committed,
        "steps_run": out.steps_total,
    });
    // Only the `programmatic` battery carries these; omitting them entirely
    // for the others keeps "did not run" distinguishable from "ran and scored
    // zero", which a defaulted 0 would erase.
    if let (Some(p), Some(obj)) = (&out.prog, rec.as_object_mut()) {
        obj.insert("cases".into(), p.cases.into());
        obj.insert("prog_pass".into(), p.pass.into());
        obj.insert("compile_fail".into(), p.compile_fail.into());
        obj.insert("wrong_output".into(), p.wrong_output.into());
        obj.insert("fenced".into(), p.fenced.into());
        obj.insert("probes".into(), p.probes.into());
    }
    if let (Some(sf), Some(obj)) = (&out.soft, rec.as_object_mut()) {
        obj.insert("soft_found".into(), sf.found.into());
        obj.insert("soft_total".into(), sf.total.into());
        obj.insert("absence_ok".into(), sf.absence_ok.into());
        obj.insert("absence_total".into(), sf.absence_total.into());
    }
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        Ok(mut f) => {
            let _ = writeln!(f, "{rec}");
        }
        Err(e) => eprintln!("census: cannot append summary to {}: {e}", path.display()),
    }
}

/// Count contested COMMITTED rows in one p_max trace file.
fn scan_trace(path: &Path, tau: f32) -> Metrics {
    let mut m = Metrics {
        passed: true,
        ..Default::default()
    };
    let Ok(text) = std::fs::read_to_string(path) else {
        return m;
    };
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(rec) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        match rec.get("kind").and_then(|k| k.as_str()) {
            Some("block_commit") => {}
            // Written by census at the end of each run so a trace is
            // self-describing: `--analyze` recovers the battery verdict and
            // step accounting without the live outcome in hand.
            Some("run_summary") => {
                let u = |k: &str| rec.get(k).and_then(serde_json::Value::as_u64).unwrap_or(0);
                m.summaries += 1;
                m.passed &= rec
                    .get("passed")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                m.prompts_passed += u("prompts_passed");
                m.prompts_total += u("prompts_total");
                m.kw_found += u("kw_found");
                m.kw_total += u("kw_total");
                m.steps_committed += u("steps_committed");
                m.steps_run += u("steps_run");
                m.cases += u("cases");
                m.prog_pass += u("prog_pass");
                m.compile_fail += u("compile_fail");
                m.wrong_output += u("wrong_output");
                m.fenced += u("fenced");
                m.probes += u("probes");
                m.soft_found += u("soft_found");
                m.soft_total += u("soft_total");
                m.absence_ok += u("absence_ok");
                m.absence_total += u("absence_total");
                continue;
            }
            _ => continue,
        }
        let pmax: Vec<f64> = rec
            .get("pmax")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(serde_json::Value::as_f64).collect())
            .unwrap_or_default();
        let argmax: Vec<i64> = rec
            .get("argmax")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(serde_json::Value::as_i64).collect())
            .unwrap_or_default();
        let kept = rec
            .get("kept")
            .and_then(serde_json::Value::as_u64)
            .map_or(pmax.len(), |k| k as usize)
            .min(pmax.len())
            .min(argmax.len());
        m.blocks += 1;
        m.committed_rows += kept as u64;
        m.steps += rec
            .get("steps")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if rec
            .get("conf_trim_row")
            .map(|v| !v.is_null())
            .unwrap_or(false)
        {
            m.trims += 1;
        }
        for i in 0..kept {
            let p = pmax[i];
            if p < 0.5 {
                m.hard += 1;
            } else if p < f64::from(tau)
                && ((i > 0 && argmax[i] == argmax[i - 1])
                    || (i + 1 < kept && argmax[i] == argmax[i + 1]))
            {
                m.dup += 1;
            }
        }
    }
    m
}

/// Scan a directory of traces WITHOUT running anything: the same report and
/// the same gates over an existing campaign's output. Keeps trace analysis
/// inside the binary (no side-car script to drift from the scanner the
/// command itself uses). Files are `<arm>.<battery>.<seed>.jsonl`.
fn analyze_dir(dir: &Path, tau: f32) -> BTreeMap<(String, String), Metrics> {
    let mut out: BTreeMap<(String, String), Metrics> = BTreeMap::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        eprintln!("census: cannot read {}", dir.display());
        return out;
    };
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "jsonl"))
        .collect();
    files.sort();
    for f in files {
        let stem = f
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let mut parts = stem.split('.');
        let arm = parts.next().unwrap_or("?").to_string();
        let battery = parts.next().unwrap_or("all").to_string();
        let m = scan_trace(&f, tau);
        out.entry((arm, battery))
            .and_modify(|acc| acc.merge(&m))
            .or_insert(m);
    }
    out
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_census_cmd(
    model_dir: &Path,
    arm_specs: &[String],
    batteries: &[String],
    seeds: &[u64],
    gate_specs: &[String],
    baseline: Option<&str>,
    out_dir: Option<&Path>,
    tau: f32,
    steps: usize,
    analyze: Option<&Path>,
) -> ExitCode {
    // Analyze-only: no arms, no GPU, just the report + gates over traces.
    if let Some(dir) = analyze {
        let results = analyze_dir(dir, tau);
        if results.is_empty() {
            eprintln!("census: no *.jsonl traces in {}", dir.display());
            return ExitCode::FAILURE;
        }
        let mut gates = Vec::new();
        for spec in gate_specs {
            match parse_gate(spec) {
                Ok(g) => gates.push(g),
                Err(e) => {
                    eprintln!("census: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
        // Traces carry no pass/fail (that is the battery's verdict), so a
        // `passed` gate is meaningless here; report and gate on counts.
        let have_verdicts = results.values().any(|m| m.summaries > 0);
        return report(
            &results,
            &gates,
            baseline,
            out_dir,
            tau,
            seeds,
            have_verdicts,
        );
    }

    // ---- Parse + VALIDATE everything before any GPU time is spent. -------
    let mut arms = Vec::new();
    for spec in arm_specs {
        match parse_arm(spec) {
            Ok(a) => arms.push(a),
            Err(e) => {
                eprintln!("census: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    if arms.is_empty() {
        eprintln!("census: at least one --arm is required (e.g. --arm 'base:')");
        return ExitCode::FAILURE;
    }
    let mut gates = Vec::new();
    for spec in gate_specs {
        match parse_gate(spec) {
            Ok(g) => gates.push(g),
            Err(e) => {
                eprintln!("census: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    // Arms go through the same flag validation as process startup: a bad
    // arm dies here, not three hours in.
    let mut configs = Vec::new();
    let mut bad = false;
    for arm in &arms {
        let (cfg, errs) = crate::flags::RuntimeConfig::from_pairs(&arm.pairs);
        for e in &errs {
            eprintln!("census: arm {}: {e}", arm.name);
            bad = true;
        }
        configs.push(cfg);
    }
    if bad {
        return ExitCode::FAILURE;
    }
    // `Battery::parse` is the single source of truth for the known names, so
    // adding a battery cannot leave this check behind.
    let mut battery_kinds = Vec::new();
    for b in batteries {
        match super::smoketest::Battery::parse(b) {
            Some(k) => battery_kinds.push(k),
            None => {
                eprintln!(
                    "census: unknown battery {b:?} (known: {})",
                    super::smoketest::Battery::KNOWN
                );
                return ExitCode::FAILURE;
            }
        }
    }
    if let Some(b) = baseline
        && !arms.iter().any(|a| a.name == b)
    {
        eprintln!("census: --baseline {b:?} is not one of the arms");
        return ExitCode::FAILURE;
    }

    let trace_dir: PathBuf = out_dir.map_or_else(std::env::temp_dir, Path::to_path_buf);
    if let Err(e) = std::fs::create_dir_all(&trace_dir) {
        eprintln!("census: cannot create {}: {e}", trace_dir.display());
        return ExitCode::FAILURE;
    }
    eprintln!(
        "census: {} arm(s) x {} batter(ies) x {} seed(s) = {} runs -> {}",
        arms.len(),
        batteries.len(),
        seeds.len(),
        arms.len() * batteries.len() * seeds.len(),
        trace_dir.display(),
    );

    // ---- Run the matrix. -------------------------------------------------
    // Per (arm, battery): merged over seeds. Batteries are separate metric
    // rows so a gate can target one of them.
    let mut results: BTreeMap<(String, String), Metrics> = BTreeMap::new();
    for (arm, cfg) in arms.iter().zip(&configs) {
        for (battery, &kind) in batteries.iter().zip(&battery_kinds) {
            for &seed in seeds {
                let tag = format!("{}.{}.seed{seed}", arm.name, battery);
                let trace = trace_dir.join(format!("{tag}.jsonl"));
                let _ = std::fs::remove_file(&trace);
                // The trace path is injected as a flag, so it rides the same
                // parse path as everything else in the arm.
                let mut pairs = arm.pairs.clone();
                pairs.retain(|(k, _)| k != "DGQ_TRACE_PMAX_JSONL");
                pairs.push((
                    "DGQ_TRACE_PMAX_JSONL".into(),
                    trace.to_string_lossy().to_string(),
                ));
                let (run_cfg, errs) = crate::flags::RuntimeConfig::from_pairs(&pairs);
                debug_assert!(errs.is_empty(), "{errs:?}");
                let _guard = crate::flags::install_scoped(run_cfg);
                eprintln!("census: [{tag}] running");
                let outcome = super::smoketest::run_smoketest(
                    model_dir,
                    None,
                    Some(seed),
                    steps,
                    None,
                    false,
                    None,
                    1,
                    kind,
                );
                if let Some(e) = &outcome.error {
                    eprintln!("census: [{tag}] {e}");
                } else if !outcome.ok() {
                    eprintln!(
                        "census: [{tag}] {}/{} passed; failed: {}",
                        outcome.passed,
                        outcome.total,
                        outcome.failed_ids().join(", "),
                    );
                }
                append_run_summary(&trace, &outcome);
                // scan_trace is now the SINGLE reader of a run's numbers, so
                // a live run and a later `--analyze` of the same trace agree
                // by construction.
                let m = scan_trace(&trace, tau);
                let _ = cfg; // arm cfg kept for reporting; run_cfg is what ran
                results
                    .entry((arm.name.clone(), battery.clone()))
                    .and_modify(|acc| acc.merge(&m))
                    .or_insert(m);
            }
        }
    }

    report(&results, &gates, baseline, out_dir, tau, seeds, true)
}

/// Print the matrix, optionally persist it, then evaluate gates.
/// `with_pass` marks whether the pass/fail column is meaningful (it is not
/// in analyze-only mode, where nothing was executed).
fn report(
    results: &BTreeMap<(String, String), Metrics>,
    gates: &[Gate],
    baseline: Option<&str>,
    out_dir: Option<&Path>,
    tau: f32,
    seeds: &[u64],
    with_pass: bool,
) -> ExitCode {
    println!();
    println!(
        "{:<12} {:<9} {:>6} {:>7} {:>12} {:>6} {:>5} {:>6} {:>9} {:>7} {:>8} {:>7}",
        "arm",
        "battery",
        "pass",
        "blocks",
        "commit_rows",
        "hard",
        "dup",
        "trims",
        "cont/1k",
        "retr%",
        "steps",
        "retry"
    );
    for ((arm, battery), m) in results {
        println!(
            "{:<12} {:<9} {:>6} {:>7} {:>12} {:>6} {:>5} {:>6} {:>9.2} {:>7.1} {:>8} {:>7}",
            arm,
            battery,
            if !with_pass {
                "-"
            } else if m.passed {
                "yes"
            } else {
                "NO"
            },
            m.blocks,
            m.committed_rows,
            m.hard,
            m.dup,
            m.trims,
            m.contested_per_1k(),
            m.retrieval_pct(),
            m.steps_run,
            m.steps_retry(),
        );
    }

    // Executable correctness gets its own block: the three outcome states are
    // only meaningful together, and bolting five columns onto the wart table
    // would make both unreadable. Printed only when something executed.
    if results.values().any(|m| m.cases > 0) {
        println!();
        println!(
            "{:<12} {:<13} {:>6} {:>6} {:>6} {:>13} {:>13} {:>7} {:>7}",
            "arm",
            "battery",
            "probes",
            "cases",
            "pass",
            "compile_fail",
            "wrong_output",
            "pass%",
            "fenced%"
        );
        for ((arm, battery), m) in results.iter().filter(|(_, m)| m.cases > 0) {
            println!(
                "{:<12} {:<13} {:>6} {:>6} {:>6} {:>13} {:>13} {:>7.1} {:>7.1}",
                arm,
                battery,
                m.probes,
                m.cases,
                m.prog_pass,
                m.compile_fail,
                m.wrong_output,
                m.prog_pass_pct(),
                m.fenced_pct(),
            );
        }
    }

    // Soft retrieval prints separately too, and shows the two rates side by
    // side: recall means little without the hallucination rate next to it,
    // since a model that answers everything scores well on one and badly on
    // the other.
    if results
        .values()
        .any(|m| m.soft_total > 0 || m.absence_total > 0)
    {
        println!();
        println!(
            "{:<12} {:<13} {:>10} {:>8} {:>12} {:>10}",
            "arm", "battery", "soft_found", "soft%", "absence_ok", "absence%"
        );
        for ((arm, battery), m) in results
            .iter()
            .filter(|(_, m)| m.soft_total > 0 || m.absence_total > 0)
        {
            println!(
                "{:<12} {:<13} {:>10} {:>8.1} {:>12} {:>10.1}",
                arm,
                battery,
                format!("{}/{}", m.soft_found, m.soft_total),
                m.soft_pct(),
                format!("{}/{}", m.absence_ok, m.absence_total),
                m.absence_pct(),
            );
        }
    }

    if let Some(dir) = out_dir {
        let json: Vec<serde_json::Value> = results
            .iter()
            .map(|((arm, battery), m)| {
                serde_json::json!({
                    "arm": arm, "battery": battery, "passed": m.passed,
                    "blocks": m.blocks, "committed_rows": m.committed_rows,
                    "hard": m.hard, "dup": m.dup, "trims": m.trims,
                    "contested_per_1k": m.contested_per_1k(),
                    "mean_steps": m.mean_steps(),
                    "prompts_passed": m.prompts_passed,
                    "prompts_total": m.prompts_total,
                    "retrieval_pct": m.retrieval_pct(),
                    "steps_committed": m.steps_committed,
                    "steps_run": m.steps_run,
                    "steps_retry": m.steps_retry(),
                    "probes": m.probes, "cases": m.cases,
                    "prog_pass": m.prog_pass,
                    "compile_fail": m.compile_fail,
                    "wrong_output": m.wrong_output,
                    "fenced": m.fenced,
                    "prog_pass_pct": m.prog_pass_pct(),
                    "compile_fail_pct": m.compile_fail_pct(),
                    "fenced_pct": m.fenced_pct(),
                    "soft_found": m.soft_found, "soft_total": m.soft_total,
                    "soft_pct": m.soft_pct(),
                    "absence_ok": m.absence_ok, "absence_total": m.absence_total,
                    "absence_pct": m.absence_pct(),
                })
            })
            .collect();
        let path = dir.join("census.json");
        match serde_json::to_string_pretty(&serde_json::json!({
            "tau": tau, "seeds": seeds, "results": json,
        })) {
            Ok(s) => {
                if let Err(e) = std::fs::write(&path, s) {
                    eprintln!("census: cannot write {}: {e}", path.display());
                } else {
                    eprintln!("census: wrote {}", path.display());
                }
            }
            Err(e) => eprintln!("census: cannot serialize results: {e}"),
        }
    }

    // ---- Gates. ----------------------------------------------------------
    if gates.is_empty() {
        return ExitCode::SUCCESS;
    }
    println!();
    let mut failed = false;
    for ((arm, battery), m) in results {
        for g in gates {
            // `battery.metric` targets one battery; a bare metric applies to all.
            let (want_battery, metric) = match g.metric.split_once('.') {
                Some((b, k)) => (Some(b), k),
                None => (None, g.metric.as_str()),
            };
            if want_battery.is_some_and(|b| b != battery) {
                continue;
            }
            let Some(lhs) = m.get(metric) else {
                eprintln!("census: gate names unknown metric {metric:?}");
                failed = true;
                continue;
            };
            let rhs = if g.vs_baseline {
                let Some(base) = baseline else {
                    eprintln!("census: gate uses `baseline` but --baseline was not given");
                    failed = true;
                    continue;
                };
                match results
                    .get(&(base.to_string(), battery.clone()))
                    .and_then(|bm| bm.get(metric))
                {
                    Some(v) => v * g.value,
                    None => {
                        eprintln!("census: no baseline result for {base}/{battery}");
                        failed = true;
                        continue;
                    }
                }
            } else {
                g.value
            };
            let ok = cmp_ok(&g.op, lhs, rhs);
            if !ok {
                failed = true;
            }
            println!(
                "{:<5} {arm}/{battery}: {metric} {lhs:.3} {} {rhs:.3}",
                if ok { "PASS" } else { "FAIL" },
                g.op,
            );
        }
    }
    if failed {
        eprintln!("census: GATES FAILED");
        return ExitCode::FAILURE;
    }
    eprintln!("census: all gates passed");
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arm_spec_parses_name_and_overrides() {
        let a = parse_arm("trim:DGQ_COMMIT_CONF_TRIM=0.9,DGQ_PREFIX_EXIT=0.05").unwrap();
        assert_eq!(a.name, "trim");
        assert_eq!(a.pairs.len(), 2);
        // A bare `name:` is a valid no-override arm (the baseline).
        assert!(parse_arm("base:").unwrap().pairs.is_empty());
        assert!(parse_arm("noколон").is_err());
        assert!(parse_arm("bad:NOEQUALS").is_err());
        assert!(parse_arm(":x=1").is_err());
    }

    #[test]
    fn gate_spec_parses_numbers_and_baseline_relatives() {
        let g = parse_gate("contested_per_1k<=0.5").unwrap();
        assert_eq!(
            (g.metric.as_str(), g.op.as_str(), g.value),
            ("contested_per_1k", "<=", 0.5)
        );
        assert!(!g.vs_baseline);
        let g = parse_gate("mean_steps<=baseline*1.10").unwrap();
        assert!(g.vs_baseline);
        assert!((g.value - 1.10).abs() < 1e-9);
        let g = parse_gate("smoke.passed==1").unwrap();
        assert_eq!(g.metric, "smoke.passed");
        assert!(parse_gate("nonsense").is_err());
        assert!(parse_gate("x<=notanumber").is_err());
    }

    #[test]
    fn contested_rate_counts_only_committed_rows() {
        // Two blocks; the second is trimmed so its bad rows never committed.
        let dir = std::env::temp_dir().join(format!("dgq-census-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("t.jsonl");
        std::fs::write(
            &f,
            format!(
                "{}\n{}\n",
                serde_json::json!({
                    "kind": "block_commit", "steps": 10, "kept": 3,
                    "pmax": [0.99, 0.40, 0.95], "argmax": [1, 2, 3],
                }),
                serde_json::json!({
                    // kept=1 => the 0.2 row at index 1 was trimmed away.
                    "kind": "block_commit", "steps": 8, "kept": 1,
                    "conf_trim_row": 1,
                    "pmax": [0.99, 0.20], "argmax": [4, 5],
                }),
            ),
        )
        .unwrap();
        let m = scan_trace(&f, 0.9);
        assert_eq!(m.blocks, 2);
        assert_eq!(m.committed_rows, 4);
        assert_eq!(m.hard, 1, "only the COMMITTED 0.40 row counts");
        assert_eq!(m.trims, 1);
        assert_eq!(m.steps, 18);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A trace carrying a `run_summary` is self-describing: `--analyze`
    /// recovers the verdict and step accounting that used to live only in
    /// the live run's outcome.
    #[test]
    fn run_summary_makes_a_trace_self_describing() {
        let dir = std::env::temp_dir().join(format!("dgq-census-sum-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("t.jsonl");
        std::fs::write(
            &f,
            format!(
                "{}\n{}\n",
                serde_json::json!({
                    "kind": "block_commit", "steps": 7, "kept": 2,
                    "pmax": [0.99, 0.99], "argmax": [1, 2],
                }),
                serde_json::json!({
                    "kind": "run_summary", "passed": true,
                    "prompts_passed": 17, "prompts_total": 17,
                    "kw_found": 8, "kw_total": 8,
                    "steps_committed": 120, "steps_run": 145,
                }),
            ),
        )
        .unwrap();
        let m = scan_trace(&f, 0.9);
        assert_eq!(m.summaries, 1);
        assert!(m.passed);
        assert_eq!(m.prompts_passed, 17);
        assert_eq!(m.retrieval_pct(), 100.0);
        assert_eq!(m.steps_run, 145);
        assert_eq!(m.steps_retry(), 25, "run - committed = discarded re-rolls");
        // The block_commit record is still counted for wart stats.
        assert_eq!(m.blocks, 1);
        assert_eq!(m.committed_rows, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The `programmatic` battery's three states survive the trace round-trip
    /// and are reachable as gate metrics — the whole point of routing them
    /// through `Metrics::get` rather than printing them and moving on.
    #[test]
    fn programmatic_states_round_trip_and_are_gateable() {
        let dir = std::env::temp_dir().join(format!("dgq-census-prog-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("t.jsonl");
        std::fs::write(
            &f,
            format!(
                "{}\n",
                serde_json::json!({
                    "kind": "run_summary", "passed": false,
                    "prompts_passed": 5, "prompts_total": 8,
                    "steps_committed": 400, "steps_run": 400,
                    "probes": 8, "cases": 16, "prog_pass": 12,
                    "compile_fail": 3, "wrong_output": 1, "fenced": 2,
                }),
            ),
        )
        .unwrap();
        let m = scan_trace(&f, 0.9);
        assert_eq!(m.cases, m.prog_pass + m.compile_fail + m.wrong_output);
        assert_eq!(m.get("compile_fail"), Some(3.0));
        assert_eq!(
            m.get("wrong_output"),
            Some(1.0),
            "distinct from compile_fail"
        );
        assert_eq!(m.get("prog_pass_pct"), Some(75.0));
        assert_eq!(m.get("fenced_pct"), Some(25.0), "over PROBES, not cases");
        // A battery that ran no cases floors at 0 rather than the 100 that
        // `retrieval_pct` uses: an unmeasured retrieval must not fail a gate,
        // but an unmeasured pass rate must not PASS one.
        assert_eq!(Metrics::default().prog_pass_pct(), 0.0);
        assert_eq!(Metrics::default().retrieval_pct(), 100.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dup_tier_needs_both_low_confidence_and_a_repeat() {
        let dir = std::env::temp_dir().join(format!("dgq-census-dup-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("t.jsonl");
        std::fs::write(
            &f,
            format!(
                "{}\n",
                serde_json::json!({
                    "kind": "block_commit", "steps": 5, "kept": 4,
                    // 0.7 repeating a neighbour = dup; 0.7 alone = benign soft
                    // row; 0.95 repeating = confident repeat, not a stutter.
                    "pmax": [0.70, 0.70, 0.95, 0.95], "argmax": [7, 7, 9, 9],
                }),
            ),
        )
        .unwrap();
        let m = scan_trace(&f, 0.9);
        assert_eq!(m.dup, 2, "both low-confidence repeats");
        assert_eq!(m.hard, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
