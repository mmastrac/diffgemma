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
//! A BATTERY is a unit of gated work (`smoke`, `longctx`). Every battery
//! yields the same metric shape, because the census numbers come from the
//! denoise path's p_max trace rather than from anything battery-specific:
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
    }
    /// The decision line: contested rows that reached KV, per 1k committed.
    fn contested_per_1k(&self) -> f64 {
        if self.committed_rows == 0 {
            0.0
        } else {
            1000.0 * (self.hard + self.dup) as f64 / self.committed_rows as f64
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
            _ => return None,
        })
    }
}

/// `NAME:KEY=VAL,KEY=VAL` (the override list may be empty: `base:`).
fn parse_arm(spec: &str) -> Result<Arm, String> {
    let (name, rest) = spec
        .split_once(':')
        .ok_or_else(|| format!("arm {spec:?} must be NAME:KEY=VAL[,KEY=VAL...] (use NAME: for no overrides)"))?;
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
                    return Err(format!("gate {spec:?}: expected `baseline` or `baseline*N`"));
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
    Err(format!("gate {spec:?} must be METRIC<OP>VALUE (<=, >=, ==, <, >)"))
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
        if rec.get("kind").and_then(|k| k.as_str()) != Some("block_commit") {
            continue;
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
        m.steps += rec.get("steps").and_then(serde_json::Value::as_u64).unwrap_or(0);
        if rec.get("conf_trim_row").map(|v| !v.is_null()).unwrap_or(false) {
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
) -> ExitCode {
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
    for b in batteries {
        if !matches!(b.as_str(), "smoke" | "longctx") {
            eprintln!("census: unknown battery {b:?} (known: smoke, longctx)");
            return ExitCode::FAILURE;
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
        for battery in batteries {
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
                let code = super::smoketest::run_smoketest_cmd(
                    model_dir,
                    None,
                    Some(seed),
                    steps,
                    None,
                    false,
                    None,
                    1,
                    battery == "longctx",
                );
                let mut m = scan_trace(&trace, tau);
                m.passed = code == ExitCode::SUCCESS;
                let _ = cfg; // arm cfg kept for reporting; run_cfg is what ran
                results
                    .entry((arm.name.clone(), battery.clone()))
                    .and_modify(|acc| acc.merge(&m))
                    .or_insert(m);
            }
        }
    }

    // ---- Report. ---------------------------------------------------------
    println!();
    println!(
        "{:<12} {:<9} {:>6} {:>7} {:>12} {:>6} {:>5} {:>6} {:>9} {:>10}",
        "arm", "battery", "pass", "blocks", "commit_rows", "hard", "dup", "trims",
        "cont/1k", "steps/blk"
    );
    for ((arm, battery), m) in &results {
        println!(
            "{:<12} {:<9} {:>6} {:>7} {:>12} {:>6} {:>5} {:>6} {:>9.2} {:>10.1}",
            arm,
            battery,
            if m.passed { "yes" } else { "NO" },
            m.blocks,
            m.committed_rows,
            m.hard,
            m.dup,
            m.trims,
            m.contested_per_1k(),
            m.mean_steps(),
        );
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
    for ((arm, battery), m) in &results {
        for g in &gates {
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
        assert_eq!((g.metric.as_str(), g.op.as_str(), g.value), ("contested_per_1k", "<=", 0.5));
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
