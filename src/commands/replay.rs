//! `replay` — re-execute an `ops.jsonl` op-log (serve `--log-dir`) against a
//! fresh pipeline and diff every event against the recorded one. Turns a
//! field session's op-log into a golden-style repro artifact: the first
//! divergence pinpoints the op whose outcome no longer reproduces (lineage
//! drift, nondeterminism, or a behavior change since the log was recorded).
//!
//! Fidelity notes: ops are reconstructed from the log's full-id records; the
//! replayable cfg core (seed/budget/steps/stops) is rebuilt exactly,
//! but DGQ_* flags come from the CURRENT environment — run the replay under
//! the same flags (and build) as the recorded session for a meaningful diff.
//! A second `meta` line marks a serve restart; replay stops there (the fresh
//! process's epochs restart, so cross-boundary comparison is meaningless).

use std::process::ExitCode;

use crate::pipeline::{Pipeline, PipelineOp};

pub(crate) fn run_replay_cmd(
    model_dir: &std::path::Path,
    log_path: &std::path::Path,
    ctx_override: Option<usize>,
    steps_override: Option<usize>,
) -> ExitCode {
    let text = match std::fs::read_to_string(log_path) {
        Ok(t) => t,
        Err(err) => {
            eprintln!("replay: read {}: {err}", log_path.display());
            return ExitCode::FAILURE;
        }
    };

    // Header + entries. Lines that fail to parse are counted, not fatal
    // (a killed process can leave a torn final line).
    let mut meta_max_seq: Option<usize> = None;
    let mut meta_steps: Option<usize> = None;
    let mut entries: Vec<(usize, serde_json::Value, serde_json::Value)> = Vec::new();
    let mut torn = 0usize;
    for (lineno, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            torn += 1;
            continue;
        };
        if let Some(meta) = v.get("meta") {
            if meta_max_seq.is_some() {
                eprintln!(
                    "replay: second meta header at line {} (serve restart); stopping here",
                    lineno + 1
                );
                break;
            }
            meta_max_seq = meta
                .get("max_seq")
                .and_then(|x| x.as_u64())
                .map(|n| n as usize);
            meta_steps = meta
                .get("steps")
                .and_then(|x| x.as_u64())
                .map(|n| n as usize);
            continue;
        }
        if let (Some(op), Some(event)) = (v.get("op"), v.get("event")) {
            entries.push((lineno + 1, op.clone(), event.clone()));
        } else {
            torn += 1;
        }
    }
    if entries.is_empty() {
        eprintln!("replay: no op entries in {}", log_path.display());
        return ExitCode::FAILURE;
    }

    let max_seq = ctx_override.or(meta_max_seq).unwrap_or(8192);
    let steps = steps_override.or(meta_steps).unwrap_or(24);
    eprintln!(
        "replay: {} ops from {} (ctx={max_seq}, steps={steps}); flags come from THIS environment",
        entries.len(),
        log_path.display()
    );
    let pipeline = Pipeline::spawn(model_dir.to_path_buf(), max_seq, steps);

    let mut ran = 0usize;
    let mut diverged = 0usize;
    let mut skipped = 0usize;
    for (lineno, op_json, recorded) in &entries {
        let Some(op) = PipelineOp::from_log_json(op_json, model_dir) else {
            eprintln!("replay: line {lineno}: unknown op shape; skipping");
            skipped += 1;
            continue;
        };
        if matches!(op, PipelineOp::Shutdown) {
            break;
        }
        let name = op_name(&op);
        let got = pipeline.call(op).log_json();
        ran += 1;
        if got == *recorded {
            println!("replay: [{lineno:>5}] {name:<14} OK");
        } else {
            diverged += 1;
            println!(
                "replay: [{lineno:>5}] {name:<14} DIVERGED\n  recorded: {recorded}\n  replayed: {got}"
            );
        }
    }

    println!(
        "replay: {ran} ops replayed, {diverged} diverged, {skipped} skipped, {torn} unparseable lines"
    );
    if diverged == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn op_name(op: &PipelineOp) -> &'static str {
    match op {
        PipelineOp::Extend(_) => "extend",
        PipelineOp::Generate { .. } => "generate",
        PipelineOp::Rewind(_) => "rewind",
        PipelineOp::SyntheticFill { .. } => "synthetic_fill",
        PipelineOp::KvFingerprint => "fingerprint",
        PipelineOp::Ping => "ping",
        PipelineOp::Mark => "mark",
        PipelineOp::Activate { .. } => "activate",
        PipelineOp::Finalize { .. } => "finalize",
        PipelineOp::AlignTo { .. } => "align_to",
        PipelineOp::Splice { .. } => "splice",
        PipelineOp::BeginTurn { .. } => "begin_turn",
        PipelineOp::ProposeBlock => "propose_block",
        PipelineOp::CommitBlock { .. } => "commit_block",
        PipelineOp::DiscardBlock => "discard_block",
        PipelineOp::EndTurn => "end_turn",
        PipelineOp::Shutdown => "shutdown",
    }
}
