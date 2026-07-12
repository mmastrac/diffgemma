//! `smoketest` gate subcommand + fixture spec types.

use super::*;

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
/// Convergence + adherence gate over a prompt set. Reuses the chat session path
/// so each prompt is a fresh single-turn generation; reports actual vs threshold
/// denoise steps (ratchet thresholds down in the JSON as the engine improves).
#[cfg(target_os = "macos")]
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
    use metal::{StepGenerateConfig, StepGenerateSession, generate_with_session};

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: smoketest requires a .dgq directory (-m /path/to/quantized-weights)");
        return ExitCode::FAILURE;
    }

    let default_path = std::path::PathBuf::from("fixtures/smoketest/prompts.json");
    let path = prompts_path.unwrap_or(default_path.as_path());
    let mut spec: SmoketestSpec = match std::fs::read_to_string(path) {
        Ok(text) => match serde_json::from_str(&text) {
            Ok(s) => s,
            Err(err) => {
                eprintln!("error: parse {}: {err}", path.display());
                return ExitCode::FAILURE;
            }
        },
        Err(err) => {
            eprintln!("error: read {}: {err}", path.display());
            return ExitCode::FAILURE;
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
        let kept = if longctx {
            spec.longctx.len()
        } else {
            spec.adherence.len() + spec.convergence.len()
        };
        if kept == 0 {
            eprintln!("smoketest: no prompts match filter {pat:?}");
            return ExitCode::FAILURE;
        }
        eprintln!("smoketest: filter {pat:?} -> {kept} prompt(s)");
    }
    if longctx && spec.longctx.is_empty() {
        eprintln!("smoketest: --longctx but the spec has no longctx probes");
        return ExitCode::FAILURE;
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
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
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
    let smoke_max_seq = if longctx {
        LONGCTX_MAX_SEQ
    } else {
        SMOKE_MAX_SEQ
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
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    let tok_path = model_dir.join("tokenizer.json");
    let tokenizer = match tokenizer::Tokenizer::load(&tok_path) {
        Ok(t) => t,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    // (denoise_steps, reply) for one fresh single-turn generation.
    let mut run_one = |prompt_text: &str| -> Result<(usize, String), safetensors::Error> {
        // Each prompt is independent — drop prior KV so we re-prefill fresh
        // (chat's KV-reuse continuation would otherwise answer the first prompt).
        session.reset_kv();
        let history = vec![chat_template::ChatTurn::user(prompt_text)];
        let prompt = build_chat_prompt_tokens(model_dir, &history, raw_prompt)?;
        let prompt_len = prompt.len();
        // Bound generation (and thus time + KV) — a gate doesn't need essays.
        step_cfg.max_new_tokens =
            SMOKE_GEN_CAP.min(smoke_max_seq.saturating_sub(prompt_len).max(1));
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
        Ok((committed_steps, reply))
    };

    // (No warm-up: cold-start is fixed at the root by the deterministic first-step
    // SC seed. Verified engine 16/16 with the warm-up removed.)

    let mut passed = 0usize;
    let mut total = 0usize;
    let mut failures: Vec<String> = Vec::new();

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
                println!(
                    "  {id:<22} {mark:<4} doc {n:>5}tok steps {st:>3}/{max:<3} {af}  | {prev}",
                    id = p.id
                );
            }
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
    if passed == total {
        ExitCode::SUCCESS
    } else {
        println!("failed: {}", failures.join(", "));
        ExitCode::FAILURE
    }
}
