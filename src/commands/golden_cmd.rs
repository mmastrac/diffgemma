//! `golden` byte-identity pack subcommand (Tier-1 refactor gate).

use super::*;

/// Golden byte-identity pack (task #73) — the Tier-1 refactor gate. Runs a fixed
/// path-matrix through the PRODUCTION `generate_with_session` path and asserts
/// byte identity (token_ids + causal-KV hash + denoise metadata) against the
/// checked-in pack. `bless` re-records instead of checking.
///
/// One f16 session at `GOLDEN_MAX_SEQ` covers the whole matrix: sliding layers
/// ring-wrap at >2048 regardless of `max_seq`, full layers stay linear, and
/// attention reads only `[0..kv_len]`, so per-case outputs are `max_seq`-invariant
/// for any prompt that fits. (See the module doc for the machine-class caveat:
/// pick `GOLDEN_MAX_SEQ` so the f16 KV stays under the q8-auto threshold on the
/// blessing box.)
#[cfg(target_os = "macos")]
pub(crate) fn run_golden_cmd(
    model_dir: &std::path::Path,
    pack_path: Option<&std::path::Path>,
    bless: bool,
    filter: Option<&str>,
) -> ExitCode {
    use metal::{StepGenerateConfig, StepGenerateSession};

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: golden requires a .dgq directory (-m /path/to/quantized-weights)");
        return ExitCode::FAILURE;
    }
    let layers = match resolve_model_layers(model_dir, None) {
        Ok(l) => l,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    // Big enough for the deepest longctx case; small enough that f16 KV stays
    // under the q8-auto fraction on a 36GB box (~20.0 GiB resident << 0.85·cap).
    const GOLDEN_MAX_SEQ: usize = 16384;

    let pack_path = pack_path.unwrap_or_else(|| golden::default_pack_path());
    let mut pack = match golden::GoldenPack::load(pack_path) {
        Ok(p) => p,
        Err(err) => {
            eprintln!("error: reading pack {}: {err}", pack_path.display());
            return ExitCode::FAILURE;
        }
    };
    let spec_dir = pack_path.parent().unwrap_or(std::path::Path::new("."));

    let tokenizer = match tokenizer::Tokenizer::load(model_dir.join("tokenizer.json")) {
        Ok(t) => t,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };
    let stop_token_ids = config::load_generation_stop_tokens(model_dir);
    let degen = chat_template::empty_reply_check(model_dir, stop_token_ids.clone());

    let sampler = sample::sampler_for_steps(steps_production_default(), false);
    let base_cfg =
        StepGenerateConfig::from_generate(0, 256, GOLDEN_MAX_SEQ, layers, sampler, false);
    let mut session = match StepGenerateSession::open(model_dir, &base_cfg, None) {
        Ok((s, compile)) => {
            eprintln!("golden: session ready ({compile:.2?}, {layers}L, max_seq {GOLDEN_MAX_SEQ})");
            s
        }
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    println!(
        "\ngolden {}: {} ({} mode)",
        pack_path.display(),
        model_dir.display(),
        if bless { "BLESS" } else { "check" }
    );

    let mut passed = 0usize;
    let mut total = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let mut doc_cache: std::collections::HashMap<String, Vec<u32>> = Default::default();

    for case in &mut pack.cases {
        if let Some(pat) = filter {
            if !case.id.to_lowercase().contains(&pat.to_lowercase()) {
                continue;
            }
        }
        total += 1;

        let observed = match run_golden_case(
            &mut session,
            &tokenizer,
            model_dir,
            &degen,
            &stop_token_ids,
            layers,
            GOLDEN_MAX_SEQ,
            spec_dir,
            &mut doc_cache,
            case,
        ) {
            Ok(r) => r,
            Err(err) => {
                println!("  {:<22} ERROR  {err}", case.id);
                failures.push(case.id.clone());
                continue;
            }
        };

        // Determinism: a second in-process run must be byte-identical to the first.
        if case.check_determinism {
            match run_golden_case(
                &mut session,
                &tokenizer,
                model_dir,
                &degen,
                &stop_token_ids,
                layers,
                GOLDEN_MAX_SEQ,
                spec_dir,
                &mut doc_cache,
                case,
            ) {
                Ok(second) => {
                    if let Err(err) = observed.compare(&second) {
                        println!(
                            "  {:<22} FAIL   nondeterministic within process: {err}",
                            case.id
                        );
                        failures.push(case.id.clone());
                        continue;
                    }
                }
                Err(err) => {
                    println!("  {:<22} ERROR  determinism re-run: {err}", case.id);
                    failures.push(case.id.clone());
                    continue;
                }
            }
        }

        if bless {
            case.golden = Some(observed.clone());
            passed += 1;
            println!(
                "  {:<22} BLESS  {} tok, kv_valid {}, {}",
                case.id,
                observed.token_ids.len(),
                observed.kv_valid_len,
                observed.kv_hash
            );
            continue;
        }

        match &case.golden {
            None => {
                println!(
                    "  {:<22} FAIL   never blessed (run `golden --bless`)",
                    case.id
                );
                failures.push(case.id.clone());
            }
            Some(golden) => match golden.compare(&observed) {
                Ok(()) => {
                    passed += 1;
                    println!(
                        "  {:<22} OK     {} tok, kv_valid {}, {}",
                        case.id,
                        observed.token_ids.len(),
                        observed.kv_valid_len,
                        observed.kv_hash
                    );
                }
                Err(err) => {
                    println!("  {:<22} FAIL   {err}", case.id);
                    failures.push(case.id.clone());
                }
            },
        }
    }

    if bless {
        if let Err(err) = pack.write(pack_path) {
            eprintln!("error: writing blessed pack: {err}");
            return ExitCode::FAILURE;
        }
        println!(
            "\ngolden: blessed {passed}/{total} cases -> {}",
            pack_path.display()
        );
        return ExitCode::SUCCESS;
    }

    println!("\ngolden: {passed}/{total} byte-identical");
    if passed == total {
        ExitCode::SUCCESS
    } else {
        println!("failed: {}", failures.join(", "));
        ExitCode::FAILURE
    }
}
/// The step-count baseline the golden sampler uses (matches the production
/// smoketest cap so the denoise trajectory is the shipped one).
pub(crate) fn steps_production_default() -> usize {
    24
}
/// Run one golden case through the production path and capture its record.
/// Resets the KV first (each case is independent) EXCEPT that a multi-turn case
/// keeps its own KV across turns (cross-turn reuse). Returns the FINAL turn's
/// output + a hash of the resident causal KV.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_golden_case(
    session: &mut metal::StepGenerateSession,
    tokenizer: &tokenizer::Tokenizer,
    model_dir: &std::path::Path,
    degen: &Option<std::sync::Arc<dyn Fn(&[u32]) -> bool + Send + Sync>>,
    stop_token_ids: &[u32],
    layers: usize,
    max_seq: usize,
    spec_dir: &std::path::Path,
    doc_cache: &mut std::collections::HashMap<String, Vec<u32>>,
    case: &golden::GoldenCase,
) -> Result<golden::GoldenRecord, crate::Error> {
    use metal::{StepGenerateConfig, generate_with_session};

    session.reset_kv();

    let sampler = sample::sampler_for_steps(steps_production_default(), false);
    let mut cfg = StepGenerateConfig::from_generate(
        case.seed,
        case.max_new_tokens,
        max_seq,
        layers,
        sampler,
        case.no_early_stop,
    );
    cfg.stop_token_ids = stop_token_ids.to_vec();
    cfg.degenerate_reply_check = degen.clone();

    let clamp_new = |prompt_len: usize| {
        case.max_new_tokens
            .min(max_seq.saturating_sub(prompt_len + metal::CANVAS).max(1))
    };

    let question = case.turns.first().cloned().unwrap_or_default();

    let (out, prompt_len) = if let Some(tool) = &case.tool {
        // Tool-call grammar path: render the message + tool declaration.
        let messages = vec![serde_json::json!({"role":"user","content": question})];
        let tools = vec![tool.clone()];
        let prompt_str = tools::render_conversation(&messages, &tools, true, false);
        let prompt = tokenizer.encode_with_specials(&prompt_str);
        cfg.max_new_tokens = clamp_new(prompt.len());
        let out = generate_with_session(session, &prompt, &cfg, "golden")?;
        (out, prompt.len())
    } else if let Some(doc) = &case.doc {
        // Longctx / fast-prefill / ring-wrap: doc excerpt + question.
        if !doc_cache.contains_key(doc) {
            let text = std::fs::read_to_string(spec_dir.join(doc)).map_err(crate::Error::Io)?;
            doc_cache.insert(doc.clone(), tokenizer.encode(&text, false));
        }
        let ids = &doc_cache[doc];
        let n = case.doc_tokens.min(ids.len());
        let excerpt = tokenizer.decode(&ids[..n]);
        let prompt_text =
            format!("{excerpt}\n\n[end of document]\nAnswer from the document above: {question}");
        let prompt = build_chat_prompt_tokens(
            model_dir,
            &[chat_template::ChatTurn::user(prompt_text)],
            false,
        )?;
        cfg.max_new_tokens = clamp_new(prompt.len());
        let out = generate_with_session(session, &prompt, &cfg, "golden")?;
        (out, prompt.len())
    } else {
        // Chat turns: >1 turn exercises cross-turn KV reuse (no reset between).
        let mut history: Vec<chat_template::ChatTurn> = Vec::new();
        let mut last: Option<(generate::GenerateOutput, usize)> = None;
        let n_turns = case.turns.len().max(1);
        for (i, turn) in case.turns.iter().enumerate() {
            history.push(chat_template::ChatTurn::user(turn.clone()));
            let prompt = build_chat_prompt_tokens(model_dir, &history, false)?;
            cfg.max_new_tokens = clamp_new(prompt.len());
            let out = generate_with_session(session, &prompt, &cfg, "golden")?;
            if i + 1 < n_turns {
                // Append the model's own reply so the next turn's prompt shares
                // the resident-KV prefix — exactly how chat drives reuse.
                let new_ids = sample::strip_degenerate_token_ids(
                    out.token_ids.get(prompt.len()..).unwrap_or(&[]),
                );
                let reply = chat_template::sanitize_model_reply(&tokenizer.decode(&new_ids));
                history.push(chat_template::ChatTurn::model(reply));
            }
            last = Some((out, prompt.len()));
        }
        last.expect("at least one turn")
    };

    let snap = session.snapshot_kv();
    // Diagnostic (MoE-error qualification): `DGQ_GOLDEN_DUMP_KV=dir` writes the
    // raw post-prefill KV bytes per case, so two block_m runs can be diffed for
    // per-region rel-error / cosine. No effect when unset.
    if let Ok(dir) = std::env::var("DGQ_GOLDEN_DUMP_KV") {
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(format!("{dir}/{}.kv", case.id), snap.kv_bytes());
    }
    Ok(golden::GoldenRecord {
        prompt_len,
        kv_valid_len: snap.tokens().len(),
        token_ids: out.token_ids,
        denoise_steps_run: out.denoise_steps_run,
        blocks_committed: out.blocks_committed,
        stopped_on_eot: out.stopped_on_eot,
        block_steps_eff: out.block_steps_eff,
        kv_hash: golden::kv_digest(snap.kv_bytes()),
    })
}
