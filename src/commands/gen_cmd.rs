//! One-shot generation subcommands (monolithic + parity) and their printers.

use super::*;

pub(crate) fn print_generate_elapsed(label: &str, elapsed: std::time::Duration) {
    let secs = elapsed.as_secs_f64();
    println!("  {label} elapsed: {secs:.2}s ({elapsed:.2?})");
}
/// Print just the clean reply (the one-shot-chat default for `ask`): decode the
/// generated tokens, strip pad/filler, sanitize chat scaffolding (thought/turn
/// markers — tool-call markers are kept), and print the FULL reply untruncated.
pub(crate) fn print_generate_reply(
    out: &generate::GenerateOutput,
    prompt_len: usize,
    model_dir: &std::path::Path,
) {
    let Ok(tokenizer) = tokenizer::Tokenizer::load(model_dir.join("tokenizer.json")) else {
        eprintln!("error: could not load tokenizer to decode reply");
        return;
    };
    let generated = out.token_ids.get(prompt_len..).unwrap_or(&[]);
    let display_ids = sample::strip_degenerate_token_ids(generated);
    let reply = chat_template::sanitize_model_reply(&tokenizer.decode(&display_ids));
    if reply.is_empty() {
        println!("(empty response)");
    } else {
        println!("{reply}");
    }
}
pub(crate) fn print_generate_output(
    label: &str,
    out: &generate::GenerateOutput,
    prompt_len: usize,
    elapsed: std::time::Duration,
    model_dir: &std::path::Path,
) {
    let new_tokens = out.token_ids.len().saturating_sub(prompt_len);
    println!("{label} ok");
    println!("  total tokens: {}", out.token_ids.len());
    println!("  new tokens:   {new_tokens}");
    println!("  denoise steps run: {}", out.denoise_steps_run);
    println!("  blocks committed:  {}", out.blocks_committed);
    if !out.block_steps_eff.is_empty() {
        println!("  block steps_eff:   {:?}", out.block_steps_eff);
    }
    if !out.last_block_accept_hist.is_empty() {
        println!("  last accept/step:  {:?}", out.last_block_accept_hist);
    }
    if !out.last_block_min_entropy_hist.is_empty() {
        println!("  min_entropy/step:  {:?}", out.last_block_min_entropy_hist);
    }
    print_generate_elapsed(label, elapsed);
    println!(
        "  prefill:  {:.2}s ({:.2?})",
        out.prefill_elapsed.as_secs_f64(),
        out.prefill_elapsed
    );
    println!(
        "  denoise:  {:.2}s ({:.2?})",
        out.denoise_elapsed.as_secs_f64(),
        out.denoise_elapsed
    );
    if out.extend_elapsed.as_secs_f64() > 0.0 {
        println!(
            "  extend:   {:.2}s ({:.2?})",
            out.extend_elapsed.as_secs_f64(),
            out.extend_elapsed
        );
    }
    if out.denoise_elapsed.as_secs_f64() > 0.0 && new_tokens > 0 {
        let tok_s = new_tokens as f64 / out.denoise_elapsed.as_secs_f64();
        println!("  throughput: {tok_s:.2} tok/s (denoise only, excludes prefill/extend)");
    }

    #[cfg(target_os = "macos")]
    if !out.session_telemetry.steps.is_empty() {
        out.session_telemetry.print_summary("  session telemetry:");
        if out.denoise_steps_run > 0 {
            let agg = out.session_telemetry.aggregate_forward();
            let n = out.denoise_steps_run as f64;
            let step_ms = out.denoise_elapsed.as_secs_f64() * 1000.0 / n;
            println!("  mean step wall:       {step_ms:.1} ms");
            println!(
                "  gpu hot path:         {:.1} syncs/step, {:.1} KiB readback/step",
                agg.gpu_syncs as f64 / n,
                agg.gpu_readback_bytes as f64 / 1024.0 / n
            );
            println!(
                "  weight bytes/step:    {:.2} GiB expert + {:.2} MiB logits",
                agg.expert_weight_bytes_touched as f64 / n / (1024.0_f64.powi(3)),
                agg.lm_head_logits_bytes as f64 / n / (1024.0 * 1024.0)
            );
        }
    }

    if let Ok(tokenizer) = tokenizer::Tokenizer::load(model_dir.join("tokenizer.json")) {
        let generated = out.token_ids.get(prompt_len..).unwrap_or(&[]);
        let display_ids = sample::strip_degenerate_token_ids(generated);
        if !display_ids.is_empty() {
            let text = tokenizer.decode(&display_ids);
            if !text.is_empty() {
                let preview: String = text.chars().take(200).collect();
                println!("  text: {preview}");
            }
        } else if !generated.is_empty() {
            let pad = generated
                .iter()
                .filter(|&&t| t == sample::PAD_TOKEN_ID)
                .count();
            let filler = generated
                .iter()
                .filter(|&&t| t == sample::FILLER_TOKEN_ID)
                .count();
            eprintln!(
                "  text: (empty after stripping pad/filler; generated={} pad={pad} filler={filler})",
                generated.len()
            );
        }
    }

    let preview: Vec<String> = out
        .token_ids
        .iter()
        .take(16)
        .map(|t| t.to_string())
        .collect();
    println!("  token_ids[0..16]: [{}]", preview.join(", "));
    if out.token_ids.len() > prompt_len {
        let gen_preview: Vec<String> = out
            .token_ids
            .iter()
            .skip(prompt_len)
            .take(16)
            .map(|t| t.to_string())
            .collect();
        println!("  generated[0..16]: [{}]", gen_preview.join(", "));
    }
}
#[cfg(target_os = "macos")]
pub(crate) fn run_generate_monolithic_cmd(
    model_dir: &std::path::Path,
    prompt_text: Option<String>,
    seed: u64,
    steps: usize,
    prompt_len: usize,
    max_new_tokens: usize,
    max_layers: Option<usize>,
    no_early_stop: bool,
    kernel_assert: bool,
    kernel_debug_deep: bool,
    write_golden: Option<String>,
    write_trace: Option<PathBuf>,
    raw_prompt: bool,
    verbose: bool,
) -> ExitCode {
    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!(
            "error: generate-monolithic requires a .dgq directory (-m /path/to/quantized-weights)"
        );
        return ExitCode::FAILURE;
    }

    crate::shaders::variant::set_runtime_kernel_debug(kernel_assert, kernel_debug_deep);

    let vocab = match crate::config::ModelConfig::load(model_dir) {
        Ok(c) => c.text_config.vocab_size,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };
    let prompt = match build_prompt_tokens(
        model_dir,
        prompt_text.as_deref(),
        prompt_len,
        vocab,
        raw_prompt,
        &[],
    ) {
        Ok(ids) => ids,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };
    let prompt_len = prompt.len();
    // Each denoise block places a CANVAS-wide (256) canvas at [kv_len..kv_len+CANVAS]
    // and writes its K/V there, so the cache must hold prompt + all generated tokens
    // PLUS one canvas block of headroom. Omitting the +CANVAS silently overflowed the
    // KV region for prompts >256 tokens (kv_len+256 > max_seq), corrupting attention
    // into word-salad. See run_chat_cmd's roomy CHAT_MAX_SEQ for the same reasoning.
    let max_seq = (prompt_len + max_new_tokens + metal::CANVAS).max(512);
    // A large prompt / --max-new-tokens sizes max_seq up; refuse configs whose
    // KV cache would swap or fail to allocate, before loading 19 GiB of weights.
    if let Err(msg) = check_ctx_budget(max_seq) {
        eprintln!("error: {msg}");
        return ExitCode::FAILURE;
    }

    let layers = match resolve_model_layers(model_dir, max_layers) {
        Ok(n) => n,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    let gen_cfg = generate::GenerateConfig {
        sampler: sample::sampler_for_steps(steps, no_early_stop),
        max_new_tokens,
        seed,
        max_layers: Some(layers),
        no_early_stop,
        deterministic: false,
        // `ask` is a single-turn chat: stop at eos (multi-block-until-done), with
        // max_new_tokens as the cap.
        full_message_stop: true,
        trace_prompt: None,
    };

    if verbose {
        let stop_note = if no_early_stop { ", no_early_stop" } else { "" };
        let assert_note = if kernel_assert { ", assert" } else { "" };
        let deep_note = if kernel_debug_deep {
            ", debug-deep"
        } else {
            ""
        };
        eprintln!(
            "running generate-monolithic (prompt_len={prompt_len}, steps={steps}, layers={layers}, max_new_tokens={max_new_tokens}, seed={seed}{stop_note}{assert_note}{deep_note})..."
        );
    }
    let started = std::time::Instant::now();

    let prompt_label = prompt_text.clone().unwrap_or_default();
    match generate::generate_monolithic_gpu(model_dir, &prompt, &gen_cfg, max_seq, &prompt_label) {
        Ok(out) => {
            if let Some(ref name) = write_golden {
                let prompt_str = prompt_text.clone().unwrap_or_default();
                if let Err(err) = write_generate_golden(
                    name,
                    &prompt_str,
                    &gen_cfg,
                    steps,
                    generate_golden::monolithic_weights_profile(),
                    &out,
                ) {
                    eprintln!("error: {err}");
                    return ExitCode::FAILURE;
                }
            }
            if let Some(ref path) = write_trace {
                if let Some(ref trace) = out.denoise_trace {
                    if let Err(err) = trace.write(path) {
                        eprintln!("error writing trace: {err}");
                        return ExitCode::FAILURE;
                    }
                    eprintln!("wrote denoise trace: {}", path.display());
                } else {
                    eprintln!("error: denoise trace unavailable on this build");
                    return ExitCode::FAILURE;
                }
            }
            if verbose {
                print_generate_output(
                    "generate-monolithic",
                    &out,
                    prompt_len,
                    started.elapsed(),
                    model_dir,
                );
            } else {
                print_generate_reply(&out, prompt_len, model_dir);
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
#[cfg(target_os = "macos")]
pub(crate) fn run_generate_monolithic_parity_cmd(
    model_dir: &std::path::Path,
    prompt_text: Option<String>,
    seed: u64,
    steps: usize,
    prompt_len: usize,
    max_new_tokens: usize,
    max_layers: Option<usize>,
    no_early_stop: bool,
    golden_name: Option<String>,
    write_golden: Option<String>,
    raw_prompt: bool,
) -> ExitCode {
    use generate_golden::GenerateGolden;

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: generate-monolithic-parity requires a .dgq directory");
        return ExitCode::FAILURE;
    }

    let vocab = match crate::config::ModelConfig::load(model_dir) {
        Ok(c) => c.text_config.vocab_size,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };
    let prompt = match build_prompt_tokens(
        model_dir,
        prompt_text.as_deref(),
        prompt_len,
        vocab,
        raw_prompt,
        &[],
    ) {
        Ok(ids) => ids,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };
    let prompt_len = prompt.len();
    // Each denoise block places a CANVAS-wide (256) canvas at [kv_len..kv_len+CANVAS]
    // and writes its K/V there, so the cache must hold prompt + all generated tokens
    // PLUS one canvas block of headroom. Omitting the +CANVAS silently overflowed the
    // KV region for prompts >256 tokens (kv_len+256 > max_seq), corrupting attention
    // into word-salad. See run_chat_cmd's roomy CHAT_MAX_SEQ for the same reasoning.
    let max_seq = (prompt_len + max_new_tokens + metal::CANVAS).max(512);
    let prompt_label = prompt_text
        .clone()
        .unwrap_or_else(|| format!("prompt_len={prompt_len}"));

    let gen_cfg = generate::GenerateConfig {
        sampler: sample::sampler_for_steps(steps, no_early_stop),
        max_new_tokens,
        seed,
        max_layers,
        no_early_stop,
        deterministic: true,
        full_message_stop: false,
        trace_prompt: None,
    };

    if let Some(n) = max_layers {
        eprintln!("generate-monolithic-parity: layers limited to {n}");
    }
    eprintln!(
        "running generate-monolithic parity (native Q4 default, prompt_len={prompt_len}, steps={steps}, seed={seed})..."
    );

    let out = match generate::generate_monolithic_gpu(
        model_dir,
        &prompt,
        &gen_cfg,
        max_seq,
        &prompt_label,
    ) {
        Ok(out) => out,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    let profile = generate_golden::monolithic_weights_profile();

    if let Some(ref name) = write_golden {
        if let Err(err) = write_generate_golden(name, &prompt_label, &gen_cfg, steps, profile, &out)
        {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    }

    let fixture_name = golden_name.or_else(|| {
        generate_golden::infer_monolithic_fixture_name(prompt_text.as_deref(), steps, max_layers)
    });

    if let Some(name) = fixture_name {
        let path = generate_golden::resolve_fixture(&name);
        let golden = match GenerateGolden::load(&path) {
            Ok(g) => g,
            Err(err) => {
                eprintln!("error: load golden {}: {err}", path.display());
                return ExitCode::FAILURE;
            }
        };
        if !golden.matches_config(&prompt_label, &gen_cfg, steps, profile) {
            eprintln!(
                "warning: golden {} config metadata differs from this run",
                path.display()
            );
        }
        match golden.compare(&out) {
            Ok(()) => {
                println!("generate-monolithic-parity ok ({name})");
                ExitCode::SUCCESS
            }
            Err(msg) => {
                eprintln!("generate-monolithic-parity failed: {msg}");
                ExitCode::FAILURE
            }
        }
    } else if write_golden.is_some() {
        println!("generate-monolithic-parity: golden written (no fixture to compare)");
        ExitCode::SUCCESS
    } else {
        eprintln!("error: no --golden fixture; use --write-golden NAME");
        ExitCode::FAILURE
    }
}
pub(crate) fn write_generate_golden(
    name: &str,
    prompt: &str,
    gen_cfg: &generate::GenerateConfig,
    steps: usize,
    weights_profile: &str,
    out: &generate::GenerateOutput,
) -> Result<(), crate::Error> {
    let golden = generate_golden::GenerateGolden::from_run(
        name,
        prompt,
        gen_cfg,
        steps,
        weights_profile,
        out,
    );
    let path = generate_golden::resolve_fixture(name);
    golden.write(&path)?;
    eprintln!(
        "wrote golden {} (profile={weights_profile}, {} tokens)",
        path.display(),
        out.token_ids.len()
    );
    Ok(())
}
