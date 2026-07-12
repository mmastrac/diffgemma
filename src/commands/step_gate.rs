//! Step-kernel verification/CI/parity gates + step-smoke.

use super::*;

#[cfg(target_os = "macos")]
pub(crate) fn run_step_verify_cmd(model_dir: &std::path::Path, layers: usize) -> ExitCode {
    use metal::run_step_verify;

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-verify requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let probe_layers = layers.max(1).min(30);
    match run_step_verify(Some(model_dir), probe_layers) {
        Ok(r) => {
            let ok = r.all_pass();
            for c in &r.checks {
                let mark = if c.pass { "ok" } else { "FAIL" };
                println!("  [{mark}] {}: {}", c.id, c.detail);
            }
            if ok {
                println!("step-verify ok ({probe_layers}L integration)");
                ExitCode::SUCCESS
            } else {
                eprintln!("step-verify failed");
                ExitCode::FAILURE
            }
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
#[cfg(target_os = "macos")]
pub(crate) fn run_step_ci_cmd(model_dir: &std::path::Path, layers: usize) -> ExitCode {
    use metal::validate_step_model;

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        println!(
            "step-ci skipped (no .dgq weights at {})",
            model_dir.display()
        );
        return ExitCode::SUCCESS;
    }

    let probe_layers = layers.max(1).min(30);
    eprintln!("step-ci: layers={probe_layers}");

    match validate_step_model(model_dir) {
        Ok(v) => metal::log_validated_step_model(&v),
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    }

    if run_step_verify_cmd(model_dir, probe_layers) != ExitCode::SUCCESS {
        eprintln!("step-ci failed at step-verify");
        return ExitCode::FAILURE;
    }

    if dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("step-ci: generate-monolithic-parity (hello, seed=42, steps=4)...");
        let parity = run_generate_monolithic_parity_cmd(
            model_dir,
            Some("hello".to_string()),
            42,
            4,
            1,
            256,
            Some(probe_layers),
            true,
            None,
            None,
            true,
        );
        if parity != ExitCode::SUCCESS {
            eprintln!("step-ci failed at generate-monolithic-parity");
            return parity;
        }
    }

    if run_chat_quality_gate(model_dir, probe_layers) != ExitCode::SUCCESS {
        eprintln!("step-ci failed at chat-quality gate");
        return ExitCode::FAILURE;
    }

    println!("step-ci ok (config + step-verify + generate-monolithic-parity + chat-quality)");
    ExitCode::SUCCESS
}
#[cfg(target_os = "macos")]
pub(crate) fn run_chat_quality_gate(model_dir: &std::path::Path, layers: usize) -> ExitCode {
    use generate_golden::{ChatQualityFixture, check_chat_quality};

    let path = generate_golden::resolve_fixture("chat_quality_hello_layers3");
    let gate = match ChatQualityFixture::load(&path) {
        Ok(g) => g,
        Err(err) => {
            eprintln!("error: load chat quality fixture {}: {err}", path.display());
            return ExitCode::FAILURE;
        }
    };

    let history: Vec<chat_template::ChatTurn> = vec![chat_template::ChatTurn::user(&gate.prompt)];
    let prompt = match build_chat_prompt_tokens(model_dir, &history, false) {
        Ok(ids) => ids,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };
    let prompt_len = prompt.len();
    let max_seq = (prompt_len + 256).max(512);
    let max_layers = gate.max_layers.unwrap_or(layers).min(layers);

    let gen_cfg = generate::GenerateConfig {
        sampler: sample::sampler_for_steps(gate.steps, false),
        max_new_tokens: 256,
        seed: gate.seed,
        max_layers: Some(max_layers),
        no_early_stop: false,
        deterministic: true,
        full_message_stop: false,
        trace_prompt: None,
    };

    eprintln!(
        "step-ci: chat-quality (templated {:?}, seed={}, steps={}, layers={max_layers})...",
        gate.prompt, gate.seed, gate.steps
    );

    let out = match generate::generate_monolithic_gpu(
        model_dir,
        &prompt,
        &gen_cfg,
        max_seq,
        &gate.prompt,
    ) {
        Ok(out) => out,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    match check_chat_quality(&out, prompt_len, &gate) {
        Ok(()) => {
            let (total, real) = generate_golden::count_new_tokens(&out, prompt_len);
            println!(
                "chat-quality ok ({}: {}/{} real new tokens, block_steps_eff={:?})",
                gate.name, real, total, out.block_steps_eff
            );
            ExitCode::SUCCESS
        }
        Err(msg) => {
            eprintln!("chat-quality failed: {msg}");
            ExitCode::FAILURE
        }
    }
}
#[cfg(target_os = "macos")]
pub(crate) fn run_step_parity_cmd(
    model_dir: &std::path::Path,
    layers: usize,
    kv_len: u32,
    seed: u64,
    max_seq: usize,
) -> ExitCode {
    use metal::{StepParityConfig, run_step_parity};

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-parity requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let cfg = StepParityConfig {
        layers: layers.max(1).min(30),
        kv_len,
        seed,
        max_seq: max_seq.max(64),
        ..StepParityConfig::default()
    };
    match run_step_parity(model_dir, &cfg) {
        Ok(r) => {
            if r.skipped {
                println!(
                    "step-parity skipped (kv_len={}): {}",
                    r.kv_len,
                    r.skip_reason.as_deref().unwrap_or("?")
                );
                return ExitCode::SUCCESS;
            }
            println!(
                "step-parity: layers={} kv_len={} seed={}",
                r.layers, r.kv_len, r.seed
            );
            println!(
                "  hidden max_abs={:.4} (tol {:.1})",
                r.hidden_max_abs, r.hidden_tol
            );
            println!(
                "  logits mean|Δ|={:.4} (tol {:.1})",
                r.logits_mean_diff, r.logits_tol
            );
            if r.pass {
                println!("step-parity ok");
                ExitCode::SUCCESS
            } else {
                eprintln!("step-parity failed");
                ExitCode::FAILURE
            }
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
#[cfg(target_os = "macos")]
pub(crate) fn run_step_smoke_cmd(
    model_dir: &std::path::Path,
    layers: usize,
    steps: usize,
    kv_len: u32,
    seed: u64,
    max_seq: usize,
    forward_only: bool,
    prompt: Option<&str>,
    raw_prompt: bool,
) -> ExitCode {
    use metal::run_step_smoke;

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-smoke requires a .dgq directory (-m /path/to/quantized-weights)");
        return ExitCode::FAILURE;
    }

    let mut cfg = step_kernel_config(layers, kv_len, seed, max_seq, forward_only);
    cfg.steps = steps.max(1);
    if let Err(err) = attach_step_prefill(&mut cfg, model_dir, kv_len, prompt, raw_prompt) {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    eprintln!(
        "step-smoke: model={} layers={layers} steps={steps} kv_len={kv_len} seed={seed} max_seq={max_seq}",
        model_dir.display()
    );
    match run_step_smoke(model_dir, cfg) {
        Ok(r) => {
            println!("step-smoke ok");
            println!("  step:          {}", r.step);
            println!("  stop_flag:     {}", r.stop_flag);
            println!("  mean_entropy:  {:.4}", r.mean_entropy);
            println!("  min_entropy:   {:.4}", r.min_entropy);
            println!("  low_ent(<0.1): {}", r.low_entropy_positions);
            println!("  logits_finite: {}", r.logits_finite);
            println!("  max_abs_logit: {:.4}", r.max_abs_logit);
            println!("  elapsed:       {:.2?}", r.elapsed);
            println!("  ids[0..8]:     {:?}", &r.ids[..8.min(r.ids.len())]);
            if r.step >= 1 {
                if !r.logits_finite {
                    eprintln!(
                        "warning: logits contain non-finite values (max_abs={:.4}); parity tuning still needed",
                        r.max_abs_logit
                    );
                }
                ExitCode::SUCCESS
            } else {
                eprintln!("error: smoke criteria not met (step={})", r.step);
                ExitCode::FAILURE
            }
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
