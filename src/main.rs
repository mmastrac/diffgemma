mod buffer;
mod chat_template;
mod config;
mod fast_slice;
mod generate;
mod generate_golden;
#[allow(dead_code)]
mod kernels;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;
mod model;
mod pack;
mod dgq;
mod sample;
mod safetensors;
mod tensor;
mod tokenizer;
mod weights;

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug)]
struct Cli {
    model_dir: PathBuf,
    command: Command,
    /// When true, `-p` text is BPE-encoded as-is (no chat template).
    raw_prompt: bool,
}

#[derive(Debug)]
enum Command {
    Summary,
    Config,
    Weights(String),
    Layer0,
    Decoder,
    Prefill,
    Generate {
        prompt: Option<String>,
        seed: u64,
        steps: usize,
        prompt_len: usize,
        max_new_tokens: usize,
        max_layers: Option<usize>,
        no_early_stop: bool,
    },
    GenerateGpu {
        prompt: Option<String>,
        seed: u64,
        steps: usize,
        prompt_len: usize,
        max_new_tokens: usize,
        max_layers: Option<usize>,
        no_early_stop: bool,
        write_golden: Option<String>,
    },
    GenerateMonolithic {
        prompt: Option<String>,
        seed: u64,
        steps: usize,
        prompt_len: usize,
        max_new_tokens: usize,
        max_layers: Option<usize>,
        no_early_stop: bool,
        engine_fallback: bool,
        write_golden: Option<String>,
    },
    GenerateMonolithicParity {
        prompt: Option<String>,
        seed: u64,
        steps: usize,
        prompt_len: usize,
        max_new_tokens: usize,
        max_layers: Option<usize>,
        no_early_stop: bool,
        golden: Option<String>,
        write_golden: Option<String>,
    },
    GenerateParity {
        prompt: Option<String>,
        seed: u64,
        steps: usize,
        prompt_len: usize,
        max_new_tokens: usize,
        max_layers: Option<usize>,
        no_early_stop: bool,
        golden: Option<String>,
        compare_cpu: bool,
        write_golden: Option<String>,
    },
    Tokenize(String),
    Gemm { size: usize },
    Attention,
    DecoderGpu {
        seq_len: usize,
        kv_len: usize,
        layers: usize,
    },
    BenchDecoder {
        seq_len: usize,
        kv_len: usize,
        layers: usize,
        iters: usize,
    },
    BenchStep {
        canvas: usize,
        layers: usize,
        iters: usize,
        prompt: Option<String>,
        seed: u64,
    },
    BenchPrefill {
        prompt_len: usize,
        layers: usize,
        iters: usize,
        seed: u64,
    },
    ProbeDevice,
    BenchGemm {
        shapes: String,
        oracle: Option<String>,
        iters: usize,
    },
    Quantize {
        output: PathBuf,
        profile: String,
    },
    ConvertModel {
        output_dir: PathBuf,
    },
    StepSmoke {
        layers: usize,
        steps: usize,
        kv_len: u32,
        seed: u64,
        max_seq: usize,
        forward_only: bool,
        prompt: Option<String>,
    },
    StepProbe {
        layers: usize,
        kv_len: u32,
        seed: u64,
        max_seq: usize,
        prompt: Option<String>,
    },
    StepKvCheck {
        kv_len: usize,
        layers: usize,
        seed: u64,
        max_seq: usize,
    },
    StepVerify {
        layers: usize,
    },
    StepCi {
        layers: usize,
    },
    StepParity {
        layers: usize,
        kv_len: u32,
        seed: u64,
        max_seq: usize,
    },
    BenchStepKernel {
        layers: usize,
        kv_len: u32,
        seed: u64,
        max_seq: usize,
        iters: usize,
        forward_only: bool,
    },
    Chat {
        seed: u64,
        steps: usize,
        max_new_tokens: usize,
        max_layers: Option<usize>,
        no_early_stop: bool,
        initial_prompt: Option<String>,
    },
}

fn main() -> ExitCode {
    let cli = parse_cli();
    match cli.command {
        Command::ConvertModel { output_dir } => run_convert_model(&cli.model_dir, &output_dir),
        Command::StepSmoke {
            layers,
            steps,
            kv_len,
            seed,
            max_seq,
            forward_only,
            prompt,
        } => run_step_smoke_cmd(
            &cli.model_dir,
            layers,
            steps,
            kv_len,
            seed,
            max_seq,
            forward_only,
            prompt.as_deref(),
            cli.raw_prompt,
        ),
        Command::StepProbe {
            layers,
            kv_len,
            seed,
            max_seq,
            prompt,
        } => run_step_probe_cmd(
            &cli.model_dir,
            layers,
            kv_len,
            seed,
            max_seq,
            prompt.as_deref(),
            cli.raw_prompt,
        ),
        Command::StepKvCheck {
            kv_len,
            layers,
            seed,
            max_seq,
        } => run_step_kv_check_cmd(&cli.model_dir, kv_len, layers, seed, max_seq),
        Command::StepVerify { layers } => run_step_verify_cmd(&cli.model_dir, layers),
        Command::StepCi { layers } => run_step_ci_cmd(&cli.model_dir, layers),
        Command::StepParity {
            layers,
            kv_len,
            seed,
            max_seq,
        } => run_step_parity_cmd(&cli.model_dir, layers, kv_len, seed, max_seq),
        Command::BenchStepKernel {
            layers,
            kv_len,
            seed,
            max_seq,
            iters,
            forward_only,
        } => run_bench_step_kernel_cmd(
            &cli.model_dir,
            layers,
            kv_len,
            seed,
            max_seq,
            iters,
            forward_only,
        ),
        Command::GenerateMonolithic {
            prompt,
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers,
            no_early_stop,
            engine_fallback,
            write_golden,
        } => run_generate_monolithic_cmd(
            &cli.model_dir,
            prompt,
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers,
            no_early_stop,
            engine_fallback,
            write_golden,
            cli.raw_prompt,
        ),
        Command::GenerateMonolithicParity {
            prompt,
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers,
            no_early_stop,
            golden,
            write_golden,
        } => run_generate_monolithic_parity_cmd(
            &cli.model_dir,
            prompt,
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers,
            no_early_stop,
            golden,
            write_golden,
            cli.raw_prompt,
        ),
        Command::Chat {
            seed,
            steps,
            max_new_tokens,
            max_layers,
            no_early_stop,
            initial_prompt,
        } => run_chat_cmd(
            &cli.model_dir,
            initial_prompt,
            seed,
            steps,
            max_new_tokens,
            max_layers,
            no_early_stop,
            cli.raw_prompt,
        ),
        Command::Quantize { output, profile } => run_quantize(&cli.model_dir, &output, &profile),
        Command::Tokenize(text) => run_tokenize(&cli.model_dir, &text, cli.raw_prompt),
        Command::Gemm { size } => run_gemm(size),
        Command::ProbeDevice => run_probe_device(),
        Command::BenchGemm { shapes, oracle, iters } => run_bench_gemm(&shapes, oracle.as_deref(), iters),
        command => {
            eprintln!("loading from {}", cli.model_dir.display());
            match model::Model::open(&cli.model_dir) {
                Ok(m) => run_command(&m, &cli.model_dir, command, cli.raw_prompt),
                Err(err) => {
                    eprintln!("error: {err}");
                    ExitCode::FAILURE
                }
            }
        }
    }
}

fn run_command(
    m: &model::Model,
    model_dir: &std::path::Path,
    command: Command,
    raw_prompt: bool,
) -> ExitCode {
    match command {
        Command::Summary => {
            print_summary(&m.weights);
            ExitCode::SUCCESS
        }
        Command::Config => {
            m.config.print_summary();
            ExitCode::SUCCESS
        }
        Command::Weights(name) => match m.weights.tensor(&name) {
            Ok(t) => {
                t.print_info();
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("error: {err}");
                ExitCode::FAILURE
            }
        },
        Command::Layer0 => run_layer0_forward(m),
        Command::Attention => run_attention_parity(m),
        Command::Decoder => run_decoder_forward(m),
        Command::DecoderGpu {
            seq_len,
            kv_len,
            layers,
        } => run_decoder_gpu_parity(m, seq_len, kv_len, layers),
        Command::BenchDecoder {
            seq_len,
            kv_len,
            layers,
            iters,
        } => run_bench_decoder(m, seq_len, kv_len, layers, iters),
        Command::BenchStep {
            canvas,
            layers,
            iters,
            prompt,
            seed,
        } => run_bench_step(m, model_dir, canvas, layers, iters, prompt, seed, raw_prompt),
        Command::BenchPrefill {
            prompt_len,
            layers,
            iters,
            seed,
        } => run_bench_prefill(m, prompt_len, layers, iters, seed),
        Command::Prefill => run_prefill(m),
        Command::Generate {
            prompt,
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers,
            no_early_stop,
        } => run_generate(
            m,
            model_dir,
            prompt,
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers,
            no_early_stop,
            false,
            None,
            raw_prompt,
        ),
        Command::GenerateGpu {
            prompt,
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers,
            no_early_stop,
            write_golden,
        } => run_generate(
            m,
            model_dir,
            prompt,
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers,
            no_early_stop,
            true,
            write_golden,
            raw_prompt,
        ),
        Command::GenerateParity {
            prompt,
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers,
            no_early_stop,
            golden,
            compare_cpu,
            write_golden,
        } => run_generate_parity(
            m,
            model_dir,
            prompt,
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers,
            no_early_stop,
            golden,
            compare_cpu,
            write_golden,
            raw_prompt,
        ),
        Command::Chat { .. } => ExitCode::FAILURE,
        Command::Tokenize(_) => ExitCode::FAILURE,
        Command::Gemm { .. } => ExitCode::FAILURE,
        Command::ProbeDevice { .. } => ExitCode::FAILURE,
        Command::ConvertModel { .. } => ExitCode::FAILURE,
        Command::Quantize { .. } => ExitCode::FAILURE,
        Command::BenchGemm { .. } => ExitCode::FAILURE,
        Command::StepSmoke { .. } => ExitCode::FAILURE,
        Command::StepProbe { .. } => ExitCode::FAILURE,
        Command::StepKvCheck { .. } => ExitCode::FAILURE,
        Command::StepVerify { .. } => ExitCode::FAILURE,
        Command::StepCi { .. } => ExitCode::FAILURE,
        Command::StepParity { .. } => ExitCode::FAILURE,
        Command::BenchStepKernel { .. } => ExitCode::FAILURE,
        Command::GenerateMonolithic { .. } => ExitCode::FAILURE,
        Command::GenerateMonolithicParity { .. } => ExitCode::FAILURE,
    }
}

fn step_kernel_config(
    layers: usize,
    kv_len: u32,
    seed: u64,
    max_seq: usize,
    forward_only: bool,
) -> metal::StepSmokeConfig {
    metal::StepSmokeConfig {
        layers,
        steps: 1,
        kv_len,
        seed,
        max_seq,
        finish: if forward_only {
            metal::StepFinishMode::ForwardOnly
        } else {
            metal::StepFinishMode::Full
        },
        use_mps_q4: None,
        prefill_token_ids: None,
    }
}

fn attach_step_prefill(
    cfg: &mut metal::StepSmokeConfig,
    model_dir: &std::path::Path,
    kv_len: u32,
    prompt: Option<&str>,
    raw_prompt: bool,
) -> Result<(), safetensors::Error> {
    if kv_len == 0 && prompt.is_none() {
        return Ok(());
    }
    let vocab = crate::config::ModelConfig::load(model_dir)?.text_config.vocab_size;
    let prompt_len = if kv_len > 0 {
        kv_len as usize
    } else {
        64
    };
    let ids = build_prompt_tokens(model_dir, prompt, prompt_len, vocab, raw_prompt, &[])?;
    eprintln!("step-kernel: prefill {} prompt tokens", ids.len());
    cfg.prefill_token_ids = Some(ids);
    Ok(())
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_step_probe_cmd(
    model_dir: &std::path::Path,
    layers: usize,
    kv_len: u32,
    seed: u64,
    max_seq: usize,
    prompt: Option<&str>,
    raw_prompt: bool,
) -> ExitCode {
    use metal::{run_step_probe, StepFinishMode, StepSmokeConfig};

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-probe requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let mut cfg = StepSmokeConfig {
        layers,
        steps: 1,
        kv_len,
        seed,
        max_seq,
        finish: StepFinishMode::ForwardOnly,
        use_mps_q4: None,
        prefill_token_ids: None,
    };
    if let Err(err) = attach_step_prefill(&mut cfg, model_dir, kv_len, prompt, raw_prompt) {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    match run_step_probe(model_dir, cfg) {
        Ok(r) => {
            println!("step-probe ok ({:.2?})", r.elapsed);
            for cp in &r.checkpoints {
                println!(
                    "  {:>16}: finite={} max_abs={:.4}",
                    cp.label, cp.finite, cp.max_abs
                );
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_step_probe_cmd(
    _model_dir: &std::path::Path,
    _layers: usize,
    _kv_len: u32,
    _seed: u64,
    _max_seq: usize,
    _prompt: Option<&str>,
    _raw_prompt: bool,
) -> ExitCode {
    eprintln!("error: step-probe requires --features metal on macOS");
    ExitCode::FAILURE
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_step_kv_check_cmd(
    model_dir: &std::path::Path,
    kv_len: usize,
    layers: usize,
    seed: u64,
    max_seq: usize,
) -> ExitCode {
    use metal::run_step_kv_audit;

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-kv-check requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    if kv_len == 0 {
        eprintln!("error: step-kv-check requires --kv-len > 0");
        return ExitCode::FAILURE;
    }
    match run_step_kv_audit(model_dir, kv_len, layers, seed, max_seq) {
        Ok(r) => {
            println!("step-kv-check ok");
            println!("  kv_len:              {}", r.kv_len);
            println!("  prefix_max_abs_l0:   {:.6}", r.prefix_max_abs_l0);
            println!("  hidden_diff_vs_kv0:  {:.6}", r.hidden_max_abs_vs_zero);
            println!("  logits_diff_vs_kv0:  {:.6}", r.logits_max_abs_vs_zero);
            if let Some(n) = r.extend_kv_len {
                println!("  extend_kv_len:       {n}");
            }
            if let Some(d) = r.extend_hidden_diff {
                println!("  extend_hidden_diff:  {d:.6}");
            }
            println!("  pass:                {}", r.pass);
            if r.pass {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_step_kv_check_cmd(
    _model_dir: &std::path::Path,
    _kv_len: usize,
    _layers: usize,
    _seed: u64,
    _max_seq: usize,
) -> ExitCode {
    eprintln!("error: step-kv-check requires --features metal on macOS");
    ExitCode::FAILURE
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_step_verify_cmd(model_dir: &std::path::Path, layers: usize) -> ExitCode {
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

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_step_verify_cmd(_model_dir: &std::path::Path, _layers: usize) -> ExitCode {
    eprintln!("error: step-verify requires --features metal on macOS");
    ExitCode::FAILURE
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_step_ci_cmd(model_dir: &std::path::Path, layers: usize) -> ExitCode {
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

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_chat_quality_gate(model_dir: &std::path::Path, layers: usize) -> ExitCode {
    use generate_golden::{check_chat_quality, ChatQualityFixture};

    let path = generate_golden::resolve_fixture("chat_quality_hello_layers3");
    let gate = match ChatQualityFixture::load(&path) {
        Ok(g) => g,
        Err(err) => {
            eprintln!("error: load chat quality fixture {}: {err}", path.display());
            return ExitCode::FAILURE;
        }
    };

    let history: Vec<chat_template::ChatTurn> =
        vec![chat_template::ChatTurn::user(&gate.prompt)];
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
    };

    eprintln!(
        "step-ci: chat-quality (templated {:?}, seed={}, steps={}, layers={max_layers})...",
        gate.prompt, gate.seed, gate.steps
    );

    let out = match generate::generate_monolithic_gpu(model_dir, &prompt, &gen_cfg, max_seq) {
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
                gate.name,
                real,
                total,
                out.block_steps_eff
            );
            ExitCode::SUCCESS
        }
        Err(msg) => {
            eprintln!("chat-quality failed: {msg}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_chat_quality_gate(_model_dir: &std::path::Path, _layers: usize) -> ExitCode {
    ExitCode::SUCCESS
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_step_ci_cmd(_model_dir: &std::path::Path, _layers: usize) -> ExitCode {
    eprintln!("error: step-ci requires --features metal on macOS");
    ExitCode::FAILURE
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_step_parity_cmd(
    model_dir: &std::path::Path,
    layers: usize,
    kv_len: u32,
    seed: u64,
    max_seq: usize,
) -> ExitCode {
    use metal::{run_step_parity, StepParityConfig};

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
                "  logits max_abs={:.4} (tol {:.1})",
                r.logits_max_abs, r.logits_tol
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

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_step_parity_cmd(
    _model_dir: &std::path::Path,
    _layers: usize,
    _kv_len: u32,
    _seed: u64,
    _max_seq: usize,
) -> ExitCode {
    eprintln!("error: step-parity requires --features metal on macOS");
    ExitCode::FAILURE
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_bench_step_kernel_cmd(
    model_dir: &std::path::Path,
    layers: usize,
    kv_len: u32,
    seed: u64,
    max_seq: usize,
    iters: usize,
    forward_only: bool,
) -> ExitCode {
    use metal::bench_step_kernel;

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: bench-step-kernel requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let cfg = step_kernel_config(layers, kv_len, seed, max_seq, forward_only);
    eprintln!(
        "bench-step-kernel: layers={layers} kv_len={kv_len} iters={iters} forward_only={forward_only}"
    );
    match bench_step_kernel(model_dir, cfg, iters) {
        Ok(r) => {
            println!("bench-step-kernel ok");
            println!("  compile:  {:.2?}", r.compile);
            println!("  warmup:   {:.2?}", r.warmup);
            println!("  per_step: {:.2?}", r.per_step);
            println!("  iters:    {}", r.iters);
            println!("  mode:     {:?}", r.finish);
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_bench_step_kernel_cmd(
    _model_dir: &std::path::Path,
    _layers: usize,
    _kv_len: u32,
    _seed: u64,
    _max_seq: usize,
    _iters: usize,
    _forward_only: bool,
) -> ExitCode {
    eprintln!("error: bench-step-kernel requires --features metal on macOS");
    ExitCode::FAILURE
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_step_smoke_cmd(
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
    use metal::{run_step_smoke, StepSmokeConfig};

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
            println!(
                "  ids[0..8]:     {:?}",
                &r.ids[..8.min(r.ids.len())]
            );
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

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_step_smoke_cmd(
    _model_dir: &std::path::Path,
    _layers: usize,
    _steps: usize,
    _kv_len: u32,
    _seed: u64,
    _max_seq: usize,
    _forward_only: bool,
    _prompt: Option<&str>,
    _raw_prompt: bool,
) -> ExitCode {
    eprintln!("error: step-smoke requires --features metal on macOS");
    ExitCode::FAILURE
}

fn run_quantize(source_dir: &std::path::Path, output: &std::path::Path, profile: &str) -> ExitCode {
    use dgq::{quantize_model, QuantizeOptions};
    use dgq::layout::QuantProfile;

    let profile_name = profile;
    let profile = match profile {
        "q4" => QuantProfile::Q4,
        "q5" => QuantProfile::Q5,
        other => {
            eprintln!("error: unknown profile {other} (use q4 or q5)");
            return ExitCode::FAILURE;
        }
    };

    let out_dir = if output.extension().is_some_and(|e| e == "dgq") {
        output.with_extension("")
    } else {
        output.to_path_buf()
    };

    eprintln!(
        "quantize: {} -> {} (profile={profile_name})",
        source_dir.display(),
        out_dir.display(),
    );
    let started = std::time::Instant::now();
    match quantize_model(QuantizeOptions {
        source_dir: source_dir.to_path_buf(),
        output_prefix: out_dir.clone(),
        profile,
    }) {
        Ok(summary) => {
            let gib = summary.blob_bytes as f64 / (1024.0_f64.powi(3));
            println!("quantize ok");
            println!("  output dir:    {}", out_dir.display());
            println!("  tensors:       {}", summary.tensor_count);
            println!("  blob size:     {gib:.2} GiB");
            println!("  q4 tensors:    {}", summary.q4_tensors);
            println!("  q8 tensors:    {}", summary.q8_tensors);
            println!("  raw tensors:   {}", summary.raw_tensors);
            println!("  elapsed:       {:.2?}", started.elapsed());
            println!("  manifest:      {}/{}", out_dir.display(), dgq::layout::MANIFEST_FILE);
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_probe_device() -> ExitCode {
    use metal::{print_probe_result, probe_device};
    match probe_device() {
        Ok(r) => {
            print_probe_result(&r);
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_probe_device() -> ExitCode {
    eprintln!("error: probe-device requires --features metal on macOS");
    ExitCode::FAILURE
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_bench_gemm(shapes: &str, oracle: Option<&str>, iters: usize) -> ExitCode {
    use metal::{bench_custom_kernel, bench_mps_oracle, parse_shapes, print_bench_rows};
    let parsed = match parse_shapes(shapes) {
        Ok(s) => s,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };
    let mut rows = match bench_custom_kernel(&parsed, iters) {
        Ok(r) => r,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };
    if oracle == Some("mps") {
        match bench_mps_oracle(&parsed, iters) {
            Ok(mut mps) => rows.append(&mut mps),
            Err(err) => {
                eprintln!("warning: {err}");
            }
        }
    }
    print_bench_rows(&rows);
    ExitCode::SUCCESS
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_bench_gemm(_shapes: &str, _oracle: Option<&str>, _iters: usize) -> ExitCode {
    eprintln!("error: bench-gemm requires --features metal on macOS");
    ExitCode::FAILURE
}

fn run_convert_model(source_dir: &std::path::Path, output_dir: &std::path::Path) -> ExitCode {
    use pack::{convert_model, ConvertOptions};
    eprintln!(
        "convert-model: {} -> {}",
        source_dir.display(),
        output_dir.display()
    );
    match convert_model(ConvertOptions {
        source_dir: source_dir.to_path_buf(),
        output_dir: output_dir.to_path_buf(),
    }) {
        Ok(summary) => {
            println!("convert-model ok");
            println!("  tensors:           {}", summary.tensor_count);
            println!("  blob size:         {:.2} GiB", summary.blob_bytes as f64 / (1024.0_f64.powi(3)));
            println!("  gemm transposed:   {}", summary.transposed_gemm);
            println!("  expert transposed: {}", summary.transposed_experts);
            println!("  raw copied:        {}", summary.raw_copied);
            println!("  manifest:          {}/{}", output_dir.display(), pack::layout::MANIFEST_FILE);
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

/// Production generate/chat default is 48 (model card); parity/bench default is 2.
fn resolve_steps(override_steps: Option<usize>, parity_default: bool) -> usize {
    override_steps.unwrap_or(if parity_default { 2 } else { 48 })
}

fn parse_cli() -> Cli {
    let mut args = env::args().skip(1);
    let mut model_dir = PathBuf::from("model/transformer");
    let mut positional = Vec::new();
    let mut seed = 42u64;
    let mut steps_override: Option<usize> = None;
    let mut prompt_len = 8usize;
    let mut max_new_tokens = 256usize;
    let mut gemm_size = 512usize;
    let mut prompt: Option<String> = None;
    let mut bench_seq = 16usize;
    let mut bench_kv = 8usize;
    let mut bench_layers = 1usize;
    let mut bench_iters = 3usize;
    let mut bench_canvas = 256usize;
    let mut parity_seq: Option<usize> = None;
    let mut parity_kv: Option<usize> = None;
    let mut parity_layers: Option<usize> = None;
    let mut golden_name: Option<String> = None;
    let mut compare_cpu = false;
    let mut write_golden: Option<String> = None;
    let mut no_early_stop = false;
    let mut output_dir: Option<PathBuf> = None;
    let mut quant_profile = String::from("q4");
    let mut bench_gemm_shapes = String::from("256x2816x2816,33x2816x1408");
    let mut bench_gemm_oracle: Option<String> = None;
    let mut bench_prefill_len = 1usize;
    let mut step_kv_len = 0u32;
    let mut step_max_seq = 512usize;
    let mut step_forward_only = false;
    let mut use_monolithic = false;
    let mut raw_prompt = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-p" | "--prompt" => {
                if let Some(text) = args.next() {
                    prompt = Some(text);
                }
            }
            "-m" | "--model" => {
                if let Some(path) = args.next() {
                    model_dir = PathBuf::from(path);
                }
            }
            "-o" | "--output" => {
                if let Some(path) = args.next() {
                    output_dir = Some(PathBuf::from(path));
                }
            }
            "--seed" => {
                if let Some(v) = args.next() {
                    seed = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --seed");
                        std::process::exit(2);
                    });
                }
            }
            "--steps" => {
                if let Some(v) = args.next() {
                    steps_override = Some(v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --steps");
                        std::process::exit(2);
                    }));
                }
            }
            "--prompt-len" => {
                if let Some(v) = args.next() {
                    prompt_len = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --prompt-len");
                        std::process::exit(2);
                    });
                }
            }
            "--prefill-len" => {
                if let Some(v) = args.next() {
                    bench_prefill_len = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --prefill-len");
                        std::process::exit(2);
                    });
                }
            }
            "--max-new-tokens" => {
                if let Some(v) = args.next() {
                    max_new_tokens = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --max-new-tokens");
                        std::process::exit(2);
                    });
                }
            }
            "--compare-cpu" => compare_cpu = true,
            "--no-early-stop" => no_early_stop = true,
            "--write-golden" => {
                if let Some(v) = args.next() {
                    write_golden = Some(v);
                }
            }
            "--golden" => {
                if let Some(v) = args.next() {
                    golden_name = Some(v);
                }
            }
            "--size" => {
                if let Some(v) = args.next() {
                    gemm_size = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --size");
                        std::process::exit(2);
                    });
                }
            }
            "--seq" => {
                if let Some(v) = args.next() {
                    let n = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --seq");
                        std::process::exit(2);
                    });
                    bench_seq = n;
                    parity_seq = Some(n);
                }
            }
            "--kv" => {
                if let Some(v) = args.next() {
                    let n = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --kv");
                        std::process::exit(2);
                    });
                    bench_kv = n;
                    parity_kv = Some(n);
                }
            }
            "--layers" => {
                if let Some(v) = args.next() {
                    let n = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --layers");
                        std::process::exit(2);
                    });
                    bench_layers = n;
                    parity_layers = Some(n);
                }
            }
            "--iters" => {
                if let Some(v) = args.next() {
                    bench_iters = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --iters");
                        std::process::exit(2);
                    });
                }
            }
            "--canvas" => {
                if let Some(v) = args.next() {
                    bench_canvas = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --canvas");
                        std::process::exit(2);
                    });
                }
            }
            "--profile" => {
                if let Some(v) = args.next() {
                    quant_profile = v;
                }
            }
            "--shapes" => {
                if let Some(v) = args.next() {
                    bench_gemm_shapes = v;
                }
            }
            "--oracle" => {
                if let Some(v) = args.next() {
                    bench_gemm_oracle = Some(v);
                }
            }
            "--forward-only" => step_forward_only = true,
            "--monolithic" => use_monolithic = true,
            "--raw" => raw_prompt = true,
            "--max-seq" => {
                if let Some(v) = args.next() {
                    step_max_seq = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --max-seq");
                        std::process::exit(2);
                    });
                }
            }
            "--kv-len" => {
                if let Some(v) = args.next() {
                    step_kv_len = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --kv-len");
                        std::process::exit(2);
                    });
                }
            }
            _ => positional.push(arg),
        }
    }

    let use_monolithic = use_monolithic || monolithic_from_env();
    let steps_production = resolve_steps(steps_override, false);
    let steps_parity = resolve_steps(steps_override, true);

    let command = match positional.first().map(String::as_str) {
        None => default_generate_command(
            &model_dir,
            prompt,
            seed,
            steps_production,
            prompt_len,
            max_new_tokens,
            parity_layers,
            no_early_stop,
            use_monolithic,
        ),
        Some("summary") => Command::Summary,
        Some("config") => Command::Config,
        Some("weights") => {
            let name = positional.get(1).cloned().unwrap_or_else(|| {
                eprintln!("usage: diffgemma-mps weights <tensor_name>");
                std::process::exit(2);
            });
            Command::Weights(name)
        }
        Some("layer0") => Command::Layer0,
        Some("decoder") => Command::Decoder,
        Some("prefill") => Command::Prefill,
        Some("generate") if use_monolithic => Command::GenerateMonolithic {
            prompt: prompt.clone(),
            seed,
            steps: steps_production,
            prompt_len,
            max_new_tokens,
            max_layers: parity_layers,
            no_early_stop,
            engine_fallback: true,
            write_golden: None,
        },
        Some("generate") => Command::Generate {
            prompt: prompt.clone(),
            seed,
            steps: steps_production,
            prompt_len,
            max_new_tokens,
            max_layers: parity_layers,
            no_early_stop,
        },
        Some("generate-gpu") if use_monolithic => Command::GenerateMonolithic {
            prompt: prompt.clone(),
            seed,
            steps: steps_production,
            prompt_len,
            max_new_tokens,
            max_layers: parity_layers,
            no_early_stop,
            engine_fallback: true,
            write_golden: None,
        },
        Some("generate-gpu") => Command::GenerateGpu {
            prompt: prompt.clone(),
            seed,
            steps: steps_production,
            prompt_len,
            max_new_tokens,
            max_layers: parity_layers,
            no_early_stop,
            write_golden,
        },
        Some("generate-monolithic") => Command::GenerateMonolithic {
            prompt: prompt.clone(),
            seed,
            steps: steps_production,
            prompt_len,
            max_new_tokens,
            max_layers: parity_layers,
            no_early_stop,
            engine_fallback: false,
            write_golden,
        },
        Some("generate-monolithic-parity") => Command::GenerateMonolithicParity {
            prompt: prompt.clone(),
            seed,
            steps: steps_parity,
            prompt_len,
            max_new_tokens,
            max_layers: parity_layers,
            no_early_stop,
            golden: golden_name,
            write_golden,
        },
        Some("chat") => Command::Chat {
            seed,
            steps: steps_production,
            max_new_tokens,
            max_layers: parity_layers,
            no_early_stop,
            initial_prompt: prompt.clone(),
        },
        Some("generate-parity") => Command::GenerateParity {
            prompt: prompt.clone(),
            seed,
            steps: steps_parity,
            prompt_len,
            max_new_tokens,
            max_layers: parity_layers,
            no_early_stop,
            golden: golden_name,
            compare_cpu,
            write_golden,
        },
        Some("tokenize") => {
            let text = positional.get(1).cloned().unwrap_or_else(|| {
                eprintln!("usage: diffgemma-mps tokenize <text>");
                std::process::exit(2);
            });
            Command::Tokenize(text)
        }
        Some("gemm") => Command::Gemm { size: gemm_size },
        Some("attention") => Command::Attention,
        Some("decoder-gpu") => Command::DecoderGpu {
            seq_len: parity_seq.unwrap_or(256),
            kv_len: parity_kv.unwrap_or(128),
            layers: parity_layers.unwrap_or(0),
        },
        Some("bench-decoder") => Command::BenchDecoder {
            seq_len: bench_seq,
            kv_len: bench_kv,
            layers: bench_layers,
            iters: bench_iters,
        },
        Some("bench-step") => Command::BenchStep {
            canvas: bench_canvas,
            layers: bench_layers.max(1),
            iters: bench_iters,
            prompt: prompt.clone(),
            seed,
        },
        Some("bench-prefill") => Command::BenchPrefill {
            prompt_len: bench_prefill_len.max(1),
            layers: bench_layers.max(1),
            iters: bench_iters.max(1),
            seed,
        },
        Some("probe-device") => Command::ProbeDevice,
        Some("bench-gemm") => Command::BenchGemm {
            shapes: bench_gemm_shapes,
            oracle: bench_gemm_oracle,
            iters: bench_iters.max(1),
        },
        Some("quantize") => {
            let out = output_dir.unwrap_or_else(|| {
                eprintln!("usage: diffgemma-mps quantize -o OUTPUT_DIR [-m SOURCE] [--profile q4|q5]");
                std::process::exit(2);
            });
            Command::Quantize {
                output: out,
                profile: quant_profile,
            }
        }
        Some("convert-model") => {
            let out = output_dir.unwrap_or_else(|| {
                eprintln!("usage: diffgemma-mps convert-model -o OUTPUT_DIR [-m SOURCE_MODEL]");
                std::process::exit(2);
            });
            Command::ConvertModel { output_dir: out }
        }
        Some("step-smoke") => Command::StepSmoke {
            layers: bench_layers.max(1).min(30),
            steps: steps_parity.max(1),
            kv_len: step_kv_len,
            seed,
            max_seq: step_max_seq.max(64),
            forward_only: step_forward_only,
            prompt: prompt.clone(),
        },
        Some("step-probe") => Command::StepProbe {
            layers: bench_layers.max(1).min(30),
            kv_len: step_kv_len,
            seed,
            max_seq: step_max_seq.max(64),
            prompt: prompt.clone(),
        },
        Some("step-kv-check") => Command::StepKvCheck {
            kv_len: step_kv_len.max(1) as usize,
            layers: bench_layers.max(1).min(30),
            seed,
            max_seq: step_max_seq.max(64),
        },
        Some("step-verify") => Command::StepVerify {
            layers: bench_layers.max(1).min(30),
        },
        Some("step-ci") => Command::StepCi {
            layers: bench_layers.max(1).min(30),
        },
        Some("step-parity") => Command::StepParity {
            layers: bench_layers.max(1).min(30),
            kv_len: step_kv_len,
            seed,
            max_seq: step_max_seq.max(64),
        },
        Some("bench-step-kernel") => Command::BenchStepKernel {
            layers: bench_layers.max(1).min(30),
            kv_len: step_kv_len,
            seed,
            max_seq: step_max_seq.max(64),
            iters: bench_iters.max(1),
            forward_only: step_forward_only,
        },
        Some(cmd) => {
            eprintln!("unknown command: {cmd}");
            eprintln!(
                "usage: diffgemma-mps [-p PROMPT] [--raw] [summary|config|weights <name>|quantize|convert-model|step-smoke|step-probe|step-kv-check|step-verify|step-ci|step-parity|bench-step-kernel|bench-step|bench-prefill|probe-device|layer0|decoder|decoder-gpu|prefill|generate|generate-gpu|generate-monolithic|generate-monolithic-parity|generate-parity|chat|tokenize <text>|gemm|attention]"
            );
            eprintln!("  default (no command): generate-gpu with --features metal (or generate-monolithic with DGQ_MONOLITHIC=1 / --monolithic on .dgq)");
            eprintln!("  prompts: chat template applied by default; use --raw for bare BPE (-p \"Hello\" -> [9259])");
            eprintln!("  chat: interactive REPL (monolithic .dgq); optional -p for first user turn");
            eprintln!("  generate-parity: GPU vs checked-in golden (use --compare-cpu for slow CPU path; use --raw for legacy goldens)");
            eprintln!("  options: ... --golden NAME --write-golden NAME --compare-cpu --no-early-stop");
            eprintln!("  gemm options: --size N (default 512, requires --features metal)");
            eprintln!("  attention: layer 0 GQA parity (requires --features metal)");
            eprintln!("  decoder-gpu: full decoder CPU vs GPU parity at seq=256 (requires --features metal)");
            std::process::exit(2);
        }
    };

    Cli {
        model_dir,
        command,
        raw_prompt,
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn monolithic_from_env() -> bool {
    match std::env::var("DGQ_MONOLITHIC") {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
        Err(_) => false,
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn monolithic_from_env() -> bool {
    false
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn default_generate_command(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    seed: u64,
    steps: usize,
    prompt_len: usize,
    max_new_tokens: usize,
    max_layers: Option<usize>,
    no_early_stop: bool,
    use_monolithic: bool,
) -> Command {
    if use_monolithic && dgq::store::looks_like_dgq_dir(model_dir) {
        return Command::GenerateMonolithic {
            prompt,
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers,
            no_early_stop,
            engine_fallback: true,
            write_golden: None,
        };
    }
    Command::GenerateGpu {
        prompt,
        seed,
        steps,
        prompt_len,
        max_new_tokens,
        max_layers,
        no_early_stop,
        write_golden: None,
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn default_generate_command(
    _model_dir: &std::path::Path,
    prompt: Option<String>,
    seed: u64,
    steps: usize,
    prompt_len: usize,
    max_new_tokens: usize,
    max_layers: Option<usize>,
    no_early_stop: bool,
    _use_monolithic: bool,
) -> Command {
    Command::Generate {
        prompt,
        seed,
        steps,
        prompt_len,
        max_new_tokens,
        max_layers,
        no_early_stop,
    }
}

fn print_summary(store: &weights::WeightStore) {
    let s = store.summarize();

    println!("DiffusionGemma weight summary");
    println!("  model dir:          {}", store.model_dir().display());
    if store.is_quantized() {
        println!("  format:             .dgq (quantized, mmap)");
    } else if store.is_packed() {
        println!("  format:             iris.pack (pre-transposed)");
    }
    println!("  shards:             {}", s.shard_count);
    println!("  tensors (index):    {}", s.tensor_count_index);
    println!("  tensors (headers):  {}", s.tensor_count_headers);
    println!("  on-disk total:      {:.2} GiB", gib(s.total_file_bytes));
    println!("  tensor payload:     {:.2} GiB", gib(s.total_data_bytes));
    println!("  total elements:     {}", s.total_elements);

    if let Some(meta) = store.safetensor_metadata() {
        if let Some(params) = meta.get("total_parameters") {
            println!("  index metadata total_parameters: {params}");
        }
        if let Some(size) = meta.get("total_size") {
            println!("  index metadata total_size:       {size}");
        }
    }

    println!("\n  dtypes:");
    for (dtype, count) in &s.dtypes {
        println!("    {dtype}: {count}");
    }

    println!("\n  top-level prefixes:");
    for (prefix, count) in s.top_prefixes.iter().take(8) {
        println!("    {prefix}: {count}");
    }

    println!("\n  per-shard:");
    match store {
        weights::WeightStore::Safetensors(s) => {
            for shard in &s.shards {
                let payload: u64 = shard.tensors.iter().map(|t| t.data_size as u64).sum();
                println!(
                    "    {}  {:>4} tensors  {:.2} GiB payload  {:.2} GiB file",
                    shard.path.file_name().unwrap().to_string_lossy(),
                    shard.tensors.len(),
                    gib(payload),
                    gib(shard.file_size() as u64),
                );
            }
        }
        weights::WeightStore::Packed(_) => {
            println!("    iris.pack.bin  (single mmap blob)");
        }
        weights::WeightStore::Dgq(_) => {
            println!("    model.dgq.bin  (quantized mmap blob)");
        }
    }

    println!("\n  largest tensors (by numel):");
    for (name, numel, shape) in &s.largest {
        println!("    {name}");
        println!("      shape={shape}  numel={numel}");
    }

    if let Ok(t) = store.tensor("model.decoder.embed_tokens.weight") {
        println!("\n  spot-check: model.decoder.embed_tokens.weight");
        println!(
            "    dtype={} shape={:?} bytes={}",
            t.dtype.as_str(),
            t.shape,
            t.byte_len()
        );
        if let Ok(bf16) = t.bf16() {
            println!("    bf16[0..4]: [{}]", bf16.preview_hex(4));
        }
    }
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn run_prefill(m: &model::Model) -> ExitCode {
    const PROMPT_LEN: usize = 128;
    let vocab = m.config.text_config.vocab_size;

    let mut token_ids = vec![0u32; PROMPT_LEN];
    for (i, id) in token_ids.iter_mut().enumerate() {
        *id = ((i * 131 + 7) % vocab.max(1)) as u32;
    }

    let mut scratch = model::encoder::EncoderScratch::new(PROMPT_LEN, &m.config);
    let input = model::encoder::EncoderPrefillInput {
        token_ids: &token_ids,
        position_offset: 0,
    };

    eprintln!("running encoder prefill (prompt_len={PROMPT_LEN}, layers={})...", m.config.text_config.num_hidden_layers);
    let started = std::time::Instant::now();
    match model::encoder::prefill(&m.weights, &m.config, &input, &mut scratch) {
        Ok(out) => {
            println!("encoder prefill ok");
            println!("  kv_len: {}", out.kv_cache.kv_len);
            println!("  hidden shape: [{PROMPT_LEN}, {}]", m.config.text_config.hidden_size);
            println!("  elapsed: {:.2?}", started.elapsed());
            println!("  {}", out.kv_cache.describe_layer(0));
            println!("  {}", out.kv_cache.describe_layer(5));
            println!("  {}", out.kv_cache.describe_layer(29));
            println!(
                "  hidden[0..4]: [{:.6}, {:.6}, {:.6}, {:.6}]",
                out.hidden_states[0],
                out.hidden_states[1],
                out.hidden_states[2],
                out.hidden_states[3]
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run_decoder_gpu_parity(m: &model::Model, seq_len: usize, kv_len: usize, layers: usize) -> ExitCode {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use metal::{
            bench_forward, estimate_decoder_forward, estimate_paged_layer_bytes,
            estimate_weight_cache, load_weight_cache, BenchConfig, GpuDecoderEngine,
            GpuDecoderScratch,
        };
        use model::decoder::{forward as cpu_forward, DecoderForwardInput, DecoderScratch};

        let canvas_len = seq_len;
        let kv_len = kv_len;
        let max_layers = if layers == 0 {
            m.config.text_config.num_hidden_layers
        } else {
            layers.min(m.config.text_config.num_hidden_layers)
        };
        let hidden = m.config.text_config.hidden_size;

        let kv_cache = match model::kv_cache::KvCache::dummy(&m.config.text_config, kv_len, 42) {
            Ok(cache) => cache,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };

        let mut token_ids = vec![0u32; canvas_len];
        let vocab = m.config.text_config.vocab_size;
        for (i, id) in token_ids.iter_mut().enumerate() {
            *id = ((i * 997 + 13) % vocab.max(1)) as u32;
        }

        let mask = model::mask::DecoderAttnMask::all_valid(canvas_len, kv_len);
        let input = DecoderForwardInput::new(&token_ids, &kv_cache);
        let mut input = DecoderForwardInput {
            mask: Some(&mask),
            compute_logits: false,
            return_hidden: true,
            ..input
        };

        let est = estimate_decoder_forward(&m.config.text_config, canvas_len, kv_len);
        est.print_summary("decoder-gpu (single-path)");

        let mut gpu_scratch = GpuDecoderScratch::new(canvas_len, &m.config);
        let mut gpu_weights =
            match load_weight_cache(&m.weights, &m.config.text_config, canvas_len, kv_len) {
            Ok(w) => w,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };
        eprintln!(
            "  weight cache: {:.1} KiB final norm; ~{:.1} MiB peak per layer (paged)",
            estimate_weight_cache(&gpu_weights) as f64 / 1024.0,
            estimate_paged_layer_bytes(&m.config.text_config) as f64 / (1024.0 * 1024.0)
        );
        let mut engine = match GpuDecoderEngine::new() {
            Ok(e) => e,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };
        if let Err(err) = gpu_scratch.ensure_gpu_kv(
            &engine.ctx.device,
            &m.config.text_config,
            kv_len,
            canvas_len,
        ) {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
        if let Err(err) = gpu_scratch.sync_gpu_kv_from_cpu(&kv_cache, canvas_len) {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }

        eprintln!(
            "running GPU decoder forward (canvas={canvas_len}, kv={kv_len}, layers={max_layers})..."
        );
        let gpu_started = std::time::Instant::now();
        let bench_cfg = BenchConfig { max_layers };
        let gpu_out = match bench_forward(
            &m.weights,
            &m.config,
            &mut input,
            &mut gpu_scratch,
            &mut gpu_weights,
            &mut engine,
            &bench_cfg,
        ) {
            Ok(out) => out,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };
        let gpu_elapsed = gpu_started.elapsed();
        let gpu_hidden = gpu_out.hidden_states;
        drop(gpu_out.logits);
        drop(gpu_scratch);
        drop(gpu_weights);
        drop(engine);

        let mut cpu_scratch = DecoderScratch::new(canvas_len, &m.config);
        eprintln!("running CPU decoder forward...");
        let cpu_started = std::time::Instant::now();
        let cpu_out = match cpu_forward(
            &m.weights,
            &m.config,
            &mut input,
            &mut cpu_scratch,
            Some(max_layers),
        ) {
            Ok(out) => out,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };
        let cpu_elapsed = cpu_started.elapsed();

        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut max_idx = 0usize;
        let mut nan_count = 0usize;
        for (i, (&c, &g)) in cpu_out
            .hidden_states
            .iter()
            .zip(gpu_hidden.iter())
            .enumerate()
        {
            if !g.is_finite() {
                nan_count += 1;
                continue;
            }
            let d = (c - g).abs();
            if !d.is_finite() {
                nan_count += 1;
                continue;
            }
            let denom = c.abs().max(g.abs()).max(1e-2);
            let rel = d / denom;
            if d > max_abs {
                max_abs = d;
                max_idx = i;
            }
            if rel > max_rel {
                max_rel = rel;
            }
        }
        if nan_count > 0 {
            eprintln!("error: GPU hidden has {nan_count} non-finite values");
            return ExitCode::FAILURE;
        }

        println!("decoder GPU parity ok");
        println!("  hidden shape: [{canvas_len}, {hidden}]");
        println!("  layers: {max_layers}");
        println!("  cpu elapsed: {cpu_elapsed:.2?}");
        println!("  gpu elapsed: {gpu_elapsed:.2?}");
        println!("  max_abs_diff: {max_abs:.6} at index {max_idx}");
        println!("  max_rel_diff: {max_rel:.6}");
        println!(
            "  cpu hidden[0..4]: [{:.6}, {:.6}, {:.6}, {:.6}]",
            cpu_out.hidden_states[0],
            cpu_out.hidden_states[1],
            cpu_out.hidden_states[2],
            cpu_out.hidden_states[3]
        );
        println!(
            "  gpu hidden[0..4]: [{:.6}, {:.6}, {:.6}, {:.6}]",
            gpu_hidden[0], gpu_hidden[1], gpu_hidden[2], gpu_hidden[3]
        );

        const TOL: f32 = 1e-2;
        if max_abs <= TOL {
            ExitCode::SUCCESS
        } else {
            eprintln!("error: max_abs_diff {max_abs} exceeds tolerance {TOL}");
            ExitCode::FAILURE
        }
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        let _ = m;
        eprintln!("error: decoder-gpu requires --features metal on macOS");
        ExitCode::FAILURE
    }
}

fn run_bench_decoder(
    m: &model::Model,
    seq_len: usize,
    kv_len: usize,
    layers: usize,
    iters: usize,
) -> ExitCode {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use metal::{
            bench_forward, load_weight_cache, BenchConfig, GpuDecoderEngine, GpuDecoderScratch,
        };
        use model::decoder::DecoderForwardInput;

        let seq_len = seq_len.max(1);
        let kv_len = kv_len.max(1);
        let layers = layers.max(1).min(m.config.text_config.num_hidden_layers);
        let iters = iters.max(1);
        let vocab = m.config.text_config.vocab_size;

        let kv_cache = match model::kv_cache::KvCache::dummy(&m.config.text_config, kv_len, 42) {
            Ok(cache) => cache,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };

        let mut token_ids = vec![0u32; seq_len];
        for (i, id) in token_ids.iter_mut().enumerate() {
            *id = ((i * 997 + 13) % vocab.max(1)) as u32;
        }

        let mask = model::mask::DecoderAttnMask::all_valid(seq_len, kv_len);
        let input = DecoderForwardInput::new(&token_ids, &kv_cache);
        let mut input = DecoderForwardInput {
            mask: Some(&mask),
            compute_logits: false,
            return_hidden: false,
            ..input
        };

        let mut scratch = GpuDecoderScratch::new(seq_len, &m.config);
        let mut gpu_weights =
            match load_weight_cache(&m.weights, &m.config.text_config, seq_len, kv_len) {
            Ok(w) => w,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };
        let mut engine = match GpuDecoderEngine::new() {
            Ok(e) => e,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };
        if let Err(err) = scratch.ensure_gpu_kv(
            &engine.ctx.device,
            &m.config.text_config,
            kv_len,
            seq_len,
        ) {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
        if let Err(err) = scratch.sync_gpu_kv_from_cpu(&kv_cache, seq_len) {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }

        let bench = BenchConfig { max_layers: layers };
        eprintln!(
            "bench-decoder warmup (seq={seq_len}, kv={kv_len}, layers={layers})..."
        );
        if let Err(err) = bench_forward(
            &m.weights,
            &m.config,
            &mut input,
            &mut scratch,
            &mut gpu_weights,
            &mut engine,
            &bench,
        ) {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }

        eprintln!("bench-decoder running {iters} iterations...");
        let started = std::time::Instant::now();
        for _ in 0..iters {
            if let Err(err) = bench_forward(
                &m.weights,
                &m.config,
                &mut input,
                &mut scratch,
                &mut gpu_weights,
                &mut engine,
                &bench,
            ) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        }
        let elapsed = started.elapsed();
        let per_fwd = elapsed / iters as u32;

        println!("bench-decoder ok");
        println!("  seq_len: {seq_len}");
        println!("  kv_len:  {kv_len}");
        println!("  layers:  {layers}");
        println!("  iters:   {iters}");
        println!("  total:   {elapsed:.2?}");
        println!("  per_fwd: {per_fwd:.2?}");
        ExitCode::SUCCESS
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        let _ = (m, seq_len, kv_len, layers, iters);
        eprintln!("error: bench-decoder requires --features metal on macOS");
        ExitCode::FAILURE
    }
}

fn run_bench_step(
    m: &model::Model,
    model_dir: &std::path::Path,
    canvas: usize,
    layers: usize,
    iters: usize,
    prompt_text: Option<String>,
    seed: u64,
    raw_prompt: bool,
) -> ExitCode {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use metal::{
            decoder_forward, load_weight_cache, GpuDecoderEngine, GpuDecoderScratch,
        };
        use model::decoder::DecoderForwardInput;
        use model::encoder::{prefill, EncoderPrefillInput, EncoderScratch};
        use sample::{initialize_canvas, Rng};

        let canvas = canvas.max(1);
        let layers = layers.max(1).min(m.config.text_config.num_hidden_layers);
        let iters = iters.max(1);
        let vocab = m.config.text_config.vocab_size;

        let prompt = match build_prompt_tokens(
            model_dir,
            prompt_text.as_deref(),
            64,
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
        let kv_len = prompt.len();

        let mut enc_scratch = EncoderScratch::new(kv_len.max(canvas), &m.config);
        eprintln!(
            "bench-step setup (canvas={canvas}, kv={kv_len}, layers={layers}, seed={seed})..."
        );

        let mut scratch = GpuDecoderScratch::new(canvas, &m.config);
        let mut gpu_weights = match load_weight_cache(
            &m.weights,
            &m.config.text_config,
            canvas,
            kv_len,
        ) {
            Ok(w) => w,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };
        let mut engine = match GpuDecoderEngine::new() {
            Ok(e) => e,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };
        if let Err(err) = scratch.ensure_gpu_kv(
            &engine.ctx.device,
            &m.config.text_config,
            kv_len,
            canvas,
        ) {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }

        let kv_cache = if m.weights.is_quantized() {
            match metal::prefill_gpu(
                &m.weights,
                &m.config,
                &EncoderPrefillInput {
                    token_ids: &prompt,
                    position_offset: 0,
                },
                &mut enc_scratch,
                &mut scratch,
                &mut gpu_weights,
                &mut engine,
                kv_len,
                canvas,
                Some(layers),
            ) {
                Ok(kv) => kv,
                Err(err) => {
                    eprintln!("error: {err}");
                    return ExitCode::FAILURE;
                }
            }
        } else {
            let prefill_out = match prefill(
                &m.weights,
                &m.config,
                &EncoderPrefillInput {
                    token_ids: &prompt,
                    position_offset: 0,
                },
                &mut enc_scratch,
            ) {
                Ok(out) => out,
                Err(err) => {
                    eprintln!("error: {err}");
                    return ExitCode::FAILURE;
                }
            };
            if let Err(err) = scratch.sync_gpu_kv_from_cpu(&prefill_out.kv_cache, canvas) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            prefill_out.kv_cache
        };

        let mut rng = Rng::new(seed);
        let mut token_ids = initialize_canvas(canvas, vocab, &mut rng);
        let mask = model::mask::DecoderAttnMask::all_valid(canvas, kv_cache.kv_len);
        let mut logits = vec![0.0f32; canvas * vocab];

        eprintln!("bench-step warmup...");
        {
            let mut input = DecoderForwardInput {
                token_ids: &token_ids,
                kv_cache: &kv_cache,
                self_conditioning_logits: None,
                mask: Some(&mask),
                logits_out: Some(&mut logits),
                compute_logits: true,
                return_hidden: false,
            };
            engine.reset_forward_telemetry();
            if let Err(err) = decoder_forward(
                &m.weights,
                &m.config,
                &mut input,
                &mut scratch,
                &mut gpu_weights,
                &mut engine,
                Some(layers),
            ) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            let _ = engine.take_forward_telemetry();
        }

        eprintln!("bench-step running {iters} iterations...");
        let mut telem_sum = metal::ForwardTelemetry::default();
        let started = std::time::Instant::now();
        for i in 0..iters {
            token_ids = initialize_canvas(canvas, vocab, &mut Rng::new(seed + i as u64 + 1));
            let mut input = DecoderForwardInput {
                token_ids: &token_ids,
                kv_cache: &kv_cache,
                self_conditioning_logits: None,
                mask: Some(&mask),
                logits_out: Some(&mut logits),
                compute_logits: true,
                return_hidden: false,
            };
            engine.reset_forward_telemetry();
            if let Err(err) = decoder_forward(
                &m.weights,
                &m.config,
                &mut input,
                &mut scratch,
                &mut gpu_weights,
                &mut engine,
                Some(layers),
            ) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            let t = engine.take_forward_telemetry();
            telem_sum.gpu_syncs += t.gpu_syncs;
            telem_sum.gpu_readback_bytes += t.gpu_readback_bytes;
            telem_sum.expert_weight_bytes_touched += t.expert_weight_bytes_touched;
            telem_sum.expert_hits += t.expert_hits;
            telem_sum.expert_misses += t.expert_misses;
            telem_sum.expert_upload_bytes += t.expert_upload_bytes;
            telem_sum.dense_gpu_upload_bytes += t.dense_gpu_upload_bytes;
            telem_sum.lm_head_logits_bytes += t.lm_head_logits_bytes;
            let n = t.expert_unique_per_layer.len();
            if telem_sum.expert_unique_per_layer.len() < n {
                telem_sum
                    .expert_unique_per_layer
                    .resize(n, 0);
            }
            for (li, &u) in t.expert_unique_per_layer.iter().enumerate() {
                telem_sum.expert_unique_per_layer[li] += u;
            }
        }
        let elapsed = started.elapsed();
        let per_step = elapsed / iters as u32;
        let n = iters as f64;
        for u in &mut telem_sum.expert_unique_per_layer {
            *u = ((*u as f64) / n).round() as u32;
        }
        telem_sum.gpu_syncs /= iters as u64;
        telem_sum.gpu_readback_bytes /= iters as u64;
        telem_sum.expert_weight_bytes_touched /= iters as u64;
        telem_sum.expert_hits /= iters as u64;
        telem_sum.expert_misses /= iters as u64;
        telem_sum.expert_upload_bytes /= iters as u64;
        telem_sum.dense_gpu_upload_bytes /= iters as u64;
        telem_sum.lm_head_logits_bytes /= iters as u64;

        let expert_stats = gpu_weights.expert_cache_stats();
        println!("bench-step ok");
        println!("  canvas:   {canvas}");
        println!("  kv_len:   {kv_len}");
        println!("  layers:   {layers}");
        println!("  iters:    {iters}");
        println!("  per_step: {per_step:.2?}");
        telem_sum.print_summary("  telemetry (per step, mean):");
        println!(
            "  expert LRU end: {:.1}/{:.1} MiB, {} entries, {} evictions",
            expert_stats.used_bytes as f64 / (1024.0 * 1024.0),
            expert_stats.budget_bytes as f64 / (1024.0 * 1024.0),
            expert_stats.entries,
            expert_stats.evictions,
        );
        ExitCode::SUCCESS
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        let _ = (m, model_dir, canvas, layers, iters, prompt_text, seed);
        eprintln!("error: bench-step requires --features metal on macOS");
        ExitCode::FAILURE
    }
}

fn run_bench_prefill(
    m: &model::Model,
    prompt_len: usize,
    layers: usize,
    iters: usize,
    seed: u64,
) -> ExitCode {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use metal::{load_weight_cache, GpuDecoderEngine, GpuDecoderScratch};
        use model::encoder::{prefill, EncoderPrefillInput, EncoderScratch};

        let prompt_len = prompt_len.max(1);
        let layers = layers.max(1).min(m.config.text_config.num_hidden_layers);
        let iters = iters.max(1);
        let canvas = m.config.canvas_length;
        let vocab = m.config.text_config.vocab_size;

        let mut token_ids = vec![0u32; prompt_len];
        for (i, id) in token_ids.iter_mut().enumerate() {
            *id = ((i as u64 * 131 + seed + 7) % vocab.max(1) as u64) as u32;
        }

        let mut enc_scratch = EncoderScratch::new(prompt_len.max(canvas), &m.config);
        let mut dec_scratch = GpuDecoderScratch::new(canvas, &m.config);
        let mut gpu_weights = match load_weight_cache(
            &m.weights,
            &m.config.text_config,
            canvas,
            prompt_len,
        ) {
            Ok(w) => w,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };
        let mut engine = match GpuDecoderEngine::new() {
            Ok(e) => e,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };

        let input = EncoderPrefillInput {
            token_ids: &token_ids,
            position_offset: 0,
        };

        eprintln!(
            "bench-prefill setup (prompt_len={prompt_len}, layers={layers}, quantized={})...",
            m.weights.is_quantized()
        );

        // Warmup
        if m.weights.is_quantized() {
            if let Err(err) = metal::prefill_gpu(
                &m.weights,
                &m.config,
                &input,
                &mut enc_scratch,
                &mut dec_scratch,
                &mut gpu_weights,
                &mut engine,
                prompt_len,
                canvas,
                Some(layers),
            ) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        } else {
            if let Err(err) = prefill(&m.weights, &m.config, &input, &mut enc_scratch) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        }

        eprintln!("bench-prefill running {iters} iterations...");
        let started = std::time::Instant::now();
        for _ in 0..iters {
            if m.weights.is_quantized() {
                if let Err(err) = metal::prefill_gpu(
                    &m.weights,
                    &m.config,
                    &input,
                    &mut enc_scratch,
                    &mut dec_scratch,
                    &mut gpu_weights,
                    &mut engine,
                    prompt_len,
                    canvas,
                    Some(layers),
                ) {
                    eprintln!("error: {err}");
                    return ExitCode::FAILURE;
                }
            } else {
                if let Err(err) = prefill(&m.weights, &m.config, &input, &mut enc_scratch) {
                    eprintln!("error: {err}");
                    return ExitCode::FAILURE;
                }
            }
        }
        let elapsed = started.elapsed();
        let per_run = elapsed / iters as u32;

        println!("bench-prefill ok");
        println!("  prompt_len: {prompt_len}");
        println!("  layers:     {layers}");
        println!("  iters:      {iters}");
        println!("  per_run:    {per_run:.2?}");
        println!("  gate (≤0.5s @ 1 tok): {}", if per_run.as_secs_f64() <= 0.5 { "pass" } else { "fail" });
        ExitCode::SUCCESS
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        let _ = (m, prompt_len, layers, iters, seed);
        eprintln!("error: bench-prefill requires --features metal on macOS");
        ExitCode::FAILURE
    }
}

fn run_decoder_forward(m: &model::Model) -> ExitCode {
    const CANVAS_LEN: usize = 256;
    const KV_LEN: usize = 128;
    let hidden = m.config.text_config.hidden_size;
    let vocab = m.config.text_config.vocab_size;

    let kv_cache = match model::kv_cache::KvCache::dummy(&m.config.text_config, KV_LEN, 42) {
        Ok(cache) => cache,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    let mut token_ids = vec![0u32; CANVAS_LEN];
    for (i, id) in token_ids.iter_mut().enumerate() {
        *id = ((i * 997 + 13) % vocab.max(1)) as u32;
    }

    let mut scratch = model::decoder::DecoderScratch::new(CANVAS_LEN, &m.config);
    let mask = model::mask::DecoderAttnMask::all_valid(CANVAS_LEN, KV_LEN);
    let input = model::decoder::DecoderForwardInput::new(&token_ids, &kv_cache);
    let mut input = model::decoder::DecoderForwardInput {
        mask: Some(&mask),
        ..input
    };

    eprintln!(
        "running full decoder forward (canvas={CANVAS_LEN}, kv={KV_LEN}, layers={})...",
        m.config.text_config.num_hidden_layers
    );
    let started = std::time::Instant::now();
    match model::decoder::forward(&m.weights, &m.config, &mut input, &mut scratch, None) {
        Ok(out) => {
            println!("decoder forward ok");
            println!("  hidden shape: [{CANVAS_LEN}, {hidden}]");
            println!("  logits shape: [{CANVAS_LEN}, {vocab}]");
            println!("  elapsed: {:.2?}", started.elapsed());
            println!(
                "  hidden[0..4]: [{:.6}, {:.6}, {:.6}, {:.6}]",
                out.hidden_states[0],
                out.hidden_states[1],
                out.hidden_states[2],
                out.hidden_states[3]
            );
            println!(
                "  logits[0,0..4]: [{:.6}, {:.6}, {:.6}, {:.6}]",
                out.logits[0], out.logits[1], out.logits[2], out.logits[3]
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run_gemm(size: usize) -> ExitCode {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use metal::{bf16_matmul_cpu, f32_to_bf16, Bf16Gemm};

        let n = size.max(1);
        let m = n;
        let k = n;
        eprintln!("running bf16 gemm on Metal ({m}x{k} @ {k}x{n})...");

        let mut a = vec![0u16; m * k];
        let mut b = vec![0u16; k * n];
        let mut cpu = vec![0.0f32; m * n];
        let mut gpu = vec![0.0f32; m * n];

        let mut state = 0xC0FFEE_u64;
        for slot in a.iter_mut().chain(b.iter_mut()) {
            state = state.wrapping_mul(6_966_169_279).wrapping_add(1);
            let v = ((state >> 32) as f32) / 65536.0 - 0.5;
            *slot = f32_to_bf16(v);
        }

        let started = std::time::Instant::now();
        bf16_matmul_cpu(&mut cpu, &a, &b, m, k, n);
        let cpu_elapsed = started.elapsed();

        let mut gemm = match Bf16Gemm::new() {
            Ok(g) => g,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };
        let started = std::time::Instant::now();
        if let Err(err) = gemm.matmul(&a, &b, &mut gpu, m, k, n) {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
        let gpu_elapsed = started.elapsed();

        let mut max_abs = 0.0f32;
        let mut max_idx = 0usize;
        for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            let d = (c - g).abs();
            if d > max_abs {
                max_abs = d;
                max_idx = i;
            }
        }

        println!("bf16 gemm ok");
        println!("  shape: {m}x{k} @ {k}x{n}");
        println!("  cpu elapsed: {cpu_elapsed:.2?}");
        println!("  gpu elapsed: {gpu_elapsed:.2?}");
        println!("  max_abs_diff: {max_abs:.6} at index {max_idx}");
        const TOL: f32 = 1e-3;
        if max_abs <= TOL {
            ExitCode::SUCCESS
        } else {
            eprintln!("error: max_abs_diff {max_abs} exceeds tolerance {TOL}");
            ExitCode::FAILURE
        }
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        let _ = size;
        eprintln!("error: gemm requires --features metal on macOS");
        ExitCode::FAILURE
    }
}

fn run_tokenize(model_dir: &PathBuf, text: &str, raw_prompt: bool) -> ExitCode {
    let path = model_dir.join("tokenizer.json");
    match tokenizer::Tokenizer::load(&path) {
        Ok(tok) => {
            let (formatted, ids) = if raw_prompt {
                (text.to_string(), tok.encode(text, false))
            } else {
                let turns = [chat_template::ChatTurn::user(text)];
                let formatted = chat_template::format_user_prompt(text);
                let ids = match chat_template::format_chat_token_ids(
                    &tok,
                    &turns,
                    &chat_template::ChatFormatOptions::default(),
                ) {
                    Ok(v) => v,
                    Err(err) => {
                        eprintln!("error: {err}");
                        return ExitCode::FAILURE;
                    }
                };
                (formatted, ids)
            };
            let payload = serde_json::json!({
                "text": text,
                "formatted": formatted,
                "chat_template": !raw_prompt,
                "ids": ids,
            });
            println!("{}", serde_json::to_string(&payload).unwrap_or_default());
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn build_prompt_tokens(
    model_dir: &std::path::Path,
    prompt_text: Option<&str>,
    prompt_len: usize,
    vocab: usize,
    raw_prompt: bool,
    history: &[chat_template::ChatTurn],
) -> Result<Vec<u32>, safetensors::Error> {
    if let Some(text) = prompt_text {
        let tok_path = model_dir.join("tokenizer.json");
        let tokenizer = tokenizer::Tokenizer::load(&tok_path)?;
        if raw_prompt {
            Ok(tokenizer.encode(text, false))
        } else {
            let mut turns = history.to_vec();
            turns.push(chat_template::ChatTurn::user(text));
            chat_template::format_chat_token_ids(
                &tokenizer,
                &turns,
                &chat_template::ChatFormatOptions::default(),
            )
        }
    } else {
        let mut prompt = vec![0u32; prompt_len];
        for (i, id) in prompt.iter_mut().enumerate() {
            *id = ((i * 131 + 7) % vocab.max(1)) as u32;
        }
        Ok(prompt)
    }
}

fn build_chat_prompt_tokens(
    model_dir: &std::path::Path,
    history: &[chat_template::ChatTurn],
    raw_prompt: bool,
) -> Result<Vec<u32>, safetensors::Error> {
    let tok_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizer::Tokenizer::load(&tok_path)?;
    if raw_prompt {
        let text = history
            .last()
            .map(|t| t.content.as_str())
            .unwrap_or("");
        Ok(tokenizer.encode(text, false))
    } else {
        chat_template::format_chat_token_ids(
            &tokenizer,
            history,
            &chat_template::ChatFormatOptions::default(),
        )
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_chat_cmd(
    model_dir: &std::path::Path,
    initial_prompt: Option<String>,
    seed: u64,
    steps: usize,
    max_new_tokens: usize,
    max_layers: Option<usize>,
    no_early_stop: bool,
    raw_prompt: bool,
) -> ExitCode {
    use metal::{generate_with_session, StepGenerateConfig, StepGenerateSession};
    use std::io::{self, Write};

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: chat requires a .dgq directory (-m /path/to/quantized-weights)");
        return ExitCode::FAILURE;
    }

    let layers = match max_layers {
        Some(n) => n.max(1).min(30),
        None => match crate::config::ModelConfig::load(model_dir) {
            Ok(c) => c.text_config.num_hidden_layers.min(30).max(1),
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        },
    };

    let sampler = sample::sampler_for_steps(steps, no_early_stop);
    let mut step_cfg = StepGenerateConfig::from_generate(
        seed,
        max_new_tokens,
        512,
        layers,
        sampler,
        no_early_stop,
    );

    let mut session = match StepGenerateSession::open(model_dir, &step_cfg) {
        Ok((s, compile)) => {
            eprintln!("chat: session ready ({compile:.2?}, layers={layers})");
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

    let mut history: Vec<chat_template::ChatTurn> = Vec::new();
    let mut turn_idx = 0u64;

    let mut run_turn = |history: &mut Vec<chat_template::ChatTurn>,
                        turn_idx: &mut u64|
     -> Result<(), safetensors::Error> {
        let prompt = build_chat_prompt_tokens(model_dir, history, raw_prompt)?;
        let prompt_len = prompt.len();
        step_cfg.max_seq = (prompt_len + max_new_tokens).max(512);
        step_cfg.seed = seed.wrapping_add(*turn_idx);
        *turn_idx = turn_idx.wrapping_add(1);

        let started = std::time::Instant::now();
        let out = generate_with_session(&mut session, &prompt, &step_cfg)?;
        let elapsed = started.elapsed();

        let new_ids = sample::strip_degenerate_token_ids(
            out.token_ids.get(prompt_len..).unwrap_or(&[]),
        );
        let reply = chat_template::sanitize_model_reply(&tokenizer.decode(&new_ids));
        if reply.is_empty() {
            println!("model> (empty response)");
        } else {
            println!("model> {reply}");
        }
        let new_tokens = out.token_ids.len().saturating_sub(prompt_len);
        eprintln!(
            "  turn: {new_tokens} new tokens, {} denoise steps, {:.2?}",
            out.denoise_steps_run, elapsed
        );
        history.push(chat_template::ChatTurn::model(reply));
        Ok(())
    };

    if let Some(first) = initial_prompt {
        let first = first.trim();
        if !first.is_empty() {
            history.push(chat_template::ChatTurn::user(first));
            if let Err(err) = run_turn(&mut history, &mut turn_idx) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        }
    }

    eprintln!("chat ready (type 'exit' or 'quit' to end; Ctrl-D also exits)");
    let stdin = io::stdin();
    loop {
        print!("you> ");
        let _ = io::stdout().flush();
        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "exit" || line == "quit" {
            break;
        }

        history.push(chat_template::ChatTurn::user(line));
        if let Err(err) = run_turn(&mut history, &mut turn_idx) {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    }

    ExitCode::SUCCESS
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_chat_cmd(
    _model_dir: &std::path::Path,
    _initial_prompt: Option<String>,
    _seed: u64,
    _steps: usize,
    _max_new_tokens: usize,
    _max_layers: Option<usize>,
    _no_early_stop: bool,
    _raw_prompt: bool,
) -> ExitCode {
    eprintln!("error: chat requires --features metal on macOS");
    ExitCode::FAILURE
}

fn print_generate_elapsed(label: &str, elapsed: std::time::Duration) {
    let secs = elapsed.as_secs_f64();
    println!("  {label} elapsed: {secs:.2}s ({elapsed:.2?})");
}

fn print_generate_timing_compare(
    cpu_elapsed: std::time::Duration,
    gpu_elapsed: std::time::Duration,
) {
    let cpu_s = cpu_elapsed.as_secs_f64();
    let gpu_s = gpu_elapsed.as_secs_f64();
    println!("timing:");
    print_generate_elapsed("cpu (generate)", cpu_elapsed);
    print_generate_elapsed("gpu (generate-gpu)", gpu_elapsed);
    if gpu_s > 0.0 {
        println!("  gpu speedup: {:.2}x", cpu_s / gpu_s);
    }
}

fn print_generate_output(
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
    println!("  prefill:  {:.2}s ({:.2?})", out.prefill_elapsed.as_secs_f64(), out.prefill_elapsed);
    println!("  denoise:  {:.2}s ({:.2?})", out.denoise_elapsed.as_secs_f64(), out.denoise_elapsed);
    if out.extend_elapsed.as_secs_f64() > 0.0 {
        println!("  extend:   {:.2}s ({:.2?})", out.extend_elapsed.as_secs_f64(), out.extend_elapsed);
    }
    if out.denoise_elapsed.as_secs_f64() > 0.0 && new_tokens > 0 {
        let tok_s = new_tokens as f64 / out.denoise_elapsed.as_secs_f64();
        println!("  throughput: {tok_s:.2} tok/s (denoise only, excludes prefill/extend)");
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    if !out.session_telemetry.steps.is_empty() {
        out.session_telemetry.print_summary("  session telemetry:");
        if out.denoise_steps_run > 0 {
            let agg = out.session_telemetry.aggregate_forward();
            let step_ms = out.denoise_elapsed.as_secs_f64() * 1000.0
                / out.denoise_steps_run as f64;
            println!("  mean step wall:       {step_ms:.1} ms");
            println!(
                "  weight bytes/step:    {:.2} GiB expert + {:.2} MiB logits",
                agg.expert_weight_bytes_touched as f64 / (1024.0_f64.powi(3)),
                agg.lm_head_logits_bytes as f64 / (1024.0 * 1024.0)
            );
        }
    }

    if let Ok(tokenizer) = tokenizer::Tokenizer::load(model_dir.join("tokenizer.json")) {
        let display_ids = sample::strip_degenerate_token_ids(
            &out.token_ids.get(prompt_len..).unwrap_or(&[]),
        );
        if !display_ids.is_empty() {
            let text = tokenizer.decode(&display_ids);
            if !text.is_empty() {
                let preview: String = text.chars().take(200).collect();
                println!("  text: {preview}");
            }
        }
    }

    let preview: Vec<String> = out
        .token_ids
        .iter()
        .take(16)
        .map(|t| t.to_string())
        .collect();
    println!("  token_ids[0..16]: [{}]", preview.join(", "));
}

fn infer_golden_name(
    prompt_text: Option<&str>,
    steps: usize,
    max_layers: Option<usize>,
    quantized: bool,
) -> Option<String> {
    generate_golden::infer_fixture_name(prompt_text, steps, max_layers, quantized)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_generate_monolithic_cmd(
    model_dir: &std::path::Path,
    prompt_text: Option<String>,
    seed: u64,
    steps: usize,
    prompt_len: usize,
    max_new_tokens: usize,
    max_layers: Option<usize>,
    no_early_stop: bool,
    engine_fallback: bool,
    write_golden: Option<String>,
    raw_prompt: bool,
) -> ExitCode {
    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: generate-monolithic requires a .dgq directory (-m /path/to/quantized-weights)");
        if engine_fallback {
            eprintln!("note: falling back to generate-gpu (not a .dgq path)");
            return run_generate_engine_fallback(
                model_dir,
                prompt_text,
                seed,
                steps,
                prompt_len,
                max_new_tokens,
                max_layers,
                no_early_stop,
                raw_prompt,
            );
        }
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
    let max_seq = (prompt_len + max_new_tokens).max(512);

    let gen_cfg = generate::GenerateConfig {
        sampler: sample::sampler_for_steps(steps, no_early_stop),
        max_new_tokens,
        seed,
        max_layers,
        no_early_stop,
        deterministic: false,
    };

    let layers_note = max_layers
        .map(|n| format!(", layers={n}"))
        .unwrap_or_default();
    let stop_note = if no_early_stop { ", no_early_stop" } else { "" };
    eprintln!(
        "running generate-monolithic (prompt_len={prompt_len}, steps={steps}, max_new_tokens={max_new_tokens}, seed={seed}{layers_note}{stop_note})..."
    );
    let started = std::time::Instant::now();

    match generate::generate_monolithic_gpu(model_dir, &prompt, &gen_cfg, max_seq) {
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
            print_generate_output(
                "generate-monolithic",
                &out,
                prompt_len,
                started.elapsed(),
                model_dir,
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            if engine_fallback {
                eprintln!("note: monolithic failed; falling back to generate-gpu");
                return run_generate_engine_fallback(
                    model_dir,
                    prompt_text,
                    seed,
                    steps,
                    prompt_len,
                    max_new_tokens,
                    max_layers,
                    no_early_stop,
                    raw_prompt,
                );
            }
            ExitCode::FAILURE
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_generate_monolithic_parity_cmd(
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
    let max_seq = (prompt_len + max_new_tokens).max(512);
    let prompt_label = prompt_text.clone().unwrap_or_else(|| format!("prompt_len={prompt_len}"));

    let gen_cfg = generate::GenerateConfig {
        sampler: sample::sampler_for_steps(steps, no_early_stop),
        max_new_tokens,
        seed,
        max_layers,
        no_early_stop,
        deterministic: true,
    };

    if let Some(n) = max_layers {
        eprintln!("generate-monolithic-parity: layers limited to {n}");
    }
    eprintln!(
        "running generate-monolithic parity (DGQ_MPS_Q4=0 native Q4, prompt_len={prompt_len}, steps={steps}, seed={seed})..."
    );

    let out = match generate::generate_monolithic_gpu(model_dir, &prompt, &gen_cfg, max_seq) {
        Ok(out) => out,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    let profile = generate_golden::monolithic_weights_profile();

    if let Some(ref name) = write_golden {
        if let Err(err) = write_generate_golden(
            name,
            &prompt_label,
            &gen_cfg,
            steps,
            profile,
            &out,
        ) {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    }

    let fixture_name = golden_name.or_else(|| {
        generate_golden::infer_monolithic_fixture_name(
            prompt_text.as_deref(),
            steps,
            max_layers,
        )
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

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_generate_monolithic_parity_cmd(
    _model_dir: &std::path::Path,
    _prompt_text: Option<String>,
    _seed: u64,
    _steps: usize,
    _prompt_len: usize,
    _max_new_tokens: usize,
    _max_layers: Option<usize>,
    _no_early_stop: bool,
    _golden_name: Option<String>,
    _write_golden: Option<String>,
    _raw_prompt: bool,
) -> ExitCode {
    eprintln!("error: generate-monolithic-parity requires --features metal on macOS");
    ExitCode::FAILURE
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_generate_engine_fallback(
    model_dir: &std::path::Path,
    prompt_text: Option<String>,
    seed: u64,
    steps: usize,
    prompt_len: usize,
    max_new_tokens: usize,
    max_layers: Option<usize>,
    no_early_stop: bool,
    raw_prompt: bool,
) -> ExitCode {
    match model::Model::open(model_dir) {
        Ok(m) => run_generate(
            &m,
            model_dir,
            prompt_text,
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers,
            no_early_stop,
            true,
            None,
            raw_prompt,
        ),
        Err(err) => {
            eprintln!("error: engine fallback failed: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_generate_monolithic_cmd(
    _model_dir: &std::path::Path,
    _prompt_text: Option<String>,
    _seed: u64,
    _steps: usize,
    _prompt_len: usize,
    _max_new_tokens: usize,
    _max_layers: Option<usize>,
    _no_early_stop: bool,
    _engine_fallback: bool,
    _write_golden: Option<String>,
    _raw_prompt: bool,
) -> ExitCode {
    eprintln!("error: generate-monolithic requires --features metal on macOS");
    ExitCode::FAILURE
}

fn run_generate(
    m: &model::Model,
    model_dir: &std::path::Path,
    prompt_text: Option<String>,
    seed: u64,
    steps: usize,
    prompt_len: usize,
    max_new_tokens: usize,
    max_layers: Option<usize>,
    no_early_stop: bool,
    use_gpu: bool,
    write_golden: Option<String>,
    raw_prompt: bool,
) -> ExitCode {
    let vocab = m.config.text_config.vocab_size;
    let canvas = m.config.canvas_length;

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

    let enc_seq = prompt_len.max(canvas);
    let mut enc_scratch = model::encoder::EncoderScratch::new(enc_seq, &m.config);

    let gen_cfg = generate::GenerateConfig {
        sampler: sample::sampler_for_steps(steps, no_early_stop),
        max_new_tokens,
        seed,
        max_layers,
        no_early_stop,
        deterministic: false,
    };

    let backend = if use_gpu { "generate-gpu" } else { "generate" };
    let layers_note = max_layers
        .map(|n| format!(", layers={n}"))
        .unwrap_or_default();
    let stop_note = if no_early_stop { ", no_early_stop" } else { "" };
    eprintln!(
        "running {backend} (prompt_len={prompt_len}, canvas={canvas}, steps={steps}, max_new_tokens={max_new_tokens}, seed={seed}{layers_note}{stop_note})..."
    );
    let started = std::time::Instant::now();

    #[cfg(all(feature = "metal", target_os = "macos"))]
    if use_gpu {
        use metal::{
            load_weight_cache, log_expert_cache_stats, GpuDecoderEngine, GpuDecoderScratch,
        };
        let mut dec_scratch = GpuDecoderScratch::new(canvas, &m.config);
        let max_kv = prompt_len + max_new_tokens;
        let mut gpu_weights =
            match load_weight_cache(&m.weights, &m.config.text_config, canvas, max_kv) {
            Ok(w) => w,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };
        let mut engine = match GpuDecoderEngine::new() {
            Ok(e) => e,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };
        return match generate::generate_gpu(
            &m.weights,
            &m.config,
            &prompt,
            &gen_cfg,
            &mut enc_scratch,
            &mut dec_scratch,
            &mut gpu_weights,
            &mut engine,
        ) {
            Ok(out) => {
                if let Some(ref name) = write_golden {
                    let golden_name = name.clone();
                    let prompt_str = prompt_text.clone().unwrap_or_default();
                    let profile =
                        generate_golden::weights_profile_name(m.weights.is_quantized());
                    if let Err(err) = write_generate_golden(
                        &golden_name,
                        &prompt_str,
                        &gen_cfg,
                        steps,
                        profile,
                        &out,
                    ) {
                        eprintln!("error: {err}");
                        return ExitCode::FAILURE;
                    }
                }
                log_expert_cache_stats(gpu_weights.expert_cache_stats());
                print_generate_output(backend, &out, prompt_len, started.elapsed(), model_dir);
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("error: {err}");
                ExitCode::FAILURE
            }
        };
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    if use_gpu {
        eprintln!("error: generate-gpu requires --features metal on macOS");
        return ExitCode::FAILURE;
    }

    let mut dec_scratch = model::decoder::DecoderScratch::new(canvas, &m.config);
    match generate::generate(
        &m.weights,
        &m.config,
        &prompt,
        &gen_cfg,
        &mut enc_scratch,
        &mut dec_scratch,
    ) {
        Ok(out) => {
            print_generate_output(backend, &out, prompt_len, started.elapsed(), model_dir);
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn write_generate_golden(
    name: &str,
    prompt: &str,
    gen_cfg: &generate::GenerateConfig,
    steps: usize,
    weights_profile: &str,
    out: &generate::GenerateOutput,
) -> Result<(), safetensors::Error> {
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

fn run_generate_parity(
    m: &model::Model,
    model_dir: &std::path::Path,
    prompt_text: Option<String>,
    seed: u64,
    steps: usize,
    prompt_len: usize,
    max_new_tokens: usize,
    max_layers: Option<usize>,
    no_early_stop: bool,
    golden_name: Option<String>,
    compare_cpu: bool,
    write_golden: Option<String>,
    raw_prompt: bool,
) -> ExitCode {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use generate_golden::GenerateGolden;
        use metal::{load_weight_cache, GpuDecoderEngine, GpuDecoderScratch};

        let vocab = m.config.text_config.vocab_size;
        let canvas = m.config.canvas_length;
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

        let gen_cfg = generate::GenerateConfig {
            sampler: sample::sampler_for_steps(steps, no_early_stop),
            max_new_tokens,
            seed,
            max_layers,
            no_early_stop,
            deterministic: true,
        };

        let prompt_label = prompt_text.clone().unwrap_or_else(|| format!("prompt_len={prompt_len}"));

        if let Some(n) = max_layers {
            eprintln!("generate-parity: decoder layers limited to {n}");
        }

        let enc_seq = prompt.len().max(canvas);
        let mut enc_gpu = model::encoder::EncoderScratch::new(enc_seq, &m.config);
        let mut dec_gpu = GpuDecoderScratch::new(canvas, &m.config);
        let max_kv = prompt.len() + max_new_tokens;
        let mut gpu_weights =
            match load_weight_cache(&m.weights, &m.config.text_config, canvas, max_kv) {
            Ok(w) => w,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };
        let mut engine = match GpuDecoderEngine::new() {
            Ok(e) => e,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };

        eprintln!("running GPU generate...");
        let gpu_started = std::time::Instant::now();
        let gpu_out = match generate::generate_gpu(
            &m.weights,
            &m.config,
            &prompt,
            &gen_cfg,
            &mut enc_gpu,
            &mut dec_gpu,
            &mut gpu_weights,
            &mut engine,
        ) {
            Ok(out) => out,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };
        let gpu_elapsed = gpu_started.elapsed();
        eprintln!("GPU generate finished in {:.2}s ({gpu_elapsed:.2?})", gpu_elapsed.as_secs_f64());

        let weights_profile = generate_golden::weights_profile_name(m.weights.is_quantized());

        if let Some(ref name) = write_golden {
            if let Err(err) = write_generate_golden(
                name,
                &prompt_label,
                &gen_cfg,
                steps,
                weights_profile,
                &gpu_out,
            ) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        }

        if compare_cpu {
            let mut enc_cpu = model::encoder::EncoderScratch::new(enc_seq, &m.config);
            let mut dec_cpu = model::decoder::DecoderScratch::new(canvas, &m.config);
            eprintln!("running CPU generate...");
            let cpu_started = std::time::Instant::now();
            let cpu_out = match generate::generate(
                &m.weights,
                &m.config,
                &prompt,
                &gen_cfg,
                &mut enc_cpu,
                &mut dec_cpu,
            ) {
                Ok(out) => out,
                Err(err) => {
                    eprintln!("error: {err}");
                    return ExitCode::FAILURE;
                }
            };
            let cpu_elapsed = cpu_started.elapsed();
            eprintln!("CPU generate finished in {:.2}s ({cpu_elapsed:.2?})", cpu_elapsed.as_secs_f64());
            if cpu_out.token_ids != gpu_out.token_ids {
                let first_diff = cpu_out
                    .token_ids
                    .iter()
                    .zip(gpu_out.token_ids.iter())
                    .position(|(a, b)| a != b);
                eprintln!(
                    "error: CPU/GPU token mismatch at index {:?}",
                    first_diff
                );
                return ExitCode::FAILURE;
            }
            print_generate_timing_compare(cpu_elapsed, gpu_elapsed);
        } else {
            let fixture = match golden_name.or_else(|| {
                infer_golden_name(
                    prompt_text.as_deref(),
                    steps,
                    max_layers,
                    m.weights.is_quantized(),
                )
            }) {
                Some(name) => name,
                None => {
                    eprintln!("error: no --golden fixture; use --write-golden NAME or --compare-cpu");
                    return ExitCode::FAILURE;
                }
            };
            let path = generate_golden::resolve_fixture(&fixture);
            eprintln!("checking golden {} (profile={weights_profile})...", path.display());
            let golden = match GenerateGolden::load(&path) {
                Ok(g) => g,
                Err(err) => {
                    eprintln!("error: {err}");
                    return ExitCode::FAILURE;
                }
            };
            if !golden.matches_config(&prompt_label, &gen_cfg, steps, weights_profile) {
                eprintln!(
                    "error: golden config mismatch for {} (expected profile={})",
                    golden.name,
                    golden.expected_weights_profile()
                );
                return ExitCode::FAILURE;
            }
            if let Err(err) = golden.compare(&gpu_out) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            println!("generate golden ok ({})", golden.name);
        }

        let new_tokens = gpu_out.token_ids.len().saturating_sub(prompt.len());
        println!("  tokens: {}", gpu_out.token_ids.len());
        println!("  denoise steps: {}", gpu_out.denoise_steps_run);
        println!("  blocks: {}", gpu_out.blocks_committed);
        if gpu_out.denoise_elapsed.as_secs_f64() > 0.0 && new_tokens > 0 {
            let gpu_tps = new_tokens as f64 / gpu_out.denoise_elapsed.as_secs_f64();
            println!("  gpu throughput: {gpu_tps:.2} tok/s (denoise only)");
        }
        if let Ok(tokenizer) = tokenizer::Tokenizer::load(model_dir.join("tokenizer.json")) {
            let text = tokenizer.decode(&gpu_out.token_ids);
            if !text.is_empty() {
                let preview: String = text.chars().take(200).collect();
                println!("  text: {preview}");
            }
        }
        ExitCode::SUCCESS
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        let _ = (
            m,
            model_dir,
            prompt_text,
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers,
            golden_name,
            compare_cpu,
            write_golden,
        );
        eprintln!("error: generate-parity requires --features metal on macOS");
        ExitCode::FAILURE
    }
}

fn run_attention_parity(m: &model::Model) -> ExitCode {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use metal::GpuAttention;
        use model::attention::{forward_to_attn_out, prepare_qkv_pre_rope, AttentionParams};

        const SEQ_LEN: usize = 16;
        let hidden = m.config.text_config.hidden_size;

        let layer = match model::layer_weights::DecoderLayerWeights::load(
            &m.weights,
            0,
            &m.config.text_config,
        ) {
            Ok(layer) => layer,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };

        let params = match AttentionParams::for_layer(&m.config.text_config, 0) {
            Ok(p) => p,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };

        let mut scratch = model::attention::AttentionScratch::new(SEQ_LEN, hidden, &params);

        let mut input = vec![0.0f32; SEQ_LEN * hidden];
        for (i, v) in input.iter_mut().enumerate() {
            *v = ((i % hidden) as f32) * 0.01 - 0.5;
        }
        let positions: Vec<i64> = (0..SEQ_LEN as i64).collect();

        if let Err(err) = prepare_qkv_pre_rope(
            &input,
            &layer,
            &m.config.text_config,
            0,
            SEQ_LEN,
            &positions,
            &mut scratch,
        ) {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }

        let pre_q = scratch.q.clone();
        let pre_k = scratch.k.clone();
        let pre_v = scratch.v.clone();
        let freqs = scratch.rope_freqs.clone();

        let q_dim = SEQ_LEN * params.n_heads * params.head_dim;
        let mut cpu_out = vec![0.0f32; q_dim];
        if let Err(err) = forward_to_attn_out(
            &mut cpu_out,
            &input,
            &layer,
            &m.config.text_config,
            0,
            SEQ_LEN,
            &positions,
            &mut scratch,
        ) {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }

        let mut gpu_q = pre_q;
        let mut gpu_k = pre_k;
        let mut gpu_out = vec![0.0f32; q_dim];
        let mut gpu = match GpuAttention::new() {
            Ok(g) => g,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };

        eprintln!(
            "running layer 0 GPU attention (seq={SEQ_LEN}, heads={}, kv_heads={}, head_dim={})...",
            params.n_heads, params.n_kv_heads, params.head_dim
        );
        let started = std::time::Instant::now();
        if let Err(err) = gpu.rope_and_gqa(
            &mut gpu_out,
            &mut gpu_q,
            &mut gpu_k,
            &pre_v,
            &freqs,
            SEQ_LEN,
            SEQ_LEN,
            &params,
            model::attention::GqaMask::CausalSliding,
        ) {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
        let gpu_elapsed = started.elapsed();

        let mut max_abs = 0.0f32;
        let mut max_idx = 0usize;
        for (i, (&c, &g)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
            let d = (c - g).abs();
            if d > max_abs {
                max_abs = d;
                max_idx = i;
            }
        }

        println!("layer 0 attention parity ok");
        println!("  shape: [{SEQ_LEN}, {}, {}]", params.n_heads, params.head_dim);
        println!("  gpu elapsed: {gpu_elapsed:.2?}");
        println!("  max_abs_diff: {max_abs:.6} at index {max_idx}");
        println!(
            "  cpu[0..4]: [{:.6}, {:.6}, {:.6}, {:.6}]",
            cpu_out[0], cpu_out[1], cpu_out[2], cpu_out[3]
        );
        println!(
            "  gpu[0..4]: [{:.6}, {:.6}, {:.6}, {:.6}]",
            gpu_out[0], gpu_out[1], gpu_out[2], gpu_out[3]
        );

        const TOL: f32 = 1e-3;
        if max_abs <= TOL {
            ExitCode::SUCCESS
        } else {
            eprintln!("error: max_abs_diff {max_abs} exceeds tolerance {TOL}");
            ExitCode::FAILURE
        }
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        let _ = m;
        eprintln!("error: attention requires --features metal on macOS");
        ExitCode::FAILURE
    }
}

fn run_layer0_forward(m: &model::Model) -> ExitCode {
    const SEQ_LEN: usize = 16;
    let hidden = m.config.text_config.hidden_size;

    let layer = match model::layer_weights::DecoderLayerWeights::load(
        &m.weights,
        0,
        &m.config.text_config,
    ) {
        Ok(layer) => layer,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    let mut scratch = match model::decoder_layer::DecoderLayerScratch::new(
        SEQ_LEN,
        &m.config.text_config,
        0,
    ) {
        Ok(scratch) => scratch,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    let mut input = vec![0.0f32; SEQ_LEN * hidden];
    let mut output = vec![0.0f32; SEQ_LEN * hidden];
    for (i, v) in input.iter_mut().enumerate() {
        *v = ((i % hidden) as f32) * 0.01 - 0.5;
    }
    let positions: Vec<i64> = (0..SEQ_LEN as i64).collect();

    eprintln!("running decoder layer 0 forward (seq={SEQ_LEN}, hidden={hidden})...");
    let started = std::time::Instant::now();
    match model::decoder_layer::forward(
        &mut output,
        &input,
        &layer,
        &m.config.text_config,
        0,
        SEQ_LEN,
        &positions,
        &mut scratch,
    ) {
        Ok(()) => {
            println!("decoder layer 0 forward ok");
            println!("  output shape: [{SEQ_LEN}, {hidden}]");
            println!("  elapsed: {:.2?}", started.elapsed());
            println!(
                "  output[0..4]: [{:.6}, {:.6}, {:.6}, {:.6}]",
                output[0], output[1], output[2], output[3]
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
