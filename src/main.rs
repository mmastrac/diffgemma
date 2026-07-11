// diffgemma-mps is an Apple-Silicon Metal inference engine. Metal is the only
// backend — there is no CPU forward/generate path — so the whole program is
// macOS-only. Fail fast on other targets with one clear message rather than a
// wall of "cannot find `metal`" errors from the GPU-referencing kernel code.
#[cfg(not(target_os = "macos"))]
compile_error!(
    "diffgemma-mps requires macOS on Apple Silicon: Metal is the only backend \
     and there is no CPU inference path."
);

mod buffer;
#[cfg(target_os = "macos")]
mod chat_protocol;
mod chat_template;
#[cfg(target_os = "macos")]
mod chat_ui;
mod config;
#[cfg(target_os = "macos")]
mod conversation;
mod denoise_trace;
mod dgq;
mod fast_slice;
mod flags;
mod generate;
mod generate_golden;
#[allow(dead_code)]
mod kernels;
#[cfg(target_os = "macos")]
mod metal;
mod model;
mod pack;
mod safetensors;
mod sample;
#[cfg(target_os = "macos")]
mod server;
mod tensor;
mod tokenizer;
mod toolcompact;
mod tools;
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
    GenerateMonolithic {
        prompt: Option<String>,
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
        /// `--verbose`: show the full generation telemetry. Default prints only
        /// the clean reply (one-shot chat).
        verbose: bool,
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
    Tokenize(String),
    Gemm {
        size: usize,
    },
    Attention,
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
    StepKvParity {
        prompt: Option<String>,
        prompt_len: usize,
        layers: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
    },
    StepKvBf16Cross {
        prompt: Option<String>,
        prompt_len: usize,
        layers: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
        bf16_ref_dir: PathBuf,
    },
    StepAttnProbe {
        prompt: Option<String>,
        prompt_len: usize,
        layer: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
    },
    StepLogitsDump {
        prompt: Option<String>,
        layers: usize,
        steps: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
        output: PathBuf,
        positions: String,
        top_k: usize,
    },
    StepBf16OracleLogitsDump {
        prompt: Option<String>,
        layers: usize,
        steps: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
        output: PathBuf,
        positions: String,
        top_k: usize,
        bf16_ref_dir: PathBuf,
        gpu_kv: bool,
    },
    StepLayerProbe {
        prompt: Option<String>,
        layers: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
        output: PathBuf,
        position: usize,
    },
    StepAttnDump {
        prompt: Option<String>,
        layers: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
        output: PathBuf,
        layer: usize,
        position: usize,
    },
    StepMoeDump {
        prompt: Option<String>,
        layers: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
        output: PathBuf,
        layer: usize,
        position: usize,
    },
    StepMoeRouteDump {
        prompt: Option<String>,
        layers: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
        output: PathBuf,
        layer: usize,
        run_grouped: bool,
    },
    StepMoeBatchedPinDump {
        prompt: Option<String>,
        layers: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
        output: PathBuf,
        layer: usize,
    },
    StepMoeSingleDump {
        prompt: Option<String>,
        layers: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
        output: PathBuf,
        layer: usize,
        position: usize,
        expert: u32,
    },
    StepPreambleDump {
        prompt: Option<String>,
        layers: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
        output: PathBuf,
        position: usize,
    },
    EmbedRowDump {
        token: u32,
        layers: usize,
        max_seq: usize,
        prompt: Option<String>,
        raw_prompt: bool,
        output: PathBuf,
        bf16_ref_dir: Option<PathBuf>,
        gpu: bool,
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
        profile: bool,
        profile_steps: usize,
        layer_profile: bool,
    },
    BenchGemmFusion {
        layers: usize,
        kv_len: u32,
        seed: u64,
        max_seq: usize,
        iters: usize,
    },
    Chat {
        seed: u64,
        steps: usize,
        max_new_tokens: usize,
        max_layers: Option<usize>,
        no_early_stop: bool,
        initial_prompt: Option<String>,
        verbose: bool,
        events_path: Option<PathBuf>,
        json: bool,
        /// Context window (max_seq) override; None = default 8192.
        ctx: Option<usize>,
    },
    /// OpenAI-compatible HTTP server (`serve`). Single-GPU-queue chat completions.
    Serve {
        addr: String,
        seed: u64,
        steps: usize,
        max_layers: Option<usize>,
        /// Context window (max_seq); default 8192.
        ctx: usize,
        /// Tool-output compaction (KV rewinder). Also `DGQ_TOOL_COMPACT=1`.
        tool_compact: bool,
    },
    Smoketest {
        prompts_path: Option<PathBuf>,
        /// None = spec-pinned gate seed; Some = explicit CLI --seed sweep.
        seed: Option<u64>,
        steps: usize,
        max_layers: Option<usize>,
        /// Substring (case-insensitive) on prompt id; only matching prompts run.
        filter: Option<String>,
        /// Repeat the whole (filtered) prompt sequence N times in ONE session
        /// (no re-warmup) — surfaces reset_kv session-state carryover.
        repeat: usize,
        /// Run ONLY the long-context doc-QA tier (bigger session; E13).
        longctx: bool,
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
        Command::StepKvParity {
            prompt,
            prompt_len,
            layers,
            seed,
            max_seq,
            raw_prompt,
        } => run_step_kv_parity_cmd(
            &cli.model_dir,
            prompt,
            prompt_len,
            layers,
            seed,
            max_seq,
            raw_prompt,
        ),
        Command::StepKvBf16Cross {
            prompt,
            prompt_len,
            layers,
            seed,
            max_seq,
            raw_prompt,
            bf16_ref_dir,
        } => run_step_kv_bf16_cross_cmd(
            &cli.model_dir,
            &bf16_ref_dir,
            prompt,
            prompt_len,
            layers,
            seed,
            max_seq,
            raw_prompt,
        ),
        Command::StepAttnProbe {
            prompt,
            prompt_len,
            layer,
            seed,
            max_seq,
            raw_prompt,
        } => run_step_attn_probe_cmd(
            &cli.model_dir,
            prompt,
            prompt_len,
            layer,
            seed,
            max_seq,
            raw_prompt,
        ),
        Command::StepLogitsDump {
            prompt,
            layers,
            steps,
            seed,
            max_seq,
            raw_prompt,
            output,
            positions,
            top_k,
        } => run_step_logits_dump_cmd(
            &cli.model_dir,
            prompt,
            layers,
            steps,
            seed,
            max_seq,
            raw_prompt,
            &output,
            &positions,
            top_k,
        ),
        Command::StepBf16OracleLogitsDump {
            prompt,
            layers,
            steps,
            seed,
            max_seq,
            raw_prompt,
            output,
            positions,
            top_k,
            bf16_ref_dir,
            gpu_kv,
        } => run_step_bf16_oracle_logits_dump_cmd(
            &cli.model_dir,
            &bf16_ref_dir,
            prompt,
            layers,
            steps,
            seed,
            max_seq,
            raw_prompt,
            &output,
            &positions,
            top_k,
            gpu_kv,
        ),
        Command::StepLayerProbe {
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            output,
            position,
        } => run_step_layer_probe_cmd(
            &cli.model_dir,
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            &output,
            position,
        ),
        Command::StepAttnDump {
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            output,
            layer,
            position,
        } => run_step_attn_dump_cmd(
            &cli.model_dir,
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            &output,
            layer,
            position,
        ),
        Command::StepMoeDump {
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            output,
            layer,
            position,
        } => run_step_moe_dump_cmd(
            &cli.model_dir,
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            &output,
            layer,
            position,
        ),
        Command::StepMoeRouteDump {
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            output,
            layer,
            run_grouped,
        } => run_step_moe_route_dump_cmd(
            &cli.model_dir,
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            &output,
            layer,
            run_grouped,
        ),
        Command::StepMoeBatchedPinDump {
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            output,
            layer,
        } => run_step_moe_batched_pin_dump_cmd(
            &cli.model_dir,
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            &output,
            layer,
        ),
        Command::StepMoeSingleDump {
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            output,
            layer,
            position,
            expert,
        } => run_step_moe_single_dump_cmd(
            &cli.model_dir,
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            &output,
            layer,
            position,
            expert,
        ),
        Command::StepPreambleDump {
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            output,
            position,
        } => run_step_preamble_dump_cmd(
            &cli.model_dir,
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            &output,
            position,
        ),
        Command::EmbedRowDump {
            token,
            layers,
            max_seq,
            prompt,
            raw_prompt,
            output,
            bf16_ref_dir,
            gpu,
        } => run_embed_row_dump_cmd(
            &cli.model_dir,
            token,
            layers,
            max_seq,
            prompt,
            raw_prompt,
            &output,
            bf16_ref_dir.as_deref(),
            gpu,
        ),
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
            profile,
            profile_steps,
            layer_profile,
        } => run_bench_step_kernel_cmd(
            &cli.model_dir,
            layers,
            kv_len,
            seed,
            max_seq,
            iters,
            forward_only,
            profile,
            profile_steps,
            layer_profile,
        ),
        Command::BenchGemmFusion {
            layers,
            kv_len,
            seed,
            max_seq,
            iters,
        } => run_bench_gemm_fusion_cmd(&cli.model_dir, layers, kv_len, seed, max_seq, iters),
        Command::GenerateMonolithic {
            prompt,
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers,
            no_early_stop,
            kernel_assert,
            kernel_debug_deep,
            write_golden,
            write_trace,
            verbose,
        } => run_generate_monolithic_cmd(
            &cli.model_dir,
            prompt,
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers,
            no_early_stop,
            kernel_assert,
            kernel_debug_deep,
            write_golden,
            write_trace,
            cli.raw_prompt,
            verbose,
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
            verbose,
            events_path,
            json,
            ctx,
        } => run_chat_cmd(
            &cli.model_dir,
            initial_prompt,
            seed,
            steps,
            max_new_tokens,
            max_layers,
            no_early_stop,
            cli.raw_prompt,
            verbose,
            events_path,
            json,
            ctx,
        ),
        Command::Serve {
            addr,
            seed,
            steps,
            max_layers,
            ctx,
            tool_compact,
        } => match server::run_serve(
            &cli.model_dir,
            &addr,
            ctx,
            seed,
            steps,
            max_layers,
            tool_compact,
        ) {
            Ok(()) => ExitCode::SUCCESS,
            Err(msg) => {
                eprintln!("error: {msg}");
                ExitCode::FAILURE
            }
        },
        Command::Smoketest {
            prompts_path,
            seed,
            steps,
            max_layers,
            filter,
            repeat,
            longctx,
        } => run_smoketest_cmd(
            &cli.model_dir,
            prompts_path.as_deref(),
            seed,
            steps,
            max_layers,
            cli.raw_prompt,
            filter.as_deref(),
            repeat,
            longctx,
        ),
        Command::Quantize { output, profile } => run_quantize(&cli.model_dir, &output, &profile),
        Command::Tokenize(text) => run_tokenize(&cli.model_dir, &text, cli.raw_prompt),
        Command::Gemm { size } => run_gemm(size),
        Command::ProbeDevice => run_probe_device(),
        Command::BenchGemm {
            shapes,
            oracle,
            iters,
        } => run_bench_gemm(&shapes, oracle.as_deref(), iters),
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
        Command::Chat { .. } => ExitCode::FAILURE,
        Command::Serve { .. } => ExitCode::FAILURE,
        Command::Smoketest { .. } => ExitCode::FAILURE,
        Command::Tokenize(_) => ExitCode::FAILURE,
        Command::Gemm { .. } => ExitCode::FAILURE,
        Command::ProbeDevice { .. } => ExitCode::FAILURE,
        Command::ConvertModel { .. } => ExitCode::FAILURE,
        Command::Quantize { .. } => ExitCode::FAILURE,
        Command::BenchGemm { .. } => ExitCode::FAILURE,
        Command::StepSmoke { .. } => ExitCode::FAILURE,
        Command::StepProbe { .. } => ExitCode::FAILURE,
        Command::StepKvCheck { .. } => ExitCode::FAILURE,
        Command::StepKvParity { .. } => ExitCode::FAILURE,
        Command::StepKvBf16Cross { .. } => ExitCode::FAILURE,
        Command::StepAttnProbe { .. } => ExitCode::FAILURE,
        Command::StepLogitsDump { .. } => ExitCode::FAILURE,
        Command::StepBf16OracleLogitsDump { .. } => ExitCode::FAILURE,
        Command::StepLayerProbe { .. } => ExitCode::FAILURE,
        Command::StepAttnDump { .. } => ExitCode::FAILURE,
        Command::StepMoeDump { .. } => ExitCode::FAILURE,
        Command::StepMoeRouteDump { .. } => ExitCode::FAILURE,
        Command::StepMoeBatchedPinDump { .. } => ExitCode::FAILURE,
        Command::StepMoeSingleDump { .. } => ExitCode::FAILURE,
        Command::StepPreambleDump { .. } => ExitCode::FAILURE,
        Command::EmbedRowDump { .. } => ExitCode::FAILURE,
        Command::StepVerify { .. } => ExitCode::FAILURE,
        Command::StepCi { .. } => ExitCode::FAILURE,
        Command::StepParity { .. } => ExitCode::FAILURE,
        Command::BenchStepKernel { .. } => ExitCode::FAILURE,
        Command::BenchGemmFusion { .. } => ExitCode::FAILURE,
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
        prefill_token_ids: None,
        no_early_stop: false,
    }
}

/// MLX parity dumps default to the full 30-layer decoder unless `--layers` is set.
fn layers_for_parity_dump(parity_layers: Option<usize>) -> usize {
    parity_layers.unwrap_or(30).max(1).min(30)
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
    let vocab = crate::config::ModelConfig::load(model_dir)?
        .text_config
        .vocab_size;
    let prompt_len = if kv_len > 0 { kv_len as usize } else { 64 };
    let ids = build_prompt_tokens(model_dir, prompt, prompt_len, vocab, raw_prompt, &[])?;
    eprintln!("step-kernel: prefill {} prompt tokens", ids.len());
    cfg.prefill_token_ids = Some(ids);
    Ok(())
}

#[cfg(target_os = "macos")]
fn run_step_probe_cmd(
    model_dir: &std::path::Path,
    layers: usize,
    kv_len: u32,
    seed: u64,
    max_seq: usize,
    prompt: Option<&str>,
    raw_prompt: bool,
) -> ExitCode {
    use metal::{StepFinishMode, StepSmokeConfig, run_step_probe};

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
        prefill_token_ids: None,
        no_early_stop: false,
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

#[cfg(target_os = "macos")]
fn run_step_logits_dump_cmd(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    layers: usize,
    steps: usize,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
    output: &std::path::Path,
    positions: &str,
    top_k: usize,
) -> ExitCode {
    use metal::{StepSmokeConfig, parse_positions, run_step_logits_dump, write_step_logits_dump};

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-logits-dump requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let mut cfg = StepSmokeConfig {
        layers,
        steps: steps.max(1),
        kv_len: 0,
        seed,
        max_seq,
        finish: metal::StepFinishMode::ForwardOnly,
        prefill_token_ids: None,
        no_early_stop: false,
    };
    if let Err(err) = attach_step_prefill(&mut cfg, model_dir, 0, prompt.as_deref(), raw_prompt) {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    let pos = match parse_positions(positions) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };
    let label = prompt.unwrap_or_else(|| "Hello".to_string());
    match run_step_logits_dump(model_dir, &cfg, &label, &pos, top_k.max(1)) {
        Ok(dump) => {
            if let Err(err) = write_step_logits_dump(output, &dump) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            println!(
                "wrote {} (T={:.4}, {} rows)",
                output.display(),
                dump.temperature,
                dump.rows.len()
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(target_os = "macos")]
fn run_step_bf16_oracle_logits_dump_cmd(
    dgq_dir: &std::path::Path,
    bf16_dir: &std::path::Path,
    prompt: Option<String>,
    layers: usize,
    steps: usize,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
    output: &std::path::Path,
    positions: &str,
    top_k: usize,
    gpu_kv: bool,
) -> ExitCode {
    use metal::{
        StepSmokeConfig, parse_positions, run_step_bf16_oracle_logits_dump,
        run_step_bf16_oracle_logits_dump_gpu_kv, write_step_logits_dump,
    };

    if !dgq::store::looks_like_dgq_dir(dgq_dir) {
        eprintln!(
            "error: step-bf16-logits-dump requires -m pointing at a .dgq directory (prefill)"
        );
        return ExitCode::FAILURE;
    }
    if !bf16_dir.join("config.json").is_file() {
        eprintln!(
            "error: step-bf16-logits-dump --bf16-ref must contain config.json ({})",
            bf16_dir.display()
        );
        return ExitCode::FAILURE;
    }
    let mut cfg = StepSmokeConfig {
        layers,
        steps: steps.max(1),
        kv_len: 0,
        seed,
        max_seq,
        finish: metal::StepFinishMode::ForwardOnly,
        prefill_token_ids: None,
        no_early_stop: false,
    };
    if let Err(err) = attach_step_prefill(&mut cfg, dgq_dir, 0, prompt.as_deref(), raw_prompt) {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    let pos = match parse_positions(positions) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };
    let label = prompt.unwrap_or_else(|| "Hello".to_string());
    let dump_result = if gpu_kv {
        run_step_bf16_oracle_logits_dump_gpu_kv(dgq_dir, bf16_dir, &cfg, &label, &pos, top_k.max(1))
    } else {
        run_step_bf16_oracle_logits_dump(bf16_dir, &cfg, &label, &pos, top_k.max(1))
    };
    match dump_result {
        Ok(dump) => {
            if let Err(err) = write_step_logits_dump(output, &dump) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            println!(
                "wrote {} ({}, T={:.4}, {} rows)",
                output.display(),
                dump.source,
                dump.temperature,
                dump.rows.len()
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(target_os = "macos")]
fn run_step_layer_probe_cmd(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    layers: usize,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
    output: &std::path::Path,
    position: usize,
) -> ExitCode {
    use metal::{StepSmokeConfig, run_step_layer_hidden_dump, write_step_layer_hidden_dump};

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-layer-probe requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let mut cfg = StepSmokeConfig {
        layers,
        steps: 2,
        kv_len: 0,
        seed,
        max_seq,
        finish: metal::StepFinishMode::ForwardOnly,
        prefill_token_ids: None,
        no_early_stop: false,
    };
    if let Err(err) = attach_step_prefill(&mut cfg, model_dir, 0, prompt.as_deref(), raw_prompt) {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    let label = prompt.unwrap_or_else(|| "Hello".to_string());
    match run_step_layer_hidden_dump(model_dir, &cfg, &label, position) {
        Ok(dump) => {
            if let Err(err) = write_step_layer_hidden_dump(output, &dump) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            println!(
                "wrote {} (pos={}, {} checkpoints)",
                output.display(),
                dump.position,
                dump.checkpoints.len()
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(target_os = "macos")]
fn run_step_attn_dump_cmd(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    layers: usize,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
    output: &std::path::Path,
    layer: usize,
    position: usize,
) -> ExitCode {
    use metal::{StepSmokeConfig, run_step_attn_layer_dump, write_step_attn_layer_dump};

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-attn-dump requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let mut cfg = StepSmokeConfig {
        layers,
        steps: 2,
        kv_len: 0,
        seed,
        max_seq,
        finish: metal::StepFinishMode::ForwardOnly,
        prefill_token_ids: None,
        no_early_stop: false,
    };
    if let Err(err) = attach_step_prefill(&mut cfg, model_dir, 0, prompt.as_deref(), raw_prompt) {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    let label = prompt.unwrap_or_else(|| "Hello".to_string());
    match run_step_attn_layer_dump(model_dir, &cfg, &label, layer, position) {
        Ok(dump) => {
            if let Err(err) = write_step_attn_layer_dump(output, &dump) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            println!(
                "wrote {} (layer={}, pos={}, kv_len={})",
                output.display(),
                dump.layer,
                dump.position,
                dump.kv_len
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(target_os = "macos")]
fn run_step_moe_dump_cmd(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    layers: usize,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
    output: &std::path::Path,
    layer: usize,
    position: usize,
) -> ExitCode {
    use metal::{StepSmokeConfig, run_step_moe_layer_dump, write_step_moe_layer_dump};

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-moe-dump requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let mut cfg = StepSmokeConfig {
        layers,
        steps: 2,
        kv_len: 0,
        seed,
        max_seq,
        finish: metal::StepFinishMode::ForwardOnly,
        prefill_token_ids: None,
        no_early_stop: false,
    };
    if let Err(err) = attach_step_prefill(&mut cfg, model_dir, 0, prompt.as_deref(), raw_prompt) {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    let label = prompt.unwrap_or_else(|| "Hello".to_string());
    match run_step_moe_layer_dump(model_dir, &cfg, &label, layer, position) {
        Ok(dump) => {
            if let Err(err) = write_step_moe_layer_dump(output, &dump) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            println!(
                "wrote {} (layer={}, pos={}, kv_len={}, experts={:?})",
                output.display(),
                dump.layer,
                dump.position,
                dump.kv_len,
                dump.experts,
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(target_os = "macos")]
fn run_step_moe_route_dump_cmd(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    layers: usize,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
    output: &std::path::Path,
    layer: usize,
    run_grouped: bool,
) -> ExitCode {
    use metal::{
        StepFinishMode, StepSmokeConfig, print_route_summary, run_step_moe_route_dump,
        write_step_moe_route_dump,
    };

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-moe-route-dump requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let mut cfg = StepSmokeConfig {
        layers,
        steps: 1,
        kv_len: 0,
        seed,
        max_seq,
        finish: StepFinishMode::ForwardOnly,
        prefill_token_ids: None,
        no_early_stop: false,
    };
    if let Err(err) = attach_step_prefill(&mut cfg, model_dir, 0, prompt.as_deref(), raw_prompt) {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    let label = prompt.unwrap_or_else(|| "Hello".to_string());
    match run_step_moe_route_dump(model_dir, &cfg, &label, layer, run_grouped) {
        Ok(dump) => {
            print_route_summary(&dump);
            if let Err(err) = write_step_moe_route_dump(output, &dump) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            println!(
                "wrote {} (layer={}, slots_ok={})",
                output.display(),
                dump.layer,
                dump.slots_ok
            );
            if !dump.slots_ok {
                eprintln!("error: MoE bucketing failed (num_slots={})", dump.num_slots);
                return ExitCode::FAILURE;
            }
            if dump.grouped_dispatched {
                if dump.moe_out_l2.unwrap_or(0.0) < 1e-6 {
                    eprintln!("error: MoE produced zero moe_out");
                    return ExitCode::FAILURE;
                }
                if dump.moe_out_gpu_cpu_cos.unwrap_or(1.0) < 0.99 {
                    eprintln!(
                        "error: moe_out vs CPU oracle cos={:.6} (need >= 0.99, style={})",
                        dump.moe_out_gpu_cpu_cos.unwrap_or(0.0),
                        dump.moe_style
                    );
                    return ExitCode::FAILURE;
                }
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
fn run_step_moe_batched_pin_dump_cmd(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    layers: usize,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
    output: &std::path::Path,
    layer: usize,
) -> ExitCode {
    use metal::{
        StepFinishMode, StepSmokeConfig, print_batched_pin_summary, run_step_moe_batched_pin_dump,
        write_step_moe_batched_pin_dump,
    };

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-moe-batched-pin-dump requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let mut cfg = StepSmokeConfig {
        layers,
        steps: 1,
        kv_len: 0,
        seed,
        max_seq,
        finish: StepFinishMode::ForwardOnly,
        prefill_token_ids: None,
        no_early_stop: false,
    };
    if let Err(err) = attach_step_prefill(&mut cfg, model_dir, 0, prompt.as_deref(), raw_prompt) {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    let label = prompt.unwrap_or_else(|| "Hello".to_string());
    match run_step_moe_batched_pin_dump(model_dir, &cfg, &label, layer) {
        Ok(dump) => {
            print_batched_pin_summary(&dump);
            if let Err(err) = write_step_moe_batched_pin_dump(output, &dump) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            println!(
                "wrote {} (layer={}, gate_up_gemm_cos={:.6}, swiglu_post_cos={:.6}, swiglu_isolated_cos={:.6})",
                output.display(),
                dump.layer,
                dump.stages.gate_up_gemm,
                dump.stages.swiglu_post,
                dump.stages.swiglu_isolated,
            );
            let fail_stage = [
                ("gate_up_gemm", dump.stages.gate_up_gemm),
                ("swiglu_post", dump.stages.swiglu_post),
                ("swiglu_isolated", dump.stages.swiglu_isolated),
                ("down", dump.stages.down),
                ("scatter", dump.stages.scatter),
            ]
            .into_iter()
            .find(|(_, cos)| *cos < 0.99);
            if let Some((name, cos)) = fail_stage {
                eprintln!("error: batched pin stage `{name}` cos={cos:.6} (need >= 0.99)");
                return ExitCode::FAILURE;
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
fn run_step_moe_single_dump_cmd(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    layers: usize,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
    output: &std::path::Path,
    layer: usize,
    position: usize,
    expert: u32,
) -> ExitCode {
    use metal::{
        StepSmokeConfig, run_step_moe_single_expert_dump, write_step_moe_single_expert_dump,
    };

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-moe-single-dump requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let mut cfg = StepSmokeConfig {
        layers,
        steps: 2,
        kv_len: 0,
        seed,
        max_seq,
        finish: metal::StepFinishMode::ForwardOnly,
        prefill_token_ids: None,
        no_early_stop: false,
    };
    if let Err(err) = attach_step_prefill(&mut cfg, model_dir, 0, prompt.as_deref(), raw_prompt) {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    let label = prompt.unwrap_or_else(|| "Hello".to_string());
    match run_step_moe_single_expert_dump(model_dir, &cfg, &label, layer, position, expert) {
        Ok(dump) => {
            if let Err(err) = write_step_moe_single_expert_dump(output, &dump) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            let mut dot = 0.0f64;
            let mut na = 0.0f64;
            let mut nb = 0.0f64;
            for (a, b) in dump.gpu_out.iter().zip(dump.cpu_out.iter()) {
                let af = *a as f64;
                let bf = *b as f64;
                dot += af * bf;
                na += af * af;
                nb += bf * bf;
            }
            let cos = if na > 0.0 && nb > 0.0 {
                (dot / (na.sqrt() * nb.sqrt())) as f32
            } else {
                0.0
            };
            println!(
                "wrote {} (layer={}, pos={}, expert={}, kv_len={}, gpu_vs_cpu_cos={:.6})",
                output.display(),
                dump.layer,
                dump.position,
                dump.expert_id,
                dump.kv_len,
                cos,
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(target_os = "macos")]
fn run_step_preamble_dump_cmd(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    layers: usize,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
    output: &std::path::Path,
    position: usize,
) -> ExitCode {
    use metal::{StepSmokeConfig, run_step_preamble_dump, write_step_preamble_dump};

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-preamble-dump requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let mut cfg = StepSmokeConfig {
        layers,
        steps: 2,
        kv_len: 0,
        seed,
        max_seq,
        finish: metal::StepFinishMode::ForwardOnly,
        prefill_token_ids: None,
        no_early_stop: false,
    };
    if let Err(err) = attach_step_prefill(&mut cfg, model_dir, 0, prompt.as_deref(), raw_prompt) {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    let label = prompt.unwrap_or_else(|| "Hello".to_string());
    match run_step_preamble_dump(model_dir, &cfg, &label, position) {
        Ok(dump) => {
            if let Err(err) = write_step_preamble_dump(output, &dump) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            println!(
                "wrote {} (pos={}, token={})",
                output.display(),
                dump.position,
                dump.canvas_token
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run_embed_row_dump_cmd(
    model_dir: &std::path::Path,
    token: u32,
    layers: usize,
    max_seq: usize,
    prompt: Option<String>,
    raw_prompt: bool,
    output: &std::path::Path,
    bf16_ref_dir: Option<&std::path::Path>,
    gpu: bool,
) -> ExitCode {
    use dgq::{run_embed_row_dump, write_embed_row_dump};

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: embed-row-dump requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let hidden = match crate::config::ModelConfig::load(model_dir) {
        Ok(c) => c.text_config.hidden_size,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    #[cfg(target_os = "macos")]
    let gpu_scaled = if gpu {
        use metal::{StepFinishMode, StepSmokeConfig, run_embed_row_gpu};
        let mut cfg = StepSmokeConfig {
            layers,
            steps: 1,
            kv_len: 0,
            seed: 0,
            max_seq,
            finish: StepFinishMode::ForwardOnly,
            prefill_token_ids: None,
            no_early_stop: false,
        };
        if let Err(err) = attach_step_prefill(&mut cfg, model_dir, 0, prompt.as_deref(), raw_prompt)
        {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
        match run_embed_row_gpu(model_dir, &cfg, token) {
            Ok(v) => Some(v),
            Err(err) => {
                eprintln!("error: gpu embed row: {err}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };

    match run_embed_row_dump(model_dir, token, hidden, bf16_ref_dir, gpu_scaled) {
        Ok(dump) => {
            if let Err(err) = write_embed_row_dump(output, &dump) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            println!(
                "wrote {} (token={}, scale={:.6}, cpu_l2={:.2})",
                output.display(),
                dump.token,
                dump.scale_f32,
                dump.dequant_scaled_l2
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
fn run_step_kv_bf16_cross_cmd(
    dgq_dir: &std::path::Path,
    bf16_dir: &std::path::Path,
    prompt: Option<String>,
    prompt_len: usize,
    layers: usize,
    _seed: u64,
    max_seq: usize,
    raw_prompt: bool,
) -> ExitCode {
    use metal::run_step_kv_bf16_cross_parity;

    if !dgq::store::looks_like_dgq_dir(dgq_dir) {
        eprintln!("error: step-kv-bf16-cross requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    if !bf16_dir.join("config.json").is_file() {
        eprintln!(
            "error: step-kv-bf16-cross --bf16-ref must contain config.json ({})",
            bf16_dir.display()
        );
        return ExitCode::FAILURE;
    }
    let vocab = match crate::config::ModelConfig::load(dgq_dir) {
        Ok(c) => c.text_config.vocab_size,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };
    let token_ids = match build_prompt_tokens(
        dgq_dir,
        prompt.as_deref(),
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
    eprintln!(
        "step-kv-bf16-cross: {} prompt tokens, layers={}",
        token_ids.len(),
        layers.max(1).min(30)
    );
    match run_step_kv_bf16_cross_parity(dgq_dir, bf16_dir, &token_ids, layers, max_seq.max(64)) {
        Ok(r) => {
            println!("step-kv-bf16-cross:");
            println!("  kv_len:              {}", r.kv_len);
            println!("  layers:              {}", r.layers);
            println!("  gpu_prefix_l0:       {:.6}", r.gpu_prefix_max_l0);
            println!("  cpu_prefix_l0:       {:.6}", r.cpu_prefix_max_l0);
            println!(
                "  max_kv_diff:         {:.6} (layer {} pos {})",
                r.max_kv_diff, r.max_kv_diff_layer, r.max_kv_diff_pos
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(target_os = "macos")]
fn run_step_kv_parity_cmd(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    prompt_len: usize,
    layers: usize,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
) -> ExitCode {
    use metal::run_step_kv_parity;

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-kv-parity requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let vocab = match crate::config::ModelConfig::load(model_dir) {
        Ok(c) => c.text_config.vocab_size,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };
    let token_ids = match build_prompt_tokens(
        model_dir,
        prompt.as_deref(),
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
    eprintln!(
        "step-kv-parity: {} prompt tokens, layers={}",
        token_ids.len(),
        layers.max(1).min(30)
    );
    match run_step_kv_parity(model_dir, &token_ids, layers, max_seq.max(64), seed) {
        Ok(r) => {
            println!("step-kv-parity:");
            println!("  kv_len:              {}", r.kv_len);
            println!("  layers:              {}", r.layers);
            println!("  prefix_l0:           {:.6}", r.prefix_max_l0);
            println!("  prefix_l0_b:         {:.6}", r.prefix_max_l0_b);
            println!(
                "  max_kv_diff:         {:.6} (layer {} pos {})",
                r.max_kv_diff, r.max_kv_diff_layer, r.max_kv_diff_pos
            );
            println!("  min_ent:             {:.4}", r.min_ent);
            println!("  min_ent_b:           {:.4}", r.min_ent_b);
            println!("  min_ent_diff:        {:.4}", r.min_ent_diff);
            println!("  entropy_pass:        {}", r.entropy_pass);
            println!("  ln_vocab:            {:.4}", r.ln_vocab);
            println!("  pass:                {}", r.pass);
            if !r.entropy_pass {
                eprintln!(
                    "  note: KV matched but forward entropy diverged — use step-moe-route-dump on denoise path"
                );
            }
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

#[cfg(target_os = "macos")]
fn run_step_attn_probe_cmd(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    prompt_len: usize,
    layer: usize,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
) -> ExitCode {
    use metal::run_step_attn_probe;

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-attn-probe requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let vocab = match crate::config::ModelConfig::load(model_dir) {
        Ok(c) => c.text_config.vocab_size,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };
    let token_ids = match build_prompt_tokens(
        model_dir,
        prompt.as_deref(),
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
    eprintln!(
        "step-attn-probe: {} prompt tokens, layer={layer}, seed={seed}",
        token_ids.len()
    );
    match run_step_attn_probe(model_dir, &token_ids, layer, seed, max_seq.max(512)) {
        Ok(r) => {
            let ln_t = r.attn_keys_t as f32;
            let ln_uniform = ln_t.ln();
            println!("step-attn-probe:");
            println!("  kv_len (prefix):        {}", r.kv_len);
            println!("  canvas_len:             {}", r.canvas_len);
            println!(
                "  attn keys T:            {} (= kv_len + canvas_len)",
                r.attn_keys_t
            );
            println!("  layer:                  {}", r.layer);
            println!("  monolithic K max L0:    {:.4}", r.k_plane_max_l0);
            println!("  monolithic V max L0:    {:.4}", r.v_plane_max_l0);
            println!(
                "  q_norm weight mean|w|: {:.4}  rms: {:.4}",
                r.q_norm_weight_mean_abs, r.q_norm_weight_rms
            );
            println!(
                "  k_norm weight mean|w|: {:.4}  rms: {:.4}",
                r.k_norm_weight_mean_abs, r.k_norm_weight_rms
            );
            println!(
                "  CPU Q·K raw max/min:    {:.2} / {:.2}",
                r.cpu_raw_dot_max, r.cpu_raw_dot_min
            );
            println!(
                "  CPU softmax row ent:    {:.4} nats (ln T = {:.4}, active keys/row = {})",
                r.cpu_mean_softmax_entropy, ln_uniform, r.cpu_keys_per_row
            );
            println!("  CPU softmax max prob:   {:.4}", r.cpu_mean_max_prob);
            println!(
                "  CPU weight sum/row:     {:.6} (expect 1.0)",
                r.cpu_mean_weight_sum
            );
            println!(
                "  CPU Q/K head RMS:       {:.4} / {:.4}",
                r.cpu_q_head_rms, r.cpu_k_head_rms
            );
            if r.cpu_mean_softmax_entropy > ln_uniform + 1e-3 {
                eprintln!(
                    "  WARNING: entropy {:.4} > ln(T) {:.4} — probe bug or invalid softmax",
                    r.cpu_mean_softmax_entropy, ln_uniform
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

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
fn run_chat_quality_gate(model_dir: &std::path::Path, layers: usize) -> ExitCode {
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
fn run_step_parity_cmd(
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
fn print_encode_subprofile(p: &metal::EncodeSubProfileResult) {
    use metal::{LayerEncodeSubProfile, MoeEncodeSubProfile};
    let layers = p.layers.max(1) as u32;
    let layer_total = p.layer.total();
    let moe_total = p.moe.total();
    let grand = layer_total + moe_total;
    let per_l = |d: std::time::Duration| d / layers;
    let pct = |d: std::time::Duration| {
        if grand.is_zero() {
            0.0
        } else {
            100.0 * d.as_secs_f64() / grand.as_secs_f64()
        }
    };
    let print_layer_rows = |label: &str, prof: &LayerEncodeSubProfile| {
        let rows: [(&str, std::time::Duration); 11] = [
            ("qkv_gemm", prof.qkv_gemm),
            ("qk_rope_kv", prof.qk_rope_kv),
            ("attention", prof.attention),
            ("o_proj_gemm", prof.o_proj_gemm),
            ("o_proj_tail", prof.o_proj_tail),
            ("dense_pre_norm", prof.dense_pre_norm),
            ("dense_gate_up", prof.dense_gate_up),
            ("dense_glu", prof.dense_glu),
            ("dense_down", prof.dense_down),
            ("dense_post_norm", prof.dense_post_norm),
            ("router", prof.router),
        ];
        println!(
            "{label} (total {:.2?}, {:.1}%):",
            prof.total(),
            pct(prof.total())
        );
        let mut ranked: Vec<_> = rows.iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        for (name, d) in ranked {
            println!(
                "  {:16} {:.2?}  ({:.1}%, {:.2?}/layer)",
                name,
                d,
                pct(*d),
                per_l(*d)
            );
        }
    };
    let print_moe_rows = |label: &str, prof: &MoeEncodeSubProfile| {
        let rows: [(&str, std::time::Duration); 7] = [
            ("half_to_f32", prof.half_to_f32),
            ("gather", prof.gather),
            ("gate_up", prof.gate_up),
            ("swiglu", prof.swiglu),
            ("down", prof.down),
            ("scatter", prof.scatter),
            ("post", prof.post),
        ];
        println!(
            "{label} (total {:.2?}, {:.1}%):",
            prof.total(),
            pct(prof.total())
        );
        let mut ranked: Vec<_> = rows.iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        for (name, d) in ranked {
            println!(
                "  {:16} {:.2?}  ({:.1}%, {:.2?}/layer)",
                name,
                d,
                pct(*d),
                per_l(*d)
            );
        }
    };
    println!("bench-step-kernel layer-profile ok");
    println!("  compile:       {:.2?}", p.compile);
    println!("  layers:        {}", p.layers);
    print_layer_rows("encode_layer", &p.layer);
    print_moe_rows("moe_grouped+post", &p.moe);
    println!("  grand_total:   {:.2?}", grand);
}

#[cfg(target_os = "macos")]
fn run_bench_step_kernel_cmd(
    model_dir: &std::path::Path,
    layers: usize,
    kv_len: u32,
    seed: u64,
    max_seq: usize,
    iters: usize,
    forward_only: bool,
    profile: bool,
    profile_steps: usize,
    layer_profile: bool,
) -> ExitCode {
    use metal::bench_step_kernel;

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: bench-step-kernel requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let cfg = step_kernel_config(layers, kv_len, seed, max_seq, forward_only);
    if layer_profile {
        use metal::bench_step_kernel_encode_subprofile;
        eprintln!(
            "bench-step-kernel --layer-profile: layers={layers} kv_len={kv_len} forward_only={forward_only}"
        );
        match bench_step_kernel_encode_subprofile(model_dir, cfg) {
            Ok(p) => {
                print_encode_subprofile(&p);
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("error: {err}");
                ExitCode::FAILURE
            }
        }
    } else if profile_steps > 0 {
        use metal::bench_step_kernel_profile_steps;
        eprintln!(
            "bench-step-kernel --profile-steps {profile_steps}: layers={layers} kv_len={kv_len}"
        );
        match bench_step_kernel_profile_steps(model_dir, &cfg, profile_steps) {
            Ok(rows) => {
                println!(
                    "bench-step-kernel profile-steps ok ({} forwards)",
                    rows.len()
                );
                for (canvas_step, p) in rows {
                    let sc = if canvas_step == 0 { "no SC" } else { "SC" };
                    let per_l = |d: std::time::Duration| d / p.layers.max(1) as u32;
                    let pct =
                        |d: std::time::Duration| 100.0 * d.as_secs_f64() / p.total.as_secs_f64();
                    println!("--- canvas st.step={canvas_step} ({sc}) ---");
                    if canvas_step == 0 {
                        println!("  compile:       {:.2?}", p.compile);
                        println!("  block_format:  {:?}", p.block_format);
                        println!("  layers:        {}", p.layers);
                    }
                    println!(
                        "  preamble:      {:.2?}  ({:.1}%)",
                        p.preamble,
                        pct(p.preamble)
                    );
                    println!(
                        "  pre_moe:       {:.2?}  ({:.1}%, {:.2?}/layer)",
                        p.layer_pre_moe,
                        pct(p.layer_pre_moe),
                        per_l(p.layer_pre_moe)
                    );
                    println!(
                        "  moe_grouped:   {:.2?}  ({:.1}%, {:.2?}/layer)",
                        p.layer_moe,
                        pct(p.layer_moe),
                        per_l(p.layer_moe)
                    );
                    println!(
                        "  moe_post:      {:.2?}  ({:.1}%, {:.2?}/layer)",
                        p.layer_post,
                        pct(p.layer_post),
                        per_l(p.layer_post)
                    );
                    println!("  finish:        {:.2?}  ({:.1}%)", p.finish, pct(p.finish));
                    println!("  total:         {:.2?}", p.total);
                }
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("error: {err}");
                ExitCode::FAILURE
            }
        }
    } else if profile {
        use metal::bench_step_kernel_profile;
        eprintln!(
            "bench-step-kernel --profile: layers={layers} kv_len={kv_len} forward_only={forward_only}"
        );
        match bench_step_kernel_profile(model_dir, cfg) {
            Ok(p) => {
                let per_l = |d: std::time::Duration| d / p.layers.max(1) as u32;
                let pct = |d: std::time::Duration| 100.0 * d.as_secs_f64() / p.total.as_secs_f64();
                println!("bench-step-kernel profile ok");
                println!("  compile:       {:.2?}", p.compile);
                println!("  block_format:  {:?}", p.block_format);
                println!("  layers:        {}", p.layers);
                println!(
                    "  preamble:      {:.2?}  ({:.1}%)",
                    p.preamble,
                    pct(p.preamble)
                );
                println!(
                    "  pre_moe (attn+dense+router): {:.2?}  ({:.1}%, {:.2?}/layer)",
                    p.layer_pre_moe,
                    pct(p.layer_pre_moe),
                    per_l(p.layer_pre_moe)
                );
                println!(
                    "  moe_grouped:   {:.2?}  ({:.1}%, {:.2?}/layer)",
                    p.layer_moe,
                    pct(p.layer_moe),
                    per_l(p.layer_moe)
                );
                println!(
                    "  moe_post:      {:.2?}  ({:.1}%, {:.2?}/layer)",
                    p.layer_post,
                    pct(p.layer_post),
                    per_l(p.layer_post)
                );
                println!("  finish:        {:.2?}  ({:.1}%)", p.finish, pct(p.finish));
                println!("  total:         {:.2?}", p.total);
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("error: {err}");
                ExitCode::FAILURE
            }
        }
    } else {
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
}

#[cfg(target_os = "macos")]
fn run_bench_gemm_fusion_cmd(
    model_dir: &std::path::Path,
    layers: usize,
    kv_len: u32,
    seed: u64,
    max_seq: usize,
    iters: usize,
) -> ExitCode {
    use metal::bench_fused_gemm_dispatches;

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: bench-gemm-fusion requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let cfg = step_kernel_config(layers, kv_len, seed, max_seq, true);
    eprintln!(
        "bench-gemm-fusion: layers={layers} kv_len={kv_len} iters={iters} (QKV + gate/up GEMM isolation)"
    );
    match bench_fused_gemm_dispatches(model_dir, cfg, iters) {
        Ok(r) => {
            let pct = |a: std::time::Duration, b: std::time::Duration| {
                if b.is_zero() {
                    0.0
                } else {
                    100.0 * (1.0 - a.as_secs_f64() / b.as_secs_f64())
                }
            };
            let per_l = |d: std::time::Duration| d / r.layers.max(1) as u32;
            println!("bench-gemm-fusion ok");
            println!("  compile:  {:.2?}", r.compile);
            println!("  layers:   {}", r.layers);
            println!("  iters:    {}", r.iters);
            println!(
                "  dispatches/pass: qkv stacked={} split={} gate_up stacked=1/L split=2/L",
                r.qkv_stacked_dispatches_per_pass, r.qkv_split_dispatches_per_pass
            );
            println!("--- QKV GEMM only (per-layer rmsnorm prep + timed GEMM submit) ---");
            println!(
                "  stacked: {:.2?}  ({:.2?}/layer)",
                r.qkv_gemm_stacked,
                per_l(r.qkv_gemm_stacked)
            );
            println!(
                "  split:   {:.2?}  ({:.2?}/layer)",
                r.qkv_gemm_split,
                per_l(r.qkv_gemm_split)
            );
            println!(
                "  delta:   {:+.1}% stacked vs split",
                pct(r.qkv_gemm_stacked, r.qkv_gemm_split)
            );
            println!("--- gate/up GEMM only (per-layer rmsnorm prep + timed GEMM submit) ---");
            println!(
                "  stacked: {:.2?}  ({:.2?}/layer)",
                r.gate_up_gemm_stacked,
                per_l(r.gate_up_gemm_stacked)
            );
            println!(
                "  split:   {:.2?}  ({:.2?}/layer)",
                r.gate_up_gemm_split,
                per_l(r.gate_up_gemm_split)
            );
            println!(
                "  delta:   {:+.1}% stacked vs split",
                pct(r.gate_up_gemm_stacked, r.gate_up_gemm_split)
            );
            println!("--- batched pass (1 CB, interleaved rmsnorm+GEMM per layer) ---");
            println!(
                "  qkv stacked: {:.2?}  split: {:.2?}  ({:+.1}%)",
                r.qkv_batched_stacked,
                r.qkv_batched_split,
                pct(r.qkv_batched_stacked, r.qkv_batched_split)
            );
            println!(
                "  gate_up stacked: {:.2?}  split: {:.2?}  ({:+.1}%)",
                r.gate_up_batched_stacked,
                r.gate_up_batched_split,
                pct(r.gate_up_batched_stacked, r.gate_up_batched_split)
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(target_os = "macos")]
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

fn run_quantize(source_dir: &std::path::Path, output: &std::path::Path, profile: &str) -> ExitCode {
    use dgq::layout::QuantProfile;
    use dgq::{QuantizeOptions, quantize_model};

    let profile_name = profile;
    let profile = match profile {
        "q4" => QuantProfile::Q4,
        "q5" => QuantProfile::Q5,
        "q6" => QuantProfile::Q6,
        "nvfp4" => QuantProfile::Nvfp4,
        other => {
            eprintln!("error: unknown profile {other} (use q4, q5, q6, or nvfp4)");
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
            println!("  q6 tensors:    {}", summary.q6_tensors);
            println!("  nvfp4 tensors: {}", summary.nvfp4_tensors);
            println!("  q8 tensors:    {}", summary.q8_tensors);
            println!("  raw tensors:   {}", summary.raw_tensors);
            println!("  elapsed:       {:.2?}", started.elapsed());
            println!(
                "  manifest:      {}/{}",
                out_dir.display(),
                dgq::layout::MANIFEST_FILE
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
fn run_bench_gemm(shapes: &str, oracle: Option<&str>, iters: usize) -> ExitCode {
    use metal::{
        bench_custom_kernel, bench_gemm_bf16, bench_gemm_block_q4, bench_mpsgraph_oracle,
        parse_shapes, print_bench_rows,
    };
    // `--shapes sparse`: block-sparse MoE bench at the production route
    // distributions (MxKxN comes from the model shapes; the spec is ignored).
    if shapes == "sparse" {
        return match metal::bench_gemm_tunable_sparse(iters) {
            Ok(rows) => {
                print_bench_rows(&rows);
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("error: {err}");
                ExitCode::FAILURE
            }
        };
    }
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
    match bench_gemm_block_q4(&parsed, iters) {
        Ok(mut q4) => rows.append(&mut q4),
        Err(err) => eprintln!("warning: gemm_block bench: {err}"),
    }
    match bench_gemm_bf16(&parsed, iters) {
        Ok(mut bf16) => rows.append(&mut bf16),
        Err(err) => eprintln!("warning: gemm_bf16 bench: {err}"),
    }
    match metal::bench_gemm_tunable(&parsed, iters) {
        Ok(mut t) => rows.append(&mut t),
        Err(err) => eprintln!("warning: gemm_tunable bench: {err}"),
    }
    if matches!(oracle, Some("mps") | Some("mpsgraph")) {
        match bench_mpsgraph_oracle(&parsed, iters) {
            Ok(mut mps) => rows.append(&mut mps),
            Err(err) => {
                eprintln!("warning: {err}");
            }
        }
    }
    print_bench_rows(&rows);
    ExitCode::SUCCESS
}

fn run_convert_model(source_dir: &std::path::Path, output_dir: &std::path::Path) -> ExitCode {
    use pack::{ConvertOptions, convert_model};
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
            println!(
                "  blob size:         {:.2} GiB",
                summary.blob_bytes as f64 / (1024.0_f64.powi(3))
            );
            println!("  gemm transposed:   {}", summary.transposed_gemm);
            println!("  expert transposed: {}", summary.transposed_experts);
            println!("  raw copied:        {}", summary.raw_copied);
            println!(
                "  manifest:          {}/{}",
                output_dir.display(),
                pack::layout::MANIFEST_FILE
            );
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

/// Layer count for generate paths: `--layers` override, else full `num_hidden_layers` from config.
fn resolve_model_layers(
    model_dir: &std::path::Path,
    override_layers: Option<usize>,
) -> Result<usize, safetensors::Error> {
    let cfg = crate::config::ModelConfig::load(model_dir)?;
    let n = cfg.text_config.num_hidden_layers.max(1);
    Ok(override_layers.unwrap_or(n).max(1).min(n))
}

fn parse_cli() -> Cli {
    let mut args = env::args().skip(1);
    let mut model_dir = PathBuf::from("model/transformer");
    let mut positional = Vec::new();
    let mut seed = 42u64;
    let mut seed_explicit = false;
    let mut chat_ctx: Option<usize> = None;
    let mut serve_addr: String = "127.0.0.1:8080".to_string();
    let mut serve_tool_compact = false;
    let mut steps_override: Option<usize> = None;
    let mut prompt_len = 8usize;
    let mut max_new_tokens = 256usize;
    let mut max_new_tokens_explicit = false;
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
    let mut smoke_filter: Option<String> = None;
    let mut smoke_repeat: usize = 1;
    let mut smoke_longctx = false;
    let mut compare_cpu = false;
    let mut write_golden: Option<String> = None;
    let mut write_trace: Option<PathBuf> = None;
    let mut no_early_stop = false;
    let mut chat_verbose = false;
    let mut chat_events_path: Option<PathBuf> = None;
    let mut chat_json = false;
    let mut kernel_assert = false;
    let mut kernel_debug_deep = false;
    let mut output_dir: Option<PathBuf> = None;
    let mut quant_profile = String::from("q4");
    let mut bench_gemm_shapes = String::from("256x2816x2816,33x2816x1408");
    let mut bench_gemm_oracle: Option<String> = None;
    let mut bench_prefill_len = 1usize;
    let mut bench_repeat_prefill = false;
    let mut step_kv_len = 0u32;
    let mut step_max_seq = 512usize;
    let mut step_forward_only = false;
    let mut step_profile = false;
    let mut step_profile_steps = 0usize;
    let mut step_layer_profile = false;
    let mut step_logit_positions = String::new();
    let mut step_logit_top_k = 10usize;
    let mut step_layer_position = 129usize;
    let mut step_attn_layer = 2usize;
    let mut step_moe_expert = 18u32;
    let mut step_moe_route_grouped = true;
    let mut embed_row_token = 71153u32;
    let mut embed_row_gpu = false;
    let mut bf16_ref_dir: Option<PathBuf> = None;
    let mut step_gpu_kv = false;
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
                    seed_explicit = true;
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
                    max_new_tokens_explicit = true;
                }
            }
            "--compare-cpu" => compare_cpu = true,
            "--repeat-prefill" => bench_repeat_prefill = true,
            "--no-early-stop" => no_early_stop = true,
            "--verbose" => chat_verbose = true,
            "--json" => chat_json = true,
            "--ctx" => {
                if let Some(v) = args.next() {
                    chat_ctx = Some(v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --ctx");
                        std::process::exit(2);
                    }));
                }
            }
            "--events" => {
                if let Some(v) = args.next() {
                    chat_events_path = Some(PathBuf::from(v));
                }
            }
            "--addr" => {
                if let Some(v) = args.next() {
                    serve_addr = v;
                }
            }
            "--tool-compact" => serve_tool_compact = true,
            "--assert" => kernel_assert = true,
            "--debug-deep" => kernel_debug_deep = true,
            "--gpu-kv" => step_gpu_kv = true,
            "--skip-grouped" => step_moe_route_grouped = false,
            "--write-golden" => {
                if let Some(v) = args.next() {
                    write_golden = Some(v);
                }
            }
            "--write-trace" => {
                if let Some(v) = args.next() {
                    write_trace = Some(PathBuf::from(v));
                }
            }
            "--golden" => {
                if let Some(v) = args.next() {
                    golden_name = Some(v);
                }
            }
            "--filter" => {
                if let Some(v) = args.next() {
                    smoke_filter = Some(v);
                }
            }
            "--repeat" => {
                if let Some(v) = args.next() {
                    smoke_repeat = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --repeat");
                        std::process::exit(2);
                    });
                }
            }
            "--longctx" => smoke_longctx = true,
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
            "--step-profile" => step_profile = true,
            "--layer-profile" => step_layer_profile = true,
            "--profile-steps" => {
                if let Some(v) = args.next() {
                    step_profile_steps = v.parse().unwrap_or_else(|_| {
                        eprintln!("error: --profile-steps requires a positive integer");
                        std::process::exit(1);
                    });
                }
            }
            "--logit-positions" => {
                if let Some(v) = args.next() {
                    step_logit_positions = v;
                }
            }
            "--logit-top-k" => {
                if let Some(v) = args.next() {
                    step_logit_top_k = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --logit-top-k");
                        std::process::exit(2);
                    });
                }
            }
            "--layer-position" => {
                if let Some(v) = args.next() {
                    step_layer_position = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --layer-position");
                        std::process::exit(2);
                    });
                }
            }
            "--attn-layer" => {
                if let Some(v) = args.next() {
                    step_attn_layer = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --attn-layer");
                        std::process::exit(2);
                    });
                }
            }
            "--expert" => {
                if let Some(v) = args.next() {
                    step_moe_expert = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --expert");
                        std::process::exit(2);
                    });
                }
            }
            "--embed-token" => {
                if let Some(v) = args.next() {
                    embed_row_token = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --embed-token");
                        std::process::exit(2);
                    });
                }
            }
            "--embed-gpu" => embed_row_gpu = true,
            "--bf16-ref" => {
                if let Some(v) = args.next() {
                    bf16_ref_dir = Some(PathBuf::from(v));
                }
            }
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

    let steps_production = resolve_steps(steps_override, false);
    let steps_parity = resolve_steps(steps_override, true);

    let command = match positional.first().map(String::as_str) {
        None => default_generate_command(
            &model_dir,
            prompt,
            seed,
            steps_production,
            prompt_len,
            if max_new_tokens_explicit {
                max_new_tokens
            } else {
                2048
            },
            parity_layers,
            no_early_stop,
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
        // `ask` is the production generation command (formerly generate-monolithic,
        // kept as an alias). `generate`/`generate-gpu` also route here — the
        // non-monolithic generate/decoder/prefill surface is retired; the f32
        // engine survives only as the step-parity validation oracle.
        Some("ask") | Some("generate-monolithic") | Some("generate") | Some("generate-gpu") => {
            Command::GenerateMonolithic {
                prompt: prompt.clone(),
                seed,
                steps: steps_production,
                prompt_len,
                // Behave like a single-turn chat: allow a full multi-block reply
                // unless the caller pinned a smaller budget with --max-new-tokens.
                max_new_tokens: if max_new_tokens_explicit {
                    max_new_tokens
                } else {
                    2048
                },
                max_layers: parity_layers,
                no_early_stop,
                kernel_assert,
                kernel_debug_deep,
                write_golden,
                write_trace,
                verbose: chat_verbose,
            }
        }
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
            verbose: chat_verbose,
            events_path: chat_events_path.clone(),
            json: chat_json,
            ctx: chat_ctx,
        },
        Some("serve") => Command::Serve {
            addr: serve_addr.clone(),
            seed,
            steps: steps_production,
            max_layers: parity_layers,
            ctx: chat_ctx.unwrap_or(8192),
            tool_compact: serve_tool_compact,
        },
        Some("smoketest") => Command::Smoketest {
            prompts_path: positional.get(1).map(PathBuf::from),
            seed: seed_explicit.then_some(seed),
            steps: steps_production,
            max_layers: parity_layers,
            filter: smoke_filter.clone(),
            repeat: smoke_repeat.max(1),
            longctx: smoke_longctx,
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
        Some("probe-device") => Command::ProbeDevice,
        Some("bench-gemm") => Command::BenchGemm {
            shapes: bench_gemm_shapes,
            oracle: bench_gemm_oracle,
            iters: bench_iters.max(1),
        },
        Some("quantize") => {
            let out = output_dir.unwrap_or_else(|| {
                eprintln!("usage: diffgemma-mps quantize -o OUTPUT_DIR -m SOURCE [--profile q4|q5|q6|nvfp4]");
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
        Some("step-kv-parity") => Command::StepKvParity {
            prompt: prompt.clone(),
            prompt_len,
            layers: bench_layers.max(1).min(30),
            seed,
            max_seq: step_max_seq.max(64),
            raw_prompt,
        },
        Some("step-kv-bf16-cross") => {
            let bf16_ref_dir = bf16_ref_dir.unwrap_or_else(|| {
                let p = PathBuf::from("model/transformer");
                if p.join("config.json").is_file() {
                    p
                } else {
                    eprintln!("error: step-kv-bf16-cross requires --bf16-ref or model/transformer");
                    std::process::exit(2);
                }
            });
            Command::StepKvBf16Cross {
                prompt: prompt.clone(),
                prompt_len,
                layers: layers_for_parity_dump(parity_layers),
                seed,
                max_seq: step_max_seq.max(64),
                raw_prompt,
                bf16_ref_dir,
            }
        }
        Some("step-attn-probe") => Command::StepAttnProbe {
            prompt: prompt.clone(),
            prompt_len,
            layer: bench_layers,
            seed,
            max_seq: step_max_seq.max(512),
            raw_prompt,
        },
        Some("step-logits-dump") => {
            let output = output_dir.unwrap_or_else(|| {
                eprintln!(
                    "usage: diffgemma-mps step-logits-dump -m MODEL -o OUT.json [-p Hello] [--layers 30] [--steps 2] [--seed 42] [--logit-positions 0,43,58] [--logit-top-k 10]"
                );
                std::process::exit(2);
            });
            Command::StepLogitsDump {
                prompt: prompt.clone(),
                layers: layers_for_parity_dump(parity_layers),
                steps: steps_parity.max(1),
                seed,
                max_seq: step_max_seq.max(64),
                raw_prompt,
                output,
                positions: step_logit_positions,
                top_k: step_logit_top_k,
            }
        }
        Some("step-bf16-logits-dump") => {
            let output = output_dir.unwrap_or_else(|| {
                eprintln!(
                    "usage: diffgemma-mps step-bf16-logits-dump -m DGQ_DIR --bf16-ref BF16_DIR -o OUT.json [-p Hello] [--layers 30] [--steps 2] [--seed 42] [--logit-positions 0,1] [--gpu-kv]"
                );
                std::process::exit(2);
            });
            let bf16_ref_dir = bf16_ref_dir.unwrap_or_else(|| {
                let p = PathBuf::from("model/transformer");
                if p.join("config.json").is_file() {
                    p
                } else {
                    eprintln!(
                        "error: step-bf16-logits-dump requires --bf16-ref or model/transformer"
                    );
                    std::process::exit(2);
                }
            });
            Command::StepBf16OracleLogitsDump {
                prompt: prompt.clone(),
                layers: layers_for_parity_dump(parity_layers),
                steps: steps_parity.max(1),
                seed,
                max_seq: step_max_seq.max(64),
                raw_prompt,
                output,
                positions: step_logit_positions,
                top_k: step_logit_top_k,
                bf16_ref_dir,
                gpu_kv: step_gpu_kv,
            }
        }
        Some("step-layer-probe") => {
            let output = output_dir.unwrap_or_else(|| {
                eprintln!(
                    "usage: diffgemma-mps step-layer-probe -m MODEL -o OUT.json [-p Hello] [--layers 30] [--seed 42] [--layer-position 129]"
                );
                std::process::exit(2);
            });
            Command::StepLayerProbe {
                prompt: prompt.clone(),
                layers: layers_for_parity_dump(parity_layers),
                seed,
                max_seq: step_max_seq.max(64),
                raw_prompt,
                output,
                position: step_layer_position,
            }
        }
        Some("step-attn-dump") => {
            let output = output_dir.unwrap_or_else(|| {
                eprintln!(
                    "usage: diffgemma-mps step-attn-dump -m MODEL -o OUT.json [-p Hello] [--layers 30] [--seed 42] [--attn-layer 2] [--layer-position 129]"
                );
                std::process::exit(2);
            });
            Command::StepAttnDump {
                prompt: prompt.clone(),
                layers: layers_for_parity_dump(parity_layers),
                seed,
                max_seq: step_max_seq.max(64),
                raw_prompt,
                output,
                layer: step_attn_layer,
                position: step_layer_position,
            }
        }
        Some("step-moe-dump") => {
            let output = output_dir.unwrap_or_else(|| {
                eprintln!(
                    "usage: diffgemma-mps step-moe-dump -m MODEL -o OUT.json [-p Hello] [--layers 30] [--seed 42] [--attn-layer 2] [--layer-position 129]"
                );
                std::process::exit(2);
            });
            Command::StepMoeDump {
                prompt: prompt.clone(),
                layers: layers_for_parity_dump(parity_layers),
                seed,
                max_seq: step_max_seq.max(64),
                raw_prompt,
                output,
                layer: step_attn_layer,
                position: step_layer_position,
            }
        }
        Some("step-moe-route-dump") => {
            let output = output_dir.unwrap_or_else(|| {
                eprintln!(
                    "usage: diffgemma-mps step-moe-route-dump -m MODEL -o OUT.json [-p Hello] [--layers 30] [--seed 42] [--attn-layer 0] [--skip-grouped]"
                );
                std::process::exit(2);
            });
            Command::StepMoeRouteDump {
                prompt: prompt.clone(),
                layers: layers_for_parity_dump(parity_layers),
                seed,
                max_seq: step_max_seq.max(64),
                raw_prompt,
                output,
                layer: step_attn_layer,
                run_grouped: step_moe_route_grouped,
            }
        }
        Some("step-moe-batched-pin-dump") => {
            let output = output_dir.unwrap_or_else(|| {
                eprintln!(
                    "usage: diffgemma-mps step-moe-batched-pin-dump -m MODEL -o OUT.json [-p Hello] [--layers 30] [--seed 42] [--attn-layer 0]"
                );
                std::process::exit(2);
            });
            Command::StepMoeBatchedPinDump {
                prompt: prompt.clone(),
                layers: layers_for_parity_dump(parity_layers),
                seed,
                max_seq: step_max_seq.max(64),
                raw_prompt,
                output,
                layer: step_attn_layer,
            }
        }
        Some("step-moe-single-dump") => {
            let output = output_dir.unwrap_or_else(|| {
                eprintln!(
                    "usage: diffgemma-mps step-moe-single-dump -m MODEL -o OUT.json [--expert 18] [-p Hello] [--layers 30] [--seed 42] [--attn-layer 2] [--layer-position 129]"
                );
                std::process::exit(2);
            });
            Command::StepMoeSingleDump {
                prompt: prompt.clone(),
                layers: layers_for_parity_dump(parity_layers),
                seed,
                max_seq: step_max_seq.max(64),
                raw_prompt,
                output,
                layer: step_attn_layer,
                position: step_layer_position,
                expert: step_moe_expert,
            }
        }
        Some("step-preamble-dump") => {
            let output = output_dir.unwrap_or_else(|| {
                eprintln!(
                    "usage: diffgemma-mps step-preamble-dump -m MODEL -o OUT.json [-p Hello] [--layers 30] [--seed 42] [--layer-position 129]"
                );
                std::process::exit(2);
            });
            Command::StepPreambleDump {
                prompt: prompt.clone(),
                layers: layers_for_parity_dump(parity_layers),
                seed,
                max_seq: step_max_seq.max(64),
                raw_prompt,
                output,
                position: step_layer_position,
            }
        }
        Some("embed-row-dump") => {
            let output = output_dir.unwrap_or_else(|| {
                eprintln!(
                    "usage: diffgemma-mps embed-row-dump -m MODEL -o OUT.json [--embed-token 71153] [--embed-gpu] [--bf16-ref DIR]"
                );
                std::process::exit(2);
            });
            let ref_dir = bf16_ref_dir.or_else(|| {
                let p = model_dir.join("../model/transformer");
                if p.is_dir() {
                    Some(p)
                } else {
                    let p2 = PathBuf::from("model/transformer");
                    p2.is_dir().then_some(p2)
                }
            });
            Command::EmbedRowDump {
                token: embed_row_token,
                layers: layers_for_parity_dump(parity_layers),
                max_seq: step_max_seq.max(64),
                prompt: prompt.clone(),
                raw_prompt,
                output,
                bf16_ref_dir: ref_dir,
                gpu: embed_row_gpu,
            }
        }
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
            profile: step_profile,
            profile_steps: step_profile_steps,
            layer_profile: step_layer_profile,
        },
        Some("bench-gemm-fusion") => Command::BenchGemmFusion {
            layers: bench_layers.max(1).min(30),
            kv_len: step_kv_len,
            seed,
            max_seq: step_max_seq.max(64),
            iters: bench_iters.max(1),
        },
        Some(cmd) => {
            eprintln!("unknown command: {cmd}");
            eprintln!(
                "usage: diffgemma-mps [-p PROMPT] [--raw] [summary|config|weights <name>|quantize|convert-model|step-smoke|step-probe|step-kv-check|step-kv-parity|step-verify|step-ci|step-parity|bench-step-kernel|bench-step|bench-prefill|probe-device|layer0|decoder|decoder-gpu|prefill|generate|generate-gpu|generate-monolithic|generate-monolithic-parity|generate-parity|chat|serve|tokenize <text>|gemm|attention]"
            );
            eprintln!(
                "  default (no command): generate-monolithic on .dgq, else generate-gpu (bf16)"
            );
            eprintln!(
                "  prompts: chat template applied by default; use --raw for bare BPE (-p \"Hello\" -> [9259])"
            );
            eprintln!(
                "  chat: interactive REPL (monolithic .dgq); optional -p for first user turn"
            );
            eprintln!(
                "  generate-parity: GPU vs checked-in golden (use --compare-cpu for slow CPU path; use --raw for legacy goldens)"
            );
            eprintln!(
                "  options: ... --golden NAME --write-golden NAME --compare-cpu --no-early-stop --assert --debug-deep"
            );
            eprintln!("  gemm options: --size N (default 512)");
            eprintln!("  attention: layer 0 GQA parity");
            eprintln!("  decoder-gpu: full decoder CPU vs GPU parity at seq=256");
            std::process::exit(2);
        }
    };

    Cli {
        model_dir,
        command,
        raw_prompt,
    }
}

#[cfg(target_os = "macos")]
fn default_generate_command(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    seed: u64,
    steps: usize,
    prompt_len: usize,
    max_new_tokens: usize,
    max_layers: Option<usize>,
    no_early_stop: bool,
) -> Command {
    let _ = model_dir;
    // The non-monolithic generate surface is retired; `ask`/`generate` always
    // run the monolithic step path.
    Command::GenerateMonolithic {
        prompt,
        seed,
        steps,
        prompt_len,
        max_new_tokens,
        max_layers,
        no_early_stop,
        kernel_assert: false,
        kernel_debug_deep: false,
        write_golden: None,
        write_trace: None,
        verbose: false,
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

fn run_gemm(size: usize) -> ExitCode {
    #[cfg(target_os = "macos")]
    {
        use metal::{Bf16Gemm, bf16_matmul_cpu, f32_to_bf16};

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

/// Fail-fast context-budget check (panic-to-error, ROADMAP 3.2): `Err(msg)`
/// when the weights + KV at `max_seq` would exceed the safe fraction of the GPU
/// working-set (estimated from physical RAM, ~72% on Apple Silicon), which
/// swaps or fails the KV allocation. Shared by `chat` (`--ctx`) and `ask`
/// (max_seq sized from the prompt + `--max-new-tokens`). Returns `Ok` when the
/// budget can't be determined (no device query) so it never blocks small runs.
fn check_ctx_budget(max_seq: usize) -> Result<(), String> {
    let phys = crate::metal::memwatch::physical_ram_bytes();
    let budget = (phys as f64 * 0.72) as u64;
    if let Some((needed, ceiling)) = flags::ctx_over_budget(max_seq, budget) {
        let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
        return Err(format!(
            "context {max_seq} needs ~{:.1} GiB GPU-resident (weights + KV), over the ~{:.1} GiB \
             safe budget on this {:.0} GiB machine — it would swap or fail to allocate. Reduce the \
             prompt / --max-new-tokens / --ctx to <= {} (or free RAM).",
            gib(needed),
            gib(ceiling),
            gib(phys),
            flags::max_feasible_ctx(budget),
        ));
    }
    Ok(())
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
        let text = history.last().map(|t| t.content.as_str()).unwrap_or("");
        Ok(tokenizer.encode(text, false))
    } else {
        chat_template::format_chat_token_ids(
            &tokenizer,
            history,
            &chat_template::ChatFormatOptions::default(),
        )
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn run_chat_cmd(
    model_dir: &std::path::Path,
    initial_prompt: Option<String>,
    seed: u64,
    steps: usize,
    max_new_tokens: usize,
    max_layers: Option<usize>,
    no_early_stop: bool,
    raw_prompt: bool,
    verbose: bool,
    events_path: Option<PathBuf>,
    json: bool,
    ctx: Option<usize>,
) -> ExitCode {
    use metal::{StepGenerateConfig, StepGenerateSession, generate_with_session};
    use std::io::{self, IsTerminal, Write};

    // Quiet by default: chat is a UI, not a log. `--verbose` restores the
    // step/prefill/session logs; `--json` routes the event stream to stdout so
    // no human chrome may pollute it.
    if !verbose && std::env::var_os("DGQ_QUIET").is_none() {
        unsafe {
            std::env::set_var("DGQ_QUIET", "1");
        }
    }
    // `--json`: JSONL to stdout, all human output suppressed. Otherwise the
    // spinner/streaming renderer runs on an interactive tty (not under
    // --verbose). `--events <path>` tees the JSONL to a file either way.
    let interactive = !json && !verbose && io::stdout().is_terminal();
    let events_file = match &events_path {
        Some(p) => match std::fs::File::create(p) {
            Ok(f) => Some(f),
            Err(err) => {
                eprintln!("error: cannot open --events {}: {err}", p.display());
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };
    // Per-turn JSON sink factory: stdout for --json, a cloned handle to the
    // events file otherwise. `None` when neither is requested.
    let make_json_sink = |turn_active: bool| -> Option<Box<dyn Write + Send>> {
        if !turn_active {
            return None;
        }
        if json {
            Some(Box::new(io::stdout()))
        } else {
            events_file
                .as_ref()
                .and_then(|f| f.try_clone().ok())
                .map(|f| Box::new(f) as Box<dyn Write + Send>)
        }
    };
    let want_json = json || events_file.is_some();

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: chat requires a .dgq directory (-m /path/to/quantized-weights)");
        return ExitCode::FAILURE;
    }

    let layers = match resolve_model_layers(model_dir, max_layers) {
        Ok(n) => n,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    // Full-message chat: generate until the model emits its end-of-turn token,
    // limited only by the context window (no per-turn token cap — long replies
    // stream freely). The KV arena is sized once at session open and fixed
    // thereafter, so pick a roomy `max_seq` up front. `--ctx N` overrides for
    // long-context sessions (ring-buffer sliding KV keeps this cheap:
    // ~20 KB/token + ~410 MiB fixed; 131072 ≈ 3 GiB). `--max-new-tokens`, if
    // the caller raises it above the default, becomes an optional hard ceiling.
    #[allow(non_snake_case)]
    let CHAT_MAX_SEQ: usize = ctx.unwrap_or(8192);
    // Fail-fast --ctx budget guard: refuse a context whose weights+KV would
    // swap / fail to allocate, before loading the model (see check_ctx_budget).
    if let Err(msg) = check_ctx_budget(CHAT_MAX_SEQ) {
        eprintln!("error: {msg}");
        return ExitCode::FAILURE;
    }
    let explicit_cap = if max_new_tokens > 256 {
        Some(max_new_tokens)
    } else {
        None
    };

    let stop_token_ids = config::load_generation_stop_tokens(model_dir);
    if verbose {
        eprintln!("chat: full-message stop tokens = {stop_token_ids:?}, cap = {explicit_cap:?}");
    }

    let sampler = sample::sampler_for_steps(steps, no_early_stop);
    let mut step_cfg = StepGenerateConfig::from_generate(
        seed,
        CHAT_MAX_SEQ,
        CHAT_MAX_SEQ,
        layers,
        sampler,
        no_early_stop,
    );
    step_cfg.degenerate_reply_check =
        chat_template::empty_reply_check(model_dir, stop_token_ids.clone());
    step_cfg.stop_token_ids = stop_token_ids;

    if interactive {
        print!("loading model… ");
        let _ = io::stdout().flush();
    }
    let open_started = std::time::Instant::now();
    let mut session = match StepGenerateSession::open(model_dir, &step_cfg, None) {
        Ok((s, compile)) => {
            if interactive {
                println!("ready ({:.1}s)", open_started.elapsed().as_secs_f64());
            } else if verbose {
                eprintln!("chat: session ready ({compile:.2?}, layers={layers})");
            }
            s
        }
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    let tok_path = model_dir.join("tokenizer.json");
    let tokenizer = match tokenizer::Tokenizer::load(&tok_path) {
        Ok(t) => std::sync::Arc::new(t),
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    // (No warm-up: COLD-START-1 — the first fresh-session generation returning an
    // empty/EOS reply — is fixed at the root by the deterministic first-step SC
    // seed. The throwaway warm-up generation is no longer needed.)

    let mut history: Vec<chat_template::ChatTurn> = Vec::new();
    let mut turn_idx = 0u64;

    let mut run_turn = |history: &mut Vec<chat_template::ChatTurn>,
                        turn_idx: &mut u64|
     -> Result<(), safetensors::Error> {
        let prompt = build_chat_prompt_tokens(model_dir, history, raw_prompt)?;
        let prompt_len = prompt.len();
        // KV arena is fixed at session open (CHAT_MAX_SEQ); the reply may use
        // the whole remaining context (no artificial cap), or the caller's
        // explicit ceiling if one was given. Reserve one CANVAS block: each
        // denoise block writes a CANVAS-wide canvas at [kv_len..kv_len+CANVAS],
        // so kv_len + CANVAS must stay within max_seq — this keeps the
        // `set_kv_len` overflow assert unreachable from a near-full context.
        let budget = CHAT_MAX_SEQ.saturating_sub(prompt_len + metal::CANVAS);
        if budget == 0 {
            if !json {
                println!(
                    "model> (prompt leaves no room for a reply within the {CHAT_MAX_SEQ}-token context; cannot generate)"
                );
            }
            history.push(chat_template::ChatTurn::model(String::new()));
            *turn_idx = turn_idx.wrapping_add(1);
            return Ok(());
        }
        step_cfg.max_new_tokens = explicit_cap.map_or(budget, |c| c.min(budget));
        step_cfg.seed = seed.wrapping_add(*turn_idx);
        let this_turn = *turn_idx;
        *turn_idx = turn_idx.wrapping_add(1);

        let started = std::time::Instant::now();
        let stream = chat_ui::ChatStream::start(
            std::sync::Arc::clone(&tokenizer),
            step_cfg.stop_token_ids.clone(),
            interactive,
            make_json_sink(want_json),
            this_turn,
            prompt_len,
        );
        step_cfg.step_observer = Some(stream.observer());
        let out = generate_with_session(&mut session, &prompt, &step_cfg, "chat")?;
        step_cfg.step_observer = None;
        let elapsed = started.elapsed();

        let new_ids =
            sample::strip_degenerate_token_ids(out.token_ids.get(prompt_len..).unwrap_or(&[]));
        let reply = chat_template::sanitize_model_reply(&tokenizer.decode(&new_ids));
        let new_tokens = out.token_ids.len().saturating_sub(prompt_len);
        let secs = elapsed.as_secs_f64();
        stream.finish(
            &reply,
            new_tokens,
            out.denoise_steps_run,
            secs,
            out.stopped_on_eot,
        );

        if !interactive && !json {
            // Non-interactive human output (piped / --verbose): print the reply
            // once, since the terminal renderer was disabled.
            if reply.is_empty() {
                println!("model> (empty response)");
            } else {
                println!("model> {reply}");
            }
        }
        if !json {
            let cap_note = if out.stopped_on_eot {
                ""
            } else {
                " · hit context limit"
            };
            println!(
                "  ({new_tokens} tok · {} steps · {secs:.1}s · {:.1} tok/s{cap_note})",
                out.denoise_steps_run,
                new_tokens as f64 / secs.max(1e-9),
            );
        }
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

    if !json {
        println!("chat ready (type 'exit' or 'quit' to end; Ctrl-D also exits)");
    }
    let stdin = io::stdin();
    loop {
        if !json {
            print!("you> ");
            let _ = io::stdout().flush();
        }
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

/// Smoketest prompt spec (`fixtures/smoketest/prompts.json`).
#[cfg(target_os = "macos")]
#[derive(serde::Deserialize)]
struct SmoketestSpec {
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
struct SmokeConvergence {
    id: String,
    prompt: String,
    max_steps: usize,
}

/// Long-context doc-QA probe: a fixture document truncated to `doc_tokens`
/// prompt tokens + a question about facts planted near the truncation edge
/// (in-window-unique, so a grounded answer proves the model READ that depth).
#[cfg(target_os = "macos")]
#[derive(serde::Deserialize)]
struct SmokeLongCtx {
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
struct SmokeAdherence {
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
fn smoke_normalize(s: &str) -> String {
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
fn smoke_answer_matches(reply: &str, answer: &str, accept: &[String]) -> bool {
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
fn run_smoketest_cmd(
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
    if std::env::var_os("DGQ_QUIET").is_none() {
        unsafe { std::env::set_var("DGQ_QUIET", "1") };
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

fn print_generate_elapsed(label: &str, elapsed: std::time::Duration) {
    let secs = elapsed.as_secs_f64();
    println!("  {label} elapsed: {secs:.2}s ({elapsed:.2?})");
}

/// Print just the clean reply (the one-shot-chat default for `ask`): decode the
/// generated tokens, strip pad/filler, sanitize chat scaffolding (thought/turn
/// markers — tool-call markers are kept), and print the FULL reply untruncated.
fn print_generate_reply(
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
fn run_generate_monolithic_cmd(
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

    crate::kernels::sub::variant::set_runtime_kernel_debug(kernel_assert, kernel_debug_deep);

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

fn run_attention_parity(m: &model::Model) -> ExitCode {
    #[cfg(target_os = "macos")]
    {
        use metal::GpuAttention;
        use model::attention::{AttentionParams, forward_to_attn_out, prepare_qkv_pre_rope};

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
        println!(
            "  shape: [{SEQ_LEN}, {}, {}]",
            params.n_heads, params.head_dim
        );
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
}

fn run_layer0_forward(m: &model::Model) -> ExitCode {
    const SEQ_LEN: usize = 16;
    let hidden = m.config.text_config.hidden_size;

    let layer =
        match model::layer_weights::DecoderLayerWeights::load(&m.weights, 0, &m.config.text_config)
        {
            Ok(layer) => layer,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };

    let mut scratch =
        match model::decoder_layer::DecoderLayerScratch::new(SEQ_LEN, &m.config.text_config, 0) {
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
