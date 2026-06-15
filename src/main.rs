mod buffer;
mod chat_template;
mod config;
mod denoise_trace;
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
        write_trace: Option<PathBuf>,
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
        write_trace: Option<PathBuf>,
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
        repeat_prefill: bool,
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
    StepKvParity {
        prompt: Option<String>,
        prompt_len: usize,
        layers: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
    },
    StepAttnProbe {
        prompt: Option<String>,
        prompt_len: usize,
        layer: usize,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
    },
    StepQ4Parity {
        prompt: Option<String>,
        prompt_len: usize,
        layers: usize,
        kv_len: u32,
        seed: u64,
        max_seq: usize,
        raw_prompt: bool,
    },
    StepNvfp4Parity {
        prompt: Option<String>,
        prompt_len: usize,
        layers: usize,
        kv_len: u32,
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
        Command::StepQ4Parity {
            prompt,
            prompt_len,
            layers,
            kv_len,
            seed,
            max_seq,
            raw_prompt,
        } => run_step_q4_parity_cmd(
            &cli.model_dir,
            prompt,
            prompt_len,
            layers,
            kv_len,
            seed,
            max_seq,
            raw_prompt,
        ),
        Command::StepNvfp4Parity {
            prompt,
            prompt_len,
            layers,
            kv_len,
            seed,
            max_seq,
            raw_prompt,
        } => run_step_nvfp4_parity_cmd(
            &cli.model_dir,
            prompt,
            prompt_len,
            layers,
            kv_len,
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
        } => run_bench_step_kernel_cmd(
            &cli.model_dir,
            layers,
            kv_len,
            seed,
            max_seq,
            iters,
            forward_only,
            profile,
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
            write_trace,
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
            write_trace,
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
            repeat_prefill,
        } => run_bench_prefill(m, model_dir, prompt_len, layers, iters, seed, repeat_prefill),
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
            write_trace,
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
            write_trace,
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
        Command::StepKvParity { .. } => ExitCode::FAILURE,
        Command::StepAttnProbe { .. } => ExitCode::FAILURE,
        Command::StepQ4Parity { .. } | Command::StepNvfp4Parity { .. } => ExitCode::FAILURE,
        Command::StepLogitsDump { .. } => ExitCode::FAILURE,
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
        no_early_stop: false,
        encoder_use_mps_q4: None,
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
        no_early_stop: false,
        encoder_use_mps_q4: None,
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
    use metal::{parse_positions, run_step_logits_dump, write_step_logits_dump, StepSmokeConfig};

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
        use_mps_q4: None,
        prefill_token_ids: None,
        no_early_stop: false,
        encoder_use_mps_q4: None,
    };
    if let Err(err) = attach_step_prefill(
        &mut cfg,
        model_dir,
        0,
        prompt.as_deref(),
        raw_prompt,
    ) {
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

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_step_logits_dump_cmd(
    _model_dir: &std::path::Path,
    _prompt: Option<String>,
    _layers: usize,
    _steps: usize,
    _seed: u64,
    _max_seq: usize,
    _raw_prompt: bool,
    _output: &std::path::Path,
    _positions: &str,
    _top_k: usize,
) -> ExitCode {
    eprintln!("error: step-logits-dump requires --features metal on macOS");
    ExitCode::FAILURE
}

#[cfg(all(feature = "metal", target_os = "macos"))]
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
    use metal::{run_step_layer_hidden_dump, write_step_layer_hidden_dump, StepSmokeConfig};

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
        use_mps_q4: None,
        prefill_token_ids: None,
        no_early_stop: false,
        encoder_use_mps_q4: None,
    };
    if let Err(err) = attach_step_prefill(
        &mut cfg,
        model_dir,
        0,
        prompt.as_deref(),
        raw_prompt,
    ) {
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

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_step_layer_probe_cmd(
    _model_dir: &std::path::Path,
    _prompt: Option<String>,
    _layers: usize,
    _seed: u64,
    _max_seq: usize,
    _raw_prompt: bool,
    _output: &std::path::Path,
    _position: usize,
) -> ExitCode {
    eprintln!("error: step-layer-probe requires --features metal on macOS");
    ExitCode::FAILURE
}

#[cfg(all(feature = "metal", target_os = "macos"))]
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
    use metal::{run_step_attn_layer_dump, write_step_attn_layer_dump, StepSmokeConfig};

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
        use_mps_q4: None,
        prefill_token_ids: None,
        no_early_stop: false,
        encoder_use_mps_q4: None,
    };
    if let Err(err) = attach_step_prefill(
        &mut cfg,
        model_dir,
        0,
        prompt.as_deref(),
        raw_prompt,
    ) {
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

#[cfg(all(feature = "metal", target_os = "macos"))]
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
    use metal::{run_step_moe_layer_dump, write_step_moe_layer_dump, StepSmokeConfig};

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
        use_mps_q4: None,
        prefill_token_ids: None,
        no_early_stop: false,
        encoder_use_mps_q4: None,
    };
    if let Err(err) = attach_step_prefill(
        &mut cfg,
        model_dir,
        0,
        prompt.as_deref(),
        raw_prompt,
    ) {
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

#[cfg(all(feature = "metal", target_os = "macos"))]
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
        print_route_summary, run_step_moe_route_dump, write_step_moe_route_dump, StepFinishMode,
        StepSmokeConfig,
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
        use_mps_q4: Some(false),
        prefill_token_ids: None,
        no_early_stop: false,
        encoder_use_mps_q4: Some(false),
    };
    if let Err(err) = attach_step_prefill(
        &mut cfg,
        model_dir,
        0,
        prompt.as_deref(),
        raw_prompt,
    ) {
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
            println!("wrote {} (layer={}, slots_ok={})", output.display(), dump.layer, dump.slots_ok);
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

#[cfg(all(feature = "metal", target_os = "macos"))]
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
        print_batched_pin_summary, run_step_moe_batched_pin_dump, write_step_moe_batched_pin_dump,
        StepFinishMode, StepSmokeConfig,
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
        use_mps_q4: Some(false),
        prefill_token_ids: None,
        no_early_stop: false,
        encoder_use_mps_q4: Some(false),
    };
    if let Err(err) = attach_step_prefill(
        &mut cfg,
        model_dir,
        0,
        prompt.as_deref(),
        raw_prompt,
    ) {
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

#[cfg(all(feature = "metal", target_os = "macos"))]
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
        run_step_moe_single_expert_dump, write_step_moe_single_expert_dump, StepSmokeConfig,
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
        use_mps_q4: None,
        prefill_token_ids: None,
        no_early_stop: false,
        encoder_use_mps_q4: None,
    };
    if let Err(err) = attach_step_prefill(
        &mut cfg,
        model_dir,
        0,
        prompt.as_deref(),
        raw_prompt,
    ) {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    let label = prompt.unwrap_or_else(|| "Hello".to_string());
    match run_step_moe_single_expert_dump(
        model_dir, &cfg, &label, layer, position, expert,
    ) {
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

#[cfg(all(feature = "metal", target_os = "macos"))]
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
    use metal::{run_step_preamble_dump, write_step_preamble_dump, StepSmokeConfig};

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
        use_mps_q4: None,
        prefill_token_ids: None,
        no_early_stop: false,
        encoder_use_mps_q4: None,
    };
    if let Err(err) = attach_step_prefill(
        &mut cfg,
        model_dir,
        0,
        prompt.as_deref(),
        raw_prompt,
    ) {
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

    #[cfg(all(feature = "metal", target_os = "macos"))]
    let gpu_scaled = if gpu {
        use metal::{run_embed_row_gpu, StepFinishMode, StepSmokeConfig};
        let mut cfg = StepSmokeConfig {
            layers,
            steps: 1,
            kv_len: 0,
            seed: 0,
            max_seq,
            finish: StepFinishMode::ForwardOnly,
            use_mps_q4: None,
            prefill_token_ids: None,
            no_early_stop: false,
            encoder_use_mps_q4: None,
        };
        if let Err(err) = attach_step_prefill(
            &mut cfg,
            model_dir,
            0,
            prompt.as_deref(),
            raw_prompt,
        ) {
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

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    let gpu_scaled: Option<Vec<f32>> = if gpu {
        eprintln!("error: --embed-gpu requires --features metal on macOS");
        return ExitCode::FAILURE;
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

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_step_preamble_dump_cmd(
    _model_dir: &std::path::Path,
    _prompt: Option<String>,
    _layers: usize,
    _seed: u64,
    _max_seq: usize,
    _raw_prompt: bool,
    _output: &std::path::Path,
    _position: usize,
) -> ExitCode {
    eprintln!("error: step-preamble-dump requires --features metal on macOS");
    ExitCode::FAILURE
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_step_attn_dump_cmd(
    _model_dir: &std::path::Path,
    _prompt: Option<String>,
    _layers: usize,
    _seed: u64,
    _max_seq: usize,
    _raw_prompt: bool,
    _output: &std::path::Path,
    _layer: usize,
    _position: usize,
) -> ExitCode {
    eprintln!("error: step-attn-dump requires --features metal on macOS");
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

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_step_kv_parity_cmd(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    prompt_len: usize,
    layers: usize,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
) -> ExitCode {
    use metal::run_step_kv_mps_parity;

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
    match run_step_kv_mps_parity(
        model_dir,
        &token_ids,
        layers,
        max_seq.max(64),
        seed,
    ) {
        Ok(r) => {
            println!("step-kv-parity:");
            println!("  kv_len:              {}", r.kv_len);
            println!("  layers:              {}", r.layers);
            println!("  native_prefix_l0:    {:.6}", r.native_prefix_max_l0);
            println!("  mps_prefix_l0:       {:.6}", r.mps_prefix_max_l0);
            println!(
                "  max_kv_diff:         {:.6} (layer {} pos {})",
                r.max_kv_diff, r.max_kv_diff_layer, r.max_kv_diff_pos
            );
            println!("  native_min_ent:      {:.4}", r.native_min_ent);
            println!("  mps_min_ent:         {:.4}", r.mps_min_ent);
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

#[cfg(all(feature = "metal", target_os = "macos"))]
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
            println!("  attn keys T:            {} (= kv_len + canvas_len)", r.attn_keys_t);
            println!("  layer:                  {}", r.layer);
            println!("  monolithic K max L0:    {:.4}", r.k_plane_max_l0);
            println!("  monolithic V max L0:    {:.4}", r.v_plane_max_l0);
            println!("  q_norm weight mean|w|: {:.4}  rms: {:.4}", r.q_norm_weight_mean_abs, r.q_norm_weight_rms);
            println!("  k_norm weight mean|w|: {:.4}  rms: {:.4}", r.k_norm_weight_mean_abs, r.k_norm_weight_rms);
            println!("  CPU Q·K raw max/min:    {:.2} / {:.2}", r.cpu_raw_dot_max, r.cpu_raw_dot_min);
            println!(
                "  CPU softmax row ent:    {:.4} nats (ln T = {:.4}, active keys/row = {})",
                r.cpu_mean_softmax_entropy, ln_uniform, r.cpu_keys_per_row
            );
            println!("  CPU softmax max prob:   {:.4}", r.cpu_mean_max_prob);
            println!("  CPU weight sum/row:     {:.6} (expect 1.0)", r.cpu_mean_weight_sum);
            println!("  CPU Q/K head RMS:       {:.4} / {:.4}", r.cpu_q_head_rms, r.cpu_k_head_rms);
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

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_step_q4_parity_cmd(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    prompt_len: usize,
    layers: usize,
    kv_len: u32,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
) -> ExitCode {
    use metal::run_step_q4_mps_parity;

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-q4-parity requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let prefill = if kv_len > 0 || prompt.is_some() {
        let vocab = match crate::config::ModelConfig::load(model_dir) {
            Ok(c) => c.text_config.vocab_size,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };
        let plen = if kv_len > 0 {
            kv_len as usize
        } else {
            prompt_len
        };
        match build_prompt_tokens(model_dir, prompt.as_deref(), plen, vocab, raw_prompt, &[]) {
            Ok(ids) => Some(ids),
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };
    let kv_len_eff = prefill.as_ref().map(|p| p.len() as u32).unwrap_or(kv_len);
    eprintln!(
        "step-q4-parity: layers={} kv_len={}",
        layers.max(1).min(30),
        kv_len_eff
    );
    match run_step_q4_mps_parity(
        model_dir,
        layers,
        kv_len_eff,
        seed,
        max_seq.max(64),
        prefill,
    ) {
        Ok(r) => {
            println!("step-q4-parity:");
            println!("  layers:              {}", r.layers);
            println!("  kv_len:              {}", r.kv_len);
            println!("  hidden_max_abs:      {:.6}", r.hidden_max_abs);
            println!("  logits_max_abs:      {:.6}", r.logits_max_abs);
            println!("  native_min_ent:      {:.4}", r.native_min_ent);
            println!("  mps_min_ent:         {:.4}", r.mps_min_ent);
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

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_step_nvfp4_parity_cmd(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    prompt_len: usize,
    layers: usize,
    kv_len: u32,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
) -> ExitCode {
    use metal::run_step_nvfp4_mps_parity;

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-nvfp4-parity requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let prefill = if kv_len > 0 || prompt.is_some() {
        let vocab = match crate::config::ModelConfig::load(model_dir) {
            Ok(c) => c.text_config.vocab_size,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };
        let plen = if kv_len > 0 {
            kv_len as usize
        } else {
            prompt_len
        };
        match build_prompt_tokens(model_dir, prompt.as_deref(), plen, vocab, raw_prompt, &[]) {
            Ok(ids) => Some(ids),
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };
    let kv_len_eff = prefill.as_ref().map(|p| p.len() as u32).unwrap_or(kv_len);
    eprintln!(
        "step-nvfp4-parity: layers={} kv_len={}",
        layers.max(1).min(30),
        kv_len_eff
    );
    match run_step_nvfp4_mps_parity(
        model_dir,
        layers,
        kv_len_eff,
        seed,
        max_seq.max(64),
        prefill,
    ) {
        Ok(r) => {
            println!("step-nvfp4-parity:");
            println!("  layers:              {}", r.layers);
            println!("  kv_len:              {}", r.kv_len);
            println!("  hidden_max_abs:      {:.6}", r.hidden_max_abs);
            println!("  logits_max_abs:      {:.6}", r.logits_max_abs);
            println!("  native_min_ent:      {:.4}", r.native_min_ent);
            println!("  mps_min_ent:         {:.4}", r.mps_min_ent);
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
fn run_step_nvfp4_parity_cmd(
    _model_dir: &std::path::Path,
    _prompt: Option<String>,
    _prompt_len: usize,
    _layers: usize,
    _kv_len: u32,
    _seed: u64,
    _max_seq: usize,
    _raw_prompt: bool,
) -> ExitCode {
    eprintln!("error: step-nvfp4-parity requires --features metal on macOS");
    ExitCode::FAILURE
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_step_q4_parity_cmd(
    _model_dir: &std::path::Path,
    _prompt: Option<String>,
    _prompt_len: usize,
    _layers: usize,
    _kv_len: u32,
    _seed: u64,
    _max_seq: usize,
    _raw_prompt: bool,
) -> ExitCode {
    eprintln!("error: step-q4-parity requires --features metal on macOS");
    ExitCode::FAILURE
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_step_kv_parity_cmd(
    _model_dir: &std::path::Path,
    _prompt: Option<String>,
    _prompt_len: usize,
    _layers: usize,
    _seed: u64,
    _max_seq: usize,
    _raw_prompt: bool,
) -> ExitCode {
    eprintln!("error: step-kv-parity requires --features metal on macOS");
    ExitCode::FAILURE
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
    profile: bool,
) -> ExitCode {
    use metal::bench_step_kernel;

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: bench-step-kernel requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let cfg = step_kernel_config(layers, kv_len, seed, max_seq, forward_only);
    if profile {
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
                println!("  preamble:      {:.2?}  ({:.1}%)", p.preamble, pct(p.preamble));
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

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn run_bench_step_kernel_cmd(
    _model_dir: &std::path::Path,
    _layers: usize,
    _kv_len: u32,
    _seed: u64,
    _max_seq: usize,
    _iters: usize,
    _forward_only: bool,
    _profile: bool,
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
    use dgq::layout::QuantProfile;
    use dgq::{quantize_model, QuantizeOptions};

    let profile_name = profile;
    let profile = match profile {
        "q4" => QuantProfile::Q4,
        "q5" => QuantProfile::Q5,
        "nvfp4" => QuantProfile::Nvfp4,
        other => {
            eprintln!("error: unknown profile {other} (use q4, q5, or nvfp4)");
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
            println!("  nvfp4 tensors: {}", summary.nvfp4_tensors);
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
    let mut write_trace: Option<PathBuf> = None;
    let mut no_early_stop = false;
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
    let mut step_logit_positions = String::new();
    let mut step_logit_top_k = 10usize;
    let mut step_layer_position = 129usize;
    let mut step_attn_layer = 2usize;
    let mut step_moe_expert = 18u32;
    let mut step_moe_route_grouped = true;
    let mut embed_row_token = 71153u32;
    let mut embed_row_gpu = false;
    let mut bf16_ref_dir: Option<PathBuf> = None;
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
            "--repeat-prefill" => bench_repeat_prefill = true,
            "--no-early-stop" => no_early_stop = true,
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
            write_trace: None,
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
            write_trace: None,
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
            write_trace,
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
            write_trace,
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
            repeat_prefill: bench_repeat_prefill,
        },
        Some("probe-device") => Command::ProbeDevice,
        Some("bench-gemm") => Command::BenchGemm {
            shapes: bench_gemm_shapes,
            oracle: bench_gemm_oracle,
            iters: bench_iters.max(1),
        },
        Some("quantize") => {
            let out = output_dir.unwrap_or_else(|| {
                eprintln!("usage: diffgemma-mps quantize -o OUTPUT_DIR -m SOURCE [--profile q4|q5|nvfp4]");
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
        Some("step-attn-probe") => Command::StepAttnProbe {
            prompt: prompt.clone(),
            prompt_len,
            layer: bench_layers,
            seed,
            max_seq: step_max_seq.max(512),
            raw_prompt,
        },
        Some("step-q4-parity") => Command::StepQ4Parity {
            prompt: prompt.clone(),
            prompt_len,
            layers: bench_layers.max(1).min(30),
            kv_len: step_kv_len,
            seed,
            max_seq: step_max_seq.max(64),
            raw_prompt,
        },
        Some("step-nvfp4-parity") => Command::StepNvfp4Parity {
            prompt: prompt.clone(),
            prompt_len,
            layers: bench_layers.max(1).min(30),
            kv_len: step_kv_len,
            seed,
            max_seq: step_max_seq.max(64),
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
                layers: bench_layers.max(1).min(30),
                steps: steps_parity.max(1),
                seed,
                max_seq: step_max_seq.max(64),
                raw_prompt,
                output,
                positions: step_logit_positions,
                top_k: step_logit_top_k,
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
                layers: bench_layers.max(1).min(30),
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
                layers: bench_layers.max(1).min(30),
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
                layers: bench_layers.max(1).min(30),
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
                layers: bench_layers.max(1).min(30),
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
                layers: bench_layers.max(1).min(30),
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
                layers: bench_layers.max(1).min(30),
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
                layers: bench_layers.max(1).min(30),
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
                layers: bench_layers.max(1).min(30),
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
        },
        Some(cmd) => {
            eprintln!("unknown command: {cmd}");
            eprintln!(
                "usage: diffgemma-mps [-p PROMPT] [--raw] [summary|config|weights <name>|quantize|convert-model|step-smoke|step-probe|step-kv-check|step-kv-parity|step-q4-parity|step-nvfp4-parity|step-verify|step-ci|step-parity|bench-step-kernel|bench-step|bench-prefill|probe-device|layer0|decoder|decoder-gpu|prefill|generate|generate-gpu|generate-monolithic|generate-monolithic-parity|generate-parity|chat|tokenize <text>|gemm|attention]"
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
            write_trace: None,
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
        write_trace: None,
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
    model_dir: &std::path::Path,
    prompt_len: usize,
    layers: usize,
    iters: usize,
    seed: u64,
    repeat_prefill: bool,
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
        let max_seq = prompt_len.max(512);

        let mut token_ids = vec![0u32; prompt_len];
        for (i, id) in token_ids.iter_mut().enumerate() {
            *id = ((i as u64 * 131 + seed + 7) % vocab.max(1) as u64) as u32;
        }

        let setup_started = std::time::Instant::now();
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
            "bench-prefill setup (prompt_len={prompt_len}, layers={layers}, quantized={}, repeat_prefill={repeat_prefill}) ({:.2?})",
            m.weights.is_quantized(),
            setup_started.elapsed()
        );

        // Warmup
        let warmup_started = std::time::Instant::now();
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
        eprintln!("bench-prefill warmup ({:.2?})", warmup_started.elapsed());

        eprintln!("bench-prefill running {iters} isolated iterations...");
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
        println!("  isolated per_run: {per_run:.2?}");
        println!(
            "  gate (≤0.5s @ 1 tok): {}",
            if per_run.as_secs_f64() <= 0.5 {
                "pass"
            } else {
                "fail"
            }
        );

        if repeat_prefill {
            use metal::{
                build_step_runtime, prefill_monolithic_kv_with_cache, MonolithicEncoderCache,
                StepFinishMode, StepSmokeConfig, CANVAS,
            };

            eprintln!("bench-prefill: repeat-prefill path (step runtime resident)...");
            let path_started = std::time::Instant::now();
            let smoke_cfg = StepSmokeConfig {
                layers,
                steps: 1,
                kv_len: 0,
                seed,
                max_seq,
                finish: StepFinishMode::Full,
                use_mps_q4: None,
                prefill_token_ids: None,
                no_early_stop: false,
                encoder_use_mps_q4: None,
            };
            let (mut rt, build) = match build_step_runtime(model_dir, &smoke_cfg) {
                Ok(v) => v,
                Err(err) => {
                    eprintln!("error: {err}");
                    return ExitCode::FAILURE;
                }
            };
            eprintln!(
                "bench-prefill: step runtime ready (total={:.2?}, compile={:.2?}) ({:.2?})",
                build.total,
                build.compile,
                path_started.elapsed()
            );

            let encoder_started = std::time::Instant::now();
            let shared_blob = rt.shared_dgq_blob();
            let mut encoder = match MonolithicEncoderCache::open_opt(
                model_dir,
                CANVAS,
                max_seq,
                Some(shared_blob),
                None,
            ) {
                Ok(c) => c,
                Err(err) => {
                    eprintln!("error: {err}");
                    return ExitCode::FAILURE;
                }
            };
            eprintln!(
                "bench-prefill: encoder open ({:.2?})",
                encoder_started.elapsed()
            );

            for turn in 1..=2 {
                let turn_started = std::time::Instant::now();
                match prefill_monolithic_kv_with_cache(
                    &mut encoder,
                    &token_ids,
                    rt.kvcache(),
                    rt.layout(),
                    max_seq,
                    layers,
                ) {
                    Ok((kv_len, timing)) => {
                        eprintln!(
                            "bench-prefill: monolithic prefill turn {turn} kv_len={kv_len} ({:.2?}, gpu_forward={:.1}ms kv_pack={:.1}ms total={:.1}ms)",
                            turn_started.elapsed(),
                            timing.gpu_forward_ms,
                            timing.kv_pack_ms,
                            timing.total_ms
                        );
                    }
                    Err(err) => {
                        eprintln!("error: {err}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            eprintln!(
                "bench-prefill: repeat-prefill path done ({:.2?})",
                path_started.elapsed()
            );
        }

        ExitCode::SUCCESS
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        let _ = (m, model_dir, prompt_len, layers, iters, seed, repeat_prefill);
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

    let layers = match resolve_model_layers(model_dir, max_layers) {
        Ok(n) => n,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
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
        let out = generate_with_session(&mut session, &prompt, &step_cfg, "chat")?;
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
    write_trace: Option<PathBuf>,
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
        trace_prompt: None,
    };

    let stop_note = if no_early_stop { ", no_early_stop" } else { "" };
    eprintln!(
        "running generate-monolithic (prompt_len={prompt_len}, steps={steps}, layers={layers}, max_new_tokens={max_new_tokens}, seed={seed}{stop_note})..."
    );
    let started = std::time::Instant::now();

    let prompt_label = prompt_text.clone().unwrap_or_default();
    match generate::generate_monolithic_gpu(
        model_dir,
        &prompt,
        &gen_cfg,
        max_seq,
        &prompt_label,
    ) {
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
    write_trace: Option<PathBuf>,
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
        trace_prompt: prompt_text.clone(),
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
            trace_prompt: None,
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
