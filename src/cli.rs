//! Command-line surface: the `Cli`/`Command` types and the `parse_cli` parser.
//! Extracted from the former monolithic main.rs; the command bodies live in
//! `crate::commands`.

use std::env;
use std::path::PathBuf;

/// MLX parity dumps default to the full 30-layer decoder unless `--layers` is set.
fn layers_for_parity_dump(parity_layers: Option<usize>) -> usize {
    parity_layers.unwrap_or(30).max(1).min(30)
}

#[derive(Debug)]
pub(crate) struct Cli {
    pub(crate) model_dir: PathBuf,
    pub(crate) command: Command,
    /// When true, `-p` text is BPE-encoded as-is (no chat template).
    pub(crate) raw_prompt: bool,
}
#[derive(Debug)]
pub(crate) enum Command {
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
    /// Golden byte-identity pack — the Tier-1 refactor gate (task #73).
    /// Print the generated kernel FC-axis manifest (TOML) to stdout.
    Manifest,
    Golden {
        /// Pack spec (default `fixtures/golden/golden.json`).
        pack_path: Option<PathBuf>,
        /// Re-record every case's golden output instead of checking it.
        bless: bool,
        /// Substring (case-insensitive) on case id; only matching cases run.
        filter: Option<String>,
    },
}
pub(crate) fn parse_cli() -> Cli {
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
    let mut bench_layers = 1usize;
    let mut bench_iters = 3usize;
    let mut parity_layers: Option<usize> = None;
    let mut golden_name: Option<String> = None;
    let mut smoke_filter: Option<String> = None;
    let mut smoke_repeat: usize = 1;
    let mut smoke_longctx = false;
    let mut golden_bless = false;
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
            "--max-new-tokens" => {
                if let Some(v) = args.next() {
                    max_new_tokens = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --max-new-tokens");
                        std::process::exit(2);
                    });
                    max_new_tokens_explicit = true;
                }
            }
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
            "--bless-golden" | "--bless" => golden_bless = true,
            "--size" => {
                if let Some(v) = args.next() {
                    gemm_size = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --size");
                        std::process::exit(2);
                    });
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
        Some("manifest") => Command::Manifest,
        Some("golden") => Command::Golden {
            pack_path: positional.get(1).map(PathBuf::from),
            bless: golden_bless,
            filter: smoke_filter.clone(),
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
                "usage: diffgemma-mps [-p PROMPT] [--raw] [summary|config|weights <name>|quantize|step-smoke|step-probe|step-kv-check|step-kv-parity|step-verify|step-ci|step-parity|bench-step-kernel|bench-step|bench-prefill|probe-device|layer0|decoder|decoder-gpu|prefill|generate|generate-gpu|generate-monolithic|generate-monolithic-parity|generate-parity|chat|serve|tokenize <text>|gemm|attention]"
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
pub(crate) fn default_generate_command(
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
/// Production generate/chat default is 48 (model card); parity/bench default is 2.
pub(crate) fn resolve_steps(override_steps: Option<usize>, parity_default: bool) -> usize {
    override_steps.unwrap_or(if parity_default { 2 } else { 48 })
}
