//! The hand-rolled arg parser. Split from cli.rs; the `Cli`/`Command`
//! types live in mod.rs.

use super::*;
use std::env;
use std::path::PathBuf;

/// MLX parity dumps default to the full 30-layer decoder unless `--layers` is set.
fn layers_for_parity_dump(parity_layers: Option<usize>) -> usize {
    parity_layers.unwrap_or(30).clamp(1, 30)
}

fn parse_cli_bool(s: &str) -> Option<bool> {
    match s.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
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
    let mut serve_tool_repair = false;
    let mut serve_tool_validate = false;
    let mut serve_log_dir: Option<PathBuf> = None;
    let mut serve_think: Option<bool> = None;
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
    let mut census_arms: Vec<String> = Vec::new();
    let mut census_batteries: Vec<String> = Vec::new();
    let mut census_seeds: Vec<u64> = Vec::new();
    let mut census_gates: Vec<String> = Vec::new();
    let mut census_baseline: Option<String> = None;
    let mut census_out: Option<PathBuf> = None;
    let mut census_tau: f32 = 0.9;
    let mut census_analyze: Option<PathBuf> = None;
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
    // `quantize --overlay` / `repack --overlay`: emit an experts-only
    // overlay instead of a self-contained pack.
    let mut overlay_flag = false;
    // `repack --monolithic`: flatten a layered pack back to self-contained
    // (the dual of `repack --overlay`).
    let mut monolithic_flag = false;
    // `quantize --set class=format` (repeatable).
    let mut quant_sets: Vec<String> = Vec::new();
    let mut hf_repo: Option<String> = None;
    let mut hf_revision: Option<String> = None;
    let mut hf_source: Option<PathBuf> = None;
    // `download` options: HF repo, revision, and full re-fetch.
    let mut download_repo: Option<String> = None;
    let mut download_revision = String::from("main");
    let mut download_force = false;
    let mut download_jobs = 4usize;
    let mut bench_gemm_shapes = String::from("256x2816x2816,33x2816x1408");
    // Tunable prefill-attention sweep (bench-prefill-attn).
    let mut ag_qk_bm = 64usize;
    let mut ag_qk_bn = 64usize;
    let mut ag_pv_bm = 64usize;
    let mut ag_pv_bn = 64usize;
    let mut ag_hc = 4usize;
    let mut ag_sm_tpg = 256usize;
    let mut ag_side = false;
    let mut ag_causal = true;
    let mut bench_stages = false;
    let mut bench_gemm_oracle: Option<String> = None;
    let mut step_kv_len = 0u32;
    let mut step_max_seq = 512usize;
    let mut step_forward_only = false;
    let mut bench_n_subs: usize = 0;
    let mut step_profile = false;
    let mut step_profile_steps = 0usize;
    let mut step_layer_profile = false;
    let mut step_logit_positions = String::new();
    let mut step_logit_top_k = 10usize;
    let mut step_layer_position = 129usize;
    // `--warm-steps N`: denoise N steps before probing, so the canvas holds
    // what the model settled on rather than seeded noise. 0 = original behaviour.
    let mut step_warm_steps = 0usize;
    // `fit-token-probe --spec <labelled prompts>`; `--layer` overrides the spec.
    let mut probe_spec: Option<PathBuf> = None;
    let mut probe_layer: Option<usize> = None;
    let mut probe_from_generation = false;
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
            // Large prompts (100k-token oracle corpora) don't fit in argv.
            "--prompt-file" => {
                if let Some(path) = args.next() {
                    match std::fs::read_to_string(&path) {
                        Ok(text) => prompt = Some(text),
                        Err(err) => {
                            eprintln!("error: --prompt-file {path}: {err}");
                            std::process::exit(2);
                        }
                    }
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
            "--tool-repair" => serve_tool_repair = true,
            "--tool-validate" => serve_tool_validate = true,
            "--think" => serve_think = Some(true),
            s if s.starts_with("--think=") => {
                let v = &s["--think=".len()..];
                serve_think = Some(parse_cli_bool(v).unwrap_or_else(|| {
                    eprintln!("invalid --think={v} (expected true/false)");
                    true
                }));
            }
            "--log-dir" => {
                if let Some(v) = args.next() {
                    serve_log_dir = Some(PathBuf::from(v));
                }
            }
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
            // census: --arm repeats; the rest are comma-lists.
            "--arm" => {
                if let Some(v) = args.next() {
                    census_arms.push(v);
                }
            }
            "--battery" => {
                if let Some(v) = args.next() {
                    census_batteries.extend(v.split(',').map(|s| s.trim().to_string()));
                }
            }
            "--seeds" => {
                if let Some(v) = args.next() {
                    for part in v.split(',') {
                        match part.trim().parse::<u64>() {
                            Ok(n) => census_seeds.push(n),
                            Err(_) => {
                                eprintln!("--seeds: {part:?} is not a number");
                                std::process::exit(2);
                            }
                        }
                    }
                }
            }
            "--gate" => {
                if let Some(v) = args.next() {
                    census_gates.push(v);
                }
            }
            "--baseline" => census_baseline = args.next(),
            "--analyze" => census_analyze = args.next().map(PathBuf::from),
            "--out" => census_out = args.next().map(PathBuf::from),
            "--tau" => {
                if let Some(v) = args.next() {
                    match v.parse::<f32>() {
                        Ok(t) if (0.0..=1.0).contains(&t) => census_tau = t,
                        _ => {
                            eprintln!("--tau: expected a number in [0, 1], got {v:?}");
                            std::process::exit(2);
                        }
                    }
                }
            }
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
            "--overlay" => overlay_flag = true,
            "--monolithic" => monolithic_flag = true,
            "--set" => {
                if let Some(v) = args.next() {
                    quant_sets.push(v);
                } else {
                    eprintln!("usage: --set CLASS=FORMAT (e.g. --set experts=nvfp4)");
                    std::process::exit(2);
                }
            }
            "--hf-repo" => {
                if let Some(v) = args.next() {
                    hf_repo = Some(v);
                }
            }
            "--hf-revision" => {
                if let Some(v) = args.next() {
                    hf_revision = Some(v);
                }
            }
            "--hf-source" => {
                if let Some(v) = args.next() {
                    hf_source = Some(PathBuf::from(v));
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
            "--qk-bm" => {
                if let Some(v) = args.next() {
                    ag_qk_bm = v.parse().unwrap_or(64);
                }
            }
            "--qk-bn" => {
                if let Some(v) = args.next() {
                    ag_qk_bn = v.parse().unwrap_or(64);
                }
            }
            "--pv-bm" => {
                if let Some(v) = args.next() {
                    ag_pv_bm = v.parse().unwrap_or(64);
                }
            }
            "--pv-bn" => {
                if let Some(v) = args.next() {
                    ag_pv_bn = v.parse().unwrap_or(64);
                }
            }
            "--hc" => {
                if let Some(v) = args.next() {
                    ag_hc = v.parse().unwrap_or(4);
                }
            }
            "--sm-tpg" => {
                if let Some(v) = args.next() {
                    ag_sm_tpg = v.parse().unwrap_or(256);
                }
            }
            "--side" => ag_side = true,
            "--non-causal" => ag_causal = false,
            "--stages" => bench_stages = true,
            "--forward-only" => step_forward_only = true,
            "--step-profile" => step_profile = true,
            "--n-subs" => {
                if let Some(v) = args.next() {
                    bench_n_subs = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --n-subs");
                        std::process::exit(2);
                    });
                }
            }
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
            "--spec" => probe_spec = args.next().map(PathBuf::from),
            "--from-generation" => probe_from_generation = true,
            "--layer" => {
                if let Some(v) = args.next() {
                    match v.parse::<usize>() {
                        Ok(n) => probe_layer = Some(n),
                        Err(_) => {
                            eprintln!("--layer: expected an integer, got {v:?}");
                            std::process::exit(2);
                        }
                    }
                }
            }
            "--warm-steps" => {
                if let Some(v) = args.next() {
                    match v.parse::<usize>() {
                        Ok(n) => step_warm_steps = n,
                        Err(_) => {
                            eprintln!("--warm-steps: expected a non-negative integer, got {v:?}");
                            std::process::exit(2);
                        }
                    }
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
            "--repo" => {
                if let Some(v) = args.next() {
                    download_repo = Some(v);
                }
            }
            "--revision" => {
                if let Some(v) = args.next() {
                    download_revision = v;
                }
            }
            "--force" => download_force = true,
            "--jobs" => {
                if let Some(v) = args.next() {
                    download_jobs = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --jobs");
                        std::process::exit(2);
                    });
                }
            }
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
            tool_repair: serve_tool_repair,
            tool_validate: serve_tool_validate,
            log_dir: serve_log_dir.clone(),
            think: serve_think.unwrap_or(true),
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
        Some("census") => Command::Census {
            arms: census_arms.clone(),
            batteries: if census_batteries.is_empty() {
                vec!["smoke".to_string()]
            } else {
                census_batteries.clone()
            },
            seeds: if census_seeds.is_empty() {
                vec![7, 42, 123]
            } else {
                census_seeds.clone()
            },
            gates: census_gates.clone(),
            baseline: census_baseline.clone(),
            out_dir: census_out.clone(),
            tau: census_tau,
            steps: steps_production,
            analyze: census_analyze.clone(),
        },
        Some("manifest") => Command::Manifest,
        Some("replay") => Command::Replay {
            log_path: positional.get(1).map(PathBuf::from).unwrap_or_else(|| {
                eprintln!(
                    "usage: diffgemma-mps replay <ops.jsonl> [-m MODEL] [--ctx N] [--steps N]"
                );
                std::process::exit(2);
            }),
            ctx: chat_ctx,
            steps: steps_override,
        },
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
        Some("bench-prefill-attn") => Command::BenchPrefillAttn {
            kv_len: if step_kv_len > 0 { step_kv_len } else { 30000 },
            qk_bm: ag_qk_bm,
            qk_bn: ag_qk_bn,
            pv_bm: ag_pv_bm,
            pv_bn: ag_pv_bn,
            hc: ag_hc,
            sm_tpg: ag_sm_tpg,
            side: ag_side,
            causal: ag_causal,
            iters: bench_iters.max(1),
        },
        Some("bench-prefill-super") => Command::BenchPrefillSuper {
            kv_len: if step_kv_len > 0 { step_kv_len } else { 15000 },
            iters: bench_iters.max(1),
            stages: bench_stages,
            n_subs: bench_n_subs,
        },
        Some("quantize") => {
            let out = output_dir.unwrap_or_else(|| {
                eprintln!(
                    "usage: diffgemma-mps quantize -o OUTPUT_DIR -m SOURCE [--profile q4|q5|q6|nvfp4|nvfp4x] \
                     [--set class=format ...] [--overlay [--hf-repo ORG/NAME --hf-revision REV]]"
                );
                std::process::exit(2);
            });
            Command::Quantize {
                output: out,
                profile: quant_profile,
                overlay: overlay_flag,
                hf_repo,
                hf_revision,
                sets: quant_sets,
            }
        }
        Some("repack") => {
            if overlay_flag == monolithic_flag {
                eprintln!(
                    "usage: diffgemma-mps repack --overlay -o OUTPUT_DIR -m SOURCE_PACK [--hf-source DIR] [--hf-repo ORG/NAME --hf-revision REV]\n   or: diffgemma-mps repack --monolithic -o OUTPUT_DIR -m SOURCE_PACK   (exactly one of --overlay/--monolithic)"
                );
                std::process::exit(2);
            }
            let out = output_dir.unwrap_or_else(|| {
                eprintln!("usage: diffgemma-mps repack --overlay|--monolithic -o OUTPUT_DIR -m SOURCE_PACK");
                std::process::exit(2);
            });
            if monolithic_flag {
                Command::RepackMonolithic { output: out }
            } else {
                Command::Repack {
                    output: out,
                    hf_source,
                    hf_repo,
                    hf_revision,
                }
            }
        }
        Some("download") => Command::Download {
            // Default repo/dest mirror the model card; -o overrides the target.
            repo: download_repo
                .unwrap_or_else(|| "mmastrac/diffgemma-26b-a4b-it-q4".to_string()),
            revision: download_revision,
            dest: output_dir.unwrap_or_else(|| PathBuf::from("model/diffgemma-26b-a4b-it-q4")),
            force: download_force,
            jobs: download_jobs,
        },
        Some("step-smoke") => Command::StepSmoke {
            layers: bench_layers.clamp(1, 30),
            steps: steps_parity.max(1),
            kv_len: step_kv_len,
            seed,
            max_seq: step_max_seq.max(64),
            forward_only: step_forward_only,
            prompt: prompt.clone(),
        },
        Some("step-probe") => Command::StepProbe {
            layers: bench_layers.clamp(1, 30),
            kv_len: step_kv_len,
            seed,
            max_seq: step_max_seq.max(64),
            prompt: prompt.clone(),
        },
        Some("step-kv-check") => Command::StepKvCheck {
            kv_len: step_kv_len.max(1) as usize,
            layers: bench_layers.clamp(1, 30),
            seed,
            max_seq: step_max_seq.max(64),
        },
        Some("step-kv-parity") => Command::StepKvParity {
            prompt: prompt.clone(),
            prompt_len,
            layers: bench_layers.clamp(1, 30),
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
        Some("fit-token-probe") => {
            let output = output_dir.unwrap_or_else(|| {
                eprintln!(
                    "usage: diffgemma-mps fit-token-probe -m MODEL --spec SPEC.json -o PROBE.json [--layer 29] [--seed 42]"
                );
                std::process::exit(2);
            });
            let Some(spec) = probe_spec.clone() else {
                eprintln!("fit-token-probe: --spec SPEC.json is required");
                std::process::exit(2);
            };
            Command::FitTokenProbe {
                spec,
                output,
                layer: probe_layer,
                seed,
                max_seq: step_max_seq.max(64),
                from_generation: probe_from_generation,
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
                warm_steps: step_warm_steps,
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
        Some("step-attn-qk-dump") => {
            let output = output_dir.unwrap_or_else(|| {
                eprintln!(
                    "usage: diffgemma-mps step-attn-qk-dump -m MODEL -o OUT_DIR \
                     [--prompt-file DOC] [--attn-layer 5] [--max-seq N] [--seed 42]"
                );
                std::process::exit(2);
            });
            Command::StepAttnQkDump {
                prompt: prompt.clone(),
                layers: layers_for_parity_dump(parity_layers),
                seed,
                max_seq: step_max_seq.max(64),
                raw_prompt,
                output,
                layer: step_attn_layer,
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
            layers: bench_layers.clamp(1, 30),
        },
        Some("step-ci") => Command::StepCi {
            layers: bench_layers.clamp(1, 30),
        },
        Some("step-parity") => Command::StepParity {
            layers: bench_layers.clamp(1, 30),
            kv_len: step_kv_len,
            seed,
            max_seq: step_max_seq.max(64),
        },
        Some("bench-step-kernel") => Command::BenchStepKernel {
            layers: bench_layers.clamp(1, 30),
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
            layers: bench_layers.clamp(1, 30),
            kv_len: step_kv_len,
            seed,
            max_seq: step_max_seq.max(64),
            iters: bench_iters.max(1),
        },
        Some(cmd) => {
            eprintln!("unknown command: {cmd}");
            eprintln!("usage: diffgemma-mps [COMMAND] [-m MODEL] [-p PROMPT] [options]");
            eprintln!(
                "  default (no command): generate-monolithic on .dgq, else generate-gpu (bf16)"
            );
            eprintln!(
                "  main: ask | chat | serve | download | replay | smoketest | census | golden | quantize | tokenize <text>"
            );
            eprintln!("  info: summary | config | manifest | weights <name> | probe-device");
            eprintln!(
                "  gates/parity: step-smoke | step-verify | step-ci | step-parity | step-probe | step-kv-check | step-kv-parity | step-kv-bf16-cross | step-attn-probe"
            );
            eprintln!(
                "  dumps: step-logits-dump | step-bf16-logits-dump | step-layer-probe | step-attn-dump | step-attn-qk-dump | step-moe-dump | step-moe-route-dump | step-moe-batched-pin-dump | step-moe-single-dump | step-preamble-dump | embed-row-dump"
            );
            eprintln!(
                "  bench: bench-step-kernel | bench-gemm | bench-gemm-fusion | bench-prefill-attn | bench-prefill-super"
            );
            eprintln!(
                "  bf16/oracle: generate | generate-gpu | generate-monolithic | generate-monolithic-parity | layer0 | gemm | attention"
            );
            eprintln!("  prompts: chat template applied by default; --raw for bare BPE");
            eprintln!("  bad args print a per-command usage line (e.g. `replay` with no file)");
            std::process::exit(2);
        }
    };

    Cli {
        model_dir,
        command,
        raw_prompt,
    }
}
