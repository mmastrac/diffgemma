mod buffer;
mod config;
mod fast_slice;
mod generate;
mod generate_golden;
#[allow(dead_code)]
mod kernels;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;
mod model;
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
    },
    GenerateGpu {
        prompt: Option<String>,
        seed: u64,
        steps: usize,
        prompt_len: usize,
        max_new_tokens: usize,
        max_layers: Option<usize>,
        write_golden: Option<String>,
    },
    GenerateParity {
        prompt: Option<String>,
        seed: u64,
        steps: usize,
        prompt_len: usize,
        max_new_tokens: usize,
        max_layers: Option<usize>,
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
}

fn main() -> ExitCode {
    let cli = parse_cli();
    match cli.command {
        Command::Tokenize(text) => run_tokenize(&cli.model_dir, &text),
        Command::Gemm { size } => run_gemm(size),
        command => {
            eprintln!("loading from {}", cli.model_dir.display());
            match model::Model::open(&cli.model_dir) {
                Ok(m) => run_command(&m, &cli.model_dir, command),
                Err(err) => {
                    eprintln!("error: {err}");
                    ExitCode::FAILURE
                }
            }
        }
    }
}

fn run_command(m: &model::Model, model_dir: &std::path::Path, command: Command) -> ExitCode {
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
        Command::Prefill => run_prefill(m),
        Command::Generate {
            prompt,
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers,
        } => run_generate(
            m,
            model_dir,
            prompt,
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers,
            false,
            None,
        ),
        Command::GenerateGpu {
            prompt,
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers,
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
            true,
            write_golden,
        ),
        Command::GenerateParity {
            prompt,
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers,
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
            golden,
            compare_cpu,
            write_golden,
        ),
        Command::Tokenize(_) => ExitCode::FAILURE,
        Command::Gemm { .. } => ExitCode::FAILURE,
    }
}

fn parse_cli() -> Cli {
    let mut args = env::args().skip(1);
    let mut model_dir = PathBuf::from("model/transformer");
    let mut positional = Vec::new();
    let mut seed = 42u64;
    let mut steps = 2usize;
    let mut prompt_len = 8usize;
    let mut max_new_tokens = 256usize;
    let mut gemm_size = 512usize;
    let mut prompt: Option<String> = None;
    let mut bench_seq = 16usize;
    let mut bench_kv = 8usize;
    let mut bench_layers = 1usize;
    let mut bench_iters = 3usize;
    let mut parity_seq: Option<usize> = None;
    let mut parity_kv: Option<usize> = None;
    let mut parity_layers: Option<usize> = None;
    let mut golden_name: Option<String> = None;
    let mut compare_cpu = false;
    let mut write_golden: Option<String> = None;

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
                    steps = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --steps");
                        std::process::exit(2);
                    });
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
                }
            }
            "--compare-cpu" => compare_cpu = true,
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
            _ => positional.push(arg),
        }
    }

    let command = match positional.first().map(String::as_str) {
        None => default_generate_command(prompt, seed, steps, prompt_len, max_new_tokens, parity_layers),
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
        Some("generate") => Command::Generate {
            prompt: prompt.clone(),
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers: parity_layers,
        },
        Some("generate-gpu") => Command::GenerateGpu {
            prompt: prompt.clone(),
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers: parity_layers,
            write_golden,
        },
        Some("generate-parity") => Command::GenerateParity {
            prompt: prompt.clone(),
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers: parity_layers,
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
        Some(cmd) => {
            eprintln!("unknown command: {cmd}");
            eprintln!(
                "usage: diffgemma-mps [-p PROMPT] [summary|config|weights <name>|layer0|decoder|decoder-gpu|prefill|generate|generate-gpu|generate-parity|tokenize <text>|gemm|attention]"
            );
            eprintln!("  default (no command): generate-gpu with --features metal");
            eprintln!("  generate-parity: GPU vs checked-in golden (use --compare-cpu for slow CPU path)");
            eprintln!("  options: ... --golden NAME --write-golden NAME --compare-cpu");
            eprintln!("  gemm options: --size N (default 512, requires --features metal)");
            eprintln!("  attention: layer 0 GQA parity (requires --features metal)");
            eprintln!("  decoder-gpu: full decoder CPU vs GPU parity at seq=256 (requires --features metal)");
            std::process::exit(2);
        }
    };

    Cli { model_dir, command }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn default_generate_command(
    prompt: Option<String>,
    seed: u64,
    steps: usize,
    prompt_len: usize,
    max_new_tokens: usize,
    max_layers: Option<usize>,
) -> Command {
    Command::GenerateGpu {
        prompt,
        seed,
        steps,
        prompt_len,
        max_new_tokens,
        max_layers,
        write_golden: None,
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn default_generate_command(
    prompt: Option<String>,
    seed: u64,
    steps: usize,
    prompt_len: usize,
    max_new_tokens: usize,
    max_layers: Option<usize>,
) -> Command {
    Command::Generate {
        prompt,
        seed,
        steps,
        prompt_len,
        max_new_tokens,
        max_layers,
    }
}

fn print_summary(store: &weights::WeightStore) {
    let s = store.summarize();

    println!("DiffusionGemma weight summary");
    println!("  model dir:          {}", store.model_dir.display());
    println!("  shards:             {}", s.shard_count);
    println!("  tensors (index):    {}", s.tensor_count_index);
    println!("  tensors (headers):  {}", s.tensor_count_headers);
    println!("  on-disk total:      {:.2} GiB", gib(s.total_file_bytes));
    println!("  tensor payload:     {:.2} GiB", gib(s.total_data_bytes));
    println!("  total elements:     {}", s.total_elements);

    if let Some(params) = store.metadata.get("total_parameters") {
        println!("  index metadata total_parameters: {params}");
    }
    if let Some(size) = store.metadata.get("total_size") {
        println!("  index metadata total_size:       {size}");
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
    for shard in &store.shards {
        let payload: u64 = shard.tensors.iter().map(|t| t.data_size as u64).sum();
        println!(
            "    {}  {:>4} tensors  {:.2} GiB payload  {:.2} GiB file",
            shard.path.file_name().unwrap().to_string_lossy(),
            shard.tensors.len(),
            gib(payload),
            gib(shard.file_size() as u64),
        );
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
        let input = DecoderForwardInput {
            token_ids: &token_ids,
            kv_cache: &kv_cache,
            self_conditioning_logits: None,
            mask: Some(&mask),
        };

        let est = estimate_decoder_forward(&m.config.text_config, canvas_len, kv_len);
        est.print_summary("decoder-gpu (single-path)");

        let mut gpu_scratch = GpuDecoderScratch::new(canvas_len, &m.config);
        let mut gpu_weights = match load_weight_cache(&m.weights, &m.config.text_config) {
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

        eprintln!(
            "running GPU decoder forward (canvas={canvas_len}, kv={kv_len}, layers={max_layers})..."
        );
        let gpu_started = std::time::Instant::now();
        let bench_cfg = BenchConfig { max_layers };
        let gpu_out = match bench_forward(
            &m.weights,
            &m.config,
            &input,
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
            &input,
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
        let input = DecoderForwardInput {
            token_ids: &token_ids,
            kv_cache: &kv_cache,
            self_conditioning_logits: None,
            mask: Some(&mask),
        };

        let mut scratch = GpuDecoderScratch::new(seq_len, &m.config);
        let mut gpu_weights = match load_weight_cache(&m.weights, &m.config.text_config) {
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

        let bench = BenchConfig { max_layers: layers };
        eprintln!(
            "bench-decoder warmup (seq={seq_len}, kv={kv_len}, layers={layers})..."
        );
        if let Err(err) = bench_forward(
            &m.weights,
            &m.config,
            &input,
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
                &input,
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
    let input = model::decoder::DecoderForwardInput {
        token_ids: &token_ids,
        kv_cache: &kv_cache,
        self_conditioning_logits: None,
        mask: Some(&mask),
    };

    eprintln!(
        "running full decoder forward (canvas={CANVAS_LEN}, kv={KV_LEN}, layers={})...",
        m.config.text_config.num_hidden_layers
    );
    let started = std::time::Instant::now();
    match model::decoder::forward(&m.weights, &m.config, &input, &mut scratch, None) {
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

fn run_tokenize(model_dir: &PathBuf, text: &str) -> ExitCode {
    let path = model_dir.join("tokenizer.json");
    match tokenizer::Tokenizer::load(&path) {
        Ok(tok) => {
            let ids = tok.encode(text, false);
            let payload = serde_json::json!({
                "text": text,
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
) -> Result<Vec<u32>, safetensors::Error> {
    if let Some(text) = prompt_text {
        let tok_path = model_dir.join("tokenizer.json");
        let tokenizer = tokenizer::Tokenizer::load(&tok_path)?;
        Ok(tokenizer.encode(text, false))
    } else {
        let mut prompt = vec![0u32; prompt_len];
        for (i, id) in prompt.iter_mut().enumerate() {
            *id = ((i * 131 + 7) % vocab.max(1)) as u32;
        }
        Ok(prompt)
    }
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

    if let Ok(tokenizer) = tokenizer::Tokenizer::load(model_dir.join("tokenizer.json")) {
        let text = tokenizer.decode(&out.token_ids);
        if !text.is_empty() {
            let preview: String = text.chars().take(200).collect();
            println!("  text: {preview}");
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

fn infer_golden_name(prompt_text: Option<&str>, steps: usize, max_layers: Option<usize>) -> Option<String> {
    if prompt_text != Some("Hello") {
        return None;
    }
    match (steps, max_layers) {
        (1, None) => Some("hello_steps1_full".into()),
        (1, Some(3)) => Some("hello_steps1_layers3".into()),
        (2, Some(3)) => Some("hello_steps2_layers3".into()),
        (2, None) => Some("hello_steps2_full".into()),
        _ => None,
    }
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
    use_gpu: bool,
    write_golden: Option<String>,
) -> ExitCode {
    let vocab = m.config.text_config.vocab_size;
    let canvas = m.config.canvas_length;

    let prompt = match build_prompt_tokens(
        model_dir,
        prompt_text.as_deref(),
        prompt_len,
        vocab,
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
        sampler: sample::SamplerConfig {
            max_denoising_steps: steps.max(1),
            ..sample::SamplerConfig::default()
        },
        max_new_tokens,
        seed,
        max_layers,
    };

    let backend = if use_gpu { "generate-gpu" } else { "generate" };
    let layers_note = max_layers
        .map(|n| format!(", layers={n}"))
        .unwrap_or_default();
    eprintln!(
        "running {backend} (prompt_len={prompt_len}, canvas={canvas}, steps={steps}, max_new_tokens={max_new_tokens}, seed={seed}{layers_note})..."
    );
    let started = std::time::Instant::now();

    #[cfg(all(feature = "metal", target_os = "macos"))]
    if use_gpu {
        use metal::{load_weight_cache, GpuDecoderEngine, GpuDecoderScratch};
        let mut dec_scratch = GpuDecoderScratch::new(canvas, &m.config);
        let mut gpu_weights = match load_weight_cache(&m.weights, &m.config.text_config) {
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
                    if let Err(err) = write_generate_golden(
                        &golden_name,
                        &prompt_str,
                        &gen_cfg,
                        steps,
                        &out,
                    ) {
                        eprintln!("error: {err}");
                        return ExitCode::FAILURE;
                    }
                }
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
    out: &generate::GenerateOutput,
) -> Result<(), safetensors::Error> {
    let golden = generate_golden::GenerateGolden::from_run(name, prompt, gen_cfg, steps, out);
    let path = generate_golden::resolve_fixture(name);
    golden.write(&path)?;
    eprintln!("wrote golden {} ({} tokens)", path.display(), out.token_ids.len());
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
    golden_name: Option<String>,
    compare_cpu: bool,
    write_golden: Option<String>,
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
        ) {
            Ok(ids) => ids,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };

        let gen_cfg = generate::GenerateConfig {
            sampler: sample::SamplerConfig {
                max_denoising_steps: steps.max(1),
                ..sample::SamplerConfig::default()
            },
            max_new_tokens,
            seed,
            max_layers,
        };

        let prompt_label = prompt_text.clone().unwrap_or_else(|| format!("prompt_len={prompt_len}"));

        if let Some(n) = max_layers {
            eprintln!("generate-parity: decoder layers limited to {n}");
        }

        let enc_seq = prompt.len().max(canvas);
        let mut enc_gpu = model::encoder::EncoderScratch::new(enc_seq, &m.config);
        let mut dec_gpu = GpuDecoderScratch::new(canvas, &m.config);
        let mut gpu_weights = match load_weight_cache(&m.weights, &m.config.text_config) {
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

        if let Some(ref name) = write_golden {
            if let Err(err) = write_generate_golden(name, &prompt_label, &gen_cfg, steps, &gpu_out) {
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
            let fixture = match golden_name
                .or_else(|| infer_golden_name(prompt_text.as_deref(), steps, max_layers))
            {
                Some(name) => name,
                None => {
                    eprintln!("error: no --golden fixture; use --write-golden NAME or --compare-cpu");
                    return ExitCode::FAILURE;
                }
            };
            let path = generate_golden::resolve_fixture(&fixture);
            eprintln!("checking golden {}...", path.display());
            let golden = match GenerateGolden::load(&path) {
                Ok(g) => g,
                Err(err) => {
                    eprintln!("error: {err}");
                    return ExitCode::FAILURE;
                }
            };
            if !golden.matches_config(&prompt_label, &gen_cfg, steps) {
                eprintln!("error: golden config mismatch for {}", golden.name);
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
