mod config;
mod generate;
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
        seed: u64,
        steps: usize,
        prompt_len: usize,
        max_new_tokens: usize,
    },
    Tokenize(String),
    Gemm { size: usize },
}

fn main() -> ExitCode {
    let cli = parse_cli();
    match cli.command {
        Command::Tokenize(text) => run_tokenize(&cli.model_dir, &text),
        Command::Gemm { size } => run_gemm(size),
        command => {
            eprintln!("loading from {}", cli.model_dir.display());
            match model::Model::open(&cli.model_dir) {
                Ok(m) => run_command(&m, command),
                Err(err) => {
                    eprintln!("error: {err}");
                    ExitCode::FAILURE
                }
            }
        }
    }
}

fn run_command(m: &model::Model, command: Command) -> ExitCode {
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
        Command::Decoder => run_decoder_forward(m),
        Command::Prefill => run_prefill(m),
        Command::Generate {
            seed,
            steps,
            prompt_len,
            max_new_tokens,
        } => run_generate(m, seed, steps, prompt_len, max_new_tokens),
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

    while let Some(arg) = args.next() {
        match arg.as_str() {
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
            "--size" => {
                if let Some(v) = args.next() {
                    gemm_size = v.parse().unwrap_or_else(|_| {
                        eprintln!("invalid --size");
                        std::process::exit(2);
                    });
                }
            }
            _ => positional.push(arg),
        }
    }

    let command = match positional.first().map(String::as_str) {
        None | Some("summary") => Command::Summary,
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
            seed,
            steps,
            prompt_len,
            max_new_tokens,
        },
        Some("tokenize") => {
            let text = positional.get(1).cloned().unwrap_or_else(|| {
                eprintln!("usage: diffgemma-mps tokenize <text>");
                std::process::exit(2);
            });
            Command::Tokenize(text)
        }
        Some("gemm") => Command::Gemm { size: gemm_size },
        Some(cmd) => {
            eprintln!("unknown command: {cmd}");
            eprintln!(
                "usage: diffgemma-mps [summary|config|weights <name>|layer0|decoder|prefill|generate|tokenize <text>|gemm]"
            );
            eprintln!("  generate options: --seed N --steps N --prompt-len N --max-new-tokens N");
            eprintln!("  gemm options: --size N (default 512, requires --features metal)");
            std::process::exit(2);
        }
    };

    Cli { model_dir, command }
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
        mask: Some(mask),
    };

    eprintln!(
        "running full decoder forward (canvas={CANVAS_LEN}, kv={KV_LEN}, layers={})...",
        m.config.text_config.num_hidden_layers
    );
    let started = std::time::Instant::now();
    match model::decoder::forward(&m.weights, &m.config, &input, &mut scratch) {
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

fn run_generate(
    m: &model::Model,
    seed: u64,
    steps: usize,
    prompt_len: usize,
    max_new_tokens: usize,
) -> ExitCode {
    let vocab = m.config.text_config.vocab_size;
    let canvas = m.config.canvas_length;

    let mut prompt = vec![0u32; prompt_len];
    for (i, id) in prompt.iter_mut().enumerate() {
        *id = ((i * 131 + 7) % vocab.max(1)) as u32;
    }

    let enc_seq = prompt_len.max(canvas);
    let mut enc_scratch = model::encoder::EncoderScratch::new(enc_seq, &m.config);
    let mut dec_scratch = model::decoder::DecoderScratch::new(canvas, &m.config);

    let gen_cfg = generate::GenerateConfig {
        sampler: sample::SamplerConfig {
            max_denoising_steps: steps.max(1),
            ..sample::SamplerConfig::default()
        },
        max_new_tokens,
        seed,
    };

    eprintln!(
        "running generate (prompt_len={prompt_len}, canvas={canvas}, steps={steps}, max_new_tokens={max_new_tokens}, seed={seed})..."
    );
    let started = std::time::Instant::now();
    match generate::generate(&m.weights, &m.config, &prompt, &gen_cfg, &mut enc_scratch, &mut dec_scratch)
    {
        Ok(out) => {
            let new_tokens = out.token_ids.len().saturating_sub(prompt_len);
            println!("generate ok");
            println!("  total tokens: {}", out.token_ids.len());
            println!("  new tokens:   {new_tokens}");
            println!("  denoise steps run: {}", out.denoise_steps_run);
            println!("  blocks committed:  {}", out.blocks_committed);
            println!("  kv after last block: {}", out.token_ids.len());
            println!("  elapsed: {:.2?}", started.elapsed());
            let preview: Vec<String> = out
                .token_ids
                .iter()
                .take(16)
                .map(|t| t.to_string())
                .collect();
            println!("  token_ids[0..16]: [{}]", preview.join(", "));
            if out.token_ids.len() > 16 {
                let tail_start = out.token_ids.len().saturating_sub(8);
                let tail: Vec<String> = out.token_ids[tail_start..]
                    .iter()
                    .map(|t| t.to_string())
                    .collect();
                println!("  token_ids[last 8]: [{}]", tail.join(", "));
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
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
