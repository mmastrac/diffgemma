mod config;
#[allow(dead_code)]
mod kernels;
mod model;
mod safetensors;
mod tensor;
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
}

fn main() -> ExitCode {
    let cli = parse_cli();
    eprintln!("loading from {}", cli.model_dir.display());

    match model::Model::open(&cli.model_dir) {
        Ok(m) => run_command(&m, cli.command),
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
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
    }
}

fn parse_cli() -> Cli {
    let mut args = env::args().skip(1);
    let mut model_dir = PathBuf::from("model/transformer");
    let mut positional = Vec::new();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-m" | "--model" => {
                if let Some(path) = args.next() {
                    model_dir = PathBuf::from(path);
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
        Some(cmd) => {
            eprintln!("unknown command: {cmd}");
            eprintln!("usage: diffgemma-mps [summary|config|weights <name>|layer0|decoder|prefill]");
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
