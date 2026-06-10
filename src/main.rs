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
        Command::Layer0 => {
            match model::layer_weights::DecoderLayerWeights::load(
                &m.weights,
                0,
                &m.config.text_config,
            ) {
                Ok(layer) => {
                    layer.print_summary();
                    println!("\n  all 22 layer-0 tensors present with expected shapes");
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("error: {err}");
                    ExitCode::FAILURE
                }
            }
        }
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
        Some(cmd) => {
            eprintln!("unknown command: {cmd}");
            eprintln!("usage: diffgemma-mps [summary|config|weights <name>|layer0]");
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
