//! Tiny character-level GPT running on the portable Metal/CUDA ops in dgops.
//!
//!   nanogpt --check                  forward parity: GPU composition vs the
//!                                    independent CPU reference
//!   nanogpt --train --steps 2000     train on the corpus, print loss
//!   nanogpt --sample --prompt "ROMEO: " --tokens 400
//!
//! --data PATH replaces the embedded tiny-shakespeare slice; --seed makes a
//! run reproducible. The backend is whichever dgops found (Metal on macOS,
//! CUDA elsewhere when built with --features cuda).

mod backward;
mod data;
mod model;
mod reference;
mod tensor;
mod train;

use dgops::backend;
use model::{GptConfig, Rng, Weights};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Default)]
struct Args {
    check: bool,
    train: bool,
    gradcheck: bool,
    steps: usize,
    batch: usize,
    lr: f32,
    sample: bool,
    prompt: String,
    tokens: usize,
    temperature: f32,
    seed: u64,
    data: Option<PathBuf>,
}

fn parse() -> Result<Args, String> {
    let mut args = Args {
        steps: 2000,
        batch: 8,
        lr: 3e-3,
        tokens: 400,
        temperature: 0.8,
        seed: 1337,
        prompt: "ROMEO: ".to_string(),
        ..Default::default()
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--check" => args.check = true,
            "--train" => args.train = true,
            "--gradcheck" => args.gradcheck = true,
            "--sample" => args.sample = true,
            "--steps" => args.steps = value()?.parse().map_err(|e| format!("--steps: {e}"))?,
            "--batch" => args.batch = value()?.parse().map_err(|e| format!("--batch: {e}"))?,
            "--lr" => args.lr = value()?.parse().map_err(|e| format!("--lr: {e}"))?,
            "--tokens" => args.tokens = value()?.parse().map_err(|e| format!("--tokens: {e}"))?,
            "--temperature" => {
                args.temperature = value()?
                    .parse()
                    .map_err(|e| format!("--temperature: {e}"))?
            }
            "--seed" => args.seed = value()?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--prompt" => args.prompt = value()?,
            "--data" => args.data = Some(PathBuf::from(value()?)),
            "--help" | "-h" => {
                println!("{}", USAGE);
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag {other}\n\n{USAGE}")),
        }
    }
    if !args.check && !args.train && !args.sample && !args.gradcheck {
        args.check = true;
    }
    Ok(args)
}

const USAGE: &str = "\
usage: nanogpt [--check] [--train] [--sample] [options]
  --check              forward parity against the CPU reference (default)
  --gradcheck          finite-difference check of the analytic gradients
  --train              train on the corpus
  --sample             generate text
  --steps N            training steps (default 2000)
  --batch N            sequences per step (default 8)
  --lr F               AdamW learning rate (default 3e-3)
  --tokens N           tokens to generate (default 400)
  --temperature F      sampling temperature (default 0.8)
  --prompt S           sampling prompt (default \"ROMEO: \")
  --seed N             RNG seed (default 1337)
  --data PATH          corpus file (default: embedded tiny shakespeare)";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("nanogpt: {e}");
            ExitCode::FAILURE
        }
    }
}

fn load_corpus(args: &Args) -> Result<data::Corpus, String> {
    let text = match &args.data {
        Some(path) => {
            std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?
        }
        None => data::DEFAULT_TEXT.to_string(),
    };
    Ok(data::Corpus::from_str(&text))
}

fn run() -> Result<(), String> {
    let args = parse()?;
    let backend = backend::available()
        .ok_or("no GPU backend in this build (build with --features cuda on a CUDA host)")?;
    println!("backend: {backend}");

    let corpus = load_corpus(&args)?;
    let cfg = GptConfig::tiny(corpus.vocab_size());
    let mut weights = Weights::random(&cfg, args.seed);
    println!(
        "corpus: {} chars, vocab {} | model: {} layers, {} heads, dim {}, block {} | params {}",
        corpus.len(),
        cfg.vocab,
        cfg.n_layer,
        cfg.n_head,
        cfg.n_embd,
        cfg.block,
        param_count(&cfg)
    );

    if args.gradcheck {
        train::gradcheck(&mut weights, &cfg, &corpus)?;
    }

    if args.train {
        train::train(&mut weights, &cfg, &corpus, &args)?;
    }

    if args.check {
        check(&weights, &cfg, &corpus, args.seed)?;
    }

    if args.sample {
        sample(&weights, &cfg, &corpus, &args)?;
    }
    Ok(())
}

fn param_count(cfg: &GptConfig) -> usize {
    let c = cfg.n_embd;
    let per_layer = c + 3 * c * c + c * c + c + cfg.mlp() * c + c * cfg.mlp();
    cfg.vocab * c + cfg.block * c + cfg.n_layer * per_layer + c + cfg.vocab * c
}

fn check(w: &Weights, cfg: &GptConfig, corpus: &data::Corpus, seed: u64) -> Result<(), String> {
    let (tokens, _) = corpus.batch(seed as usize, 2, cfg.block);
    let gpu = model::forward(w, cfg, &tokens, cfg.block).map_err(|e| e.to_string())?;
    let cpu = reference::forward(w, cfg, &tokens, cfg.block);
    let max = gpu
        .logits
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cos = cosine(&gpu.logits, &cpu);
    println!(
        "check: {} logits | max_abs {max:.3e} | cos {cos:.6}",
        gpu.logits.len()
    );
    if cos < 0.9999 || max > 1e-2 {
        return Err(format!(
            "GPU forward diverges from the CPU reference (max_abs {max:.3e}, cos {cos:.6})"
        ));
    }
    println!("check: OK");
    Ok(())
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += (x as f64) * (y as f64);
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn sample(w: &Weights, cfg: &GptConfig, corpus: &data::Corpus, args: &Args) -> Result<(), String> {
    let mut rng = Rng::new(args.seed ^ 0x5eed);
    let mut ids = corpus.encode(&args.prompt);
    if ids.is_empty() {
        ids.push(0);
    }
    print!("{}", args.prompt);
    for _ in 0..args.tokens {
        let ctx_len = ids.len().min(cfg.block);
        let ctx = &ids[ids.len() - ctx_len..];
        let cache = model::forward(w, cfg, ctx, ctx_len).map_err(|e| e.to_string())?;
        let base = (ctx_len - 1) * cfg.vocab;
        let mut probs: Vec<f32> = cache.logits[base..base + cfg.vocab]
            .iter()
            .map(|v| v / args.temperature)
            .collect();
        let max = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for p in probs.iter_mut() {
            *p = (*p - max).exp();
            sum += *p;
        }
        for p in probs.iter_mut() {
            *p /= sum;
        }
        let r = rng.uniform() as f32;
        let mut acc = 0.0f32;
        let mut next = cfg.vocab - 1;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if r <= acc {
                next = i;
                break;
            }
        }
        ids.push(next as u32);
        print!("{}", corpus.decode(&[next as u32]));
    }
    println!();
    Ok(())
}
