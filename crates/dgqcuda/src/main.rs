//! `dgqcuda` — the CUDA inference path for a DiffusionGemma `.dgq` pack.
//!
//! Subcommands:
//!   forward   run the forward pass and print the next-token distribution
//!   parity    CPU oracle vs CUDA forward pass on the same prompt

mod config;
mod denoise;
mod forward;
mod gpu;
mod moe_grouped;
mod weights;

use config::ModelConfig;
use forward::Scratch;
use weights::Weights;

/// fixtures/golden/golden.json -> case `engine_prefill` (seed 7).
pub const GOLDEN_PROMPT_IDS: &[u32] = &[
    2, 105, 2364, 107, 3689, 563, 506, 5279, 529, 7001, 236881, 106, 107, 105, 4368, 107, 100,
    45518, 107, 101,
];

struct Args {
    cmd: String,
    model: String,
    ids: Vec<u32>,
    layers: Option<usize>,
    at: u8,
    seed: u64,
    /// `--gpu`: run the device path (default is the CPU oracle).
    gpu: bool,
    /// `--rows all`: compute logits for every canvas position.
    rows_all: bool,
    /// Diagnostic stop point for the device path (see `forward_stop`).
    stop_after: u8,
    /// `--row N`: single-row logits at position N.
    only_row: Option<usize>,
    /// `--canvas N`: denoise canvas width (default 256).
    canvas: usize,
    /// `--steps N`: denoise steps (default 48).
    steps: Option<usize>,
    /// `--parity`: also run the CPU oracle for each step and compare logits.
    parity: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut cmd = "forward".to_string();
    let mut model = String::new();
    let mut ids: Vec<u32> = GOLDEN_PROMPT_IDS.to_vec();
    let mut layers = None;
    let mut at = 0u8;
    let mut seed = 7u64;
    let mut gpu = false;
    let mut rows_all = false;
    let mut only_row: Option<usize> = None;
    let mut canvas = 256usize;
    let mut steps: Option<usize> = None;
    let mut parity = false;
    let mut stop_after = 0u8;
    let mut it = std::env::args().skip(1);
    if let Some(first) = it.next() {
        if !first.starts_with('-') {
            cmd = first;
        } else {
            return Err(format!("unexpected argument {first}"));
        }
    }
    while let Some(a) = it.next() {
        match a.as_str() {
            "-m" | "--model" => model = it.next().ok_or("--model needs a value")?,
            "--ids" => {
                let s = it.next().ok_or("--ids needs a value")?;
                ids = s
                    .split(',')
                    .map(|t| t.trim().parse::<u32>().map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--at" => {
                let s = it.next().ok_or("--at needs a value")?;
                at = s.parse::<u8>().map_err(|e| e.to_string())?;
            }
            "--layers" => {
                let s = it.next().ok_or("--layers needs a value")?;
                layers = Some(s.parse::<usize>().map_err(|e| e.to_string())?);
            }
            "--seed" => {
                let s = it.next().ok_or("--seed needs a value")?;
                seed = s.parse::<u64>().map_err(|e| e.to_string())?;
            }
            "--canvas" => {
                let s = it.next().ok_or("--canvas needs a value")?;
                canvas = s.parse::<usize>().map_err(|e| e.to_string())?;
            }
            "--steps" => {
                let s = it.next().ok_or("--steps needs a value")?;
                steps = Some(s.parse::<usize>().map_err(|e| e.to_string())?);
            }
            "--parity" => parity = true,
            "--gpu" => gpu = true,
            "--stop-after" => {
                let s = it.next().ok_or("--stop-after needs a value")?;
                stop_after = s.parse::<u8>().map_err(|e| e.to_string())?;
            }
            "--rows" => {
                let s = it.next().ok_or("--rows needs a value")?;
                rows_all = match s.as_str() {
                    "all" => true,
                    "last" => false,
                    other => return Err(format!("--rows takes all|last, got {other}")),
                };
            }
            "--row" => {
                let s = it.next().ok_or("--row needs a value")?;
                only_row = Some(s.parse::<usize>().map_err(|e| e.to_string())?);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if model.is_empty() {
        return Err("pass -m <pack dir>".to_string());
    }
    Ok(Args {
        cmd,
        model,
        ids,
        layers,
        at,
        seed,
        gpu,
        rows_all,
        stop_after,
        only_row,
        canvas,
        steps,
        parity,
    })
}

fn top_k(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx.into_iter().map(|i| (i as u32, logits[i])).collect()
}

fn entropy(logits: &[f32]) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    let mut acc = 0.0f32;
    for &v in logits {
        let p = (v - max).exp();
        sum += p;
        acc += p * (v - max);
    }
    -(acc / sum) + sum.ln()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!(
                "usage: dgqcuda [forward|parity] -m <pack dir> [--ids 1,2,3] [--layers N] [--seed S]"
            );
            std::process::exit(2);
        }
    };
    if let Err(e) = run(&args) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(args: &Args) -> Result<(), config::Error> {
    let cfg = ModelConfig::load(&args.model)?;
    let t = &cfg.text_config;
    eprintln!(
        "model: {} layers, hidden {}, {} experts top-{}, vocab {}",
        t.num_hidden_layers, t.hidden_size, t.num_experts, t.top_k_experts, t.vocab_size
    );
    let load_start = std::time::Instant::now();
    let w = Weights::open(&args.model, &cfg)?;
    eprintln!("pack opened in {:.1}s", load_start.elapsed().as_secs_f32());

    let seq = args.ids.len();
    let mut sc = Scratch::new(seq, &cfg);

    match args.cmd.as_str() {
        "forward" => {
            let start = std::time::Instant::now();
            let rows = match args.only_row {
                Some(r) => forward::LogitRows::Only(r),
                None if args.rows_all => forward::LogitRows::All,
                None => forward::LogitRows::Last,
            };
            let out = if args.gpu {
                gpu::forward_stop(
                    &w,
                    &cfg,
                    &args.ids,
                    args.layers,
                    rows,
                    &mut sc,
                    args.stop_after,
                )?
            } else {
                forward::forward(&w, &cfg, &args.ids, args.layers, rows, &mut sc)?
            };
            let elapsed = start.elapsed();
            let rows_done = out.logits.len() / t.vocab_size;
            let last_row = args
                .only_row
                .unwrap_or(if args.rows_all { seq - 1 } else { seq - 1 });
            let last = &out.logits[last_row.min(rows_done - 1) * t.vocab_size
                ..(last_row.min(rows_done - 1) + 1) * t.vocab_size];
            println!(
                "forward: {} tokens x {} layers in {:.1}s ({} logit rows)",
                seq,
                args.layers.unwrap_or(t.num_hidden_layers),
                elapsed.as_secs_f32(),
                rows_done
            );
            println!(
                "  logits: mean {:.4} std {:.4} entropy {:.4} nats",
                last.iter().sum::<f32>() / last.len() as f32,
                {
                    let m = last.iter().sum::<f32>() / last.len() as f32;
                    (last.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / last.len() as f32).sqrt()
                },
                entropy(last)
            );
            println!("  next-token top-5:");
            for (id, v) in top_k(last, 5) {
                println!("    {id:>7}  {v:+.4}");
            }
            if args.rows_all {
                println!("  per-row argmax / entropy (row 0, 1, last):");
                for r in [0usize, 1, seq - 1] {
                    let row = &out.logits[r * t.vocab_size..(r + 1) * t.vocab_size];
                    println!(
                        "    row {r:>3}: argmax {:>7} entropy {:.4}",
                        top_k(row, 1)[0].0,
                        entropy(row)
                    );
                }
            }
        }
        "parity" => {
            let cpu = forward::forward(
                &w,
                &cfg,
                &args.ids,
                args.layers,
                forward::LogitRows::All,
                &mut sc,
            )?;
            let mut gpu_sc = Scratch::new(seq, &cfg);
            let gpu = gpu::forward(
                &w,
                &cfg,
                &args.ids,
                args.layers,
                forward::LogitRows::All,
                &mut gpu_sc,
            )?;
            let n = cpu.logits.len();
            let (a, b) = if gpu.logits.len() == cpu.logits.len() {
                (cpu.logits.as_slice(), gpu.logits.as_slice())
            } else {
                (
                    &cpu.logits[(seq - 1) * t.vocab_size..seq * t.vocab_size],
                    gpu.logits.as_slice(),
                )
            };
            let cos = cosine(a, b);
            let max_abs = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            println!("parity over {n} logits: cos {cos:.7} max_abs {max_abs:.3e}");
            let stat = |v: &[f32]| {
                let m = v.iter().sum::<f32>() / v.len() as f32;
                let s = (v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / v.len() as f32).sqrt();
                (m, s)
            };
            let (cm, cs) = stat(a);
            let (gm, gs) = stat(b);
            println!("  cpu logits mean {cm:.4} std {cs:.4} | gpu mean {gm:.4} std {gs:.4}");
            let last_cpu = &cpu.logits[(seq - 1) * t.vocab_size..seq * t.vocab_size];
            let last_gpu = if gpu.logits.len() == t.vocab_size {
                &gpu.logits[..]
            } else {
                &gpu.logits[(seq - 1) * t.vocab_size..seq * t.vocab_size]
            };
            println!("  cpu top-5: {:?}", top_k(last_cpu, 5));
            println!("  gpu top-5: {:?}", top_k(last_gpu, 5));
            let same = top_k(last_cpu, 1)[0].0 == top_k(last_gpu, 1)[0].0;
            println!("  argmax match: {same}");
            if cos < 0.999 {
                return Err(config::Error::Msg(format!("parity failed: cos {cos}")));
            }
        }
        "stage" => {
            let n = args
                .layers
                .unwrap_or(t.num_hidden_layers)
                .min(t.num_hidden_layers);
            let cpu = forward::hidden_after(&w, &cfg, &args.ids, n, args.at, &mut sc)?;
            let gpu = gpu::hidden_after(
                &w,
                &cfg,
                &args.ids,
                n,
                args.at,
                &mut Scratch::new(seq, &cfg),
            )?;
            let cos = cosine(&cpu, &gpu);
            let mad = cpu
                .iter()
                .zip(gpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let scale = cpu.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
            println!(
                "stage {n} (at {}): hidden cos {cos:.7} max_abs {mad:.3e} rel {:.2e}",
                args.at,
                mad / scale
            );
            println!(
                "  len cpu {} gpu {} | cpu[0..4]={:?} gpu[0..4]={:?}",
                cpu.len(),
                gpu.len(),
                &cpu[..4.min(cpu.len())],
                &gpu[..4.min(gpu.len())]
            );
            let mut worst = (0usize, 0.0f32);
            for i in 0..cpu.len().min(gpu.len()) {
                let d = (cpu[i] - gpu[i]).abs();
                if d > worst.1 {
                    worst = (i, d);
                }
            }
            println!(
                "  worst idx {} cpu={} gpu={}",
                worst.0, cpu[worst.0], gpu[worst.0]
            );
        }
        "attn" => {
            let cpu = forward::attn_stage(&w, &cfg, &args.ids, &mut sc)?;
            let gpu = gpu::attn_stage(&w, &cfg, &args.ids, &mut Scratch::new(seq, &cfg))?;
            let cos = cosine(&cpu, &gpu);
            let mad = cpu
                .iter()
                .zip(gpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let scale = cpu.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
            println!(
                "attn stage: cos {cos:.7} max_abs {mad:.3e} rel {:.2e}",
                mad / scale
            );
            for i in 0..4 {
                println!("  i={i} cpu={} gpu={}", cpu[i], gpu[i]);
            }
        }
        "denoise" => {
            let canvas = args.canvas;
            let prompt = &args.ids;
            let mut dcfg = denoise::SamplerConfig::default();
            if let Some(n) = args.steps {
                dcfg.max_denoising_steps = n;
            }
            let mut st = denoise::DenoiseState::new(dcfg, args.seed, canvas, t.vocab_size);
            let mut sess = gpu::session::Session::open(&w, &cfg, prompt.len(), canvas)?;
            if let Some(n) = args.layers {
                sess.set_layers(n);
            }
            let load = std::time::Instant::now();
            let ph = sess.prompt_hidden(prompt)?;
            eprintln!("prompt prefill in {:.1}s", load.elapsed().as_secs_f32());
            let mut total = std::time::Duration::ZERO;
            let mut prev: Option<Vec<f32>> = None;
            for _ in 0..st.cfg.max_denoising_steps {
                let start = std::time::Instant::now();
                let logits = sess.step(prompt, &ph, &st.ids)?;
                sess.set_prev_logits(&logits)?;
                if args.parity {
                    // The same step on the CPU oracle: prompt + canvas, the
                    // SC MLP over the canvas rows, all layers, LM head. Its
                    // logits are indexed by sequence position, so the canvas
                    // rows start at `prompt.len()`.
                    let mut ids = prompt.clone();
                    ids.extend_from_slice(&st.ids);
                    let mut csc = Scratch::new(ids.len(), &cfg);
                    let cpu = forward::forward_sc(
                        &w,
                        &cfg,
                        &ids,
                        args.layers,
                        forward::LogitRows::All,
                        &mut csc,
                        prompt.len(),
                        canvas,
                        prev.as_deref(),
                        Some(&ph),
                    )?;
                    // The device step returns pre-softcap logits (the sampler
                    // needs them raw); the CPU oracle leaves them raw too, so
                    // compare both under the same final softcap.
                    let cap = t.final_logit_softcapping as f32;
                    let softcap = |v: f32| if cap > 0.0 { (v / cap).tanh() * cap } else { v };
                    let cpu_raw = &cpu.logits[prompt.len() * t.vocab_size..];
                    let gpu_capped: Vec<f32> = logits.iter().copied().map(softcap).collect();
                    let cpu_capped: Vec<f32> = cpu_raw.iter().copied().map(softcap).collect();
                    let gpu_row = gpu_capped.as_slice();
                    let cpu_row = cpu_capped.as_slice();
                    let cos = cosine(gpu_row, cpu_row);
                    let mad = gpu_row
                        .iter()
                        .zip(cpu_row.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    let (gi, _) = top_k(gpu_row, 1)[0];
                    let (ci, _) = top_k(cpu_row, 1)[0];
                    for row in 0..canvas {
                        let g = &gpu_row[row * t.vocab_size..(row + 1) * t.vocab_size];
                        let c = &cpu_row[row * t.vocab_size..(row + 1) * t.vocab_size];
                        let (gr, _) = top_k(g, 1)[0];
                        let (cr, _) = top_k(c, 1)[0];
                        println!(
                            "    row {row}: cos {:.7} argmax gpu {gr} cpu {cr}",
                            cosine(g, c)
                        );
                    }
                    println!(
                        "  parity vs CPU oracle: cos {cos:.7} max_abs {mad:.3e} argmax gpu {gi} cpu {ci} {}",
                        if gi == ci { "match" } else { "MISMATCH" }
                    );
                }
                prev = Some(logits.clone());
                let (stats, stop) = st.step(&logits, t.vocab_size);
                let dt = start.elapsed();
                total += dt;
                println!(
                    "step {:>2} t={:.3} accept {:>3} mean_H {:.4} min_H {:.4} max_H {:.4} changed {:>3}  {:.1}s",
                    stats.step,
                    stats.temperature,
                    stats.accept_count,
                    stats.mean_entropy,
                    stats.min_entropy,
                    stats.max_entropy,
                    stats.changed,
                    dt.as_secs_f32()
                );
                if let Some(reason) = stop {
                    println!(
                        "stop: {reason:?} after {} steps ({:.1}s total)",
                        stats.step,
                        total.as_secs_f32()
                    );
                    break;
                }
            }
            println!("canvas ({} ids):", st.ids.len());
            let ids: Vec<String> = st.ids.iter().map(|v| v.to_string()).collect();
            println!("{}", ids.join(","));
            let active = st
                .ids
                .iter()
                .filter(|&&v| v != denoise::PAD_TOKEN_ID)
                .count();
            println!("active tokens: {active}/{}", st.ids.len());
        }
        "gemm-probe" => {
            let m = args.layers.unwrap_or(1);
            let v = gpu::gemm_probe(m, t.hidden_size, t.vocab_size)?;
            println!("gemm probe ok: c[0] = {v}");
        }
        other => return Err(config::Error::Msg(format!("unknown command {other}"))),
    }
    Ok(())
}
