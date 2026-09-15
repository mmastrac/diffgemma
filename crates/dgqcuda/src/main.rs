//! `dgqcuda` — the CUDA inference path for a DiffusionGemma `.dgq` pack.
//!
//! Subcommands:
//!   forward   run the forward pass and print the next-token distribution
//!   parity    CPU oracle vs CUDA forward pass on the same prompt

mod chat_template;
mod config;
mod denoise;
mod forward;
mod gpu;
mod moe_grouped;
mod tokenizer;
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
    /// `hidden-parity --causal`: compare against a short causal pass over the
    /// prompt rows (the device prompt path's shape) instead of the step.
    causal: bool,
    /// `--prompt TEXT`: render a text prompt through the chat template instead
    /// of using `--ids`.
    prompt: Option<String>,
    /// `--diag`: per-step logit statistics.
    diag: bool,
    /// `--dump-step PATH`: write the first step's raw canvas logits as JSON,
    /// in the shape of the engine's `step-logits-dump`, so the two can be
    /// diffed token by token.
    dump_step: Option<String>,
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
    let mut causal = false;
    let mut prompt = None;
    let mut diag = false;
    let mut dump_step: Option<String> = None;
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
            "--causal" => causal = true,
            "--prompt" => prompt = Some(it.next().ok_or("--prompt needs a value")?),
            "--diag" => diag = true,
            "--dump-step" => dump_step = Some(it.next().ok_or("--dump-step needs a path")?),
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
        causal,
        prompt,
        diag,
        dump_step,
    })
}

/// Write one step's raw canvas logits in the engine's step-logits-dump shape.
fn dump_step_json(
    path: &str,
    canvas_ids: &[u32],
    logits: &[f32],
    vocab: usize,
    prompt_ids: &[u32],
    prompt_text: &str,
    softcap: Option<f32>,
) -> Result<(), config::Error> {
    let rows = canvas_ids.len();
    let mut out = String::new();
    out.push_str(&format!("{{\"rows\": ["));
    for r in 0..rows {
        if r > 0 {
            out.push(',');
        }
        let raw = &logits[r * vocab..(r + 1) * vocab];
        let row: Vec<f32> = match softcap {
            Some(cap) => raw.iter().map(|v| (v / cap).tanh() * cap).collect(),
            None => raw.to_vec(),
        };
        let row = &row[..];
        let mut idx: Vec<usize> = (0..vocab).collect();
        idx.sort_by(|&a, &b| {
            row[b]
                .partial_cmp(&row[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let am = idx[0];
        let top: Vec<String> = idx[..16]
            .iter()
            .map(|&i| format!("{{\"token\":{i},\"logit_raw\":{}}}", row[i]))
            .collect();
        out.push_str(&format!(
            "{{\"position\":{r},\"canvas_token\":{},\"argmax_raw\":{am},\"logit_raw_at_argmax\":{},\"token_logits\":[{}]}}",
            canvas_ids[r],
            row[am],
            top.join(",")
        ));
    }
    out.push_str(&format!(
        "], \"prompt_token_ids\": [{}], \"prompt\": {:?}, \"vocab\": {vocab}}}",
        prompt_ids
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(","),
        prompt_text
    ));
    std::fs::write(path, out)?;
    eprintln!("wrote {path} ({rows} rows)");
    Ok(())
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

    // A text prompt is rendered through the chat template; without one the
    // golden token ids are used, so the existing probes keep working.
    let tok = if args.prompt.is_some() {
        Some(tokenizer::Tokenizer::load(
            std::path::Path::new(&args.model).join("tokenizer.json"),
        )?)
    } else {
        None
    };
    let ids: Vec<u32> = match (&args.prompt, &tok) {
        (Some(text), Some(tok)) => {
            let rendered = chat_template::user_prompt_ids(tok, text)?;
            eprintln!("prompt: {} ids", rendered.len());
            eprintln!("  ids {rendered:?}");
            eprintln!("  text {:?}", tok.decode(&rendered));
            rendered
        }
        _ => args.ids.clone(),
    };
    let ids = ids.as_slice();
    let seq = ids.len();
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
                gpu::forward_stop(&w, &cfg, ids, args.layers, rows, &mut sc, args.stop_after)?
            } else {
                forward::forward(&w, &cfg, ids, args.layers, rows, &mut sc)?
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
        // Bisect the device prompt path against the CPU oracle one layer at a
        // time: both run the same causal pass over the prompt tokens, so the
        // first layer whose cosine drops below 1 is where they diverge.
        "hidden-parity" => {
            let mut sess = gpu::session::Session::open(&w, &cfg, seq, 1)?;
            let n = args.layers.unwrap_or(t.num_hidden_layers);
            let prev = std::cell::Cell::new(1.0f32);
            for layer in 1..=n {
                let g = sess.prompt_hidden_after(ids, layer)?;
                let mut csc = Scratch::new(seq, &cfg);
                let c = if args.causal {
                    // The device prompt path runs a short causal pass over the
                    // prompt rows only; compare it against the same pass, not
                    // against the oracle's [prompt][canvas] step.
                    forward::causal_hidden_after(&w, &cfg, ids, layer, &mut csc)?
                } else {
                    forward::hidden_after(&w, &cfg, ids, layer, 0, &mut csc)?
                };
                let cos = cosine(&g, &c);
                let mad = g
                    .iter()
                    .zip(c.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                println!("  layer {layer:>2}: cos {cos:.7} max_abs {mad:.3e}");
                if cos < 0.9999 && prev.get() >= 0.9999 {
                    println!("  first divergence at layer {layer}");
                    for i in 0..8 {
                        println!("    gpu[{i}] {:.6} cpu[{i}] {:.6}", g[i], c[i]);
                    }
                }
                prev.set(cos);
            }
        }
        "parity" => {
            let cpu =
                forward::forward(&w, &cfg, ids, args.layers, forward::LogitRows::All, &mut sc)?;
            let mut gpu_sc = Scratch::new(seq, &cfg);
            let gpu = gpu::forward(
                &w,
                &cfg,
                ids,
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
            let cpu = forward::hidden_after(&w, &cfg, ids, n, args.at, &mut sc)?;
            let gpu = gpu::hidden_after(&w, &cfg, ids, n, args.at, &mut Scratch::new(seq, &cfg))?;
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
            let cpu = forward::attn_stage(&w, &cfg, ids, &mut sc)?;
            let gpu = gpu::attn_stage(&w, &cfg, ids, &mut Scratch::new(seq, &cfg))?;
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
            let prompt = ids;
            let mut dcfg = denoise::SamplerConfig::default();
            if let Some(n) = args.steps {
                dcfg.max_denoising_steps = n;
            }
            let mut st = denoise::DenoiseState::new(dcfg, args.seed, canvas, t.vocab_size);
            let prompt_ids = prompt.to_vec();
            let mut sess = gpu::session::Session::open(&w, &cfg, prompt.len(), canvas)?;
            if let Some(n) = args.layers {
                sess.set_layers(n);
            }
            let load = std::time::Instant::now();
            // Warm the device (module load and first-touch allocs land on the
            // first step) and report the CPU causal prefill scale. Note what
            // this does NOT do: it never compares `prompt_hidden`. That helper
            // runs the prompt standalone, which is a different code path from
            // the step's [prompt][canvas] sequence, and probing it instead of
            // the step is how a double-applied prompt residual stream hid here.
            sess.warm(prompt)?;
            eprintln!("prompt prefill in {:.1}s", load.elapsed().as_secs_f32());
            if args.diag {
                let mut csc = Scratch::new(prompt_ids.len(), &cfg);
                let c = forward::causal_hidden_after(
                    &w,
                    &cfg,
                    &prompt_ids,
                    t.num_hidden_layers,
                    &mut csc,
                )?;
                let scale = c.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                eprintln!(
                    "  [diag] cpu causal prompt hidden rows {} max_abs {scale:.3} row0[0..4] {:?}",
                    c.len() / t.hidden_size,
                    &c[..4]
                );
            }
            let mut total = std::time::Duration::ZERO;
            let mut prev: Option<Vec<f32>> = None;
            let mut step_no = 0usize;
            for _ in 0..st.cfg.max_denoising_steps {
                let start = std::time::Instant::now();
                let mut logits = sess.step(prompt, &st.ids)?;
                // The engine softcaps inside the step, BEFORE the sampler: its
                // StepStage::Softcap runs ahead of SampleRowstats, and the
                // comment there notes that sample_rowstats reads post-softcap
                // logits. Sampling the raw row samples a far sharper
                // distribution than the model means to expose -- the softcap's
                // derivative tapers the tail, so the raw row spans about 175
                // where the capped one spans about 84.
                let cap = t.final_logit_softcapping as f32;
                if cap > 0.0 {
                    for v in logits.iter_mut() {
                        *v = (*v / cap).tanh() * cap;
                    }
                }
                // `--dump-step PATH` writes step 1 to PATH.
                // `DGQCUDA_DUMP_STEP_ALL=1` also writes every later step, to
                // PATH with `.stepN` inserted before the extension. A defect
                // that appears BETWEEN steps is invisible in a step-1 dump,
                // and step 1 is all either side could dump until now -- the
                // engine's `step-logits-dump --steps N` runs one forward
                // whatever N says, and this filter did the same here.
                if let Some(path) = args.dump_step.as_ref() {
                    let all = std::env::var("DGQCUDA_DUMP_STEP_ALL").as_deref() == Ok("1");
                    let target = if step_no == 0 {
                        Some(path.clone())
                    } else if all {
                        Some(match path.rsplit_once('.') {
                            Some((stem, ext)) => format!("{stem}.step{}.{ext}", step_no + 1),
                            None => format!("{path}.step{}", step_no + 1),
                        })
                    } else {
                        None
                    };
                    if let Some(target) = target {
                        dump_step_json(
                            &target,
                            &st.ids,
                            &logits,
                            t.vocab_size,
                            prompt,
                            args.prompt.as_deref().unwrap_or(""),
                            None,
                        )?;
                    }
                }
                step_no += 1;
                sess.set_prev_logits(&logits)?;
                if args.diag {
                    // Device post-layer hidden for the canvas row vs the CPU
                    // oracle's, plus the same buffer's prompt row.
                    let hidden_n = t.hidden_size;
                    let h = sess.read_hidden_b((prompt.len() + canvas) * hidden_n)?;
                    let base = prompt.len() * hidden_n;
                    eprintln!(
                        "  [diag] step hidden prompt0[0..4] {:?} canvas0[0..4] {:?}",
                        &h[..4],
                        &h[base..base + 4]
                    );
                    let mut step_ids = prompt.to_vec();
                    step_ids.extend_from_slice(&st.ids);
                    let mut hsc = Scratch::new(step_ids.len(), &cfg);
                    let cpu = forward::forward_sc(
                        &w,
                        &cfg,
                        &step_ids,
                        args.layers,
                        forward::LogitRows::All,
                        &mut hsc,
                        prompt.len(),
                        canvas,
                        prev.as_deref(),
                    )?;
                    let cbase = prompt.len() * hidden_n;
                    eprintln!(
                        "  [diag] cpu  hidden canvas0[0..4] {:?} cos {:.5}",
                        &hsc.hidden_b[cbase..cbase + 4],
                        cosine(
                            &h[base..base + hidden_n],
                            &hsc.hidden_b[cbase..cbase + hidden_n]
                        )
                    );
                    let row = &logits[..t.vocab_size];
                    let mean = row.iter().sum::<f32>() / row.len() as f32;
                    let var =
                        row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / row.len() as f32;
                    let (am, av) = top_k(row, 1)[0];
                    eprintln!(
                        "  [diag] canvas row0 logits mean {mean:.3} std {:.3} entropy {:.3} argmax {am} ({av:.3})",
                        var.sqrt(),
                        entropy(row)
                    );
                }
                if args.parity {
                    // The same step on the CPU oracle: prompt + canvas, the
                    // SC MLP over the canvas rows, all layers, LM head. Its
                    // logits are indexed by sequence position, so the canvas
                    // rows start at `prompt.len()`.
                    let mut step_ids = prompt.to_vec();
                    step_ids.extend_from_slice(&st.ids);
                    let mut csc = Scratch::new(step_ids.len(), &cfg);
                    let cpu = forward::forward_sc(
                        &w,
                        &cfg,
                        &step_ids,
                        args.layers,
                        forward::LogitRows::All,
                        &mut csc,
                        prompt.len(),
                        canvas,
                        prev.as_deref(),
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
            // What gets EMITTED is the argmax canvas, not the sampled one.
            // `step_generate::turn` commits `st.prev_argmax`, and the two are
            // not the same object: `ids` carries the categorical draw that
            // drives the next step's denoise, and any position the accept mask
            // left out still holds `rng.uniform_below(vocab)` -- a uniform
            // random token. Emitting `ids` therefore leaks noise into the
            // reply at exactly the positions the sampler was least sure about.
            println!("canvas ({} ids):", st.argmax.len());
            let ids: Vec<String> = st.argmax.iter().map(|v| v.to_string()).collect();
            println!("{}", ids.join(","));
            let active = st
                .argmax
                .iter()
                .filter(|&&v| v != denoise::PAD_TOKEN_ID)
                .count();
            println!("active tokens: {active}/{}", st.argmax.len());
            if let Some(tok) = &tok {
                // `reply_ids` owns the argmax-not-ids decision and is pinned by
                // tests/denoise.rs; do not inline it back to a field access.
                let text_ids = st.reply_ids(&t.eos_token_ids());
                let raw = tok.decode(&text_ids);
                println!("--- reply ---");
                println!("{}", chat_template::sanitize_model_reply(&raw));
            }
        }
        // One layer on the engine's `layer0` synthetic input, for a direct
        // body-vs-body comparison against `diffgemma layer0`.
        "layer0" => {
            let row = args.only_row.unwrap_or(0);
            // `--at <n>` picks the layer, matching the engine's
            // `DGQ_LAYER0_INDEX`. The input is synthetic at every layer, so a
            // deep layer's divergence here is its own and not inherited.
            let layer = (args.at as usize).min(cfg.text_config.num_hidden_layers - 1);
            let (hin, attn, out) = gpu::layer0_synthetic(&w, &cfg, row, layer)?;
            let field = |name: &str, v: &[f32]| {
                let vals: Vec<String> = v.iter().map(|x| format!("{x}")).collect();
                format!("\"{name}\":[{}]", vals.join(","))
            };
            let path = std::env::var("DGQCUDA_LAYER0_DUMP")
                .unwrap_or_else(|_| "/tmp/port_layer0.json".to_string());
            let body = [
                field("hidden_in", &hin),
                field("attn_out", &attn),
                field("output", &out),
                format!("\"row\":{row},\"layer\":{layer}"),
            ]
            .join(",");
            std::fs::write(&path, format!("{{{body}}}"))?;
            eprintln!("wrote {path} (layer {layer}, row {row})");
            println!("  output[0..4]: {:?}", &out[..4]);
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
