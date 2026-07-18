//! Layer/attn/MoE/embed parity DUMP + PROBE subcommands. Manual diagnostics
//! paired with `python/scripts/{dump,compare}_*.py`; not invoked by tests/CI.

use super::*;

#[cfg(target_os = "macos")]
pub(crate) fn run_step_probe_cmd(
    model_dir: &std::path::Path,
    layers: usize,
    kv_len: u32,
    seed: u64,
    max_seq: usize,
    prompt: Option<&str>,
    raw_prompt: bool,
) -> ExitCode {
    use metal::{StepFinishMode, StepSmokeConfig, run_step_probe};

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
        prefill_token_ids: None,
        no_early_stop: false,
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
#[cfg(target_os = "macos")]
pub(crate) fn run_step_kv_check_cmd(
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
#[cfg(target_os = "macos")]
pub(crate) fn run_step_kv_bf16_cross_cmd(
    dgq_dir: &std::path::Path,
    bf16_dir: &std::path::Path,
    prompt: Option<String>,
    prompt_len: usize,
    layers: usize,
    _seed: u64,
    max_seq: usize,
    raw_prompt: bool,
) -> ExitCode {
    use metal::run_step_kv_bf16_cross_parity;

    if !dgq::store::looks_like_dgq_dir(dgq_dir) {
        eprintln!("error: step-kv-bf16-cross requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    if !bf16_dir.join("config.json").is_file() {
        eprintln!(
            "error: step-kv-bf16-cross --bf16-ref must contain config.json ({})",
            bf16_dir.display()
        );
        return ExitCode::FAILURE;
    }
    let vocab = match crate::config::ModelConfig::load(dgq_dir) {
        Ok(c) => c.text_config.vocab_size,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };
    let token_ids = match build_prompt_tokens(
        dgq_dir,
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
        "step-kv-bf16-cross: {} prompt tokens, layers={}",
        token_ids.len(),
        layers.clamp(1, 30)
    );
    match run_step_kv_bf16_cross_parity(dgq_dir, bf16_dir, &token_ids, layers, max_seq.max(64)) {
        Ok(r) => {
            println!("step-kv-bf16-cross:");
            println!("  kv_len:              {}", r.kv_len);
            println!("  layers:              {}", r.layers);
            println!("  gpu_prefix_l0:       {:.6}", r.gpu_prefix_max_l0);
            println!("  cpu_prefix_l0:       {:.6}", r.cpu_prefix_max_l0);
            println!(
                "  max_kv_diff:         {:.6} (layer {} pos {})",
                r.max_kv_diff, r.max_kv_diff_layer, r.max_kv_diff_pos
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
#[cfg(target_os = "macos")]
pub(crate) fn run_step_kv_parity_cmd(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    prompt_len: usize,
    layers: usize,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
) -> ExitCode {
    use metal::run_step_kv_parity;

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
        layers.clamp(1, 30)
    );
    match run_step_kv_parity(model_dir, &token_ids, layers, max_seq.max(64), seed) {
        Ok(r) => {
            println!("step-kv-parity:");
            println!("  kv_len:              {}", r.kv_len);
            println!("  layers:              {}", r.layers);
            println!("  prefix_l0:           {:.6}", r.prefix_max_l0);
            println!("  prefix_l0_b:         {:.6}", r.prefix_max_l0_b);
            println!(
                "  max_kv_diff:         {:.6} (layer {} pos {})",
                r.max_kv_diff, r.max_kv_diff_layer, r.max_kv_diff_pos
            );
            println!("  min_ent:             {:.4}", r.min_ent);
            println!("  min_ent_b:           {:.4}", r.min_ent_b);
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
#[cfg(target_os = "macos")]
pub(crate) fn run_step_attn_probe_cmd(
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
            println!(
                "  attn keys T:            {} (= kv_len + canvas_len)",
                r.attn_keys_t
            );
            println!("  layer:                  {}", r.layer);
            println!("  monolithic K max L0:    {:.4}", r.k_plane_max_l0);
            println!("  monolithic V max L0:    {:.4}", r.v_plane_max_l0);
            println!(
                "  q_norm weight mean|w|: {:.4}  rms: {:.4}",
                r.q_norm_weight_mean_abs, r.q_norm_weight_rms
            );
            println!(
                "  k_norm weight mean|w|: {:.4}  rms: {:.4}",
                r.k_norm_weight_mean_abs, r.k_norm_weight_rms
            );
            println!(
                "  CPU Q·K raw max/min:    {:.2} / {:.2}",
                r.cpu_raw_dot_max, r.cpu_raw_dot_min
            );
            println!(
                "  CPU softmax row ent:    {:.4} nats (ln T = {:.4}, active keys/row = {})",
                r.cpu_mean_softmax_entropy, ln_uniform, r.cpu_keys_per_row
            );
            println!("  CPU softmax max prob:   {:.4}", r.cpu_mean_max_prob);
            println!(
                "  CPU weight sum/row:     {:.6} (expect 1.0)",
                r.cpu_mean_weight_sum
            );
            println!(
                "  CPU Q/K head RMS:       {:.4} / {:.4}",
                r.cpu_q_head_rms, r.cpu_k_head_rms
            );
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
