//! step-*-dump subcommands (logits/layer/attn/MoE/preamble/embed-row).
//! Split from step_debug.rs; paired with python/scripts/{dump,compare}_*.py.

use super::*;

#[cfg(target_os = "macos")]
pub(crate) fn run_step_logits_dump_cmd(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    layers: usize,
    steps: usize,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
    output: &std::path::Path,
    positions: &str,
    top_k: usize,
) -> ExitCode {
    use metal::{StepSmokeConfig, parse_positions, run_step_logits_dump, write_step_logits_dump};

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-logits-dump requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let mut cfg = StepSmokeConfig {
        layers,
        steps: steps.max(1),
        kv_len: 0,
        seed,
        max_seq,
        finish: metal::StepFinishMode::ForwardOnly,
        prefill_token_ids: None,
        no_early_stop: false,
    };
    if let Err(err) = attach_step_prefill(&mut cfg, model_dir, 0, prompt.as_deref(), raw_prompt) {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    let pos = match parse_positions(positions) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };
    let label = prompt.unwrap_or_else(|| "Hello".to_string());
    match run_step_logits_dump(model_dir, &cfg, &label, &pos, top_k.max(1)) {
        Ok(dump) => {
            if let Err(err) = write_step_logits_dump(output, &dump) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            println!(
                "wrote {} (T={:.4}, {} rows)",
                output.display(),
                dump.temperature,
                dump.rows.len()
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
pub(crate) fn run_step_bf16_oracle_logits_dump_cmd(
    dgq_dir: &std::path::Path,
    bf16_dir: &std::path::Path,
    prompt: Option<String>,
    layers: usize,
    steps: usize,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
    output: &std::path::Path,
    positions: &str,
    top_k: usize,
    gpu_kv: bool,
) -> ExitCode {
    use metal::{
        StepSmokeConfig, parse_positions, run_step_bf16_oracle_logits_dump,
        run_step_bf16_oracle_logits_dump_gpu_kv, write_step_logits_dump,
    };

    if !dgq::store::looks_like_dgq_dir(dgq_dir) {
        eprintln!(
            "error: step-bf16-logits-dump requires -m pointing at a .dgq directory (prefill)"
        );
        return ExitCode::FAILURE;
    }
    if !bf16_dir.join("config.json").is_file() {
        eprintln!(
            "error: step-bf16-logits-dump --bf16-ref must contain config.json ({})",
            bf16_dir.display()
        );
        return ExitCode::FAILURE;
    }
    let mut cfg = StepSmokeConfig {
        layers,
        steps: steps.max(1),
        kv_len: 0,
        seed,
        max_seq,
        finish: metal::StepFinishMode::ForwardOnly,
        prefill_token_ids: None,
        no_early_stop: false,
    };
    if let Err(err) = attach_step_prefill(&mut cfg, dgq_dir, 0, prompt.as_deref(), raw_prompt) {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    let pos = match parse_positions(positions) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };
    let label = prompt.unwrap_or_else(|| "Hello".to_string());
    let dump_result = if gpu_kv {
        run_step_bf16_oracle_logits_dump_gpu_kv(dgq_dir, bf16_dir, &cfg, &label, &pos, top_k.max(1))
    } else {
        run_step_bf16_oracle_logits_dump(bf16_dir, &cfg, &label, &pos, top_k.max(1))
    };
    match dump_result {
        Ok(dump) => {
            if let Err(err) = write_step_logits_dump(output, &dump) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            println!(
                "wrote {} ({}, T={:.4}, {} rows)",
                output.display(),
                dump.source,
                dump.temperature,
                dump.rows.len()
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
pub(crate) fn run_step_layer_probe_cmd(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    layers: usize,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
    output: &std::path::Path,
    position: usize,
    warm_steps: usize,
) -> ExitCode {
    use metal::{StepSmokeConfig, run_step_layer_hidden_dump, write_step_layer_hidden_dump};

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-layer-probe requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let mut cfg = StepSmokeConfig {
        layers,
        steps: 2,
        kv_len: 0,
        seed,
        max_seq,
        finish: metal::StepFinishMode::ForwardOnly,
        prefill_token_ids: None,
        no_early_stop: false,
    };
    if let Err(err) = attach_step_prefill(&mut cfg, model_dir, 0, prompt.as_deref(), raw_prompt) {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    let label = prompt.unwrap_or_else(|| "Hello".to_string());
    match run_step_layer_hidden_dump(model_dir, &cfg, &label, position, warm_steps) {
        Ok(dump) => {
            if let Err(err) = write_step_layer_hidden_dump(output, &dump) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            println!(
                "wrote {} (pos={}, warm_steps={}, {} checkpoints)",
                output.display(),
                dump.position,
                dump.warm_steps,
                dump.checkpoints.len()
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
pub(crate) fn run_step_attn_dump_cmd(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    layers: usize,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
    output: &std::path::Path,
    layer: usize,
    position: usize,
) -> ExitCode {
    use metal::{StepSmokeConfig, run_step_attn_layer_dump, write_step_attn_layer_dump};

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-attn-dump requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let mut cfg = StepSmokeConfig {
        layers,
        steps: 2,
        kv_len: 0,
        seed,
        max_seq,
        finish: metal::StepFinishMode::ForwardOnly,
        prefill_token_ids: None,
        no_early_stop: false,
    };
    if let Err(err) = attach_step_prefill(&mut cfg, model_dir, 0, prompt.as_deref(), raw_prompt) {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    let label = prompt.unwrap_or_else(|| "Hello".to_string());
    match run_step_attn_layer_dump(model_dir, &cfg, &label, layer, position) {
        Ok(dump) => {
            if let Err(err) = write_step_attn_layer_dump(output, &dump) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            println!(
                "wrote {} (layer={}, pos={}, kv_len={})",
                output.display(),
                dump.layer,
                dump.position,
                dump.kv_len
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
/// E22-M0 (task #101): prefill a document, run one denoise-step forward to
/// `--attn-layer`, and dump the full canvas Q plane + the layer's K cache as
/// raw f32 binaries (+ meta json) into `-o OUT_DIR`. Consumed by
/// `python/scripts/e22_block_mass.py`.
#[cfg(target_os = "macos")]
pub(crate) fn run_step_attn_qk_dump_cmd(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    layers: usize,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
    output: &std::path::Path,
    layer: usize,
) -> ExitCode {
    use metal::{StepSmokeConfig, run_step_attn_qk_plane_dump};

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-attn-qk-dump requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let mut cfg = StepSmokeConfig {
        layers,
        steps: 2,
        kv_len: 0,
        seed,
        max_seq,
        finish: metal::StepFinishMode::ForwardOnly,
        prefill_token_ids: None,
        no_early_stop: false,
    };
    if let Err(err) = attach_step_prefill(&mut cfg, model_dir, 0, prompt.as_deref(), raw_prompt) {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    match run_step_attn_qk_plane_dump(model_dir, &cfg, layer, output) {
        Ok((kv_len, total_kv)) => {
            println!(
                "wrote q/k planes to {} (layer={layer}, kv_len={kv_len}, total_kv={total_kv})",
                output.display()
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
pub(crate) fn run_step_moe_dump_cmd(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    layers: usize,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
    output: &std::path::Path,
    layer: usize,
    position: usize,
) -> ExitCode {
    use metal::{StepSmokeConfig, run_step_moe_layer_dump, write_step_moe_layer_dump};

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-moe-dump requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let mut cfg = StepSmokeConfig {
        layers,
        steps: 2,
        kv_len: 0,
        seed,
        max_seq,
        finish: metal::StepFinishMode::ForwardOnly,
        prefill_token_ids: None,
        no_early_stop: false,
    };
    if let Err(err) = attach_step_prefill(&mut cfg, model_dir, 0, prompt.as_deref(), raw_prompt) {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    let label = prompt.unwrap_or_else(|| "Hello".to_string());
    match run_step_moe_layer_dump(model_dir, &cfg, &label, layer, position) {
        Ok(dump) => {
            if let Err(err) = write_step_moe_layer_dump(output, &dump) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            println!(
                "wrote {} (layer={}, pos={}, kv_len={}, experts={:?})",
                output.display(),
                dump.layer,
                dump.position,
                dump.kv_len,
                dump.experts,
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
pub(crate) fn run_step_moe_route_dump_cmd(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    layers: usize,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
    output: &std::path::Path,
    layer: usize,
    run_grouped: bool,
) -> ExitCode {
    use metal::{
        StepFinishMode, StepSmokeConfig, print_route_summary, run_step_moe_route_dump,
        write_step_moe_route_dump,
    };

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-moe-route-dump requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let mut cfg = StepSmokeConfig {
        layers,
        steps: 1,
        kv_len: 0,
        seed,
        max_seq,
        finish: StepFinishMode::ForwardOnly,
        prefill_token_ids: None,
        no_early_stop: false,
    };
    if let Err(err) = attach_step_prefill(&mut cfg, model_dir, 0, prompt.as_deref(), raw_prompt) {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    let label = prompt.unwrap_or_else(|| "Hello".to_string());
    match run_step_moe_route_dump(model_dir, &cfg, &label, layer, run_grouped) {
        Ok(dump) => {
            print_route_summary(&dump);
            if let Err(err) = write_step_moe_route_dump(output, &dump) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            println!(
                "wrote {} (layer={}, slots_ok={})",
                output.display(),
                dump.layer,
                dump.slots_ok
            );
            if !dump.slots_ok {
                eprintln!("error: MoE bucketing failed (num_slots={})", dump.num_slots);
                return ExitCode::FAILURE;
            }
            if dump.grouped_dispatched {
                if dump.moe_out_l2.unwrap_or(0.0) < 1e-6 {
                    eprintln!("error: MoE produced zero moe_out");
                    return ExitCode::FAILURE;
                }
                if dump.moe_out_gpu_cpu_cos.unwrap_or(1.0) < 0.99 {
                    eprintln!(
                        "error: moe_out vs CPU oracle cos={:.6} (need >= 0.99, style={})",
                        dump.moe_out_gpu_cpu_cos.unwrap_or(0.0),
                        dump.moe_style
                    );
                    return ExitCode::FAILURE;
                }
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
pub(crate) fn run_step_moe_batched_pin_dump_cmd(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    layers: usize,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
    output: &std::path::Path,
    layer: usize,
) -> ExitCode {
    use metal::{
        StepFinishMode, StepSmokeConfig, print_batched_pin_summary, run_step_moe_batched_pin_dump,
        write_step_moe_batched_pin_dump,
    };

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-moe-batched-pin-dump requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let mut cfg = StepSmokeConfig {
        layers,
        steps: 1,
        kv_len: 0,
        seed,
        max_seq,
        finish: StepFinishMode::ForwardOnly,
        prefill_token_ids: None,
        no_early_stop: false,
    };
    if let Err(err) = attach_step_prefill(&mut cfg, model_dir, 0, prompt.as_deref(), raw_prompt) {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    let label = prompt.unwrap_or_else(|| "Hello".to_string());
    match run_step_moe_batched_pin_dump(model_dir, &cfg, &label, layer) {
        Ok(dump) => {
            print_batched_pin_summary(&dump);
            if let Err(err) = write_step_moe_batched_pin_dump(output, &dump) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            println!(
                "wrote {} (layer={}, gate_up_gemm_cos={:.6}, swiglu_post_cos={:.6}, swiglu_isolated_cos={:.6})",
                output.display(),
                dump.layer,
                dump.stages.gate_up_gemm,
                dump.stages.swiglu_post,
                dump.stages.swiglu_isolated,
            );
            let fail_stage = [
                ("gate_up_gemm", dump.stages.gate_up_gemm),
                ("swiglu_post", dump.stages.swiglu_post),
                ("swiglu_isolated", dump.stages.swiglu_isolated),
                ("down", dump.stages.down),
                ("scatter", dump.stages.scatter),
            ]
            .into_iter()
            .find(|(_, cos)| *cos < 0.99);
            if let Some((name, cos)) = fail_stage {
                eprintln!("error: batched pin stage `{name}` cos={cos:.6} (need >= 0.99)");
                return ExitCode::FAILURE;
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
pub(crate) fn run_step_moe_single_dump_cmd(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    layers: usize,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
    output: &std::path::Path,
    layer: usize,
    position: usize,
    expert: u32,
) -> ExitCode {
    use metal::{
        StepSmokeConfig, run_step_moe_single_expert_dump, write_step_moe_single_expert_dump,
    };

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-moe-single-dump requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let mut cfg = StepSmokeConfig {
        layers,
        steps: 2,
        kv_len: 0,
        seed,
        max_seq,
        finish: metal::StepFinishMode::ForwardOnly,
        prefill_token_ids: None,
        no_early_stop: false,
    };
    if let Err(err) = attach_step_prefill(&mut cfg, model_dir, 0, prompt.as_deref(), raw_prompt) {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    let label = prompt.unwrap_or_else(|| "Hello".to_string());
    match run_step_moe_single_expert_dump(model_dir, &cfg, &label, layer, position, expert) {
        Ok(dump) => {
            if let Err(err) = write_step_moe_single_expert_dump(output, &dump) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            let mut dot = 0.0f64;
            let mut na = 0.0f64;
            let mut nb = 0.0f64;
            for (a, b) in dump.gpu_out.iter().zip(dump.cpu_out.iter()) {
                let af = *a as f64;
                let bf = *b as f64;
                dot += af * bf;
                na += af * af;
                nb += bf * bf;
            }
            let cos = if na > 0.0 && nb > 0.0 {
                (dot / (na.sqrt() * nb.sqrt())) as f32
            } else {
                0.0
            };
            println!(
                "wrote {} (layer={}, pos={}, expert={}, kv_len={}, gpu_vs_cpu_cos={:.6})",
                output.display(),
                dump.layer,
                dump.position,
                dump.expert_id,
                dump.kv_len,
                cos,
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
pub(crate) fn run_step_preamble_dump_cmd(
    model_dir: &std::path::Path,
    prompt: Option<String>,
    layers: usize,
    seed: u64,
    max_seq: usize,
    raw_prompt: bool,
    output: &std::path::Path,
    position: usize,
) -> ExitCode {
    use metal::{StepSmokeConfig, run_step_preamble_dump, write_step_preamble_dump};

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: step-preamble-dump requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let mut cfg = StepSmokeConfig {
        layers,
        steps: 2,
        kv_len: 0,
        seed,
        max_seq,
        finish: metal::StepFinishMode::ForwardOnly,
        prefill_token_ids: None,
        no_early_stop: false,
    };
    if let Err(err) = attach_step_prefill(&mut cfg, model_dir, 0, prompt.as_deref(), raw_prompt) {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    let label = prompt.unwrap_or_else(|| "Hello".to_string());
    match run_step_preamble_dump(model_dir, &cfg, &label, position) {
        Ok(dump) => {
            if let Err(err) = write_step_preamble_dump(output, &dump) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            println!(
                "wrote {} (pos={}, token={})",
                output.display(),
                dump.position,
                dump.canvas_token
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
pub(crate) fn run_embed_row_dump_cmd(
    model_dir: &std::path::Path,
    token: u32,
    layers: usize,
    max_seq: usize,
    prompt: Option<String>,
    raw_prompt: bool,
    output: &std::path::Path,
    bf16_ref_dir: Option<&std::path::Path>,
    gpu: bool,
) -> ExitCode {
    use dgq::{run_embed_row_dump, write_embed_row_dump};

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: embed-row-dump requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let hidden = match crate::config::ModelConfig::load(model_dir) {
        Ok(c) => c.text_config.hidden_size,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    #[cfg(target_os = "macos")]
    let gpu_scaled = if gpu {
        use metal::{StepFinishMode, StepSmokeConfig, run_embed_row_gpu};
        let mut cfg = StepSmokeConfig {
            layers,
            steps: 1,
            kv_len: 0,
            seed: 0,
            max_seq,
            finish: StepFinishMode::ForwardOnly,
            prefill_token_ids: None,
            no_early_stop: false,
        };
        if let Err(err) = attach_step_prefill(&mut cfg, model_dir, 0, prompt.as_deref(), raw_prompt)
        {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
        match run_embed_row_gpu(model_dir, &cfg, token) {
            Ok(v) => Some(v),
            Err(err) => {
                eprintln!("error: gpu embed row: {err}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };

    match run_embed_row_dump(model_dir, token, hidden, bf16_ref_dir, gpu_scaled) {
        Ok(dump) => {
            if let Err(err) = write_embed_row_dump(output, &dump) {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
            println!(
                "wrote {} (token={}, scale={:.6}, cpu_l2={:.2})",
                output.display(),
                dump.token,
                dump.scale_f32,
                dump.dequant_scaled_l2
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
