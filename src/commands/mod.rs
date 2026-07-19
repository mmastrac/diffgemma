//! CLI command dispatch. Each subcommand's body lives in a sibling module here;
//! argument parsing is `crate::cli`. Extracted from the former monolithic main.rs.

pub(crate) use crate::cli::{Cli, Command};

// Re-export the crate modules + std types the command bodies reference, so each
// submodule can `use super::*` and keep the crate-root call style it grew up with.
pub(crate) use crate::{
    chat_template, chat_ui, config, dgq, flags, generate, generate_golden, golden, metal, model,
    sample, server, tokenizer, tools, weights,
};
pub(crate) use std::path::PathBuf;
pub(crate) use std::process::ExitCode;

mod bench;
mod census;
mod chat;
mod common;
mod gen_cmd;
mod golden_cmd;
mod model_ops;
mod replay;
mod smoketest;
mod step_debug;
mod step_debug_dumps;
mod step_gate;

pub(crate) use bench::*;
pub(crate) use chat::*;
pub(crate) use common::*;
pub(crate) use gen_cmd::*;
pub(crate) use golden_cmd::*;
pub(crate) use model_ops::*;
pub(crate) use replay::*;
pub(crate) use smoketest::*;
pub(crate) use step_debug::*;
pub(crate) use step_debug_dumps::*;
pub(crate) use step_gate::*;

pub(crate) fn dispatch(cli: Cli) -> ExitCode {
    match cli.command {
        Command::StepSmoke {
            layers,
            steps,
            kv_len,
            seed,
            max_seq,
            forward_only,
            prompt,
        } => run_step_smoke_cmd(
            &cli.model_dir,
            layers,
            steps,
            kv_len,
            seed,
            max_seq,
            forward_only,
            prompt.as_deref(),
            cli.raw_prompt,
        ),
        Command::StepProbe {
            layers,
            kv_len,
            seed,
            max_seq,
            prompt,
        } => run_step_probe_cmd(
            &cli.model_dir,
            layers,
            kv_len,
            seed,
            max_seq,
            prompt.as_deref(),
            cli.raw_prompt,
        ),
        Command::StepKvCheck {
            kv_len,
            layers,
            seed,
            max_seq,
        } => run_step_kv_check_cmd(&cli.model_dir, kv_len, layers, seed, max_seq),
        Command::StepKvParity {
            prompt,
            prompt_len,
            layers,
            seed,
            max_seq,
            raw_prompt,
        } => run_step_kv_parity_cmd(
            &cli.model_dir,
            prompt,
            prompt_len,
            layers,
            seed,
            max_seq,
            raw_prompt,
        ),
        Command::StepKvBf16Cross {
            prompt,
            prompt_len,
            layers,
            seed,
            max_seq,
            raw_prompt,
            bf16_ref_dir,
        } => run_step_kv_bf16_cross_cmd(
            &cli.model_dir,
            &bf16_ref_dir,
            prompt,
            prompt_len,
            layers,
            seed,
            max_seq,
            raw_prompt,
        ),
        Command::StepAttnProbe {
            prompt,
            prompt_len,
            layer,
            seed,
            max_seq,
            raw_prompt,
        } => run_step_attn_probe_cmd(
            &cli.model_dir,
            prompt,
            prompt_len,
            layer,
            seed,
            max_seq,
            raw_prompt,
        ),
        Command::StepLogitsDump {
            prompt,
            layers,
            steps,
            seed,
            max_seq,
            raw_prompt,
            output,
            positions,
            top_k,
        } => run_step_logits_dump_cmd(
            &cli.model_dir,
            prompt,
            layers,
            steps,
            seed,
            max_seq,
            raw_prompt,
            &output,
            &positions,
            top_k,
        ),
        Command::StepBf16OracleLogitsDump {
            prompt,
            layers,
            steps,
            seed,
            max_seq,
            raw_prompt,
            output,
            positions,
            top_k,
            bf16_ref_dir,
            gpu_kv,
        } => run_step_bf16_oracle_logits_dump_cmd(
            &cli.model_dir,
            &bf16_ref_dir,
            prompt,
            layers,
            steps,
            seed,
            max_seq,
            raw_prompt,
            &output,
            &positions,
            top_k,
            gpu_kv,
        ),
        Command::StepLayerProbe {
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            output,
            position,
        } => run_step_layer_probe_cmd(
            &cli.model_dir,
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            &output,
            position,
        ),
        Command::StepAttnDump {
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            output,
            layer,
            position,
        } => run_step_attn_dump_cmd(
            &cli.model_dir,
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            &output,
            layer,
            position,
        ),
        Command::StepAttnQkDump {
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            output,
            layer,
        } => run_step_attn_qk_dump_cmd(
            &cli.model_dir,
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            &output,
            layer,
        ),
        Command::StepMoeDump {
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            output,
            layer,
            position,
        } => run_step_moe_dump_cmd(
            &cli.model_dir,
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            &output,
            layer,
            position,
        ),
        Command::StepMoeRouteDump {
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            output,
            layer,
            run_grouped,
        } => run_step_moe_route_dump_cmd(
            &cli.model_dir,
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            &output,
            layer,
            run_grouped,
        ),
        Command::StepMoeBatchedPinDump {
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            output,
            layer,
        } => run_step_moe_batched_pin_dump_cmd(
            &cli.model_dir,
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            &output,
            layer,
        ),
        Command::StepMoeSingleDump {
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            output,
            layer,
            position,
            expert,
        } => run_step_moe_single_dump_cmd(
            &cli.model_dir,
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            &output,
            layer,
            position,
            expert,
        ),
        Command::StepPreambleDump {
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            output,
            position,
        } => run_step_preamble_dump_cmd(
            &cli.model_dir,
            prompt,
            layers,
            seed,
            max_seq,
            raw_prompt,
            &output,
            position,
        ),
        Command::EmbedRowDump {
            token,
            layers,
            max_seq,
            prompt,
            raw_prompt,
            output,
            bf16_ref_dir,
            gpu,
        } => run_embed_row_dump_cmd(
            &cli.model_dir,
            token,
            layers,
            max_seq,
            prompt,
            raw_prompt,
            &output,
            bf16_ref_dir.as_deref(),
            gpu,
        ),
        Command::StepVerify { layers } => run_step_verify_cmd(&cli.model_dir, layers),
        Command::StepCi { layers } => run_step_ci_cmd(&cli.model_dir, layers),
        Command::StepParity {
            layers,
            kv_len,
            seed,
            max_seq,
        } => run_step_parity_cmd(&cli.model_dir, layers, kv_len, seed, max_seq),
        Command::BenchStepKernel {
            layers,
            kv_len,
            seed,
            max_seq,
            iters,
            forward_only,
            profile,
            profile_steps,
            layer_profile,
        } => run_bench_step_kernel_cmd(
            &cli.model_dir,
            layers,
            kv_len,
            seed,
            max_seq,
            iters,
            forward_only,
            profile,
            profile_steps,
            layer_profile,
        ),
        Command::BenchPrefillSuper {
            kv_len,
            iters,
            stages,
            n_subs,
        } => run_bench_prefill_super_cmd(&cli.model_dir, kv_len, iters, stages, n_subs),
        Command::BenchGemmFusion {
            layers,
            kv_len,
            seed,
            max_seq,
            iters,
        } => run_bench_gemm_fusion_cmd(&cli.model_dir, layers, kv_len, seed, max_seq, iters),
        Command::GenerateMonolithic {
            prompt,
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers,
            no_early_stop,
            kernel_assert,
            kernel_debug_deep,
            write_golden,
            write_trace,
            verbose,
        } => run_generate_monolithic_cmd(
            &cli.model_dir,
            prompt,
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers,
            no_early_stop,
            kernel_assert,
            kernel_debug_deep,
            write_golden,
            write_trace,
            cli.raw_prompt,
            verbose,
        ),
        Command::GenerateMonolithicParity {
            prompt,
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers,
            no_early_stop,
            golden,
            write_golden,
        } => run_generate_monolithic_parity_cmd(
            &cli.model_dir,
            prompt,
            seed,
            steps,
            prompt_len,
            max_new_tokens,
            max_layers,
            no_early_stop,
            golden,
            write_golden,
            cli.raw_prompt,
        ),
        Command::Chat {
            seed,
            steps,
            max_new_tokens,
            max_layers,
            no_early_stop,
            initial_prompt,
            verbose,
            events_path,
            json,
            ctx,
        } => run_chat_cmd(
            &cli.model_dir,
            initial_prompt,
            seed,
            steps,
            max_new_tokens,
            max_layers,
            no_early_stop,
            cli.raw_prompt,
            verbose,
            events_path,
            json,
            ctx,
        ),
        Command::Serve {
            addr,
            seed,
            steps,
            max_layers,
            ctx,
            tool_compact,
            tool_repair,
            tool_validate,
            log_dir,
            think,
        } => match server::run_serve(
            &cli.model_dir,
            &addr,
            ctx,
            seed,
            steps,
            max_layers,
            tool_compact,
            tool_repair,
            tool_validate,
            log_dir,
            think,
        ) {
            Ok(()) => ExitCode::SUCCESS,
            Err(msg) => {
                eprintln!("error: {msg}");
                ExitCode::FAILURE
            }
        },
        Command::Smoketest {
            prompts_path,
            seed,
            steps,
            max_layers,
            filter,
            repeat,
            longctx,
        } => run_smoketest_cmd(
            &cli.model_dir,
            prompts_path.as_deref(),
            seed,
            steps,
            max_layers,
            cli.raw_prompt,
            filter.as_deref(),
            repeat,
            longctx,
        ),
        #[cfg(target_os = "macos")]
        Command::Census {
            arms,
            batteries,
            seeds,
            gates,
            baseline,
            out_dir,
            tau,
            steps,
        } => census::run_census_cmd(
            &cli.model_dir,
            &arms,
            &batteries,
            &seeds,
            &gates,
            baseline.as_deref(),
            out_dir.as_deref(),
            tau,
            steps,
        ),
        Command::Manifest => {
            print!("{}", crate::shaders::manifest::render_toml());
            ExitCode::SUCCESS
        }
        #[cfg(target_os = "macos")]
        Command::Golden {
            pack_path,
            bless,
            filter,
        } => run_golden_cmd(
            &cli.model_dir,
            pack_path.as_deref(),
            bless,
            filter.as_deref(),
        ),
        #[cfg(not(target_os = "macos"))]
        Command::Golden { .. } => ExitCode::FAILURE,
        #[cfg(target_os = "macos")]
        Command::Replay {
            log_path,
            ctx,
            steps,
        } => run_replay_cmd(&cli.model_dir, &log_path, ctx, steps),
        #[cfg(not(target_os = "macos"))]
        Command::Replay { .. } => ExitCode::FAILURE,
        Command::Quantize { output, profile } => run_quantize(&cli.model_dir, &output, &profile),
        Command::Tokenize(text) => run_tokenize(&cli.model_dir, &text, cli.raw_prompt),
        Command::Gemm { size } => run_gemm(size),
        Command::ProbeDevice => run_probe_device(),
        Command::BenchGemm {
            shapes,
            oracle,
            iters,
        } => run_bench_gemm(&shapes, oracle.as_deref(), iters),
        Command::BenchPrefillAttn {
            kv_len,
            qk_bm,
            qk_bn,
            pv_bm,
            pv_bn,
            hc,
            sm_tpg,
            side,
            causal,
            iters,
        } => run_bench_prefill_attn(
            kv_len, qk_bm, qk_bn, pv_bm, pv_bn, hc, sm_tpg, side, causal, iters,
        ),
        command => {
            eprintln!("loading from {}", cli.model_dir.display());
            match model::Model::open(&cli.model_dir) {
                Ok(m) => run_command(&m, &cli.model_dir, command, cli.raw_prompt),
                Err(err) => {
                    eprintln!("error: {err}");
                    ExitCode::FAILURE
                }
            }
        }
    }
}

pub(crate) fn run_command(
    m: &model::Model,
    _model_dir: &std::path::Path,
    command: Command,
    _raw_prompt: bool,
) -> ExitCode {
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
        Command::Chat { .. } => ExitCode::FAILURE,
        Command::Serve { .. } => ExitCode::FAILURE,
        Command::Smoketest { .. } | Command::Census { .. } => ExitCode::FAILURE,
        Command::Golden { .. } => ExitCode::FAILURE,
        Command::Replay { .. } => ExitCode::FAILURE,
        Command::Manifest => {
            print!("{}", crate::shaders::manifest::render_toml());
            ExitCode::SUCCESS
        }
        Command::Tokenize(_) => ExitCode::FAILURE,
        Command::Gemm { .. } => ExitCode::FAILURE,
        Command::ProbeDevice { .. } => ExitCode::FAILURE,
        Command::Quantize { .. } => ExitCode::FAILURE,
        Command::BenchGemm { .. } => ExitCode::FAILURE,
        Command::BenchPrefillAttn { .. } => ExitCode::FAILURE,
        Command::StepSmoke { .. } => ExitCode::FAILURE,
        Command::StepProbe { .. } => ExitCode::FAILURE,
        Command::StepKvCheck { .. } => ExitCode::FAILURE,
        Command::StepKvParity { .. } => ExitCode::FAILURE,
        Command::StepKvBf16Cross { .. } => ExitCode::FAILURE,
        Command::StepAttnProbe { .. } => ExitCode::FAILURE,
        Command::StepLogitsDump { .. } => ExitCode::FAILURE,
        Command::StepBf16OracleLogitsDump { .. } => ExitCode::FAILURE,
        Command::StepLayerProbe { .. } => ExitCode::FAILURE,
        Command::StepAttnDump { .. } => ExitCode::FAILURE,
        Command::StepAttnQkDump { .. } => ExitCode::FAILURE,
        Command::StepMoeDump { .. } => ExitCode::FAILURE,
        Command::StepMoeRouteDump { .. } => ExitCode::FAILURE,
        Command::StepMoeBatchedPinDump { .. } => ExitCode::FAILURE,
        Command::StepMoeSingleDump { .. } => ExitCode::FAILURE,
        Command::StepPreambleDump { .. } => ExitCode::FAILURE,
        Command::EmbedRowDump { .. } => ExitCode::FAILURE,
        Command::StepVerify { .. } => ExitCode::FAILURE,
        Command::StepCi { .. } => ExitCode::FAILURE,
        Command::StepParity { .. } => ExitCode::FAILURE,
        Command::BenchStepKernel { .. } => ExitCode::FAILURE,
        Command::BenchPrefillSuper { .. } => ExitCode::FAILURE,
        Command::BenchGemmFusion { .. } => ExitCode::FAILURE,
        Command::GenerateMonolithic { .. } => ExitCode::FAILURE,
        Command::GenerateMonolithicParity { .. } => ExitCode::FAILURE,
    }
}
