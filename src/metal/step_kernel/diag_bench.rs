//! step_kernel bench + forward/smoke harnesses (step-smoke gate, bench-*).
//! Extracted from step_kernel_diagnostics.rs (see diag_probe for the family).

use crate::Error;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use std::path::Path;
use std::time::Instant;

use super::*;

pub fn bench_step_kernel(
    model_dir: &Path,
    cfg: StepSmokeConfig,
    iters: usize,
) -> Result<StepBenchResult, Error> {
    let iters = iters.max(1);
    let (mut rt, build) = build_step_runtime(model_dir, &cfg)?;
    let finish = cfg.finish;

    let warmup_started = Instant::now();
    rt.run_forward_once(finish)?;
    let warmup = warmup_started.elapsed();

    let started = Instant::now();
    for _ in 0..iters {
        rt.run_forward_once(finish)?;
    }
    let elapsed = started.elapsed();
    let per_step = elapsed / iters as u32;

    Ok(StepBenchResult {
        compile: build.compile,
        warmup,
        per_step,
        iters,
        finish,
    })
}

pub fn bench_step_kernel_profile(
    model_dir: &Path,
    cfg: StepSmokeConfig,
) -> Result<StepProfileResult, Error> {
    let (mut rt, build) = build_step_runtime(model_dir, &cfg)?;
    let finish = cfg.finish;
    rt.run_forward_once(finish)?;
    let mut prof = rt.profile_forward_once(finish)?;
    prof.compile = build.compile;
    Ok(prof)
}

pub fn bench_step_kernel_encode_subprofile(
    model_dir: &Path,
    cfg: StepSmokeConfig,
) -> Result<EncodeSubProfileResult, Error> {
    let (mut rt, build) = build_step_runtime(model_dir, &cfg)?;
    rt.run_forward_once(cfg.finish)?;
    if crate::flags::trace_ranges_enabled() {
        // Warm to a steady-state (denoised) step, then trace one step's ranges.
        // Under DGQ_PREFILL_F16 the traced step runs on the fp16 pipeline set
        // (per-stage localization for the E11 arena dtype flip).
        rt.arena_f16_mode = rt.pipelines_prefill_f16.is_some();
        rt.run_forward_once(cfg.finish)?;
        rt.trace_step_ranges()?;
        rt.arena_f16_mode = false;
    }
    let mut prof = rt.profile_encode_subprofile()?;
    prof.compile = build.compile;
    Ok(prof)
}

/// Holistic prefill proxy (task #87): build the runtime (compiling pipelines
/// per the current tile flags) and time one M=1024 super-chunk at `kv_len`.
/// Returns mean ms/super-chunk.
pub fn bench_step_kernel_prefill_super(
    model_dir: &Path,
    cfg: StepSmokeConfig,
    kv_len: u32,
    iters: usize,
    n_subs: usize,
) -> Result<std::time::Duration, Error> {
    let (mut rt, _build) = build_step_runtime(model_dir, &cfg)?;
    rt.bench_prefill_super(kv_len, iters, n_subs)
}

/// Floor decomposition: full super-chunk time + per-stage-group cost (ablation).
pub fn bench_step_kernel_prefill_super_stages(
    model_dir: &Path,
    cfg: StepSmokeConfig,
    kv_len: u32,
    iters: usize,
    n_subs: usize,
) -> Result<(f64, Vec<(&'static str, f64)>), Error> {
    let (mut rt, _build) = build_step_runtime(model_dir, &cfg)?;
    rt.bench_prefill_super_stages(kv_len, iters, n_subs)
}

/// Profile the first `n_steps` denoise forwards (canvas `st.step` 0..n_steps-1).
/// Step 0 has no SC preamble; step >= 1 includes self-conditioning.
pub fn bench_step_kernel_profile_steps(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
    n_steps: usize,
) -> Result<Vec<(u32, StepProfileResult)>, Error> {
    if cfg.finish != StepFinishMode::Full {
        return Err(Error::Format(
            "bench_step_kernel_profile_steps requires StepFinishMode::Full",
        ));
    }
    let n_steps = n_steps.max(1);
    let watch = crate::flags::mem_watch_enabled();
    let mem_before = watch.then(crate::metal::memwatch::snapshot);
    let (mut rt, build) = build_step_runtime(model_dir, cfg)?;
    if let Some(before) = mem_before {
        crate::metal::memwatch::report_section("session-open", before, &rt.ctx.device);
    }
    let finish = StepFinishMode::Full;
    let mut out = Vec::with_capacity(n_steps);
    for i in 0..n_steps {
        let st: CanvasState = read_struct(&rt.bufs.state);
        let mem_before = watch.then(crate::metal::memwatch::snapshot);
        let mut prof = rt.profile_forward_once(finish)?;
        if i == 0 {
            prof.compile = build.compile;
        }
        if let Some(before) = mem_before {
            crate::metal::memwatch::report_section(
                &format!("profile-step {i}"),
                before,
                &rt.ctx.device,
            );
        }
        out.push((st.step, prof));
    }
    Ok(out)
}

/// Wall-clock timing for fused vs split QKV / gate_up GEMM dispatches (all layers).
#[derive(Debug, Clone)]
pub struct FusedGemmDispatchBenchResult {
    pub compile: std::time::Duration,
    pub layers: usize,
    pub iters: usize,
    /// One command buffer per layer: untimed rmsnorm submit, timed GEMM submit.
    pub qkv_gemm_stacked: std::time::Duration,
    pub qkv_gemm_split: std::time::Duration,
    pub gate_up_gemm_stacked: std::time::Duration,
    pub gate_up_gemm_split: std::time::Duration,
    /// One command buffer: interleaved per-layer rmsnorm + GEMM (production ordering).
    pub qkv_batched_stacked: std::time::Duration,
    pub qkv_batched_split: std::time::Duration,
    pub gate_up_batched_stacked: std::time::Duration,
    pub gate_up_batched_split: std::time::Duration,
    pub qkv_stacked_dispatches_per_pass: usize,
    pub qkv_split_dispatches_per_pass: usize,
    #[allow(dead_code)]
    pub gate_up_dispatches_per_pass: usize,
}

fn qkv_split_dispatch_count(layout: &ModelLayout, layers: usize) -> usize {
    (0..layers)
        .map(|layer| {
            let l = &layout.layers[layer];
            if l.v_proj != 0 { 3 } else { 2 }
        })
        .sum()
}

fn time_qkv_gemm_only(
    rt: &mut StepRuntime,
    layers: usize,
    stacked: bool,
) -> Result<std::time::Duration, Error> {
    use std::time::Instant;
    let layout = rt.layout;
    let mut total = std::time::Duration::ZERO;
    for layer in 0..layers {
        let l = &layout.layers[layer];
        rt.dispatch_and_wait(|enc| {
            enc.rmsnorm(
                enc.arena().hidden_off(),
                enc.arena().tmp_off(),
                l.input_ln,
                HID as u32,
                CANVAS,
            );
            Ok(())
        })?;
        let t0 = Instant::now();
        rt.dispatch_and_wait(|enc| enc.dispatch_qkv_gemms(layer, &layout, stacked))?;
        total += t0.elapsed();
    }
    Ok(total)
}

fn time_gate_up_gemm_only(
    rt: &mut StepRuntime,
    layers: usize,
    stacked: bool,
) -> Result<std::time::Duration, Error> {
    use std::time::Instant;
    let layout = rt.layout;
    let mut total = std::time::Duration::ZERO;
    for layer in 0..layers {
        let l = &layout.layers[layer];
        rt.dispatch_and_wait(|enc| {
            enc.rmsnorm(
                enc.arena().stream_off(),
                enc.arena().tmp_off(),
                l.pre_ff_ln,
                HID as u32,
                CANVAS,
            );
            Ok(())
        })?;
        let t0 = Instant::now();
        rt.dispatch_and_wait(|enc| enc.dispatch_gate_up_gemms(layer, &layout, stacked))?;
        total += t0.elapsed();
    }
    Ok(total)
}

fn time_qkv_batched(
    rt: &mut StepRuntime,
    layers: usize,
    stacked: bool,
) -> Result<std::time::Duration, Error> {
    use std::time::Instant;
    let layout = rt.layout;
    let t0 = Instant::now();
    rt.dispatch_and_wait(|enc| {
        for layer in 0..layers {
            let l = &layout.layers[layer];
            enc.rmsnorm(
                enc.arena().hidden_off(),
                enc.arena().tmp_off(),
                l.input_ln,
                HID as u32,
                CANVAS,
            );
            enc.dispatch_qkv_gemms(layer, &layout, stacked)?;
        }
        Ok(())
    })?;
    Ok(t0.elapsed())
}

fn time_gate_up_batched(
    rt: &mut StepRuntime,
    layers: usize,
    stacked: bool,
) -> Result<std::time::Duration, Error> {
    use std::time::Instant;
    let layout = rt.layout;
    let t0 = Instant::now();
    rt.dispatch_and_wait(|enc| {
        for layer in 0..layers {
            let l = &layout.layers[layer];
            enc.rmsnorm(
                enc.arena().stream_off(),
                enc.arena().tmp_off(),
                l.pre_ff_ln,
                HID as u32,
                CANVAS,
            );
            enc.dispatch_gate_up_gemms(layer, &layout, stacked)?;
        }
        Ok(())
    })?;
    Ok(t0.elapsed())
}

/// Profile QKV and dense gate/up GEMM dispatches in isolation (real weights + canvas activations).
pub fn bench_fused_gemm_dispatches(
    model_dir: &Path,
    cfg: StepSmokeConfig,
    iters: usize,
) -> Result<FusedGemmDispatchBenchResult, Error> {
    let iters = iters.max(1);
    let (mut rt, build) = build_step_runtime(model_dir, &cfg)?;
    let layout = rt.layout;
    let layers = rt.layers;

    // Realistic activations: one forward preamble + embed (no MoE/finish).
    rt.dispatch_and_wait(|enc| {
        let st: CanvasState = read_struct(&enc.bufs.state);
        let first_step = if st.step == 0 { 1u32 } else { 0u32 };
        enc.encode_step_preamble(&layout, first_step)?;
        for layer in 0..layers {
            enc.encode_layer(layer, &layout)?;
        }
        Ok(())
    })?;

    let qkv_split_dispatches = qkv_split_dispatch_count(&layout, layers);
    let gate_up_dispatches = layers * 2;

    let mut qkv_gemm_stacked = std::time::Duration::ZERO;
    let mut qkv_gemm_split = std::time::Duration::ZERO;
    let mut gate_up_gemm_stacked = std::time::Duration::ZERO;
    let mut gate_up_gemm_split = std::time::Duration::ZERO;
    let mut qkv_batched_stacked = std::time::Duration::ZERO;
    let mut qkv_batched_split = std::time::Duration::ZERO;
    let mut gate_up_batched_stacked = std::time::Duration::ZERO;
    let mut gate_up_batched_split = std::time::Duration::ZERO;

    // Warmup each mode once.
    let _ = time_qkv_gemm_only(&mut rt, layers, true)?;
    let _ = time_qkv_gemm_only(&mut rt, layers, false)?;
    let _ = time_gate_up_gemm_only(&mut rt, layers, true)?;
    let _ = time_gate_up_gemm_only(&mut rt, layers, false)?;

    for _ in 0..iters {
        qkv_gemm_stacked += time_qkv_gemm_only(&mut rt, layers, true)?;
        qkv_gemm_split += time_qkv_gemm_only(&mut rt, layers, false)?;
        gate_up_gemm_stacked += time_gate_up_gemm_only(&mut rt, layers, true)?;
        gate_up_gemm_split += time_gate_up_gemm_only(&mut rt, layers, false)?;
        qkv_batched_stacked += time_qkv_batched(&mut rt, layers, true)?;
        qkv_batched_split += time_qkv_batched(&mut rt, layers, false)?;
        gate_up_batched_stacked += time_gate_up_batched(&mut rt, layers, true)?;
        gate_up_batched_split += time_gate_up_batched(&mut rt, layers, false)?;
    }

    let div = |d: std::time::Duration| d / iters as u32;
    Ok(FusedGemmDispatchBenchResult {
        compile: build.compile,
        layers,
        iters,
        qkv_gemm_stacked: div(qkv_gemm_stacked),
        qkv_gemm_split: div(qkv_gemm_split),
        gate_up_gemm_stacked: div(gate_up_gemm_stacked),
        gate_up_gemm_split: div(gate_up_gemm_split),
        qkv_batched_stacked: div(qkv_batched_stacked),
        qkv_batched_split: div(qkv_batched_split),
        gate_up_batched_stacked: div(gate_up_batched_stacked),
        gate_up_batched_split: div(gate_up_batched_split),
        qkv_stacked_dispatches_per_pass: layers,
        qkv_split_dispatches_per_pass: qkv_split_dispatches,
        gate_up_dispatches_per_pass: gate_up_dispatches,
    })
}

/// Read `elems` bf16 arena values from a shared Metal buffer as f32.
pub fn read_arena_buffer_f32(
    buf: &ProtocolObject<dyn MTLBuffer>,
    byte_off: usize,
    elems: usize,
) -> Vec<f32> {
    use crate::shaders::bf16;
    let ptr = unsafe { buf.contents().as_ptr().add(byte_off) as *const u16 };
    (0..elems)
        .map(|i| unsafe { bf16::bf16_bits_to_f32(*ptr.add(i)) })
        .collect()
}

/// Read `elems` bf16 values from a shared Metal buffer as f32 (logits / KV cache).
pub fn read_half_buffer_f32(
    buf: &ProtocolObject<dyn MTLBuffer>,
    byte_off: usize,
    elems: usize,
) -> Vec<f32> {
    use crate::shaders::bf16;
    let ptr = unsafe { buf.contents().as_ptr().add(byte_off) as *const u16 };
    (0..elems)
        .map(|i| unsafe { bf16::bf16_bits_to_f32(*ptr.add(i)) })
        .collect()
}

#[derive(Debug)]
pub struct StepForwardOutput {
    pub norm_hidden: Vec<f32>,
    pub logits: Vec<f32>,
    pub token_ids: Vec<u32>,
}

/// Forward-only monolithic pass: final norm hidden + softcapped logits.
pub fn run_step_forward(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
) -> Result<StepForwardOutput, Error> {
    let (mut rt, _) = build_step_runtime(model_dir, cfg)?;
    let layout = rt.layout;
    let layers = rt.layers;
    let st_before: CanvasState = read_struct(&rt.bufs.state);
    let first_step = if st_before.step == 0 { 1u32 } else { 0u32 };

    rt.dispatch_and_wait(|enc| enc.encode_step_preamble(&layout, first_step))?;
    for layer in 0..layers {
        rt.dispatch_and_wait(|enc| enc.encode_full_layer(layer, &layout))?;
    }
    // Snapshot final norm before lm_head; gemm_q8_logits clobbers self.arena().tmp_off() on GPU.
    rt.dispatch_and_wait(|enc| {
        enc.rmsnorm(
            enc.arena().hidden_off(),
            enc.arena().tmp_off(),
            layout.final_norm,
            HID as u32,
            CANVAS,
        );
        Ok(())
    })?;
    let norm_hidden = read_arena_buffer_f32(
        &rt.bufs.arena,
        rt.bufs.arena_map.tmp_off() as usize,
        CANVAS * HID,
    );
    rt.dispatch_and_wait(|enc| {
        enc.gemm_q8_logits(
            enc.arena().tmp_off(),
            layout.embed,
            CANVAS as u32,
            VOCAB as u32,
            HID as u32,
            0,
        )?;
        enc.dispatch_softcap();
        Ok(())
    })?;
    let state: CanvasState = read_struct(&rt.bufs.state);
    Ok(StepForwardOutput {
        norm_hidden,
        logits: read_half_buffer_f32(&rt.bufs.logits, 0, CANVAS * VOCAB),
        token_ids: state.ids.to_vec(),
    })
}

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub struct DenoiseStepStats {
    pub accept_count: u32,
    pub mean_entropy: f32,
    pub min_entropy: f32,
    pub low_entropy_positions: u32,
}

/// Chat-templated `-p Hello` prefill token ids (matches `generate-monolithic` default).
#[cfg(test)]
pub fn hello_chat_prefill_token_ids(model_dir: &Path) -> Result<Vec<u32>, Error> {
    use crate::chat_template::{ChatFormatOptions, ChatTurn, format_chat_token_ids};
    use crate::tokenizer::Tokenizer;
    let tok = Tokenizer::load(&model_dir.join("tokenizer.json"))?;
    format_chat_token_ids(
        &tok,
        &[ChatTurn::user("Hello")],
        &ChatFormatOptions::default(),
    )
}

/// Run `cfg.steps` monolithic denoise iterations; one stats record per iteration.
#[cfg(test)]
pub fn run_denoise_steps(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
) -> Result<Vec<DenoiseStepStats>, Error> {
    if cfg.finish != StepFinishMode::Full {
        return Err(Error::Format(
            "run_denoise_steps requires StepFinishMode::Full",
        ));
    }
    use crate::sample::StableConfidentStopper;
    let steps = cfg.steps.max(1);
    let (mut rt, _) = build_step_runtime(model_dir, cfg)?;
    let sampler = crate::sample::sampler_for_steps(steps, cfg.no_early_stop);
    let params = step_params_from_sampler(
        &sampler,
        rt.read_params().kv_len,
        cfg.no_early_stop,
        rt.read_params().eos_token_id,
    );
    let mut rng = Rng::new(cfg.seed);
    rt.reset_block(VOCAB, &mut rng, params);
    let mut stopper = StableConfidentStopper::new(
        sampler.stability_threshold,
        if cfg.no_early_stop {
            f32::MAX
        } else {
            sampler.confidence_threshold
        },
    );
    stopper.reset();
    let mut out = Vec::with_capacity(steps);
    for _ in 0..steps {
        rt.run_denoise_step()?;
        let st = rt.read_canvas_state();
        let ent = crate::sample::step_entropy_stats(&st.entropy, &st.accept);
        out.push(DenoiseStepStats {
            accept_count: ent.accept_count,
            mean_entropy: st.mean_entropy,
            min_entropy: ent.min_entropy,
            low_entropy_positions: ent.low_entropy_positions,
        });
        let max_steps_reached = st.step >= params.max_steps;
        let confident_stop = !cfg.no_early_stop && st.stop_flag != 0;
        if confident_stop || max_steps_reached {
            break;
        }
    }
    Ok(out)
}

pub fn run_step_smoke(model_dir: &Path, cfg: StepSmokeConfig) -> Result<StepSmokeResult, Error> {
    let finish = cfg.finish;
    let steps = cfg.steps;
    let (mut rt, _) = build_step_runtime(model_dir, &cfg)?;
    let started = Instant::now();
    for step_i in 0..steps {
        rt.run_forward_once(finish)?;
        eprintln!(
            "step-smoke: completed denoise step {}/{}",
            step_i + 1,
            steps
        );
        if finish == StepFinishMode::Full {
            let st: CanvasState = read_struct(&rt.bufs.state);
            if st.stop_flag != 0 && !cfg.no_early_stop {
                eprintln!("step-smoke: early stop at step {}", st.step);
                break;
            }
        }
    }
    let elapsed = started.elapsed();

    let final_state: CanvasState = read_struct(&rt.bufs.state);
    let (logits_finite, max_abs_logit) = check_logits_finite(&rt.bufs.logits);
    let ent_stats = crate::sample::step_entropy_stats(&final_state.entropy, &final_state.accept);

    Ok(StepSmokeResult {
        step: final_state.step,
        stop_flag: final_state.stop_flag,
        mean_entropy: final_state.mean_entropy,
        min_entropy: ent_stats.min_entropy,
        low_entropy_positions: ent_stats.low_entropy_positions,
        ids: final_state.ids[..CANVAS].try_into().expect("canvas prefix"),
        logits_finite,
        max_abs_logit,
        elapsed,
    })
}
