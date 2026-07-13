//! Micro-benchmark + device-probe subcommands.

use super::*;

#[cfg(target_os = "macos")]
pub(crate) fn print_encode_subprofile(p: &metal::EncodeSubProfileResult) {
    use metal::{LayerEncodeSubProfile, MoeEncodeSubProfile};
    let layers = p.layers.max(1) as u32;
    let layer_total = p.layer.total();
    let moe_total = p.moe.total();
    let grand = layer_total + moe_total;
    let per_l = |d: std::time::Duration| d / layers;
    let pct = |d: std::time::Duration| {
        if grand.is_zero() {
            0.0
        } else {
            100.0 * d.as_secs_f64() / grand.as_secs_f64()
        }
    };
    let print_layer_rows = |label: &str, prof: &LayerEncodeSubProfile| {
        let rows: [(&str, std::time::Duration); 11] = [
            ("qkv_gemm", prof.qkv_gemm),
            ("qk_rope_kv", prof.qk_rope_kv),
            ("attention", prof.attention),
            ("o_proj_gemm", prof.o_proj_gemm),
            ("o_proj_tail", prof.o_proj_tail),
            ("dense_pre_norm", prof.dense_pre_norm),
            ("dense_gate_up", prof.dense_gate_up),
            ("dense_glu", prof.dense_glu),
            ("dense_down", prof.dense_down),
            ("dense_post_norm", prof.dense_post_norm),
            ("router", prof.router),
        ];
        println!(
            "{label} (total {:.2?}, {:.1}%):",
            prof.total(),
            pct(prof.total())
        );
        let mut ranked: Vec<_> = rows.iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        for (name, d) in ranked {
            println!(
                "  {:16} {:.2?}  ({:.1}%, {:.2?}/layer)",
                name,
                d,
                pct(*d),
                per_l(*d)
            );
        }
    };
    let print_moe_rows = |label: &str, prof: &MoeEncodeSubProfile| {
        let rows: [(&str, std::time::Duration); 7] = [
            ("half_to_f32", prof.half_to_f32),
            ("gather", prof.gather),
            ("gate_up", prof.gate_up),
            ("swiglu", prof.swiglu),
            ("down", prof.down),
            ("scatter", prof.scatter),
            ("post", prof.post),
        ];
        println!(
            "{label} (total {:.2?}, {:.1}%):",
            prof.total(),
            pct(prof.total())
        );
        let mut ranked: Vec<_> = rows.iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        for (name, d) in ranked {
            println!(
                "  {:16} {:.2?}  ({:.1}%, {:.2?}/layer)",
                name,
                d,
                pct(*d),
                per_l(*d)
            );
        }
    };
    println!("bench-step-kernel layer-profile ok");
    println!("  compile:       {:.2?}", p.compile);
    println!("  layers:        {}", p.layers);
    print_layer_rows("encode_layer", &p.layer);
    print_moe_rows("moe_grouped+post", &p.moe);
    println!("  grand_total:   {:.2?}", grand);
}
#[cfg(target_os = "macos")]
pub(crate) fn run_bench_step_kernel_cmd(
    model_dir: &std::path::Path,
    layers: usize,
    kv_len: u32,
    seed: u64,
    max_seq: usize,
    iters: usize,
    forward_only: bool,
    profile: bool,
    profile_steps: usize,
    layer_profile: bool,
) -> ExitCode {
    use metal::bench_step_kernel;

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: bench-step-kernel requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let cfg = step_kernel_config(layers, kv_len, seed, max_seq, forward_only);
    if layer_profile {
        use metal::bench_step_kernel_encode_subprofile;
        eprintln!(
            "bench-step-kernel --layer-profile: layers={layers} kv_len={kv_len} forward_only={forward_only}"
        );
        match bench_step_kernel_encode_subprofile(model_dir, cfg) {
            Ok(p) => {
                print_encode_subprofile(&p);
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("error: {err}");
                ExitCode::FAILURE
            }
        }
    } else if profile_steps > 0 {
        use metal::bench_step_kernel_profile_steps;
        eprintln!(
            "bench-step-kernel --profile-steps {profile_steps}: layers={layers} kv_len={kv_len}"
        );
        match bench_step_kernel_profile_steps(model_dir, &cfg, profile_steps) {
            Ok(rows) => {
                println!(
                    "bench-step-kernel profile-steps ok ({} forwards)",
                    rows.len()
                );
                for (canvas_step, p) in rows {
                    let sc = if canvas_step == 0 { "no SC" } else { "SC" };
                    let per_l = |d: std::time::Duration| d / p.layers.max(1) as u32;
                    let pct =
                        |d: std::time::Duration| 100.0 * d.as_secs_f64() / p.total.as_secs_f64();
                    println!("--- canvas st.step={canvas_step} ({sc}) ---");
                    if canvas_step == 0 {
                        println!("  compile:       {:.2?}", p.compile);
                        println!("  block_format:  {:?}", p.block_format);
                        println!("  layers:        {}", p.layers);
                    }
                    println!(
                        "  preamble:      {:.2?}  ({:.1}%)",
                        p.preamble,
                        pct(p.preamble)
                    );
                    println!(
                        "  pre_moe:       {:.2?}  ({:.1}%, {:.2?}/layer)",
                        p.layer_pre_moe,
                        pct(p.layer_pre_moe),
                        per_l(p.layer_pre_moe)
                    );
                    println!(
                        "  moe_grouped:   {:.2?}  ({:.1}%, {:.2?}/layer)",
                        p.layer_moe,
                        pct(p.layer_moe),
                        per_l(p.layer_moe)
                    );
                    println!(
                        "  moe_post:      {:.2?}  ({:.1}%, {:.2?}/layer)",
                        p.layer_post,
                        pct(p.layer_post),
                        per_l(p.layer_post)
                    );
                    println!("  finish:        {:.2?}  ({:.1}%)", p.finish, pct(p.finish));
                    println!("  total:         {:.2?}", p.total);
                }
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("error: {err}");
                ExitCode::FAILURE
            }
        }
    } else if profile {
        use metal::bench_step_kernel_profile;
        eprintln!(
            "bench-step-kernel --profile: layers={layers} kv_len={kv_len} forward_only={forward_only}"
        );
        match bench_step_kernel_profile(model_dir, cfg) {
            Ok(p) => {
                let per_l = |d: std::time::Duration| d / p.layers.max(1) as u32;
                let pct = |d: std::time::Duration| 100.0 * d.as_secs_f64() / p.total.as_secs_f64();
                println!("bench-step-kernel profile ok");
                println!("  compile:       {:.2?}", p.compile);
                println!("  block_format:  {:?}", p.block_format);
                println!("  layers:        {}", p.layers);
                println!(
                    "  preamble:      {:.2?}  ({:.1}%)",
                    p.preamble,
                    pct(p.preamble)
                );
                println!(
                    "  pre_moe (attn+dense+router): {:.2?}  ({:.1}%, {:.2?}/layer)",
                    p.layer_pre_moe,
                    pct(p.layer_pre_moe),
                    per_l(p.layer_pre_moe)
                );
                println!(
                    "  moe_grouped:   {:.2?}  ({:.1}%, {:.2?}/layer)",
                    p.layer_moe,
                    pct(p.layer_moe),
                    per_l(p.layer_moe)
                );
                println!(
                    "  moe_post:      {:.2?}  ({:.1}%, {:.2?}/layer)",
                    p.layer_post,
                    pct(p.layer_post),
                    per_l(p.layer_post)
                );
                println!("  finish:        {:.2?}  ({:.1}%)", p.finish, pct(p.finish));
                println!("  total:         {:.2?}", p.total);
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("error: {err}");
                ExitCode::FAILURE
            }
        }
    } else {
        eprintln!(
            "bench-step-kernel: layers={layers} kv_len={kv_len} iters={iters} forward_only={forward_only}"
        );
        match bench_step_kernel(model_dir, cfg, iters) {
            Ok(r) => {
                println!("bench-step-kernel ok");
                println!("  compile:  {:.2?}", r.compile);
                println!("  warmup:   {:.2?}", r.warmup);
                println!("  per_step: {:.2?}", r.per_step);
                println!("  iters:    {}", r.iters);
                println!("  mode:     {:?}", r.finish);
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("error: {err}");
                ExitCode::FAILURE
            }
        }
    }
}
#[cfg(target_os = "macos")]
pub(crate) fn run_bench_gemm_fusion_cmd(
    model_dir: &std::path::Path,
    layers: usize,
    kv_len: u32,
    seed: u64,
    max_seq: usize,
    iters: usize,
) -> ExitCode {
    use metal::bench_fused_gemm_dispatches;

    if !dgq::store::looks_like_dgq_dir(model_dir) {
        eprintln!("error: bench-gemm-fusion requires a .dgq directory");
        return ExitCode::FAILURE;
    }
    let cfg = step_kernel_config(layers, kv_len, seed, max_seq, true);
    eprintln!(
        "bench-gemm-fusion: layers={layers} kv_len={kv_len} iters={iters} (QKV + gate/up GEMM isolation)"
    );
    match bench_fused_gemm_dispatches(model_dir, cfg, iters) {
        Ok(r) => {
            let pct = |a: std::time::Duration, b: std::time::Duration| {
                if b.is_zero() {
                    0.0
                } else {
                    100.0 * (1.0 - a.as_secs_f64() / b.as_secs_f64())
                }
            };
            let per_l = |d: std::time::Duration| d / r.layers.max(1) as u32;
            println!("bench-gemm-fusion ok");
            println!("  compile:  {:.2?}", r.compile);
            println!("  layers:   {}", r.layers);
            println!("  iters:    {}", r.iters);
            println!(
                "  dispatches/pass: qkv stacked={} split={} gate_up stacked=1/L split=2/L",
                r.qkv_stacked_dispatches_per_pass, r.qkv_split_dispatches_per_pass
            );
            println!("--- QKV GEMM only (per-layer rmsnorm prep + timed GEMM submit) ---");
            println!(
                "  stacked: {:.2?}  ({:.2?}/layer)",
                r.qkv_gemm_stacked,
                per_l(r.qkv_gemm_stacked)
            );
            println!(
                "  split:   {:.2?}  ({:.2?}/layer)",
                r.qkv_gemm_split,
                per_l(r.qkv_gemm_split)
            );
            println!(
                "  delta:   {:+.1}% stacked vs split",
                pct(r.qkv_gemm_stacked, r.qkv_gemm_split)
            );
            println!("--- gate/up GEMM only (per-layer rmsnorm prep + timed GEMM submit) ---");
            println!(
                "  stacked: {:.2?}  ({:.2?}/layer)",
                r.gate_up_gemm_stacked,
                per_l(r.gate_up_gemm_stacked)
            );
            println!(
                "  split:   {:.2?}  ({:.2?}/layer)",
                r.gate_up_gemm_split,
                per_l(r.gate_up_gemm_split)
            );
            println!(
                "  delta:   {:+.1}% stacked vs split",
                pct(r.gate_up_gemm_stacked, r.gate_up_gemm_split)
            );
            println!("--- batched pass (1 CB, interleaved rmsnorm+GEMM per layer) ---");
            println!(
                "  qkv stacked: {:.2?}  split: {:.2?}  ({:+.1}%)",
                r.qkv_batched_stacked,
                r.qkv_batched_split,
                pct(r.qkv_batched_stacked, r.qkv_batched_split)
            );
            println!(
                "  gate_up stacked: {:.2?}  split: {:.2?}  ({:+.1}%)",
                r.gate_up_batched_stacked,
                r.gate_up_batched_split,
                pct(r.gate_up_batched_stacked, r.gate_up_batched_split)
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
pub(crate) fn run_probe_device() -> ExitCode {
    use metal::{print_probe_result, probe_device};
    match probe_device() {
        Ok(r) => {
            print_probe_result(&r);
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
#[cfg(target_os = "macos")]
pub(crate) fn run_bench_gemm(shapes: &str, oracle: Option<&str>, iters: usize) -> ExitCode {
    use metal::{
        bench_custom_kernel, bench_gemm_bf16, bench_gemm_block_q4, bench_mpsgraph_oracle,
        parse_shapes, print_bench_rows,
    };
    // `--shapes sparse`: block-sparse MoE bench at the production route
    // distributions (MxKxN comes from the model shapes; the spec is ignored).
    if shapes == "sparse" {
        return match metal::bench_gemm_tunable_sparse(iters) {
            Ok(rows) => {
                print_bench_rows(&rows);
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("error: {err}");
                ExitCode::FAILURE
            }
        };
    }
    let parsed = match parse_shapes(shapes) {
        Ok(s) => s,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };
    let mut rows = match bench_custom_kernel(&parsed, iters) {
        Ok(r) => r,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };
    match bench_gemm_block_q4(&parsed, iters) {
        Ok(mut q4) => rows.append(&mut q4),
        Err(err) => eprintln!("warning: gemm_block bench: {err}"),
    }
    match bench_gemm_bf16(&parsed, iters) {
        Ok(mut bf16) => rows.append(&mut bf16),
        Err(err) => eprintln!("warning: gemm_bf16 bench: {err}"),
    }
    match metal::bench_gemm_tunable(&parsed, iters) {
        Ok(mut t) => rows.append(&mut t),
        Err(err) => eprintln!("warning: gemm_tunable bench: {err}"),
    }
    if matches!(oracle, Some("mps") | Some("mpsgraph")) {
        match bench_mpsgraph_oracle(&parsed, iters) {
            Ok(mut mps) => rows.append(&mut mps),
            Err(err) => {
                eprintln!("warning: {err}");
            }
        }
    }
    print_bench_rows(&rows);
    ExitCode::SUCCESS
}
pub(crate) fn run_gemm(size: usize) -> ExitCode {
    #[cfg(target_os = "macos")]
    {
        use metal::{Bf16Gemm, bf16_matmul_cpu, f32_to_bf16};

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
}

/// Tunable E17 prefill-attention bench (task #87). Compiles the kernels for a
/// tile/HC/TPG config and times the full head-chunked prefill sequence at model
/// shape (canvas=256, 16 Q / 2 KV, hd=512). Prints a human line + a machine
/// `RESULT {json}` line (ms/layer, TF/s) for the Optuna sweep driver. Needs no
/// model weights — a synthetic fixture drives the kernels.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_bench_prefill_attn(
    kv_len: u32,
    qk_bm: usize,
    qk_bn: usize,
    pv_bm: usize,
    pv_bn: usize,
    hc: usize,
    sm_tpg: usize,
    side: bool,
    iters: usize,
) -> ExitCode {
    use crate::shaders::attention_gemm::{TuneCfg, bench_tuned, model_full_fixture};
    let cfg = TuneCfg {
        hc,
        qk_bm,
        qk_bn,
        pv_bm,
        pv_bn,
        sm_tpg,
    };
    if !cfg.valid() {
        // Machine-parseable failure so the sweep driver can prune invalid points.
        println!("RESULT {{\"ok\": false, \"reason\": \"invalid config\", \"cfg\": {cfg:?}}}");
        return ExitCode::FAILURE;
    }
    let f = model_full_fixture(kv_len);
    let (canvas, n_q_heads, hd) = (256usize, 16usize, 512usize);
    let t_total = kv_len as usize + canvas;
    // QK (2*M*T*hd) + PV (2*M*T*hd) MAC pairs per head, x heads.
    let flops = (n_q_heads as f64) * 4.0 * (canvas as f64) * (t_total as f64) * (hd as f64);
    match bench_tuned(&f, iters, cfg, side) {
        Ok(ms) => {
            let tf_s = flops / (ms / 1e3) / 1e12;
            println!(
                "prefill-attn kv={kv_len} qk={qk_bm}x{qk_bn} pv={pv_bm}x{pv_bn} hc={hc} \
                 tpg={sm_tpg} side={side}: {ms:.3} ms/layer, {tf_s:.2} TF/s"
            );
            println!(
                "RESULT {{\"ok\": true, \"ms\": {ms:.4}, \"tf_s\": {tf_s:.4}, \"kv_len\": {kv_len}, \
                 \"qk_bm\": {qk_bm}, \"qk_bn\": {qk_bn}, \"pv_bm\": {pv_bm}, \"pv_bn\": {pv_bn}, \
                 \"hc\": {hc}, \"sm_tpg\": {sm_tpg}, \"side\": {side}}}"
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            // Compile/dispatch failure (e.g. register spill) — prune, don't crash.
            println!("RESULT {{\"ok\": false, \"reason\": \"{err}\", \"cfg\": {cfg:?}}}");
            ExitCode::FAILURE
        }
    }
}
