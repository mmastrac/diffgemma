//! Model-level utility subcommands (quantize/convert/tokenize) + CPU-forward probes.

use super::*;

pub(crate) fn run_quantize(
    source_dir: &std::path::Path,
    output: &std::path::Path,
    profile: &str,
    overlay: bool,
    hf_repo: Option<&str>,
    hf_revision: Option<&str>,
) -> ExitCode {
    use dgq::layout::QuantProfile;
    use dgq::{QuantizeOptions, quantize_model};

    let profile_name = profile;
    let profile = match profile {
        "q4" => QuantProfile::Q4,
        "q5" => QuantProfile::Q5,
        "q6" => QuantProfile::Q6,
        "nvfp4" => QuantProfile::Nvfp4,
        // Perf-isolation only (see QuantProfile::Nvfp4Experts doc comment):
        // experts nvfp4, everything else classified exactly like q4.
        "nvfp4x" => QuantProfile::Nvfp4Experts,
        other => {
            eprintln!("error: unknown profile {other} (use q4, q5, q6, nvfp4, or nvfp4x)");
            return ExitCode::FAILURE;
        }
    };

    let out_dir = if output.extension().is_some_and(|e| e == "dgq") {
        output.with_extension("")
    } else {
        output.to_path_buf()
    };

    let overlay_base = if overlay {
        let hf_store = match weights::SafetensorStore::open(source_dir) {
            Ok(s) => s,
            Err(err) => {
                eprintln!("error: opening HF source {}: {err}", source_dir.display());
                return ExitCode::FAILURE;
            }
        };
        match dgq::overlay::auto_or_override_base_model(
            source_dir,
            &hf_store,
            hf_repo.map(str::to_string),
            hf_revision.map(str::to_string),
        ) {
            Ok(base) => Some(base),
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };

    eprintln!(
        "quantize: {} -> {} (profile={profile_name}{})",
        source_dir.display(),
        out_dir.display(),
        if overlay { ", overlay" } else { "" },
    );
    let started = std::time::Instant::now();
    match quantize_model(QuantizeOptions {
        source_dir: source_dir.to_path_buf(),
        output_prefix: out_dir.clone(),
        profile,
        overlay_base,
    }) {
        Ok(summary) => {
            let gib = summary.blob_bytes as f64 / (1024.0_f64.powi(3));
            let local_gib = summary.local_blob_bytes as f64 / (1024.0_f64.powi(3));
            println!("quantize ok");
            println!("  output dir:    {}", out_dir.display());
            println!("  tensors:       {}", summary.tensor_count);
            println!("  canonical size:{gib:.2} GiB");
            if overlay {
                println!("  local blob:    {local_gib:.2} GiB");
            }
            println!("  q4 tensors:    {}", summary.q4_tensors);
            println!("  q6 tensors:    {}", summary.q6_tensors);
            println!("  nvfp4 tensors: {}", summary.nvfp4_tensors);
            println!("  q8 tensors:    {}", summary.q8_tensors);
            println!("  raw tensors:   {}", summary.raw_tensors);
            println!("  elapsed:       {:.2?}", started.elapsed());
            println!(
                "  manifest:      {}/{}",
                out_dir.display(),
                dgq::layout::MANIFEST_FILE
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

pub(crate) fn run_repack_overlay(
    pack_dir: &std::path::Path,
    output: &std::path::Path,
    hf_source: Option<&std::path::Path>,
    hf_repo: Option<&str>,
    hf_revision: Option<&str>,
) -> ExitCode {
    use dgq::overlay::{RepackOverlayOptions, repack_overlay};

    let out_dir = if output.extension().is_some_and(|e| e == "dgq") {
        output.with_extension("")
    } else {
        output.to_path_buf()
    };

    eprintln!(
        "repack --overlay: {} -> {}",
        pack_dir.display(),
        out_dir.display(),
    );
    let started = std::time::Instant::now();
    match repack_overlay(RepackOverlayOptions {
        pack_dir: pack_dir.to_path_buf(),
        output_dir: out_dir.clone(),
        hf_source_dir: hf_source.map(std::path::Path::to_path_buf),
        hf_repo_override: hf_repo.map(str::to_string),
        hf_revision_override: hf_revision.map(str::to_string),
    }) {
        Ok(summary) => {
            let gib = summary.local_blob_bytes as f64 / (1024.0_f64.powi(3));
            println!("repack --overlay ok");
            println!("  output dir:       {}", out_dir.display());
            println!("  tensors:          {}", summary.total_tensors);
            println!("  external (HF):    {}", summary.external_tensors);
            println!("  local:            {}", summary.local_tensors);
            println!("  local blob:       {gib:.2} GiB");
            println!(
                "  base model:       {}@{}",
                summary.base_model.repo, summary.base_model.revision
            );
            println!("  HF shards refd:   {}", summary.shard_count);
            println!(
                "  verbatim mismatches: {}",
                summary.verbatim_mismatches.len()
            );
            println!("  elapsed:          {:.2?}", started.elapsed());
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
pub(crate) fn run_tokenize(model_dir: &std::path::Path, text: &str, raw_prompt: bool) -> ExitCode {
    let path = model_dir.join("tokenizer.json");
    match tokenizer::Tokenizer::load(&path) {
        Ok(tok) => {
            let (formatted, ids) = if raw_prompt {
                (text.to_string(), tok.encode(text, false))
            } else {
                let turns = [chat_template::ChatTurn::user(text)];
                let formatted = chat_template::format_user_prompt(text);
                let ids = match chat_template::format_chat_token_ids(
                    &tok,
                    &turns,
                    &chat_template::ChatFormatOptions::default(),
                ) {
                    Ok(v) => v,
                    Err(err) => {
                        eprintln!("error: {err}");
                        return ExitCode::FAILURE;
                    }
                };
                (formatted, ids)
            };
            let payload = serde_json::json!({
                "text": text,
                "formatted": formatted,
                "chat_template": !raw_prompt,
                "ids": ids,
            });
            println!("{}", serde_json::to_string(&payload).unwrap_or_default());
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
pub(crate) fn run_attention_parity(m: &model::Model) -> ExitCode {
    #[cfg(target_os = "macos")]
    {
        use metal::GpuAttention;
        use model::attention::{AttentionParams, forward_to_attn_out, prepare_qkv_pre_rope};

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

        let params = match AttentionParams::for_layer(&m.config.text_config, 0) {
            Ok(p) => p,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };

        let mut scratch = model::attention::AttentionScratch::new(SEQ_LEN, hidden, &params);

        let mut input = vec![0.0f32; SEQ_LEN * hidden];
        for (i, v) in input.iter_mut().enumerate() {
            *v = ((i % hidden) as f32) * 0.01 - 0.5;
        }
        let positions: Vec<i64> = (0..SEQ_LEN as i64).collect();

        if let Err(err) = prepare_qkv_pre_rope(
            &input,
            &layer,
            &m.config.text_config,
            0,
            SEQ_LEN,
            &positions,
            &mut scratch,
        ) {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }

        let pre_q = scratch.q.clone();
        let pre_k = scratch.k.clone();
        let pre_v = scratch.v.clone();
        let freqs = scratch.rope_freqs.clone();

        let q_dim = SEQ_LEN * params.n_heads * params.head_dim;
        let mut cpu_out = vec![0.0f32; q_dim];
        if let Err(err) = forward_to_attn_out(
            &mut cpu_out,
            &input,
            &layer,
            &m.config.text_config,
            0,
            SEQ_LEN,
            &positions,
            &mut scratch,
        ) {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }

        let mut gpu_q = pre_q;
        let mut gpu_k = pre_k;
        let mut gpu_out = vec![0.0f32; q_dim];
        let mut gpu = match GpuAttention::new() {
            Ok(g) => g,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };

        eprintln!(
            "running layer 0 GPU attention (seq={SEQ_LEN}, heads={}, kv_heads={}, head_dim={})...",
            params.n_heads, params.n_kv_heads, params.head_dim
        );
        let started = std::time::Instant::now();
        if let Err(err) = gpu.rope_and_gqa(
            &mut gpu_out,
            &mut gpu_q,
            &mut gpu_k,
            &pre_v,
            &freqs,
            SEQ_LEN,
            SEQ_LEN,
            &params,
            model::attention::GqaMask::CausalSliding,
        ) {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
        let gpu_elapsed = started.elapsed();

        let mut max_abs = 0.0f32;
        let mut max_idx = 0usize;
        for (i, (&c, &g)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
            let d = (c - g).abs();
            if d > max_abs {
                max_abs = d;
                max_idx = i;
            }
        }

        println!("layer 0 attention parity ok");
        println!(
            "  shape: [{SEQ_LEN}, {}, {}]",
            params.n_heads, params.head_dim
        );
        println!("  gpu elapsed: {gpu_elapsed:.2?}");
        println!("  max_abs_diff: {max_abs:.6} at index {max_idx}");
        println!(
            "  cpu[0..4]: [{:.6}, {:.6}, {:.6}, {:.6}]",
            cpu_out[0], cpu_out[1], cpu_out[2], cpu_out[3]
        );
        println!(
            "  gpu[0..4]: [{:.6}, {:.6}, {:.6}, {:.6}]",
            gpu_out[0], gpu_out[1], gpu_out[2], gpu_out[3]
        );

        const TOL: f32 = 1e-3;
        if max_abs <= TOL {
            ExitCode::SUCCESS
        } else {
            eprintln!("error: max_abs_diff {max_abs} exceeds tolerance {TOL}");
            ExitCode::FAILURE
        }
    }
}
pub(crate) fn run_layer0_forward(m: &model::Model) -> ExitCode {
    const SEQ_LEN: usize = 16;
    let hidden = m.config.text_config.hidden_size;

    let layer =
        match model::layer_weights::DecoderLayerWeights::load(&m.weights, 0, &m.config.text_config)
        {
            Ok(layer) => layer,
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        };

    let mut scratch =
        match model::decoder_layer::DecoderLayerScratch::new(SEQ_LEN, &m.config.text_config, 0) {
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
