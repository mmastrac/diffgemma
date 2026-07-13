//! Shared helpers used across the command modules.

use super::*;

pub(crate) fn step_kernel_config(
    layers: usize,
    kv_len: u32,
    seed: u64,
    max_seq: usize,
    forward_only: bool,
) -> metal::StepSmokeConfig {
    metal::StepSmokeConfig {
        layers,
        steps: 1,
        kv_len,
        seed,
        max_seq,
        finish: if forward_only {
            metal::StepFinishMode::ForwardOnly
        } else {
            metal::StepFinishMode::Full
        },
        prefill_token_ids: None,
        no_early_stop: false,
    }
}
pub(crate) fn attach_step_prefill(
    cfg: &mut metal::StepSmokeConfig,
    model_dir: &std::path::Path,
    kv_len: u32,
    prompt: Option<&str>,
    raw_prompt: bool,
) -> Result<(), crate::Error> {
    if kv_len == 0 && prompt.is_none() {
        return Ok(());
    }
    let vocab = crate::config::ModelConfig::load(model_dir)?
        .text_config
        .vocab_size;
    let prompt_len = if kv_len > 0 { kv_len as usize } else { 64 };
    let ids = build_prompt_tokens(model_dir, prompt, prompt_len, vocab, raw_prompt, &[])?;
    eprintln!("step-kernel: prefill {} prompt tokens", ids.len());
    cfg.prefill_token_ids = Some(ids);
    Ok(())
}
/// Layer count for generate paths: `--layers` override, else full `num_hidden_layers` from config.
pub(crate) fn resolve_model_layers(
    model_dir: &std::path::Path,
    override_layers: Option<usize>,
) -> Result<usize, crate::Error> {
    let cfg = crate::config::ModelConfig::load(model_dir)?;
    let n = cfg.text_config.num_hidden_layers.max(1);
    Ok(override_layers.unwrap_or(n).max(1).min(n))
}
pub(crate) fn build_prompt_tokens(
    model_dir: &std::path::Path,
    prompt_text: Option<&str>,
    prompt_len: usize,
    vocab: usize,
    raw_prompt: bool,
    history: &[chat_template::ChatTurn],
) -> Result<Vec<u32>, crate::Error> {
    if let Some(text) = prompt_text {
        let tok_path = model_dir.join("tokenizer.json");
        let tokenizer = tokenizer::Tokenizer::load(&tok_path)?;
        if raw_prompt {
            Ok(tokenizer.encode(text, false))
        } else {
            let mut turns = history.to_vec();
            turns.push(chat_template::ChatTurn::user(text));
            chat_template::format_chat_token_ids(
                &tokenizer,
                &turns,
                &chat_template::ChatFormatOptions::default(),
            )
        }
    } else {
        let mut prompt = vec![0u32; prompt_len];
        for (i, id) in prompt.iter_mut().enumerate() {
            *id = ((i * 131 + 7) % vocab.max(1)) as u32;
        }
        Ok(prompt)
    }
}
pub(crate) fn build_chat_prompt_tokens(
    model_dir: &std::path::Path,
    history: &[chat_template::ChatTurn],
    raw_prompt: bool,
) -> Result<Vec<u32>, crate::Error> {
    let tok_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizer::Tokenizer::load(&tok_path)?;
    if raw_prompt {
        let text = history.last().map(|t| t.content.as_str()).unwrap_or("");
        Ok(tokenizer.encode(text, false))
    } else {
        chat_template::format_chat_token_ids(
            &tokenizer,
            history,
            &chat_template::ChatFormatOptions::default(),
        )
    }
}
/// Fail-fast context-budget check (panic-to-error, ROADMAP 3.2): `Err(msg)`
/// when the weights + KV at `max_seq` would exceed the safe fraction of the GPU
/// working-set (estimated from physical RAM, ~72% on Apple Silicon), which
/// swaps or fails the KV allocation. Shared by `chat` (`--ctx`) and `ask`
/// (max_seq sized from the prompt + `--max-new-tokens`). Returns `Ok` when the
/// budget can't be determined (no device query) so it never blocks small runs.
pub(crate) fn check_ctx_budget(max_seq: usize) -> Result<(), String> {
    let phys = crate::metal::memwatch::physical_ram_bytes();
    let budget = (phys as f64 * 0.72) as u64;
    if let Some((needed, ceiling)) = flags::ctx_over_budget(max_seq, budget) {
        let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
        return Err(format!(
            "context {max_seq} needs ~{:.1} GiB GPU-resident (weights + KV), over the ~{:.1} GiB \
             safe budget on this {:.0} GiB machine — it would swap or fail to allocate. Reduce the \
             prompt / --max-new-tokens / --ctx to <= {} (or free RAM).",
            gib(needed),
            gib(ceiling),
            gib(phys),
            flags::max_feasible_ctx(budget),
        ));
    }
    Ok(())
}
pub(crate) fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}
pub(crate) fn print_summary(store: &weights::WeightStore) {
    let s = store.summarize();

    println!("DiffusionGemma weight summary");
    println!("  model dir:          {}", store.model_dir().display());
    if store.is_quantized() {
        println!("  format:             .dgq (quantized, mmap)");
    }
    println!("  shards:             {}", s.shard_count);
    println!("  tensors (index):    {}", s.tensor_count_index);
    println!("  tensors (headers):  {}", s.tensor_count_headers);
    println!("  on-disk total:      {:.2} GiB", gib(s.total_file_bytes));
    println!("  tensor payload:     {:.2} GiB", gib(s.total_data_bytes));
    println!("  total elements:     {}", s.total_elements);

    if let Some(meta) = store.safetensor_metadata() {
        if let Some(params) = meta.get("total_parameters") {
            println!("  index metadata total_parameters: {params}");
        }
        if let Some(size) = meta.get("total_size") {
            println!("  index metadata total_size:       {size}");
        }
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
    match store {
        weights::WeightStore::Safetensors(s) => {
            for shard in &s.shards {
                let payload: u64 = shard.tensors.iter().map(|t| t.data_size as u64).sum();
                println!(
                    "    {}  {:>4} tensors  {:.2} GiB payload  {:.2} GiB file",
                    shard.path.file_name().unwrap().to_string_lossy(),
                    shard.tensors.len(),
                    gib(payload),
                    gib(shard.file_size() as u64),
                );
            }
        }
        weights::WeightStore::Dgq(_) => {
            println!("    model.dgq.bin  (quantized mmap blob)");
        }
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
