use std::sync::Mutex;

pub(crate) fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// One-line preview of `text`, max `max_chars` Unicode scalars, newlines flattened.
pub(crate) fn trunc_preview(text: &str, max_chars: usize) -> String {
    let one_line = text.replace(['\n', '\r'], " ");
    let mut chars = one_line.chars();
    let head: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{head}...")
    } else {
        head
    }
}

/// Write a pretty-printed JSON object to `path`.
pub(crate) fn write_json_log(
    path: &std::path::Path,
    record: &serde_json::Value,
) -> Result<(), String> {
    let body = serde_json::to_string_pretty(record).map_err(|e| e.to_string())?;
    std::fs::write(path, body).map_err(|e| format!("{}: {e}", path.display()))
}

/// Same as [`write_json_log`], then `fsync` so a killed process still leaves a
/// durable file (used for per-block model logs flushed mid-generation).
pub(crate) fn write_json_log_sync(
    path: &std::path::Path,
    record: &serde_json::Value,
) -> Result<(), String> {
    use std::io::Write;
    let body = serde_json::to_string_pretty(record).map_err(|e| e.to_string())?;
    let mut f = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    f.write_all(body.as_bytes())
        .map_err(|e| format!("{}: {e}", path.display()))?;
    f.sync_all()
        .map_err(|e| format!("{}: sync: {e}", path.display()))?;
    Ok(())
}

/// `serve-NNNNN.json` — server-side turn record for response `seq`.
pub(crate) fn write_serve_log(
    dir: &std::path::Path,
    seq: u64,
    record: &serde_json::Value,
) -> Result<std::path::PathBuf, String> {
    let path = dir.join(format!("serve-{seq:05}.json"));
    write_json_log(&path, record)?;
    Ok(path)
}

/// `model-NNNNN-MMMMM.json` — per-block denoise stats for response `seq`, block `block`.
/// Syncs to disk so mid-turn kills still leave completed blocks.
pub(crate) fn write_model_block_log(
    dir: &std::path::Path,
    seq: u64,
    block: usize,
    record: &serde_json::Value,
) -> Result<std::path::PathBuf, String> {
    let path = dir.join(format!("model-{seq:05}-{block:05}.json"));
    write_json_log_sync(&path, record)?;
    Ok(path)
}

pub(crate) fn block_stats_to_json(
    stats: &crate::generate::BlockDenoiseStats,
    tokenizer: &crate::tokenizer::Tokenizer,
    stop_token: Option<&str>,
) -> serde_json::Value {
    let tokens: Vec<serde_json::Value> = stats
        .token_ids
        .iter()
        .map(|&id| {
            let text = tokenizer.id_to_token(id).unwrap_or("");
            serde_json::json!([id, text])
        })
        .collect();
    serde_json::json!({
        "block": stats.block_idx,
        "max_blocks": stats.max_blocks,
        "steps_eff": stats.steps_eff,
        "denoise_secs": stats.denoise_secs,
        "accept_per_step": stats.accept_per_step,
        "min_ent_per_step": stats.min_ent_per_step,
        "mean_ent_per_step": stats.mean_ent_per_step,
        "low_ent_per_step": stats.low_ent_per_step,
        "late_window": {
            "accept_sum": stats.late_accept_sum,
            "min_ent": stats.late_min_ent,
            "mean_ent": stats.late_mean_ent,
            "max_low_ent": stats.late_max_low_ent,
        },
        "denoise_stop": stats.denoise_stop,
        "kept_tokens": stats.kept_tokens,
        "tokens": tokens,
        "stop_token_id": stats.stop_token_id,
        "stop_token": stop_token,
        "stop_offset": stats.stop_offset,
        "continued_past_stop": stats.continued_past_stop,
    })
}
