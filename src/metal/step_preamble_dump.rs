//! Preamble (canvas embed) hidden dumps for MLX parity before layer 0.

use crate::metal::step_kernel::{run_step_preamble_capture, PreambleCapture, StepSmokeConfig};
use crate::safetensors::Error;
use serde::Serialize;
use std::path::Path;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize)]
pub struct StepPreambleDump {
    pub schema_version: u32,
    pub source: String,
    pub prompt: String,
    pub prompt_token_ids: Vec<u32>,
    pub initial_canvas_ids: Vec<u32>,
    pub seed: u64,
    pub position: usize,
    pub canvas_token: u32,
    pub kv_len: u32,
    /// `embed_tokens(id) * sqrt(hidden)` before self-conditioning / no-scale norm.
    pub embed_scaled: Vec<f32>,
    pub embed_scaled_l2: f32,
    pub embed_scaled_max_abs: f32,
    /// Monolithic step-1 preamble output (embed + SC zeros + no-scale RMSNorm).
    pub after_preamble: Vec<f32>,
    pub after_preamble_l2: f32,
    pub after_preamble_max_abs: f32,
}

fn vec_l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn vec_max_abs(v: &[f32]) -> f32 {
    v.iter().map(|x| x.abs()).fold(0.0f32, f32::max)
}

fn dump_from_capture(
    cap: &PreambleCapture,
    prompt: &str,
    prompt_token_ids: &[u32],
    seed: u64,
) -> StepPreambleDump {
    StepPreambleDump {
        schema_version: SCHEMA_VERSION,
        source: "rust-monolithic".into(),
        prompt: prompt.to_string(),
        prompt_token_ids: prompt_token_ids.to_vec(),
        initial_canvas_ids: cap.token_ids.clone(),
        seed,
        position: cap.position,
        canvas_token: cap.canvas_token,
        kv_len: cap.kv_len,
        embed_scaled: cap.embed_scaled.clone(),
        embed_scaled_l2: vec_l2(&cap.embed_scaled),
        embed_scaled_max_abs: vec_max_abs(&cap.embed_scaled),
        after_preamble: cap.after_preamble.clone(),
        after_preamble_l2: vec_l2(&cap.after_preamble),
        after_preamble_max_abs: vec_max_abs(&cap.after_preamble),
    }
}

pub fn run_step_preamble_dump(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
    prompt_label: &str,
    position: usize,
) -> Result<StepPreambleDump, Error> {
    let cap = run_step_preamble_capture(model_dir, cfg, position)?;
    Ok(dump_from_capture(
        &cap,
        prompt_label,
        cfg.prefill_token_ids.as_deref().unwrap_or(&[]),
        cfg.seed,
    ))
}

pub fn write_step_preamble_dump(path: &Path, dump: &StepPreambleDump) -> Result<(), Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(Error::Io)?;
    }
    let text = serde_json::to_string_pretty(dump).map_err(Error::Json)?;
    std::fs::write(path, text).map_err(Error::Io)
}
