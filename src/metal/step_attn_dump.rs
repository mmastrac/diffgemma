//! Per-layer attention dumps (Q/K/scores/attn_out) for MLX parity at a canvas row.

use crate::Error;
use crate::metal::step_kernel::{LayerAttnCapture, StepSmokeConfig, run_step_attn_layer_capture};
use serde::Serialize;
use std::path::Path;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize)]
pub struct AttnKeyDump {
    pub position: usize,
    pub k: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StepAttnLayerDump {
    pub schema_version: u32,
    pub source: String,
    pub prompt: String,
    pub prompt_token_ids: Vec<u32>,
    pub initial_canvas_ids: Vec<u32>,
    pub seed: u64,
    pub layer: usize,
    pub position: usize,
    pub canvas_token: u32,
    pub kv_len: u32,
    pub total_kv: usize,
    pub head_dim: u32,
    pub n_heads: usize,
    pub n_kv_heads: u32,
    pub is_full: bool,
    pub hidden_in: Vec<f32>,
    pub hidden_ln: Vec<f32>,
    pub q_raw_proj: Vec<f32>,
    pub q_pre_rope: Vec<f32>,
    pub q_post_rope: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub raw_scores: Vec<f32>,
    pub attn_probs: Vec<f32>,
    pub k_samples: Vec<AttnKeyDump>,
}

fn dump_from_capture(
    cap: &LayerAttnCapture,
    prompt: &str,
    prompt_token_ids: &[u32],
    seed: u64,
) -> StepAttnLayerDump {
    StepAttnLayerDump {
        schema_version: SCHEMA_VERSION,
        source: "rust-monolithic".into(),
        prompt: prompt.to_string(),
        prompt_token_ids: prompt_token_ids.to_vec(),
        initial_canvas_ids: cap.token_ids.clone(),
        seed,
        layer: cap.layer,
        position: cap.position,
        canvas_token: cap.canvas_token,
        kv_len: cap.kv_len,
        total_kv: cap.total_kv,
        head_dim: cap.head_dim,
        n_heads: cap.n_heads,
        n_kv_heads: cap.n_kv_heads,
        is_full: cap.is_full,
        hidden_in: cap.hidden_in.clone(),
        hidden_ln: cap.hidden_ln.clone(),
        q_raw_proj: cap.q_raw_proj.clone(),
        q_pre_rope: cap.q_pre_rope.clone(),
        q_post_rope: cap.q_post_rope.clone(),
        attn_out: cap.attn_out.clone(),
        raw_scores: cap.raw_scores.clone(),
        attn_probs: cap.attn_probs.clone(),
        k_samples: cap
            .k_samples
            .iter()
            .map(|(pos, k)| AttnKeyDump {
                position: *pos,
                k: k.clone(),
            })
            .collect(),
    }
}

pub fn run_step_attn_layer_dump(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
    prompt_label: &str,
    layer: usize,
    position: usize,
) -> Result<StepAttnLayerDump, Error> {
    let cap = run_step_attn_layer_capture(model_dir, cfg, layer, position)?;
    Ok(dump_from_capture(
        &cap,
        prompt_label,
        cfg.prefill_token_ids.as_deref().unwrap_or(&[]),
        cfg.seed,
    ))
}

pub fn write_step_attn_layer_dump(path: &Path, dump: &StepAttnLayerDump) -> Result<(), Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(Error::Io)?;
    }
    let text = serde_json::to_string_pretty(dump).map_err(Error::Json)?;
    std::fs::write(path, text).map_err(Error::Io)
}
