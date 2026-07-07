//! Per-layer MoE/FFN dumps (router picks, dense + MoE outputs) for MLX parity.

use crate::metal::step_kernel::{LayerMoeCapture, StepSmokeConfig, run_step_moe_layer_capture};
use crate::safetensors::Error;
use serde::Serialize;
use std::path::Path;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize)]
pub struct StepMoeLayerDump {
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
    pub experts: Vec<u32>,
    pub expert_weights: Vec<u16>,
    pub post_attn: Vec<f32>,
    pub dense_out: Vec<f32>,
    pub router_logits: Vec<f32>,
    pub moe_out: Vec<f32>,
    pub moe_out_ln: Vec<f32>,
    pub layer_out: Vec<f32>,
}

fn dump_from_capture(
    cap: &LayerMoeCapture,
    prompt: &str,
    prompt_token_ids: &[u32],
    seed: u64,
) -> StepMoeLayerDump {
    StepMoeLayerDump {
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
        experts: cap.experts.clone(),
        expert_weights: cap.expert_weights.clone(),
        post_attn: cap.post_attn.clone(),
        dense_out: cap.dense_out.clone(),
        router_logits: cap.router_logits.clone(),
        moe_out: cap.moe_out.clone(),
        moe_out_ln: cap.moe_out_ln.clone(),
        layer_out: cap.layer_out.clone(),
    }
}

pub fn run_step_moe_layer_dump(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
    prompt_label: &str,
    layer: usize,
    position: usize,
) -> Result<StepMoeLayerDump, Error> {
    let cap = run_step_moe_layer_capture(model_dir, cfg, layer, position)?;
    Ok(dump_from_capture(
        &cap,
        prompt_label,
        cfg.prefill_token_ids.as_deref().unwrap_or(&[]),
        cfg.seed,
    ))
}

pub fn write_step_moe_layer_dump(path: &Path, dump: &StepMoeLayerDump) -> Result<(), Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(Error::Io)?;
    }
    let text = serde_json::to_string_pretty(dump).map_err(Error::Json)?;
    std::fs::write(path, text).map_err(Error::Io)
}
