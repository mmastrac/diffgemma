//! MoE router bucketing state on the fused denoise path.

use crate::metal::step_kernel::{
    CANVAS, HID, N_EXPERTS, StepSmokeConfig, run_step_moe_route_capture,
};
use crate::safetensors::Error;
use serde::Serialize;
use std::path::Path;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize)]
pub struct StepMoeRouteDump {
    pub schema_version: u32,
    pub source: String,
    pub prompt: String,
    pub prompt_token_ids: Vec<u32>,
    pub seed: u64,
    pub layer: usize,
    pub step: u32,
    pub kv_len: u32,
    pub moe_style: String,
    pub expected_num_slots: u32,
    pub num_slots: u32,
    pub row_start: Vec<u32>,
    pub count: Vec<u32>,
    pub per_expert_slots: Vec<u32>,
    pub experts_used: u32,
    pub slots_ok: bool,
    pub grouped_dispatched: bool,
    pub moe_out_l2: Option<f32>,
    pub moe_out_nonzero: Option<usize>,
    pub moe_out_gpu_cpu_cos: Option<f32>,
    pub moe_out_gpu_cpu_rel_l2: Option<f32>,
}

impl StepMoeRouteDump {
    fn from_capture(
        cap: &crate::metal::step_kernel::MoeRouteCapture,
        prompt: &str,
        prompt_token_ids: &[u32],
        seed: u64,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            source: "rust-monolithic-denoise".into(),
            prompt: prompt.to_string(),
            prompt_token_ids: prompt_token_ids.to_vec(),
            seed,
            layer: cap.layer,
            step: cap.step,
            kv_len: cap.kv_len,
            moe_style: cap.moe_style.clone(),
            expected_num_slots: cap.route.expected_num_slots,
            num_slots: cap.route.num_slots,
            row_start: cap.route.row_start.clone(),
            count: cap.route.count.clone(),
            per_expert_slots: cap.route.per_expert_slots.clone(),
            experts_used: cap.route.experts_used,
            slots_ok: cap.route.slots_ok,
            grouped_dispatched: cap.grouped_dispatched,
            moe_out_l2: cap.moe_out_l2,
            moe_out_nonzero: cap.moe_out_nonzero,
            moe_out_gpu_cpu_cos: cap.moe_out_gpu_cpu_cos,
            moe_out_gpu_cpu_rel_l2: cap.moe_out_gpu_cpu_rel_l2,
        }
    }
}

pub fn run_step_moe_route_dump(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
    prompt_label: &str,
    layer: usize,
    run_grouped: bool,
) -> Result<StepMoeRouteDump, Error> {
    let cap = run_step_moe_route_capture(model_dir, cfg, layer, run_grouped)?;
    Ok(StepMoeRouteDump::from_capture(
        &cap,
        prompt_label,
        cfg.prefill_token_ids.as_deref().unwrap_or(&[]),
        cfg.seed,
    ))
}

pub fn write_step_moe_route_dump(path: &Path, dump: &StepMoeRouteDump) -> Result<(), Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(Error::Io)?;
    }
    let text = serde_json::to_string_pretty(dump).map_err(Error::Json)?;
    std::fs::write(path, text).map_err(Error::Io)
}

pub fn print_route_summary(dump: &StepMoeRouteDump) {
    eprintln!(
        "step-moe-route-dump: layer={} step={} kv_len={} moe_style={}",
        dump.layer, dump.step, dump.kv_len, dump.moe_style
    );
    eprintln!(
        "  num_slots={} expected={} slots_ok={} experts_used={}/{}",
        dump.num_slots, dump.expected_num_slots, dump.slots_ok, dump.experts_used, N_EXPERTS
    );
    if dump.num_slots == 0 {
        eprintln!("  FAIL: num_slots=0 → grouped GEMM sees M=0, MoE contributes nothing");
    }
    let sum: u32 = dump.per_expert_slots.iter().sum();
    if sum != dump.num_slots {
        eprintln!(
            "  WARN: per_expert_slots sum {sum} != num_slots {}",
            dump.num_slots
        );
    }
    if dump.grouped_dispatched {
        let path = if dump.moe_style == "scalar_per_expert" {
            "scalar per-expert"
        } else {
            "batched grouped"
        };
        match (dump.moe_out_l2, dump.moe_out_nonzero) {
            (Some(l2), Some(nz)) => {
                eprintln!(
                    "  {path} GEMM: moe_out_l2={l2:.4} nonzero_elems={nz}/{}",
                    CANVAS * HID
                );
                if l2 < 1e-6 {
                    eprintln!("  FAIL: moe_out near zero after {path} dispatch");
                }
            }
            _ => eprintln!("  {path} GEMM: dispatched (no readback stats)"),
        }
        match (dump.moe_out_gpu_cpu_cos, dump.moe_out_gpu_cpu_rel_l2) {
            (Some(cos), Some(rel)) => {
                eprintln!(
                    "  GPU vs CPU oracle ({path}): cos={cos:.6} rel_l2={rel:.6} (full canvas {CANVAS}×{HID})"
                );
                if cos < 0.99 {
                    eprintln!("  FAIL: GPU moe_out diverges from fill_moe_out_dgq_cpu");
                }
            }
            _ => eprintln!("  GPU vs CPU oracle: (not computed)"),
        }
    }
}
