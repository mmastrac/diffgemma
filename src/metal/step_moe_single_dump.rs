//! Single-expert MoE probe: bypass router, compare GPU grouped kernel vs CPU Q4 ref.

use crate::Error;
use crate::config::ModelConfig;
use crate::dgq::DgqStore;
use crate::metal::device::MetalContext;
use crate::metal::moe::{
    expert_forward_k_moe_grouped_mirror, expert_forward_staged_bf16, expert_forward_staged_dgq,
    experts_forward_dgq_cpu, load_expert_bf16_slices,
};
use crate::metal::step_kernel::{
    CANVAS, HID, SingleExpertMoeCapture, StepSmokeConfig, run_step_moe_single_expert_capture,
};
use crate::metal::weights::GpuDecoderWeightCache;
use crate::model::moe::{MoeScratch, RouteResult};
use crate::weights::WeightStore;
use serde::Serialize;
use std::path::Path;

pub const SCHEMA_VERSION: u32 = 4;

#[derive(Debug, Clone, Serialize)]
pub struct GpuKernelInputProbeDump {
    pub tok: u32,
    pub slot: u32,
    pub expert: u32,
    pub weight: f32,
    pub x_head: [f32; 8],
    pub moe_in_row0_head: [f32; 8],
    pub down_o_head: [f32; 8],
    pub moe_out_tok_row_head: [f32; 8],
}

fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// Fraction of elements where relative error exceeds `rel_thresh`.
fn divergent_fraction(a: &[f32], b: &[f32], rel_thresh: f32) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut bad = 0usize;
    for i in 0..n {
        let denom = a[i].abs().max(b[i].abs()).max(1e-6);
        if ((a[i] - b[i]) / denom).abs() > rel_thresh {
            bad += 1;
        }
    }
    bad as f32 / n as f32
}

#[derive(Debug, Clone, Serialize)]
pub struct StepMoeSingleExpertDump {
    pub schema_version: u32,
    pub source: String,
    pub prompt: String,
    pub prompt_token_ids: Vec<u32>,
    pub initial_canvas_ids: Vec<u32>,
    pub seed: u64,
    pub layer: usize,
    pub position: usize,
    pub expert_id: u32,
    pub canvas_token: u32,
    pub kv_len: u32,
    pub moe_in: Vec<f32>,
    pub gpu_out: Vec<f32>,
    pub cpu_out: Vec<f32>,
    pub bf16_out: Vec<f32>,
    pub grouped_mirror_out: Vec<f32>,
    pub gate_act: Vec<f32>,
    pub gpu_act_after_barrier: Vec<f32>,
    pub gpu_act_at_down_read: Vec<f32>,
    pub gpu_input: GpuKernelInputProbeDump,
    pub cos_gpu_x_head_cpu_moe_in: f32,
    pub cos_gpu_out_cpu: f32,
    pub cos_gpu_readback_vs_kernel_moe_out: f32,
    pub cos_kernel_moe_out_cpu: f32,
    pub cos_kernel_moe_out_mirror: f32,
    pub cos_gpu_act_after_cpu_gate: f32,
    pub cos_gpu_act_down_cpu_gate: f32,
    pub cos_gpu_act_after_down: f32,
    pub gpu_act_divergent_frac_vs_cpu: f32,
}

fn single_expert_cpu_ref(
    model_dir: &Path,
    layer: usize,
    position: usize,
    expert_id: u32,
    moe_in_row: &[f32],
) -> Result<Vec<f32>, Error> {
    let text = ModelConfig::load(model_dir)?.text_config;
    let store = WeightStore::open(model_dir)?;
    let ctx = MetalContext::new()?;
    let cache = GpuDecoderWeightCache::load(&store, &text, 0, &ctx.device)?;
    if !cache.is_dgq() {
        return Err(Error::Runtime(
            "single-expert cpu ref requires .dgq weights",
        ));
    }

    let mut moe_in = vec![0.0f32; CANVAS * HID];
    moe_in[position * HID..(position + 1) * HID].copy_from_slice(moe_in_row);

    let mut routes: Vec<RouteResult> = (0..CANVAS)
        .map(|_| RouteResult {
            indices: Vec::new(),
            weights: Vec::new(),
        })
        .collect();
    routes[position] = RouteResult {
        indices: vec![expert_id as usize],
        weights: vec![1.0],
    };

    let mut out = vec![0.0f32; CANVAS * HID];
    let mut scratch = MoeScratch::new(CANVAS, &text);
    experts_forward_dgq_cpu(
        &mut out,
        &moe_in,
        &cache,
        layer,
        &text,
        CANVAS,
        &routes,
        &mut scratch,
    )?;
    Ok(out[position * HID..(position + 1) * HID].to_vec())
}

fn single_expert_staged_refs(
    model_dir: &Path,
    layer: usize,
    expert_id: u32,
    moe_in_row: &[f32],
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), Error> {
    let text = ModelConfig::load(model_dir)?.text_config;
    let store = DgqStore::open(model_dir)?;
    let ws = WeightStore::open(model_dir)?;
    let ctx = MetalContext::new()?;
    let cache = GpuDecoderWeightCache::load(&ws, &text, 0, &ctx.device)?;
    let hidden = text.hidden_size;
    let moe_inter = text.moe_intermediate_size;
    let expert = expert_id as usize;
    let gate_up = cache.expert_gate_up_q4(layer, expert);
    let down = cache.expert_down_q4(layer, expert);
    let mut scratch = MoeScratch::new(1, &text);
    let q4 =
        expert_forward_staged_dgq(moe_in_row, &gate_up, &down, moe_inter, hidden, &mut scratch);
    let (gu_bf16, dn_bf16) = load_expert_bf16_slices(&store, layer, expert, moe_inter, hidden)?;
    let bf16 = expert_forward_staged_bf16(
        moe_in_row,
        &gu_bf16,
        &dn_bf16,
        moe_inter,
        hidden,
        &mut scratch,
    );
    let mirror =
        expert_forward_k_moe_grouped_mirror(moe_in_row, &gate_up, &down, moe_inter, hidden);
    Ok((bf16.out, mirror, q4.gate_act))
}

fn dump_from_capture(
    cap: &SingleExpertMoeCapture,
    cpu_out: Vec<f32>,
    bf16_out: Vec<f32>,
    grouped_mirror_out: Vec<f32>,
    gate_act: Vec<f32>,
    prompt: &str,
    prompt_token_ids: &[u32],
    seed: u64,
) -> StepMoeSingleExpertDump {
    let cos_gpu_out_cpu = cosine_f32(&cap.gpu_out, &cpu_out);
    let cos_gpu_act_after_cpu_gate = cosine_f32(&cap.gpu_act_after_barrier, &gate_act);
    let cos_gpu_act_down_cpu_gate = cosine_f32(&cap.gpu_act_at_down_read, &gate_act);
    let cos_gpu_act_after_down = cosine_f32(&cap.gpu_act_after_barrier, &cap.gpu_act_at_down_read);
    let gpu_act_divergent_frac_vs_cpu =
        divergent_fraction(&cap.gpu_act_after_barrier, &gate_act, 0.05);
    let cpu_moe_in_head: [f32; 8] = cap.moe_in[..8].try_into().expect("moe_in head");
    let cos_gpu_x_head_cpu_moe_in = cosine_f32(&cap.gpu_input.x_head, &cpu_moe_in_head);
    let gpu_readback_head: [f32; 8] = cap.gpu_out[..8].try_into().expect("gpu_out head");
    let cos_gpu_readback_vs_kernel_moe_out =
        cosine_f32(&gpu_readback_head, &cap.gpu_input.moe_out_tok_row_head);
    let cos_kernel_moe_out_cpu = cosine_f32(&cap.gpu_input.moe_out_tok_row_head, &cpu_out[..8]);
    let cos_kernel_moe_out_mirror = cosine_f32(
        &cap.gpu_input.moe_out_tok_row_head,
        &grouped_mirror_out[..8],
    );
    eprintln!(
        "moe-single kernel input: tok={} slot={} expert={} weight={:.4} \
         cos(x_head,cpu_moe_in)={cos_gpu_x_head_cpu_moe_in:.4}",
        cap.gpu_input.tok, cap.gpu_input.slot, cap.gpu_input.expert, cap.gpu_input.weight,
    );
    let mirror_head: [f32; 8] = grouped_mirror_out[..8].try_into().expect("mirror head");
    eprintln!(
        "moe-single moe_out row {}: kernel_readback={:?} cpu_readback={:?} mirror={:?}",
        cap.gpu_input.tok, cap.gpu_input.moe_out_tok_row_head, gpu_readback_head, mirror_head,
    );
    eprintln!(
        "moe-single moe_out cos: readback_vs_kernel={cos_gpu_readback_vs_kernel_moe_out:.4} \
         kernel_vs_cpu={cos_kernel_moe_out_cpu:.4} kernel_vs_mirror={cos_kernel_moe_out_mirror:.4} \
         down_o_head[0]={:.5}",
        cap.gpu_input.down_o_head[0],
    );
    eprintln!(
        "moe-single act probe: cos(gpu_out,cpu)={cos_gpu_out_cpu:.4} \
         cos(act@barrier,cpu_gate)={cos_gpu_act_after_cpu_gate:.4} \
         cos(act@down,cpu_gate)={cos_gpu_act_down_cpu_gate:.4} \
         cos(act@barrier,act@down)={cos_gpu_act_after_down:.4} \
         act_divergent_frac={gpu_act_divergent_frac_vs_cpu:.3}"
    );
    StepMoeSingleExpertDump {
        schema_version: SCHEMA_VERSION,
        source: "rust-monolithic".into(),
        prompt: prompt.to_string(),
        prompt_token_ids: prompt_token_ids.to_vec(),
        initial_canvas_ids: cap.token_ids.clone(),
        seed,
        layer: cap.layer,
        position: cap.position,
        expert_id: cap.expert_id,
        canvas_token: cap.canvas_token,
        kv_len: cap.kv_len,
        moe_in: cap.moe_in.clone(),
        gpu_out: cap.gpu_out.clone(),
        cpu_out,
        bf16_out,
        grouped_mirror_out,
        gate_act,
        gpu_act_after_barrier: cap.gpu_act_after_barrier.clone(),
        gpu_act_at_down_read: cap.gpu_act_at_down_read.clone(),
        gpu_input: GpuKernelInputProbeDump {
            tok: cap.gpu_input.tok,
            slot: cap.gpu_input.slot,
            expert: cap.gpu_input.expert,
            weight: cap.gpu_input.weight,
            x_head: cap.gpu_input.x_head,
            moe_in_row0_head: cap.gpu_input.moe_in_row0_head,
            down_o_head: cap.gpu_input.down_o_head,
            moe_out_tok_row_head: cap.gpu_input.moe_out_tok_row_head,
        },
        cos_gpu_x_head_cpu_moe_in,
        cos_gpu_out_cpu,
        cos_gpu_readback_vs_kernel_moe_out,
        cos_kernel_moe_out_cpu,
        cos_kernel_moe_out_mirror,
        cos_gpu_act_after_cpu_gate,
        cos_gpu_act_down_cpu_gate,
        cos_gpu_act_after_down,
        gpu_act_divergent_frac_vs_cpu,
    }
}

pub fn run_step_moe_single_expert_dump(
    model_dir: &Path,
    cfg: &StepSmokeConfig,
    prompt_label: &str,
    layer: usize,
    position: usize,
    expert_id: u32,
) -> Result<StepMoeSingleExpertDump, Error> {
    let cap = run_step_moe_single_expert_capture(model_dir, cfg, layer, position, expert_id)?;
    let cpu_out = single_expert_cpu_ref(model_dir, layer, position, expert_id, &cap.moe_in)?;
    let (bf16_out, grouped_mirror_out, gate_act) =
        single_expert_staged_refs(model_dir, layer, expert_id, &cap.moe_in)?;
    Ok(dump_from_capture(
        &cap,
        cpu_out,
        bf16_out,
        grouped_mirror_out,
        gate_act,
        prompt_label,
        cfg.prefill_token_ids.as_deref().unwrap_or(&[]),
        cfg.seed,
    ))
}

pub fn write_step_moe_single_expert_dump(
    path: &Path,
    dump: &StepMoeSingleExpertDump,
) -> Result<(), Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(Error::Io)?;
    }
    let text = serde_json::to_string_pretty(dump).map_err(Error::Json)?;
    std::fs::write(path, text).map_err(Error::Io)
}
