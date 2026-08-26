//! Tier-1 pin: batched MoE pipeline stages with real routed jobs from denoise capture.

use crate::metal::{LayerOffsets, MOE_FF, N_EXPERTS, RouteScratch, layer_moe_block_jobs};
use crate::shaders::QuantFormat;
use crate::shaders::cpu::gemm_linear_grouped::gemm_linear_grouped_cpu;
use crate::shaders::cpu::moe_scatter_weighted::moe_scatter_weighted;
use crate::shaders::swiglu::InterleavedFixture;
use serde::{Deserialize, Serialize};

fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        let xf = a[i] as f64;
        let yf = b[i] as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    if na > 0.0 && nb > 0.0 {
        (dot / (na.sqrt() * nb.sqrt())) as f32
    } else {
        0.0
    }
}

/// Element-wise gate_up GEMM diff structure (GPU vs CPU oracle).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateUpDiffAnalysis {
    pub cos_gate_half: f32,
    pub cos_up_half: f32,
    pub rel_l2_gate_half: f32,
    pub rel_l2_up_half: f32,
    /// Mean |gpu-cpu| per output column, grouped by `col % 32` (N-tile boundary probe).
    pub col_mod32_mean_abs_err: [f32; 32],
    /// Worst slot rows by cosine (slot_idx, cos).
    pub worst_slots: Vec<(usize, f32)>,
    /// `dot(gpu,cpu)/dot(cpu,cpu)` — ~1.0 if error is uniform scale.
    pub scale_ratio: f32,
}

pub fn analyze_gate_up_diff(
    gpu: &[f32],
    cpu: &[f32],
    slots: usize,
    n: usize,
) -> GateUpDiffAnalysis {
    assert_eq!(gpu.len(), cpu.len());
    assert_eq!(gpu.len(), slots * n);
    let moe_ff = n / 2;

    let half_slice = |buf: &[f32], gate: bool| -> Vec<f32> {
        let mut out = Vec::with_capacity(slots * moe_ff);
        for row in 0..slots {
            let base = row * n;
            let (lo, hi) = if gate {
                (base, base + moe_ff)
            } else {
                (base + moe_ff, base + n)
            };
            out.extend_from_slice(&buf[lo..hi]);
        }
        out
    };

    let gpu_gate = half_slice(gpu, true);
    let cpu_gate = half_slice(cpu, true);
    let gpu_up = half_slice(gpu, false);
    let cpu_up = half_slice(cpu, false);

    let mut col_mod32 = [0.0f32; 32];
    let mut col_mod32_count = [0u32; 32];
    for row in 0..slots {
        for col in 0..n {
            let i = row * n + col;
            let err = (gpu[i] - cpu[i]).abs();
            let m = col % 32;
            col_mod32[m] += err;
            col_mod32_count[m] += 1;
        }
    }
    for m in 0..32 {
        if col_mod32_count[m] > 0 {
            col_mod32[m] /= col_mod32_count[m] as f32;
        }
    }

    let mut slot_cos: Vec<(usize, f32)> = (0..slots)
        .map(|row| {
            let base = row * n;
            let c = cosine_f32(&gpu[base..base + n], &cpu[base..base + n]);
            (row, c)
        })
        .collect();
    slot_cos.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let worst_slots: Vec<(usize, f32)> = slot_cos.into_iter().take(8).collect();

    let mut dot_gc = 0.0f64;
    let mut dot_cc = 0.0f64;
    for (&g, &c) in gpu.iter().zip(cpu.iter()) {
        dot_gc += g as f64 * c as f64;
        dot_cc += c as f64 * c as f64;
    }
    let scale_ratio = if dot_cc > 0.0 {
        (dot_gc / dot_cc) as f32
    } else {
        0.0
    };

    GateUpDiffAnalysis {
        cos_gate_half: cosine_f32(&gpu_gate, &cpu_gate),
        cos_up_half: cosine_f32(&gpu_up, &cpu_up),
        rel_l2_gate_half: rel_l2_f32(&cpu_gate, &gpu_gate),
        rel_l2_up_half: rel_l2_f32(&cpu_up, &gpu_up),
        col_mod32_mean_abs_err: col_mod32,
        worst_slots,
        scale_ratio,
    }
}

pub fn print_gate_up_diff(analysis: &GateUpDiffAnalysis) {
    eprintln!("  gate_up element-wise structure (GPU vs CPU oracle):");
    eprintln!(
        "    gate cols [0..{}):   cos={:.6} rel_l2={:.6}",
        MOE_FF, analysis.cos_gate_half, analysis.rel_l2_gate_half,
    );
    eprintln!(
        "    up cols [{}..{}):     cos={:.6} rel_l2={:.6}",
        MOE_FF,
        MOE_FF * 2,
        analysis.cos_up_half,
        analysis.rel_l2_up_half,
    );
    eprintln!(
        "    scale_ratio dot(g,c)/dot(c,c)={:.6}",
        analysis.scale_ratio
    );
    let mut peaks: Vec<(usize, f32)> = analysis
        .col_mod32_mean_abs_err
        .iter()
        .enumerate()
        .map(|(m, &e)| (m, e))
        .collect();
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!(
        "    col%32 mean|err| top4: {:?}",
        &peaks[..4]
            .iter()
            .map(|(m, e)| format!("{m}:{e:.4}"))
            .collect::<Vec<_>>()
    );
    let min_m = peaks[31].1;
    let max_m = peaks[0].1;
    if max_m > 0.0 && min_m > 0.0 && max_m / min_m > 2.0 {
        eprintln!(
            "    col%32 spread {:.2}x — possible N-tile boundary pattern",
            max_m / min_m
        );
    }
    eprintln!(
        "    worst slots (idx,cos): {:?}",
        analysis
            .worst_slots
            .iter()
            .map(|(i, c)| format!("{i}:{c:.3}"))
            .collect::<Vec<_>>()
    );
}

fn rel_l2_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut diff2 = 0.0f64;
    let mut na = 0.0f64;
    for i in 0..n {
        let d = b[i] as f64 - a[i] as f64;
        diff2 += d * d;
        na += a[i] as f64 * a[i] as f64;
    }
    if na > 0.0 {
        (diff2.sqrt() / na.sqrt()) as f32
    } else {
        0.0
    }
}

/// Truncated slot count for committed routing fixtures (full canvas routing shape).
pub const PIN_SLOTS: usize = 64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeBatchedPinRoute {
    pub num_slots: u32,
    pub token_list: Vec<u32>,
    pub slot_list: Vec<u32>,
    pub row_start: Vec<u32>,
}

impl MoeBatchedPinRoute {
    pub fn from_route(route: &RouteScratch) -> Self {
        let slots = route.num_slots as usize;
        Self {
            num_slots: route.num_slots,
            token_list: route.token_list[..slots].to_vec(),
            slot_list: route.slot_list[..slots].to_vec(),
            row_start: route.row_start[..=N_EXPERTS].to_vec(),
        }
    }

    pub fn pin_route(&self) -> Self {
        let n = PIN_SLOTS.min(self.token_list.len());
        Self {
            num_slots: self.num_slots,
            token_list: self.token_list[..n].to_vec(),
            slot_list: self.slot_list[..n].to_vec(),
            row_start: self.row_start.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeBatchedPinLayout {
    /// Bytes: gathered A rows live at `gemm_b[0..moe_w_off_gu)`.
    pub moe_w_off_gu_bytes: usize,
    /// Gate/up GEMM output `[num_slots, moe_ff*2]` f32 at `gemm_b[moe_w_off_gu..]`.
    pub gate_up_elems: usize,
    /// Post-swiglu activations `[num_slots, moe_ff]` f32 at `gemm_a[0..]`.
    pub swiglu_elems: usize,
    pub hidden: usize,
    pub moe_ff: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeBatchedPinStageCos {
    pub gather: f32,
    /// Raw gate_up GEMM: `gemm_b@gu_off` vs CPU `gemm_block_grouped` (before swiglu).
    pub gate_up_gemm: f32,
    /// Post-swiglu: `gemm_a` vs CPU swiglu(CPU gate_up GEMM).
    pub swiglu_post: f32,
    /// Swiglu isolated: `gemm_a` vs CPU swiglu(GPU gate_up GEMM readback).
    pub swiglu_isolated: f32,
    pub down: f32,
    pub scatter: f32,
    /// CPU scatter(GPU down) vs GPU scatter — isolates scatter kernel from down GEMM drift.
    pub scatter_isolated: f32,
    pub pipeline: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeBatchedPinDump {
    pub schema_version: u32,
    pub prompt: String,
    pub seed: u64,
    pub layer: usize,
    pub kv_len: u32,
    pub format: String,
    pub layout: MoeBatchedPinLayout,
    pub route: MoeBatchedPinRoute,
    pub stages: MoeBatchedPinStageCos,
    pub rel_l2: MoeBatchedPinStageCos,
    /// Element-wise gate_up GEMM diff (GPU vs CPU oracle).
    pub gate_up_diff: GateUpDiffAnalysis,
    /// `gate_up_gemm` | `swiglu_post` | `swiglu_isolated` | `down` | `scatter` | `none`
    pub first_divergent_stage: String,
}

pub fn cpu_gather_slots(moe_in: &[f32], token_list: &[u32], hidden: usize) -> Vec<f32> {
    let slots = token_list.len();
    let mut out = vec![0.0f32; slots * hidden];
    for (slot, &tok) in token_list.iter().enumerate() {
        let src = tok as usize * hidden;
        let dst = slot * hidden;
        out[dst..dst + hidden].copy_from_slice(&moe_in[src..src + hidden]);
    }
    out
}

pub fn cpu_swiglu_gate_up(gate_up: &[f32], slots: usize, moe_ff: usize) -> Vec<f32> {
    let f = InterleavedFixture {
        gate_up: gate_up.to_vec(),
        batch_size: slots,
        moe_inter: moe_ff,
    };
    crate::shaders::swiglu::cpu_interleaved(&f)
}

pub fn verify_batched_stages_cpu(
    moe_in: &[f32],
    route: &RouteScratch,
    blob: &[u8],
    layer_off: &LayerOffsets,
    format: QuantFormat,
    gpu_gather: &[f32],
    gpu_gate_up: &[f32],
    gpu_swiglu: &[f32],
    gpu_down: &[f32],
    gpu_scatter: &[f32],
) -> (
    MoeBatchedPinStageCos,
    MoeBatchedPinStageCos,
    GateUpDiffAnalysis,
) {
    let dims = crate::metal::ModelDims::reference();
    let hidden = dims.hid;
    let moe_ff = dims.moe_ff;
    let slots = route.num_slots as usize;
    let token_list = &route.token_list[..slots];

    let gather_cpu = cpu_gather_slots(moe_in, token_list, hidden);
    let (gate_jobs, down_jobs) = layer_moe_block_jobs(layer_off, format, &dims);
    let row_start = &route.row_start[..=N_EXPERTS];

    let gate_up_cpu = gemm_linear_grouped_cpu(
        &gather_cpu,
        slots,
        hidden,
        moe_ff * 2,
        blob,
        &gate_jobs,
        row_start,
        format,
    );
    let swiglu_cpu = cpu_swiglu_gate_up(&gate_up_cpu, slots, moe_ff);
    let swiglu_from_gpu_gate = cpu_swiglu_gate_up(gpu_gate_up, slots, moe_ff);
    let down_cpu = gemm_linear_grouped_cpu(
        &swiglu_cpu,
        slots,
        moe_ff,
        hidden,
        blob,
        &down_jobs,
        row_start,
        format,
    );
    let scatter_cpu = moe_scatter_weighted(&down_cpu, route, hidden);
    let scatter_from_gpu_down = moe_scatter_weighted(gpu_down, route, hidden);

    let cos = |a: &[f32], b: &[f32]| cosine_f32(a, b);
    let rel = |a: &[f32], b: &[f32]| rel_l2_f32(a, b);

    let stages = MoeBatchedPinStageCos {
        gather: cos(&gather_cpu, gpu_gather),
        gate_up_gemm: cos(&gate_up_cpu, gpu_gate_up),
        swiglu_post: cos(&swiglu_cpu, gpu_swiglu),
        swiglu_isolated: cos(&swiglu_from_gpu_gate, gpu_swiglu),
        down: cos(&down_cpu, gpu_down),
        scatter: cos(&scatter_cpu, gpu_scatter),
        scatter_isolated: cos(&scatter_from_gpu_down, gpu_scatter),
        pipeline: cos(&scatter_cpu, gpu_scatter),
    };
    let rel_l2 = MoeBatchedPinStageCos {
        gather: rel(&gather_cpu, gpu_gather),
        gate_up_gemm: rel(&gate_up_cpu, gpu_gate_up),
        swiglu_post: rel(&swiglu_cpu, gpu_swiglu),
        swiglu_isolated: rel(&swiglu_from_gpu_gate, gpu_swiglu),
        down: rel(&down_cpu, gpu_down),
        scatter: rel(&scatter_cpu, gpu_scatter),
        scatter_isolated: rel(&scatter_from_gpu_down, gpu_scatter),
        pipeline: rel(&scatter_cpu, gpu_scatter),
    };
    let gate_up_diff = analyze_gate_up_diff(gpu_gate_up, &gate_up_cpu, slots, moe_ff * 2);
    (stages, rel_l2, gate_up_diff)
}

fn first_divergent_stage(stages: &MoeBatchedPinStageCos) -> &'static str {
    const THRESH: f32 = 0.99;
    if stages.gather < THRESH {
        "gather"
    } else if stages.gate_up_gemm < THRESH {
        "gate_up_gemm"
    } else if stages.swiglu_post < THRESH {
        "swiglu_post"
    } else if stages.swiglu_isolated < THRESH {
        "swiglu_isolated"
    } else if stages.down < THRESH {
        "down"
    } else if stages.scatter < THRESH {
        "scatter"
    } else {
        "none"
    }
}

pub fn decisive_verdict(stages: &MoeBatchedPinStageCos) -> &'static str {
    const THRESH: f32 = 0.99;
    if stages.gate_up_gemm < THRESH {
        "BUG: gemm_block_grouped gate_up GEMM (gemm_b@gu_off vs CPU oracle)"
    } else if stages.swiglu_isolated < THRESH {
        "BUG: swiglu_moe_gate_up (GPU gate_up OK; post-swiglu gemm_a diverges from CPU∘GPU_gemm)"
    } else if stages.swiglu_post < THRESH {
        "BUG: swiglu chain (unlikely if isolated passed; check CPU gate_up oracle)"
    } else if stages.down < THRESH {
        "BUG: gemm_block_grouped down GEMM"
    } else if stages.scatter < THRESH {
        "BUG: moe_scatter_weighted"
    } else {
        "OK: batched pipeline stages match CPU oracle"
    }
}

pub fn verify_batched_stages_cpu_with_verdict(
    moe_in: &[f32],
    route: &RouteScratch,
    blob: &[u8],
    layer_off: &LayerOffsets,
    format: QuantFormat,
    gpu_gather: &[f32],
    gpu_gate_up: &[f32],
    gpu_swiglu: &[f32],
    gpu_down: &[f32],
    gpu_scatter: &[f32],
) -> (
    MoeBatchedPinStageCos,
    MoeBatchedPinStageCos,
    GateUpDiffAnalysis,
    String,
) {
    let (stages, rel_l2, gate_up_diff) = verify_batched_stages_cpu(
        moe_in,
        route,
        blob,
        layer_off,
        format,
        gpu_gather,
        gpu_gate_up,
        gpu_swiglu,
        gpu_down,
        gpu_scatter,
    );
    let first = first_divergent_stage(&stages).to_string();
    (stages, rel_l2, gate_up_diff, first)
}

pub fn print_pin_summary(dump: &MoeBatchedPinDump) {
    let lay = &dump.layout;
    eprintln!(
        "moe-batched-pin: layer={} kv_len={} num_slots={} format={}",
        dump.layer, dump.kv_len, dump.route.num_slots, dump.format
    );
    eprintln!(
        "  gemm_b layout: A[0..{}B] gathered slots×hid | gate_up@{}B [{} elems = slots×2×moe_ff]",
        lay.moe_w_off_gu_bytes, lay.moe_w_off_gu_bytes, lay.gate_up_elems,
    );
    eprintln!(
        "  gemm_a layout: swiglu_out[0..{} elems = slots×moe_ff] (hid={}, moe_ff={})",
        lay.swiglu_elems, lay.hidden, lay.moe_ff,
    );
    let s = &dump.stages;
    eprintln!("  DECISIVE gate_up / swiglu split:");
    eprintln!(
        "    gate_up GEMM  gemm_b@gu_off vs CPU gemm_block_grouped: cos={:.6} rel_l2={:.6}",
        s.gate_up_gemm, dump.rel_l2.gate_up_gemm,
    );
    print_gate_up_diff(&dump.gate_up_diff);
    eprintln!(
        "    swiglu post   gemm_a vs CPU swiglu(CPU gemm):         cos={:.6} rel_l2={:.6}",
        s.swiglu_post, dump.rel_l2.swiglu_post,
    );
    eprintln!(
        "    swiglu isol.  gemm_a vs CPU swiglu(GPU gemm readback): cos={:.6} rel_l2={:.6}",
        s.swiglu_isolated, dump.rel_l2.swiglu_isolated,
    );
    eprintln!(
        "  other cos  gather={:.6} down={:.6} scatter={:.6} scatter_isolated={:.6}",
        s.gather, s.down, s.scatter, s.scatter_isolated,
    );
    eprintln!("  VERDICT: {}", decisive_verdict(s));
    if dump.first_divergent_stage != "none" {
        eprintln!("  first_divergent_stage={}", dump.first_divergent_stage);
    }
}

/// Real L0 token gather order from Calgary capture (seed 42, /tmp/quantized-weights).
pub fn calgary_l0_token_list() -> [u32; PIN_SLOTS] {
    [
        205, 67, 194, 13, 64, 128, 184, 38, 105, 223, 71, 66, 39, 106, 148, 104, 151, 166, 145,
        185, 101, 0, 146, 192, 102, 167, 103, 236, 96, 196, 241, 6, 186, 100, 136, 8, 97, 99, 243,
        159, 202, 91, 161, 139, 28, 89, 163, 109, 11, 228, 176, 112, 76, 20, 21, 156, 78, 35, 217,
        154, 79, 116, 60, 72,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::CANVAS;
    use crate::shaders::gather_rows::Fixture as GatherFixture;

    #[test]
    fn pin_gather_routing_matches_cpu_oracle() {
        let hidden = 64usize;
        let num_tokens = CANVAS;
        let token_list = calgary_l0_token_list();
        let src: Vec<f32> = (0..num_tokens * hidden)
            .map(|i| ((i as f32) * 0.0023).sin() * 0.5 + 0.25)
            .collect();
        let gather = GatherFixture {
            src,
            indices: token_list.to_vec(),
            hidden,
            num_tokens,
        };
        let cpu = crate::shaders::gather_rows::cpu(&gather);
        let pin = cpu_gather_slots(&gather.src, &token_list, hidden);
        assert_eq!(cpu, pin);
    }
}
