//! Tier-1 pin: batched MoE pipeline stages with real routed jobs from denoise capture.

use crate::kernels::cpu::gemm_linear_grouped::gemm_linear_grouped_cpu;
use crate::kernels::cpu::moe_scatter_weighted::moe_scatter_weighted;
use crate::kernels::sub::swiglu::InterleavedFixture;
use crate::kernels::sub::QuantFormat;
use crate::metal::{layer_moe_block_jobs, BlockGroupedJob, LayerOffsets, RouteScratch, CANVAS, HID, MOE_FF, N_EXPERTS};
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
pub struct MoeBatchedPinStageCos {
    pub gather: f32,
    pub gate_up: f32,
    pub swiglu: f32,
    pub down: f32,
    pub scatter: f32,
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
    pub route: MoeBatchedPinRoute,
    pub stages: MoeBatchedPinStageCos,
    pub rel_l2: MoeBatchedPinStageCos,
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
    crate::kernels::sub::swiglu::cpu_interleaved(&f)
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
) -> (MoeBatchedPinStageCos, MoeBatchedPinStageCos) {
    let hidden = HID;
    let moe_ff = MOE_FF as usize;
    let slots = route.num_slots as usize;
    let token_list = &route.token_list[..slots];

    let gather_cpu = cpu_gather_slots(moe_in, token_list, hidden);
    let (gate_jobs, down_jobs) = layer_moe_block_jobs(layer_off, format);
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

    let cos = |a: &[f32], b: &[f32]| cosine_f32(a, b);
    let rel = |a: &[f32], b: &[f32]| rel_l2_f32(a, b);

    let stages = MoeBatchedPinStageCos {
        gather: cos(&gather_cpu, gpu_gather),
        gate_up: cos(&gate_up_cpu, gpu_gate_up),
        swiglu: cos(&swiglu_cpu, gpu_swiglu),
        down: cos(&down_cpu, gpu_down),
        scatter: cos(&scatter_cpu, gpu_scatter),
        pipeline: cos(&scatter_cpu, gpu_scatter),
    };
    let rel_l2 = MoeBatchedPinStageCos {
        gather: rel(&gather_cpu, gpu_gather),
        gate_up: rel(&gate_up_cpu, gpu_gate_up),
        swiglu: rel(&swiglu_cpu, gpu_swiglu),
        down: rel(&down_cpu, gpu_down),
        scatter: rel(&scatter_cpu, gpu_scatter),
        pipeline: rel(&scatter_cpu, gpu_scatter),
    };
    (stages, rel_l2)
}

pub fn print_pin_summary(dump: &MoeBatchedPinDump) {
    eprintln!(
        "moe-batched-pin: layer={} kv_len={} num_slots={} format={}",
        dump.layer, dump.kv_len, dump.route.num_slots, dump.format
    );
    let s = &dump.stages;
    eprintln!(
        "  cos  gather={:.6} gate_up={:.6} swiglu={:.6} down={:.6} scatter={:.6}",
        s.gather, s.gate_up, s.swiglu, s.down, s.scatter
    );
    let r = &dump.rel_l2;
    eprintln!(
        "  rel_l2 gather={:.6} gate_up={:.6} swiglu={:.6} down={:.6} scatter={:.6}",
        r.gather, r.gate_up, r.swiglu, r.down, r.scatter
    );
    for (name, cos) in [
        ("gather", s.gather),
        ("gate_up", s.gate_up),
        ("swiglu", s.swiglu),
        ("down", s.down),
        ("scatter", s.scatter),
    ] {
        if cos < 0.99 {
            eprintln!("  FAIL: first divergent stage likely `{name}` (cos={cos:.6})");
            return;
        }
    }
    eprintln!("  OK: all batched stages >= 0.99 cos vs CPU oracle");
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
    use crate::kernels::sub::gather_rows::Fixture as GatherFixture;

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
        let cpu = crate::kernels::sub::gather_rows::cpu(&gather);
        let pin = cpu_gather_slots(&gather.src, &token_list, hidden);
        assert_eq!(cpu, pin);
    }
}
