//! CPU reference for monolithic MoE router + expert bucketing.

use crate::kernels::cpu::{linear, rms_norm_no_scale};
use crate::model::moe::top_k_route_from_raw_logits;

pub const RMS_EPS: f32 = 1e-6;

#[derive(Debug, Clone, Copy)]
pub struct RouterDims {
    pub canvas: u32,
    pub hidden: u32,
    pub n_experts: u32,
    pub top_k: u32,
    pub router_hscale: f32,
}

#[derive(Debug, Clone)]
pub struct RouteRow {
    pub experts: Vec<u32>,
    pub weights: Vec<f32>,
}

/// Monolithic `moe_router` per canvas row.
pub fn moe_router_rows(
    stream: &[f32],
    router_scale: &[f32],
    router_proj: &[f32],
    per_expert_scale: &[f32],
    dims: RouterDims,
) -> Vec<RouteRow> {
    let canvas = dims.canvas as usize;
    let hidden = dims.hidden as usize;
    let experts = dims.n_experts as usize;
    let top_k = dims.top_k as usize;
    let mut normed = vec![0.0f32; canvas * hidden];
    for tok in 0..canvas {
        let off = tok * hidden;
        rms_norm_no_scale(
            &mut normed[off..off + hidden],
            &stream[off..off + hidden],
            RMS_EPS,
        );
        for d in 0..hidden {
            normed[off + d] *= router_scale[d] * dims.router_hscale;
        }
    }
    let mut logits = vec![0.0f32; canvas * experts];
    linear(
        &mut logits,
        &normed,
        router_proj,
        None,
        canvas,
        hidden,
        experts,
    );
    let mut out = Vec::with_capacity(canvas);
    for tok in 0..canvas {
        let row = &logits[tok * experts..(tok + 1) * experts];
        let route = top_k_route_from_raw_logits(row, top_k, per_expert_scale);
        out.push(RouteRow {
            experts: route.indices.iter().map(|&i| i as u32).collect(),
            weights: route.weights,
        });
    }
    out
}

/// Raw router logits for one canvas row (before top-k / softmax).
pub fn router_logits_row(
    stream_row: &[f32],
    router_scale: &[f32],
    router_proj: &[f32],
    hidden: usize,
    n_experts: usize,
) -> Vec<f32> {
    let hscale = (hidden as f32).powf(-0.5);
    let mut normed = vec![0.0f32; hidden];
    rms_norm_no_scale(&mut normed, stream_row, RMS_EPS);
    for d in 0..hidden {
        normed[d] *= router_scale[d] * hscale;
    }
    let mut logits = vec![0.0f32; n_experts];
    linear(
        &mut logits,
        &normed,
        router_proj,
        None,
        1,
        hidden,
        n_experts,
    );
    logits
}

#[derive(Debug, Clone)]
pub struct BucketState {
    pub count: Vec<u32>,
    pub offset: Vec<u32>,
    pub num_slots: u32,
    pub token_list: Vec<u32>,
    pub slot_list: Vec<u32>,
}

/// Run bucketing phases 0→1→2 on route assignments (CPU-serial atomics).
pub fn moe_bucket_phases(experts: &[Vec<u32>], n_experts: u32, top_k: u32) -> BucketState {
    let canvas = experts.len();
    let n_experts = n_experts as usize;
    let top_k = top_k as usize;
    let mut count = vec![0u32; n_experts];
    for tok in 0..canvas {
        for kk in 0..top_k {
            count[experts[tok][kk] as usize] += 1;
        }
    }
    let mut offset = vec![0u32; n_experts];
    let mut num_slots = 0u32;
    for e in 0..n_experts {
        offset[e] = num_slots;
        num_slots += count[e];
        count[e] = 0;
    }
    let slots = num_slots as usize;
    let mut token_list = vec![0u32; slots];
    let mut slot_list = vec![0u32; slots];
    for tok in 0..canvas {
        for kk in 0..top_k {
            let e = experts[tok][kk] as usize;
            let slot = offset[e] + count[e];
            count[e] += 1;
            token_list[slot as usize] = tok as u32;
            slot_list[slot as usize] = kk as u32;
        }
    }
    BucketState {
        count,
        offset,
        num_slots,
        token_list,
        slot_list,
    }
}

pub fn pack_route_rows(rows: &[RouteRow]) -> Vec<f32> {
    let mut out = Vec::new();
    for row in rows {
        for &e in &row.experts {
            out.push(e as f32);
        }
        out.extend_from_slice(&row.weights);
    }
    out
}

pub fn pack_bucket_state(state: &BucketState, n_experts: usize) -> Vec<f32> {
    let mut out: Vec<f32> = state.offset[..n_experts]
        .iter()
        .chain(std::iter::once(&state.num_slots))
        .map(|&v| v as f32)
        .collect();
    out.extend(state.token_list.iter().map(|&v| v as f32));
    out.extend(state.slot_list.iter().map(|&v| v as f32));
    out
}
