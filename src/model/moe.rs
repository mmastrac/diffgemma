use crate::config::TextConfig;
use crate::kernels::cpu::{
    gelu_pytorch_tanh, linear_bf16_slice, rms_norm_no_scale, rms_norm_rows,
};
use crate::model::layer_weights::DecoderLayerWeights;
use crate::safetensors::Error;
use crate::tensor::Bf16Slice;

pub struct RouteResult {
    pub weights: Vec<f32>,
    pub indices: Vec<usize>,
}

pub struct MoeScratch {
    pub router_input: Vec<f32>,
    pub router_logits: Vec<f32>,
    pub gate_act: Vec<f32>,
    pub gate_up: Vec<f32>,
    pub expert_out: Vec<f32>,
    pub router_proj_w: Vec<f32>,
    pub router_scale: Vec<f32>,
    pub per_expert_scale: Vec<f32>,
}

impl MoeScratch {
    pub fn new(seq_len: usize, cfg: &TextConfig) -> Self {
        let hidden = cfg.hidden_size;
        let experts = cfg.num_experts;
        let moe_inter = cfg.moe_intermediate_size;
        Self {
            router_input: vec![0.0; seq_len * hidden],
            router_logits: vec![0.0; seq_len * experts],
            gate_act: vec![0.0; moe_inter],
            gate_up: vec![0.0; moe_inter * 2],
            expert_out: vec![0.0; hidden],
            router_proj_w: Vec::new(),
            router_scale: Vec::new(),
            per_expert_scale: Vec::new(),
        }
    }

    fn load_router_weights(&mut self, weights: &DecoderLayerWeights<'_>) -> Result<(), Error> {
        self.router_proj_w = weights.router_proj.bf16()?.to_f32_vec();
        self.router_scale = weights.router_scale.bf16()?.to_f32_vec();
        self.per_expert_scale = weights.router_per_expert_scale.bf16()?.to_f32_vec();
        Ok(())
    }
}

/// Gemma4 router: RMSNorm(no scale) → scale * hidden^-0.5 → linear → top-k → softmax(top-k).
pub fn route(
    residual: &[f32],
    weights: &DecoderLayerWeights<'_>,
    cfg: &TextConfig,
    seq_len: usize,
    scratch: &mut MoeScratch,
) -> Result<Vec<RouteResult>, Error> {
    let hidden = cfg.hidden_size;
    let experts = cfg.num_experts;
    let top_k = cfg.top_k_experts;
    let eps = cfg.rms_norm_eps as f32;
    let root = (hidden as f32).powf(-0.5);

    assert_eq!(residual.len(), seq_len * hidden);
    scratch.load_router_weights(weights)?;

    for s in 0..seq_len {
        let off = s * hidden;
        rms_norm_no_scale(
            &mut scratch.router_input[off..off + hidden],
            &residual[off..off + hidden],
            eps,
        );
        for i in 0..hidden {
            scratch.router_input[off + i] *= scratch.router_scale[i] * root;
        }
    }

    crate::kernels::cpu::linear(
        &mut scratch.router_logits,
        &scratch.router_input,
        &scratch.router_proj_w,
        None,
        seq_len,
        hidden,
        experts,
    );
    let mut routes = Vec::with_capacity(seq_len);
    for s in 0..seq_len {
        let row = &scratch.router_logits[s * experts..(s + 1) * experts];
        routes.push(top_k_route_from_raw_logits(
            row,
            top_k,
            &scratch.per_expert_scale,
        ));
    }
    Ok(routes)
}

/// Router using pre-cached f32 weights (no per-forward dequant).
pub fn route_with_cached_weights(
    residual: &[f32],
    router_proj: &[f32],
    router_scale: &[f32],
    per_expert_scale: &[f32],
    cfg: &TextConfig,
    seq_len: usize,
    scratch: &mut MoeScratch,
) -> Result<Vec<RouteResult>, Error> {
    let hidden = cfg.hidden_size;
    let experts = cfg.num_experts;
    let top_k = cfg.top_k_experts;
    let eps = cfg.rms_norm_eps as f32;
    let root = (hidden as f32).powf(-0.5);

    assert_eq!(residual.len(), seq_len * hidden);

    for s in 0..seq_len {
        let off = s * hidden;
        rms_norm_no_scale(
            &mut scratch.router_input[off..off + hidden],
            &residual[off..off + hidden],
            eps,
        );
        for i in 0..hidden {
            scratch.router_input[off + i] *= router_scale[i] * root;
        }
    }

    crate::kernels::cpu::linear(
        &mut scratch.router_logits,
        &scratch.router_input,
        router_proj,
        None,
        seq_len,
        hidden,
        experts,
    );
    let mut routes = Vec::with_capacity(seq_len);
    for s in 0..seq_len {
        let row = &scratch.router_logits[s * experts..(s + 1) * experts];
        routes.push(top_k_route_from_raw_logits(row, top_k, per_expert_scale));
    }
    Ok(routes)
}

/// MLX/Gemma4: argpartition top-k on raw logits, softmax only over selected experts.
pub fn top_k_route_from_raw_logits(
    logits: &[f32],
    k: usize,
    per_expert_scale: &[f32],
) -> RouteResult {
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|(ia, pa), (ib, pb)| {
        pb.partial_cmp(pa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(ia.cmp(ib))
    });
    let top = &ranked[..k];
    let indices: Vec<usize> = top.iter().map(|(i, _)| *i).collect();
    let raw: Vec<f32> = top.iter().map(|(_, s)| *s).collect();
    let mx = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = raw.iter().map(|&x| (x - mx).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut weights: Vec<f32> = if sum > 0.0 {
        exps.iter().map(|e| e / sum).collect()
    } else {
        vec![1.0 / k as f32; k]
    };
    for (w, &idx) in weights.iter_mut().zip(indices.iter()) {
        *w *= per_expert_scale[idx];
    }
    RouteResult { weights, indices }
}

pub fn experts_forward(
    out: &mut [f32],
    expert_input: &[f32],
    weights: &DecoderLayerWeights<'_>,
    cfg: &TextConfig,
    seq_len: usize,
    routes: &[RouteResult],
    scratch: &mut MoeScratch,
) -> Result<(), Error> {
    let hidden = cfg.hidden_size;
    let moe_inter = cfg.moe_intermediate_size;
    let gate_up = weights.experts_gate_up.bf16()?;
    let down = weights.experts_down.bf16()?;
    let gate_up_stride = moe_inter * 2 * hidden;
    let down_stride = hidden * moe_inter;

    assert_eq!(expert_input.len(), seq_len * hidden);
    assert_eq!(out.len(), seq_len * hidden);
    assert_eq!(routes.len(), seq_len);

    out.fill(0.0);
    for s in 0..seq_len {
        let x = &expert_input[s * hidden..(s + 1) * hidden];
        let route = &routes[s];
        let o = &mut out[s * hidden..(s + 1) * hidden];
        for (&expert, &weight) in route.indices.iter().zip(route.weights.iter()) {
            expert_forward_one(
                &mut scratch.expert_out,
                x,
                gate_up,
                down,
                expert,
                hidden,
                moe_inter,
                gate_up_stride,
                down_stride,
                &mut scratch.gate_up,
                &mut scratch.gate_act,
            );
            for i in 0..hidden {
                o[i] += weight * scratch.expert_out[i];
            }
        }
    }
    Ok(())
}

fn expert_forward_one(
    out: &mut [f32],
    x: &[f32],
    gate_up: Bf16Slice<'_>,
    down: Bf16Slice<'_>,
    expert: usize,
    hidden: usize,
    moe_inter: usize,
    gate_up_stride: usize,
    down_stride: usize,
    gate_up_buf: &mut [f32],
    gate_act: &mut [f32],
) {
    let gate_up_off = expert * gate_up_stride;
    linear_bf16_slice(
        gate_up_buf,
        x,
        gate_up,
        gate_up_off,
        1,
        hidden,
        moe_inter * 2,
    );
    let (gate, up) = gate_up_buf.split_at_mut(moe_inter);
    gelu_pytorch_tanh(gate);
    for i in 0..moe_inter {
        gate_act[i] = gate[i] * up[i];
    }
    let down_off = expert * down_stride;
    linear_bf16_slice(
        out,
        gate_act,
        down,
        down_off,
        1,
        moe_inter,
        hidden,
    );
}

/// `pre_feedforward_layernorm_2(residual)` — expert input after routing.
pub fn prepare_expert_input(
    out: &mut [f32],
    residual: &[f32],
    norm_w: &[f32],
    seq_len: usize,
    hidden: usize,
    eps: f32,
) {
    rms_norm_rows(out, residual, norm_w, seq_len, hidden, eps);
}
