use crate::config::TextConfig;
use crate::kernels::cpu::{gelu_pytorch_tanh, softmax_rows};
use crate::metal::device::{ComputePipeline, MetalContext};
use crate::metal::gemm::Bf16Gemm;
use crate::metal::kernels::GpuKernels;
use crate::metal::linear::{linear_bf16, transpose_weight_bf16};
use crate::model::layer_weights::DecoderLayerWeights;
use crate::model::moe::RouteResult;
use crate::safetensors::Error;
use crate::tensor::Bf16Slice;

pub struct GpuMoeScratch {
    pub router_input: Vec<f32>,
    pub router_logits: Vec<f32>,
    pub gate_up: Vec<f32>,
    pub gate_act: Vec<f32>,
    pub expert_out: Vec<f32>,
    pub batch_input: Vec<f32>,
    pub batch_gate_up: Vec<f32>,
    pub batch_gate_act: Vec<f32>,
    pub batch_out: Vec<f32>,
}

impl GpuMoeScratch {
    pub fn new(seq_len: usize, cfg: &TextConfig) -> Self {
        let hidden = cfg.hidden_size;
        let experts = cfg.num_experts;
        let moe_inter = cfg.moe_intermediate_size;
        Self {
            router_input: vec![0.0; seq_len * hidden],
            router_logits: vec![0.0; seq_len * experts],
            gate_up: vec![0.0; moe_inter * 2],
            gate_act: vec![0.0; moe_inter],
            expert_out: vec![0.0; hidden],
            batch_input: Vec::new(),
            batch_gate_up: Vec::new(),
            batch_gate_act: Vec::new(),
            batch_out: Vec::new(),
        }
    }
}

pub fn route_gpu(
    residual: &[f32],
    weights: &DecoderLayerWeights<'_>,
    cfg: &TextConfig,
    seq_len: usize,
    scratch: &mut GpuMoeScratch,
    ctx: &MetalContext,
    pool: &mut crate::metal::buffer::BufferPool,
    kernels: &GpuKernels,
    gemm: &mut Bf16Gemm,
) -> Result<Vec<RouteResult>, Error> {
    let hidden = cfg.hidden_size;
    let experts = cfg.num_experts;
    let top_k = cfg.top_k_experts;
    let eps = cfg.rms_norm_eps as f32;
    let root = (hidden as f32).powf(-0.5);

    kernels.rms_norm_rows_no_scale(
        &ctx.queue,
        pool,
        &ctx.device,
        &mut scratch.router_input,
        residual,
        seq_len,
        hidden,
        eps,
    )?;
    let router_scale = weights.router_scale.bf16()?.to_f32_vec();
    kernels.router_scale_rows(
        &ctx.queue,
        pool,
        &ctx.device,
        &mut scratch.router_input,
        &router_scale,
        seq_len,
        hidden,
        root,
    )?;

    linear_bf16(
        gemm,
        &mut scratch.router_logits,
        &scratch.router_input,
        weights.router_proj.bf16()?,
        seq_len,
        hidden,
        experts,
    )?;
    softmax_rows(&mut scratch.router_logits, seq_len, experts);

    let per_expert_scale = weights.router_per_expert_scale.bf16()?.to_f32_vec();
    let mut routes = Vec::with_capacity(seq_len);
    for s in 0..seq_len {
        let row = &scratch.router_logits[s * experts..(s + 1) * experts];
        routes.push(crate::model::moe::top_k_route(row, top_k, &per_expert_scale));
    }
    Ok(routes)
}

pub fn experts_forward_gpu(
    out: &mut [f32],
    expert_input: &[f32],
    weights: &DecoderLayerWeights<'_>,
    cfg: &TextConfig,
    seq_len: usize,
    routes: &[RouteResult],
    scratch: &mut GpuMoeScratch,
    ctx: &MetalContext,
    pool: &mut crate::metal::buffer::BufferPool,
    _kernels: &GpuKernels,
    gemm_pipeline: &ComputePipeline,
) -> Result<(), Error> {
    let hidden = cfg.hidden_size;
    let moe_inter = cfg.moe_intermediate_size;
    let experts = cfg.num_experts;
    let gate_up = weights.experts_gate_up.bf16()?;
    let down = weights.experts_down.bf16()?;
    let gate_up_stride = moe_inter * 2 * hidden;
    let down_stride = hidden * moe_inter;

    out.fill(0.0);

    let mut buckets: Vec<Vec<(usize, f32)>> = vec![Vec::new(); experts];
    for (s, route) in routes.iter().enumerate() {
        for (&expert, &weight) in route.indices.iter().zip(route.weights.iter()) {
            buckets[expert].push((s, weight));
        }
    }

    for expert in 0..experts {
        let tokens = &buckets[expert];
        if tokens.is_empty() {
            continue;
        }
        let batch = tokens.len();
        scratch.batch_input.resize(batch * hidden, 0.0);
        scratch.batch_gate_up.resize(batch * moe_inter * 2, 0.0);
        scratch.batch_gate_act.resize(batch * moe_inter, 0.0);
        scratch.batch_out.resize(batch * hidden, 0.0);

        for (bi, &(tok, _)) in tokens.iter().enumerate() {
            let src = tok * hidden;
            let dst = bi * hidden;
            scratch.batch_input[dst..dst + hidden]
                .copy_from_slice(&expert_input[src..src + hidden]);
        }

        expert_gate_up_batch(
            &mut scratch.batch_gate_up,
            &scratch.batch_input,
            gate_up,
            expert,
            batch,
            hidden,
            moe_inter,
            gate_up_stride,
            ctx,
            pool,
            gemm_pipeline,
        )?;

        for row in 0..batch {
            let off = row * moe_inter * 2;
            let (gate, up) = scratch.batch_gate_up[off..off + moe_inter * 2].split_at_mut(moe_inter);
            gelu_pytorch_tanh(gate);
            let act_off = row * moe_inter;
            for i in 0..moe_inter {
                scratch.batch_gate_act[act_off + i] = gate[i] * up[i];
            }
        }

        expert_down_batch(
            &mut scratch.batch_out,
            &scratch.batch_gate_act,
            down,
            expert,
            batch,
            hidden,
            moe_inter,
            down_stride,
            ctx,
            pool,
            gemm_pipeline,
        )?;

        for (bi, &(_, weight)) in tokens.iter().enumerate() {
            let src = bi * hidden;
            let dst = tokens[bi].0 * hidden;
            for i in 0..hidden {
                out[dst + i] += weight * scratch.batch_out[src + i];
            }
        }
    }
    Ok(())
}

fn expert_gate_up_batch(
    out: &mut [f32],
    x: &[f32],
    gate_up: Bf16Slice<'_>,
    expert: usize,
    batch: usize,
    hidden: usize,
    moe_inter: usize,
    gate_up_stride: usize,
    ctx: &MetalContext,
    pool: &mut crate::metal::buffer::BufferPool,
    gemm_pipeline: &ComputePipeline,
) -> Result<(), Error> {
    let out_dim = moe_inter * 2;
    let w_off = expert * gate_up_stride;
    let w_raw: Vec<u16> = (0..out_dim * hidden)
        .map(|i| gate_up.get(w_off + i))
        .collect();
    let w_t = transpose_weight_bf16(&w_raw, out_dim, hidden);
    Bf16Gemm::matmul_f32_bf16_shared(
        ctx,
        pool,
        gemm_pipeline,
        x,
        &w_t,
        out,
        batch,
        hidden,
        out_dim,
    )
}

fn expert_down_batch(
    out: &mut [f32],
    gate_act_rows: &[f32],
    down: Bf16Slice<'_>,
    expert: usize,
    batch: usize,
    hidden: usize,
    moe_inter: usize,
    down_stride: usize,
    ctx: &MetalContext,
    pool: &mut crate::metal::buffer::BufferPool,
    gemm_pipeline: &ComputePipeline,
) -> Result<(), Error> {
    let w_off = expert * down_stride;
    let w_raw: Vec<u16> = (0..hidden * moe_inter)
        .map(|i| down.get(w_off + i))
        .collect();
    let w_t = transpose_weight_bf16(&w_raw, hidden, moe_inter);
    Bf16Gemm::matmul_f32_bf16_shared(
        ctx,
        pool,
        gemm_pipeline,
        gate_act_rows,
        &w_t,
        out,
        batch,
        moe_inter,
        hidden,
    )
}
