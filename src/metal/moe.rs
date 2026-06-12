use crate::config::TextConfig;
use crate::metal::batched_kernels::{self as bk};
use crate::metal::kernels::GpuKernels;
use crate::metal::batch::GpuBatch;
use crate::metal::device::MetalContext;
use crate::metal::weights::GpuDecoderWeightCache;
use crate::model::layer_weights::DecoderLayerWeights;
use crate::model::moe::RouteResult;
use crate::safetensors::Error;

struct ExpertJob {
    expert: usize,
    tokens: Vec<(usize, f32)>,
    gate_up_off: usize,
    gate_act_off: usize,
    out_off: usize,
}

pub fn experts_forward_gpu_batched(
    out: &mut [f32],
    residual: &[f32],
    pre_ff_norm_2: &[f32],
    eps: f32,
    weights: &DecoderLayerWeights<'_>,
    expert_cache: &GpuDecoderWeightCache,
    layer: usize,
    cfg: &TextConfig,
    seq_len: usize,
    routes: &[RouteResult],
    token_indices: &mut Vec<u32>,
    out_arena: &mut Vec<f32>,
    ctx: &MetalContext,
    pool: &mut crate::metal::buffer::BufferPool,
    kernels: &GpuKernels,
    gemm_pipeline: &crate::metal::device::ComputePipeline,
) -> Result<(), Error> {
    let hidden = cfg.hidden_size;
    let moe_inter = cfg.moe_intermediate_size;
    let experts = cfg.num_experts;
    let gate_up = weights.experts_gate_up.bf16()?;
    let down = weights.experts_down.bf16()?;

    out.fill(0.0);

    let mut buckets: Vec<Vec<(usize, f32)>> = vec![Vec::new(); experts];
    for (s, route) in routes.iter().enumerate() {
        for (&expert, &weight) in route.indices.iter().zip(route.weights.iter()) {
            buckets[expert].push((s, weight));
        }
    }

    let mut jobs: Vec<ExpertJob> = Vec::new();
    let mut gate_up_len = 0usize;
    let mut gate_act_len = 0usize;
    let mut out_len = 0usize;
    for expert in 0..experts {
        let tokens = &buckets[expert];
        if tokens.is_empty() {
            continue;
        }
        let batch_size = tokens.len();
        jobs.push(ExpertJob {
            expert,
            tokens: tokens.clone(),
            gate_up_off: gate_up_len,
            gate_act_off: gate_act_len,
            out_off: out_len,
        });
        gate_up_len += batch_size * moe_inter * 2;
        gate_act_len += batch_size * moe_inter;
        out_len += batch_size * hidden;
    }

    if jobs.is_empty() {
        return Ok(());
    }

    let _ = (gate_up_len, gate_act_len);
    out_arena.resize(out_len, 0.0);

    let pipeline = &gemm_pipeline.pipeline;

    for job in &jobs {
        expert_cache.prefetch_expert(
            layer,
            gate_up,
            down,
            job.expert,
            &ctx.device,
            pool,
        );
    }

    // One sync: pre_ff norm → gather → gate_up GEMM → gelu/swiglu → down GEMM per expert job.
    {
        let mut batch = GpuBatch::begin(&ctx.queue, pool, &ctx.device)?;
        let buf_res = batch.alloc_f32(residual)?;
        let buf_moe_in = bk::rms_norm_rows_gpu_buf(
            &mut batch,
            kernels,
            &buf_res,
            pre_ff_norm_2,
            seq_len,
            hidden,
            eps,
        )?;
        for job in &jobs {
            let batch_size = job.tokens.len();
            token_indices.clear();
            token_indices.extend(job.tokens.iter().map(|&(tok, _)| tok as u32));
            let buf_a = bk::gather_rows_gpu(
                &mut batch,
                kernels,
                &buf_moe_in,
                token_indices,
                hidden,
            )?;
            let act_len = batch_size * moe_inter;
            let w_gate = expert_cache.expert_gate_up_buf(layer, job.expert);
            let buf_gu = bk::f32_bf16_linear_gpu_bufs(
                &mut batch,
                pipeline,
                &buf_a,
                &w_gate,
                batch_size,
                hidden,
                moe_inter * 2,
            )?;
            let buf_act = bk::gelu_swiglu_gate_up_gpu(
                &mut batch,
                kernels,
                &buf_gu,
                act_len,
                batch_size,
                moe_inter,
            )?;
            let w_down = expert_cache.expert_down_buf(layer, job.expert);
            let out_slice = &mut out_arena[job.out_off..job.out_off + batch_size * hidden];
            let buf_out = bk::f32_bf16_linear_gpu_bufs(
                &mut batch,
                pipeline,
                &buf_act,
                &w_down,
                batch_size,
                moe_inter,
                hidden,
            )?;
            batch.register_read(buf_out, out_slice);
        }
        batch.end()?;
    }

    for job in &jobs {
        for (bi, &(_, weight)) in job.tokens.iter().enumerate() {
            let src = job.out_off + bi * hidden;
            let dst = job.tokens[bi].0 * hidden;
            for i in 0..hidden {
                out[dst + i] += weight * out_arena[src + i];
            }
        }
    }
    Ok(())
}
