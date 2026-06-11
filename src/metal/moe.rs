use crate::config::TextConfig;
use crate::kernels::cpu::gelu_pytorch_tanh;
use crate::metal::batched_kernels::f32_bf16_gemm;
use crate::metal::batch::GpuBatch;
use crate::metal::device::MetalContext;
use crate::metal::weights::GpuLayerWeightCache;
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
    expert_input: &[f32],
    weights: &DecoderLayerWeights<'_>,
    cached: &GpuLayerWeightCache,
    cfg: &TextConfig,
    _seq_len: usize,
    routes: &[RouteResult],
    batch_input: &mut Vec<f32>,
    gate_up_arena: &mut Vec<f32>,
    gate_act_arena: &mut Vec<f32>,
    out_arena: &mut Vec<f32>,
    ctx: &MetalContext,
    pool: &mut crate::metal::buffer::BufferPool,
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

    gate_up_arena.resize(gate_up_len, 0.0);
    gate_act_arena.resize(gate_act_len, 0.0);
    out_arena.resize(out_len, 0.0);

    let pipeline = &gemm_pipeline.pipeline;

    // Phase 1: all expert gate_up GEMMs in one GPU batch (one sync).
    {
        let mut batch = GpuBatch::begin(&ctx.queue, pool, &ctx.device)?;
        for job in &jobs {
            let batch_size = job.tokens.len();
            batch_input.resize(batch_size * hidden, 0.0);
            for (bi, &(tok, _)) in job.tokens.iter().enumerate() {
                let src = tok * hidden;
                let dst = bi * hidden;
                batch_input[dst..dst + hidden].copy_from_slice(&expert_input[src..src + hidden]);
            }
            let gate_slice = &mut gate_up_arena[job.gate_up_off..job.gate_up_off + batch_size * moe_inter * 2];
            cached.with_expert_gate_up_t(gate_up, job.expert, |w_t| {
                f32_bf16_gemm(
                    &mut batch,
                    pipeline,
                    gate_slice,
                    batch_input,
                    w_t,
                    batch_size,
                    hidden,
                    moe_inter * 2,
                )
            })?;
        }
        batch.end()?;
    }

    for job in &jobs {
        let batch_size = job.tokens.len();
        for row in 0..batch_size {
            let gu_off = job.gate_up_off + row * moe_inter * 2;
            let (gate, up) = gate_up_arena[gu_off..gu_off + moe_inter * 2].split_at_mut(moe_inter);
            gelu_pytorch_tanh(gate);
            let act_off = job.gate_act_off + row * moe_inter;
            for i in 0..moe_inter {
                gate_act_arena[act_off + i] = gate[i] * up[i];
            }
        }
    }

    // Phase 2: all expert down GEMMs in one GPU batch (one sync).
    {
        let mut batch = GpuBatch::begin(&ctx.queue, pool, &ctx.device)?;
        for job in &jobs {
            let batch_size = job.tokens.len();
            let act_slice = &gate_act_arena[job.gate_act_off..job.gate_act_off + batch_size * moe_inter];
            let out_slice = &mut out_arena[job.out_off..job.out_off + batch_size * hidden];
            cached.with_expert_down_t(down, job.expert, |w_t| {
                f32_bf16_gemm(
                    &mut batch,
                    pipeline,
                    out_slice,
                    act_slice,
                    w_t,
                    batch_size,
                    moe_inter,
                    hidden,
                )
            })?;
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
