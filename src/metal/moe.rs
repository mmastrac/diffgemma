use crate::config::TextConfig;
use crate::kernels::cpu::gelu_pytorch_tanh;
use crate::metal::batched_kernels::f32_bf16_gemm;
use crate::metal::batch::GpuBatch;
use crate::metal::device::MetalContext;
use crate::metal::weights::GpuLayerWeightCache;
use crate::model::layer_weights::DecoderLayerWeights;
use crate::model::moe::RouteResult;
use crate::safetensors::Error;

pub fn experts_forward_gpu_batched(
    out: &mut [f32],
    expert_input: &[f32],
    weights: &DecoderLayerWeights<'_>,
    cached: &GpuLayerWeightCache,
    cfg: &TextConfig,
    _seq_len: usize,
    routes: &[RouteResult],
    batch_input: &mut Vec<f32>,
    batch_gate_up: &mut Vec<f32>,
    batch_gate_act: &mut Vec<f32>,
    batch_out: &mut Vec<f32>,
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

    let pipeline = &gemm_pipeline.pipeline;

    for expert in 0..experts {
        let tokens = &buckets[expert];
        if tokens.is_empty() {
            continue;
        }
        let batch_size = tokens.len();
        batch_input.resize(batch_size * hidden, 0.0);
        batch_gate_up.resize(batch_size * moe_inter * 2, 0.0);
        batch_gate_act.resize(batch_size * moe_inter, 0.0);
        batch_out.resize(batch_size * hidden, 0.0);

        for (bi, &(tok, _)) in tokens.iter().enumerate() {
            let src = tok * hidden;
            let dst = bi * hidden;
            batch_input[dst..dst + hidden].copy_from_slice(&expert_input[src..src + hidden]);
        }

        {
            let mut batch = GpuBatch::begin(&ctx.queue, pool, &ctx.device)?;
            cached.with_expert_gate_up_t(gate_up, expert, |w_t| {
                f32_bf16_gemm(
                    &mut batch,
                    pipeline,
                    batch_gate_up,
                    batch_input,
                    w_t,
                    batch_size,
                    hidden,
                    moe_inter * 2,
                )
            })?;
            batch.end()?;
        }

        for row in 0..batch_size {
            let off = row * moe_inter * 2;
            let (gate, up) = batch_gate_up[off..off + moe_inter * 2].split_at_mut(moe_inter);
            gelu_pytorch_tanh(gate);
            let act_off = row * moe_inter;
            for i in 0..moe_inter {
                batch_gate_act[act_off + i] = gate[i] * up[i];
            }
        }

        {
            let mut batch = GpuBatch::begin(&ctx.queue, pool, &ctx.device)?;
            cached.with_expert_down_t(down, expert, |w_t| {
                f32_bf16_gemm(
                    &mut batch,
                    pipeline,
                    batch_out,
                    batch_gate_act,
                    w_t,
                    batch_size,
                    moe_inter,
                    hidden,
                )
            })?;
            batch.end()?;
        }

        for (bi, &(_, weight)) in tokens.iter().enumerate() {
            let src = bi * hidden;
            let dst = tokens[bi].0 * hidden;
            for i in 0..hidden {
                out[dst + i] += weight * batch_out[src + i];
            }
        }
    }
    Ok(())
}
