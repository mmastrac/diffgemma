use crate::config::TextConfig;
use crate::metal::batched_kernels::{self as bk};
use crate::metal::batch::GpuBatch;
use crate::metal::device::ComputePipeline;
use crate::metal::kernels::GpuKernels;
use crate::metal::linear::f32_q4_linear_gpu_bufs;
use crate::metal::telemetry::ForwardTelemetry;
use crate::metal::weights::GpuDecoderWeightCache;
use std::cell::RefCell;
use std::rc::Rc;
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

#[repr(C)]
#[derive(Clone, Copy)]
struct Q4GroupedJob {
    w_byte_off: u32,
    groups_per_row: u32,
}

pub fn experts_forward_gpu_batched(
    out: &mut [f32],
    residual: &[f32],
    pre_ff_norm_2: &[f32],
    eps: f32,
    _weights: Option<&DecoderLayerWeights<'_>>,
    expert_cache: &GpuDecoderWeightCache,
    layer: usize,
    cfg: &TextConfig,
    seq_len: usize,
    routes: &[RouteResult],
    token_indices: &mut Vec<u32>,
    out_arena: &mut Vec<f32>,
    ctx: &crate::metal::device::MetalContext,
    pool: &mut crate::metal::buffer::BufferPool,
    kernels: &GpuKernels,
    bf16_pipeline: &ComputePipeline,
    q4_pipeline: &ComputePipeline,
    q4_grouped_pipeline: &ComputePipeline,
    telemetry: Option<Rc<RefCell<ForwardTelemetry>>>,
) -> Result<(), Error> {
    let hidden = cfg.hidden_size;
    let moe_inter = cfg.moe_intermediate_size;
    let experts = cfg.num_experts;
    let dgq = expert_cache.is_dgq();

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

    if let Some(cell) = &telemetry {
        cell.borrow_mut()
            .record_expert_layer(layer, jobs.len(), cfg);
    }

    let _ = (gate_up_len, gate_act_len);
    out_arena.resize(out_len, 0.0);

    if !dgq {
        let weights = _weights.ok_or(Error::Format("bf16 moe needs layer weights"))?;
        let gate_up = weights.experts_gate_up.bf16()?;
        let down = weights.experts_down.bf16()?;
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
    }

    if dgq {
        experts_forward_gpu_grouped_dgq(
            out_arena,
            residual,
            pre_ff_norm_2,
            eps,
            expert_cache,
            layer,
            cfg,
            seq_len,
            &jobs,
            token_indices,
            ctx,
            pool,
            kernels,
            q4_grouped_pipeline,
            telemetry,
        )?;
    } else {
        experts_forward_gpu_per_job(
            out_arena,
            residual,
            pre_ff_norm_2,
            eps,
            expert_cache,
            layer,
            hidden,
            moe_inter,
            seq_len,
            &jobs,
            token_indices,
            ctx,
            pool,
            kernels,
            bf16_pipeline,
            q4_pipeline,
            telemetry,
        )?;
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

fn experts_forward_gpu_grouped_dgq(
    out_arena: &mut [f32],
    residual: &[f32],
    pre_ff_norm_2: &[f32],
    eps: f32,
    expert_cache: &GpuDecoderWeightCache,
    layer: usize,
    cfg: &TextConfig,
    seq_len: usize,
    jobs: &[ExpertJob],
    token_indices: &mut Vec<u32>,
    ctx: &crate::metal::device::MetalContext,
    pool: &mut crate::metal::buffer::BufferPool,
    kernels: &GpuKernels,
    q4_grouped_pipeline: &ComputePipeline,
    telemetry: Option<Rc<RefCell<ForwardTelemetry>>>,
) -> Result<(), Error> {
    let hidden = cfg.hidden_size;
    let moe_inter = cfg.moe_intermediate_size;
    let num_jobs = jobs.len();
    let w_blob = expert_cache
        .dgq_blob()
        .ok_or(Error::Format("grouped moe requires .dgq blob"))?;

    let mut gate_jobs = Vec::with_capacity(num_jobs);
    let mut down_jobs = Vec::with_capacity(num_jobs);
    let mut row_starts = Vec::with_capacity(num_jobs + 1);
    row_starts.push(0);
    token_indices.clear();
    for job in jobs {
        let w_gu = expert_cache.expert_gate_up_q4(layer, job.expert);
        gate_jobs.push(Q4GroupedJob {
            w_byte_off: w_gu.byte_offset as u32,
            groups_per_row: w_gu.groups_per_row(),
        });
        let w_dn = expert_cache.expert_down_q4(layer, job.expert);
        down_jobs.push(Q4GroupedJob {
            w_byte_off: w_dn.byte_offset as u32,
            groups_per_row: w_dn.groups_per_row(),
        });
        token_indices.extend(job.tokens.iter().map(|&(tok, _)| tok as u32));
        row_starts.push(token_indices.len() as u32);
    }
    let total_m = token_indices.len();
    if total_m == 0 {
        return Ok(());
    }

    let gate_jobs_bytes = unsafe {
        std::slice::from_raw_parts(
            gate_jobs.as_ptr().cast::<u8>(),
            gate_jobs.len() * std::mem::size_of::<Q4GroupedJob>(),
        )
    };
    let down_jobs_bytes = unsafe {
        std::slice::from_raw_parts(
            down_jobs.as_ptr().cast::<u8>(),
            down_jobs.len() * std::mem::size_of::<Q4GroupedJob>(),
        )
    };
    let row_starts_bytes = unsafe {
        std::slice::from_raw_parts(
            row_starts.as_ptr().cast::<u8>(),
            row_starts.len() * 4,
        )
    };

    {
        let mut batch = GpuBatch::begin_with_telemetry(
            &ctx.queue,
            pool,
            &ctx.device,
            telemetry.clone(),
        )?;
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
        let buf_a = bk::gather_rows_gpu(
            &mut batch,
            kernels,
            &buf_moe_in,
            token_indices,
            hidden,
        )?;
        let buf_gate_jobs = batch.alloc_bytes(gate_jobs_bytes)?;
        let buf_down_jobs = batch.alloc_bytes(down_jobs_bytes)?;
        let buf_row_starts = batch.alloc_bytes(row_starts_bytes)?;

        let gate_n = moe_inter * 2;
        let buf_gu = batch.alloc_f32_out(total_m * gate_n)?;
        batch.dispatch_q4_linear_grouped(
            &q4_grouped_pipeline.pipeline,
            &buf_a,
            w_blob,
            &buf_gu,
            &buf_gate_jobs,
            &buf_row_starts,
            total_m,
            hidden,
            gate_n,
            num_jobs,
        );

        let act_len = total_m * moe_inter;
        let buf_act = bk::gelu_swiglu_gate_up_gpu(
            &mut batch,
            kernels,
            &buf_gu,
            act_len,
            total_m,
            moe_inter,
        )?;

        let buf_out = batch.alloc_f32_out(total_m * hidden)?;
        batch.dispatch_q4_linear_grouped(
            &q4_grouped_pipeline.pipeline,
            &buf_act,
            w_blob,
            &buf_out,
            &buf_down_jobs,
            &buf_row_starts,
            total_m,
            moe_inter,
            hidden,
            num_jobs,
        );
        batch.register_read(buf_out, out_arena);
        batch.end()?;
    }
    Ok(())
}

fn experts_forward_gpu_per_job(
    out_arena: &mut [f32],
    residual: &[f32],
    pre_ff_norm_2: &[f32],
    eps: f32,
    expert_cache: &GpuDecoderWeightCache,
    layer: usize,
    hidden: usize,
    moe_inter: usize,
    seq_len: usize,
    jobs: &[ExpertJob],
    token_indices: &mut Vec<u32>,
    ctx: &crate::metal::device::MetalContext,
    pool: &mut crate::metal::buffer::BufferPool,
    kernels: &GpuKernels,
    bf16_pipeline: &ComputePipeline,
    q4_pipeline: &ComputePipeline,
    telemetry: Option<Rc<RefCell<ForwardTelemetry>>>,
) -> Result<(), Error> {
    let dgq = expert_cache.is_dgq();
    let bf16_ps = &bf16_pipeline.pipeline;
    {
        let mut batch = GpuBatch::begin_with_telemetry(
            &ctx.queue,
            pool,
            &ctx.device,
            telemetry,
        )?;
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
        for job in jobs {
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
            let buf_gu = if dgq {
                let w = expert_cache.expert_gate_up_q4(layer, job.expert);
                f32_q4_linear_gpu_bufs(
                    &mut batch,
                    q4_pipeline,
                    &buf_a,
                    &w,
                    batch_size,
                    hidden,
                    moe_inter * 2,
                )?
            } else {
                let w_gate = expert_cache.expert_gate_up_buf(layer, job.expert);
                bk::f32_bf16_linear_gpu_bufs(
                    &mut batch,
                    bf16_ps,
                    &buf_a,
                    &w_gate,
                    batch_size,
                    hidden,
                    moe_inter * 2,
                )?
            };
            let buf_act = bk::gelu_swiglu_gate_up_gpu(
                &mut batch,
                kernels,
                &buf_gu,
                act_len,
                batch_size,
                moe_inter,
            )?;
            let out_slice = &mut out_arena[job.out_off..job.out_off + batch_size * hidden];
            let buf_out = if dgq {
                let w = expert_cache.expert_down_q4(layer, job.expert);
                f32_q4_linear_gpu_bufs(
                    &mut batch,
                    q4_pipeline,
                    &buf_act,
                    &w,
                    batch_size,
                    moe_inter,
                    hidden,
                )?
            } else {
                let w_down = expert_cache.expert_down_buf(layer, job.expert);
                bk::f32_bf16_linear_gpu_bufs(
                    &mut batch,
                    bf16_ps,
                    &buf_act,
                    &w_down,
                    batch_size,
                    moe_inter,
                    hidden,
                )?
            };
            batch.register_read(buf_out, out_slice);
        }
        batch.end()?;
    }
    Ok(())
}
