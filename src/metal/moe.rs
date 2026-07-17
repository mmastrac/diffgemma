use crate::Error;
use crate::config::TextConfig;
use crate::dgq::DgqStore;
use crate::dgq::block::{q4_gemm_cpu, q4_weight_at};
use crate::dgq::layout::NVFP4_HEADER_BYTES;
use crate::dgq::nvfp4::nvfp4_gemm_cpu;
use crate::metal::batch::GpuBatch;
use crate::metal::batched_kernels::{self as bk};
use crate::metal::device::ComputePipeline;
use crate::metal::dgq_gpu::Q4LinearGpu;
use crate::metal::kernels::GpuKernels;
use crate::metal::linear::f32_q4_linear_gpu_bufs;
use crate::metal::telemetry::ForwardTelemetry;
use crate::metal::weights::GpuDecoderWeightCache;
use crate::model::layer_weights::DecoderLayerWeights;
use crate::model::moe::{MoeScratch, RouteResult};
use crate::shaders::cpu::{gelu_pytorch_tanh, gelu_pytorch_tanh_f32, linear};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) struct ExpertJob {
    pub(crate) expert: usize,
    pub(crate) tokens: Vec<(usize, f32)>,
    pub(crate) gate_up_off: usize,
    pub(crate) gate_act_off: usize,
    pub(crate) out_off: usize,
}

pub fn build_expert_jobs(routes: &[RouteResult], experts: usize) -> Vec<ExpertJob> {
    let mut buckets: Vec<Vec<(usize, f32)>> = vec![Vec::new(); experts];
    for (s, route) in routes.iter().enumerate() {
        for (&expert, &weight) in route.indices.iter().zip(route.weights.iter()) {
            buckets[expert].push((s, weight));
        }
    }

    let mut jobs = Vec::new();
    for (expert, tokens) in buckets.iter().enumerate() {
        if tokens.is_empty() {
            continue;
        }
        jobs.push(ExpertJob {
            expert,
            tokens: tokens.clone(),
            gate_up_off: 0,
            gate_act_off: 0,
            out_off: 0,
        });
    }
    jobs
}

use crate::metal::BlockGroupedJob;

fn block_gemm_cpu(a: &[f32], m: usize, k: usize, w: &Q4LinearGpu, n: usize, out: &mut [f32]) {
    if w.is_nvfp4() {
        let body = &w.src_slice()[NVFP4_HEADER_BYTES..];
        nvfp4_gemm_cpu(a, m, k, body, n, w.global_scale_f32(), out);
    } else if w.quant_kind() == crate::dgq::layout::QuantKind::Q6Block {
        crate::dgq::block::q6_gemm_cpu(a, m, k, w.src_slice(), n, out);
    } else {
        q4_gemm_cpu(a, m, k, w.src_slice(), n, out);
    }
}

#[derive(Debug, Clone)]
pub struct ExpertSingleStaged {
    pub gate_act: Vec<f32>,
    pub out: Vec<f32>,
}

/// One expert, one token: gate_up → GELU×up → down (Q4 `.dgq` weights).
pub fn expert_forward_staged_dgq(
    x: &[f32],
    gate_up_w: &Q4LinearGpu,
    down_w: &Q4LinearGpu,
    moe_inter: usize,
    hidden: usize,
    scratch: &mut MoeScratch,
) -> ExpertSingleStaged {
    assert_eq!(x.len(), hidden);
    block_gemm_cpu(x, 1, hidden, gate_up_w, moe_inter * 2, &mut scratch.gate_up);
    let (gate, up) = scratch.gate_up.split_at_mut(moe_inter);
    gelu_pytorch_tanh(gate);
    for i in 0..moe_inter {
        scratch.gate_act[i] = gate[i] * up[i];
    }
    block_gemm_cpu(
        &scratch.gate_act,
        1,
        moe_inter,
        down_w,
        hidden,
        &mut scratch.expert_out,
    );
    ExpertSingleStaged {
        gate_act: scratch.gate_act.clone(),
        out: scratch.expert_out.clone(),
    }
}

/// BF16-dequant oracle for one expert (same layout as `model::moe::expert_forward_one`).
pub fn expert_forward_staged_bf16(
    x: &[f32],
    gate_up_f32: &[f32],
    down_f32: &[f32],
    moe_inter: usize,
    hidden: usize,
    scratch: &mut MoeScratch,
) -> ExpertSingleStaged {
    assert_eq!(x.len(), hidden);
    assert_eq!(gate_up_f32.len(), moe_inter * 2 * hidden);
    assert_eq!(down_f32.len(), hidden * moe_inter);
    linear(
        &mut scratch.gate_up,
        x,
        gate_up_f32,
        None,
        1,
        hidden,
        moe_inter * 2,
    );
    let (gate, up) = scratch.gate_up.split_at_mut(moe_inter);
    gelu_pytorch_tanh(gate);
    for i in 0..moe_inter {
        scratch.gate_act[i] = gate[i] * up[i];
    }
    linear(
        &mut scratch.expert_out,
        &scratch.gate_act,
        down_f32,
        None,
        1,
        moe_inter,
        hidden,
    );
    ExpertSingleStaged {
        gate_act: scratch.gate_act.clone(),
        out: scratch.expert_out.clone(),
    }
}

/// CPU mirror of `k_moe_grouped` per-token math (dequant loop, not batched GEMM).
pub fn expert_forward_k_moe_grouped_mirror(
    x: &[f32],
    gate_up_w: &Q4LinearGpu,
    down_w: &Q4LinearGpu,
    moe_inter: usize,
    hidden: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), hidden);
    let gu = gate_up_w.src_slice();
    let dn = down_w.src_slice();
    let mut act = vec![0.0f32; moe_inter];
    for r in 0..moe_inter {
        let mut g = 0.0f32;
        let mut u = 0.0f32;
        for k in 0..hidden {
            g += q4_weight_at(gu, r, k, hidden) * x[k];
            u += q4_weight_at(gu, r + moe_inter, k, hidden) * x[k];
        }
        act[r] = gelu_pytorch_tanh_f32(g) * u;
    }
    let mut out = vec![0.0f32; hidden];
    for d in 0..hidden {
        let mut o = 0.0f32;
        for k in 0..moe_inter {
            o += q4_weight_at(dn, d, k, moe_inter) * act[k];
        }
        out[d] = o;
    }
    out
}

pub fn load_expert_bf16_slices(
    store: &DgqStore,
    layer: usize,
    expert: usize,
    moe_inter: usize,
    hidden: usize,
) -> Result<(Vec<f32>, Vec<f32>), Error> {
    let p = format!("model.decoder.layers.{layer}.");
    let gu_name = format!("{p}experts.gate_up_proj");
    let dn_name = format!("{p}experts.down_proj");
    let gu_all = store.tensor_f32(&gu_name)?;
    let dn_all = store.tensor_f32(&dn_name)?;
    let gu_stride = moe_inter * 2 * hidden;
    let dn_stride = hidden * moe_inter;
    let gu_off = expert * gu_stride;
    let dn_off = expert * dn_stride;
    if gu_off + gu_stride > gu_all.len() || dn_off + dn_stride > dn_all.len() {
        return Err(Error::Runtime("expert bf16 slice out of range"));
    }
    Ok((
        gu_all[gu_off..gu_off + gu_stride].to_vec(),
        dn_all[dn_off..dn_off + dn_stride].to_vec(),
    ))
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
    nvfp4_pipeline: &ComputePipeline,
    q4_grouped_pipeline: &ComputePipeline,
    nvfp4_grouped_pipeline: &ComputePipeline,
    telemetry: Option<Rc<RefCell<ForwardTelemetry>>>,
) -> Result<(), Error> {
    let hidden = cfg.hidden_size;
    let moe_inter = cfg.moe_intermediate_size;
    let experts = cfg.num_experts;
    let dgq = expert_cache.is_dgq();

    out.fill(0.0);

    let mut jobs = build_expert_jobs(routes, experts);

    let mut gate_up_len = 0usize;
    let mut gate_act_len = 0usize;
    let mut out_len = 0usize;
    for job in &mut jobs {
        let batch_size = job.tokens.len();
        job.gate_up_off = gate_up_len;
        job.gate_act_off = gate_act_len;
        job.out_off = out_len;
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
        let weights = _weights.ok_or(Error::Runtime("bf16 moe needs layer weights"))?;
        let gate_up = weights.experts_gate_up.bf16()?;
        let down = weights.experts_down.bf16()?;
        for job in &jobs {
            expert_cache.prefetch_expert(layer, gate_up, down, job.expert, &ctx.device, pool);
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
            nvfp4_grouped_pipeline,
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
            nvfp4_pipeline,
            telemetry,
        )?;
    }

    scatter_weighted_expert_outputs(out, out_arena, &jobs, hidden);
    Ok(())
}

pub fn scatter_weighted_expert_outputs(
    out: &mut [f32],
    out_arena: &[f32],
    jobs: &[ExpertJob],
    hidden: usize,
) {
    out.fill(0.0);
    for job in jobs {
        for (bi, &(_, weight)) in job.tokens.iter().enumerate() {
            let src = job.out_off + bi * hidden;
            let dst = job.tokens[bi].0 * hidden;
            for i in 0..hidden {
                out[dst + i] += weight * out_arena[src + i];
            }
        }
    }
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
    nvfp4_grouped_pipeline: &ComputePipeline,
    telemetry: Option<Rc<RefCell<ForwardTelemetry>>>,
) -> Result<(), Error> {
    let mut batch =
        GpuBatch::begin_with_telemetry(&ctx.queue, pool, &ctx.device, telemetry.clone())?;
    let buf_res = batch.alloc_f32(residual)?;
    experts_forward_gpu_grouped_in_batch(
        &mut batch,
        out_arena,
        &buf_res,
        pre_ff_norm_2,
        eps,
        expert_cache,
        layer,
        cfg,
        seq_len,
        jobs,
        token_indices,
        kernels,
        q4_grouped_pipeline,
        nvfp4_grouped_pipeline,
    )?;
    batch.end()?;
    Ok(())
}

pub fn experts_forward_gpu_grouped_in_batch(
    batch: &mut GpuBatch<'_>,
    out_arena: &mut [f32],
    residual_buf: &ProtocolObject<dyn MTLBuffer>,
    pre_ff_norm_2: &[f32],
    eps: f32,
    expert_cache: &GpuDecoderWeightCache,
    layer: usize,
    cfg: &TextConfig,
    seq_len: usize,
    jobs: &[ExpertJob],
    token_indices: &mut Vec<u32>,
    kernels: &GpuKernels,
    q4_grouped_pipeline: &ComputePipeline,
    nvfp4_grouped_pipeline: &ComputePipeline,
) -> Result<(), Error> {
    if let Some(buf_out) = experts_forward_gpu_grouped_in_batch_buf(
        batch,
        residual_buf,
        pre_ff_norm_2,
        eps,
        expert_cache,
        layer,
        cfg,
        seq_len,
        jobs,
        token_indices,
        kernels,
        q4_grouped_pipeline,
        nvfp4_grouped_pipeline,
    )? {
        batch.register_read(buf_out, out_arena);
    }
    Ok(())
}

/// Same as `experts_forward_gpu_grouped_in_batch` but leaves the `[total_m, hidden]`
/// expert-output arena on GPU (no readback) — for the GPU-resident prefill path.
/// Returns `None` when there is no routed work.
pub fn experts_forward_gpu_grouped_in_batch_buf(
    batch: &mut GpuBatch<'_>,
    residual_buf: &ProtocolObject<dyn MTLBuffer>,
    pre_ff_norm_2: &[f32],
    eps: f32,
    expert_cache: &GpuDecoderWeightCache,
    layer: usize,
    cfg: &TextConfig,
    seq_len: usize,
    jobs: &[ExpertJob],
    token_indices: &mut Vec<u32>,
    kernels: &GpuKernels,
    q4_grouped_pipeline: &ComputePipeline,
    nvfp4_grouped_pipeline: &ComputePipeline,
) -> Result<Option<Retained<ProtocolObject<dyn MTLBuffer>>>, Error> {
    let _hidden = cfg.hidden_size;
    let moe_inter = cfg.moe_intermediate_size;
    let num_jobs = jobs.len();
    let w_blob = expert_cache
        .dgq_expert_region()
        .ok_or(Error::Format("grouped moe requires .dgq blob"))?
        .0;

    let mut gate_jobs = Vec::with_capacity(num_jobs);
    let mut down_jobs = Vec::with_capacity(num_jobs);
    let mut row_starts = Vec::with_capacity(num_jobs + 1);
    row_starts.push(0);
    token_indices.clear();
    for job in jobs {
        // Region-rebased offsets: jobs index the bound (possibly split-off
        // experts) buffer, not the absolute blob.
        let w_gu = expert_cache.expert_gate_up_q4(layer, job.expert);
        gate_jobs.push(BlockGroupedJob {
            w_byte_off: w_gu.weight_buffer().1,
            groups_per_row: w_gu.groups_per_row(),
            _pad: 0,
        });
        let w_dn = expert_cache.expert_down_q4(layer, job.expert);
        down_jobs.push(BlockGroupedJob {
            w_byte_off: w_dn.weight_buffer().1,
            groups_per_row: w_dn.groups_per_row(),
            _pad: 0,
        });
        token_indices.extend(job.tokens.iter().map(|&(tok, _)| tok as u32));
        row_starts.push(token_indices.len() as u32);
    }
    let total_m = token_indices.len();
    if total_m == 0 {
        return Ok(None);
    }

    let gate_jobs_bytes = unsafe {
        std::slice::from_raw_parts(
            gate_jobs.as_ptr().cast::<u8>(),
            gate_jobs.len() * std::mem::size_of::<BlockGroupedJob>(),
        )
    };
    let down_jobs_bytes = unsafe {
        std::slice::from_raw_parts(
            down_jobs.as_ptr().cast::<u8>(),
            down_jobs.len() * std::mem::size_of::<BlockGroupedJob>(),
        )
    };
    let row_starts_bytes = unsafe {
        std::slice::from_raw_parts(row_starts.as_ptr().cast::<u8>(), row_starts.len() * 4)
    };

    let buf_moe_in = bk::rms_norm_rows_gpu_buf(
        batch,
        kernels,
        residual_buf,
        pre_ff_norm_2,
        seq_len,
        cfg.hidden_size,
        eps,
    )?;
    let buf_a = bk::gather_rows_gpu(batch, kernels, &buf_moe_in, token_indices, cfg.hidden_size)?;
    let buf_gate_jobs = batch.alloc_bytes(gate_jobs_bytes)?;
    let buf_down_jobs = batch.alloc_bytes(down_jobs_bytes)?;
    let buf_row_starts = batch.alloc_bytes(row_starts_bytes)?;

    let grouped_pipeline = if expert_cache.expert_gate_up_q4(layer, 0).is_nvfp4() {
        nvfp4_grouped_pipeline
    } else {
        q4_grouped_pipeline
    };

    let gate_n = moe_inter * 2;
    let buf_gu = batch.alloc_f32_out(total_m * gate_n)?;
    batch.dispatch_q4_linear_grouped(
        &grouped_pipeline.pipeline,
        &buf_a,
        w_blob,
        &buf_gu,
        &buf_gate_jobs,
        &buf_row_starts,
        total_m,
        cfg.hidden_size,
        gate_n,
        num_jobs,
    );

    let act_len = total_m * moe_inter;
    let buf_act =
        bk::gelu_swiglu_gate_up_gpu(batch, kernels, &buf_gu, act_len, total_m, moe_inter)?;

    let buf_out = batch.alloc_f32_out(total_m * cfg.hidden_size)?;
    batch.dispatch_q4_linear_grouped(
        &grouped_pipeline.pipeline,
        &buf_act,
        w_blob,
        &buf_out,
        &buf_down_jobs,
        &buf_row_starts,
        total_m,
        moe_inter,
        cfg.hidden_size,
        num_jobs,
    );
    Ok(Some(buf_out))
}

/// Per-token `(arena_row, weight)` lists for `scatter_rows_weighted`, flattened
/// to `[seq_len * top_k]`. Rows are the positions each token's entries occupy in
/// `token_indices` (i.e. the grouped-GEMM arena rows), listed in expert-JOB
/// order — the exact accumulation order of `scatter_weighted_expert_outputs`,
/// so the GPU scatter's f32 sums are bit-identical to the CPU scatter. Tokens
/// with fewer than `top_k` routed entries (shouldn't happen: the router always
/// emits top_k) are padded with `(row 0, weight 0.0)` — an exact f32 no-op.
pub(crate) fn build_scatter_lists(
    jobs: &[ExpertJob],
    seq_len: usize,
    top_k: usize,
) -> (Vec<u32>, Vec<f32>) {
    let mut rows = vec![0u32; seq_len * top_k];
    let mut weights = vec![0.0f32; seq_len * top_k];
    let mut fill = vec![0usize; seq_len];
    let mut arena_row = 0u32;
    for job in jobs {
        for &(tok, w) in &job.tokens {
            let k = fill[tok];
            debug_assert!(k < top_k, "token {tok} routed to more than top_k experts");
            if k < top_k {
                rows[tok * top_k + k] = arena_row;
                weights[tok * top_k + k] = w;
                fill[tok] = k + 1;
            }
            arena_row += 1;
        }
    }
    (rows, weights)
}

/// Deterministic CPU expert forward for `.dgq` (native Q4 GEMM matches Metal kernel).
pub fn experts_forward_dgq_cpu(
    out: &mut [f32],
    expert_input: &[f32],
    expert_cache: &GpuDecoderWeightCache,
    layer: usize,
    cfg: &TextConfig,
    seq_len: usize,
    routes: &[RouteResult],
    scratch: &mut MoeScratch,
) -> Result<(), Error> {
    let hidden = cfg.hidden_size;
    let moe_inter = cfg.moe_intermediate_size;
    assert_eq!(expert_input.len(), seq_len * hidden);
    assert_eq!(out.len(), seq_len * hidden);
    assert_eq!(routes.len(), seq_len);

    out.fill(0.0);
    for s in 0..seq_len {
        let x = &expert_input[s * hidden..(s + 1) * hidden];
        let route = &routes[s];
        let o = &mut out[s * hidden..(s + 1) * hidden];
        for (&expert, &weight) in route.indices.iter().zip(route.weights.iter()) {
            let gate_up = expert_cache.expert_gate_up_q4(layer, expert);
            let down = expert_cache.expert_down_q4(layer, expert);
            let staged = expert_forward_staged_dgq(x, &gate_up, &down, moe_inter, hidden, scratch);
            for i in 0..hidden {
                o[i] += weight * staged.out[i];
            }
        }
    }
    Ok(())
}

#[cfg(all(test, target_os = "macos"))]
mod dgq_expert_tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::dgq::DgqStore;
    use crate::dgq::layout::q4_matrix_bytes;
    use crate::metal::device::MetalContext;
    use crate::metal::weights::GpuDecoderWeightCache;
    use crate::weights::WeightStore;
    use std::sync::Arc;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let xf = *x as f64;
            let yf = *y as f64;
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

    #[test]
    fn dgq_expert_q4_matches_bf16_oracle_and_grouped_mirror() {
        let dgq_dir_buf = match crate::shaders::test_util::dgq_model_dir() {
            Some(d) => d,
            None => std::path::PathBuf::from("/tmp/quantized-weights"),
        };
        let dgq_dir = dgq_dir_buf.as_path();
        if !dgq_dir.join("model.dgq.json").exists() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let text = ModelConfig::load(dgq_dir).expect("config").text_config;
        let store = DgqStore::open(dgq_dir).expect("dgq");
        let ws = WeightStore::open(dgq_dir).expect("ws");
        let ctx = MetalContext::new().expect("metal");
        let cache = GpuDecoderWeightCache::load(&ws, &text, 0, &ctx.device).expect("cache");
        let layer = 2usize;
        let expert = 18usize;
        let hidden = text.hidden_size;
        let moe_inter = text.moe_intermediate_size;

        let (gu_bf16, dn_bf16) =
            load_expert_bf16_slices(&store, layer, expert, moe_inter, hidden).expect("bf16 slices");
        let gate_up = cache.expert_gate_up_q4(layer, expert);
        let down = cache.expert_down_q4(layer, expert);
        assert_eq!(
            gate_up.q4_byte_len(),
            q4_matrix_bytes(moe_inter * 2, hidden)
        );
        assert_eq!(down.q4_byte_len(), q4_matrix_bytes(hidden, moe_inter));

        let x: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 * 0.013 + 0.7).sin() * 0.05) as f32)
            .collect();
        let mut scratch = MoeScratch::new(1, &text);

        let q4 = expert_forward_staged_dgq(&x, &gate_up, &down, moe_inter, hidden, &mut scratch);
        let bf16 =
            expert_forward_staged_bf16(&x, &gu_bf16, &dn_bf16, moe_inter, hidden, &mut scratch);
        let mirror = expert_forward_k_moe_grouped_mirror(&x, &gate_up, &down, moe_inter, hidden);

        let cos_q4_bf16 = cosine(&q4.out, &bf16.out);
        let cos_mirror_q4 = cosine(&mirror, &q4.out);
        eprintln!(
            "expert L{layer} E{expert}: cos(q4,bf16)={cos_q4_bf16:.6} cos(mirror,q4)={cos_mirror_q4:.6}"
        );
        assert!(cos_q4_bf16 > 0.95, "q4 vs bf16 oracle cos={cos_q4_bf16}");
        assert!(
            cos_mirror_q4 > 0.999,
            "grouped mirror vs q4 cos={cos_mirror_q4}"
        );
    }
}

/// Per-expert Q4/BF16 MoE in one batch (deterministic; avoids grouped `simd_sum` kernel).
pub fn experts_forward_gpu_per_job(
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
    nvfp4_pipeline: &ComputePipeline,
    telemetry: Option<Rc<RefCell<ForwardTelemetry>>>,
) -> Result<(), Error> {
    let dgq = expert_cache.is_dgq();
    let bf16_ps = &bf16_pipeline.pipeline;
    for job in jobs {
        let batch_size = job.tokens.len();
        token_indices.clear();
        token_indices.extend(job.tokens.iter().map(|&(tok, _)| tok as u32));
        let out_slice = &mut out_arena[job.out_off..job.out_off + batch_size * hidden];
        let mut batch =
            GpuBatch::begin_with_telemetry(&ctx.queue, pool, &ctx.device, telemetry.clone())?;
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
        let buf_a = bk::gather_rows_gpu(&mut batch, kernels, &buf_moe_in, token_indices, hidden)?;
        let act_len = batch_size * moe_inter;
        let buf_gu = if dgq {
            let w = expert_cache.expert_gate_up_q4(layer, job.expert);
            f32_q4_linear_gpu_bufs(
                &mut batch,
                q4_pipeline,
                nvfp4_pipeline,
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
            &mut batch, kernels, &buf_gu, act_len, batch_size, moe_inter,
        )?;
        let buf_out = if dgq {
            let w = expert_cache.expert_down_q4(layer, job.expert);
            f32_q4_linear_gpu_bufs(
                &mut batch,
                q4_pipeline,
                nvfp4_pipeline,
                &buf_act,
                &w,
                batch_size,
                moe_inter,
                hidden,
            )?
        } else {
            let w_down = expert_cache.expert_down_buf(layer, job.expert);
            bk::f32_bf16_linear_gpu_bufs(
                &mut batch, bf16_ps, &buf_act, &w_down, batch_size, moe_inter, hidden,
            )?
        };
        batch.register_read(buf_out, out_slice);
        batch.end()?;
        pool.trim(0);
    }
    Ok(())
}
