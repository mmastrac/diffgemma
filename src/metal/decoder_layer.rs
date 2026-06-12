use crate::config::TextConfig;
use crate::metal::batched_kernels::{self as bk};
use crate::metal::batch::begin_engine_batch;
use crate::metal::engine::GpuDecoderEngine;
use crate::metal::linear::linear_cached_batched_in_buf;
use crate::metal::moe::{
    build_expert_jobs, experts_forward_gpu_batched, experts_forward_gpu_grouped_in_batch,
    scatter_weighted_expert_outputs,
};
use crate::metal::router::{pack_routes, route_gpu_in_batch, GpuRouteScratch};
use crate::metal::weights::{GpuDecoderWeightCache, GpuLayerWeightCache};
use crate::metal::decoder_attention::{
    forward_decoder_attention, forward_encoder_extend_attention, forward_encoder_prefill_attention,
};
use crate::metal::kv_cache::GpuKvCache;
use crate::model::attention::{AttentionParams, AttentionScratch};
use crate::model::decoder_layer::{forward_decoder as cpu_forward_decoder, DecoderLayerScratch};
use crate::model::layer_weights::DecoderLayerWeights;
use crate::model::kv_cache::LayerKvView;
use crate::model::mask::DecoderAttnMask;
use crate::safetensors::Error;

pub struct GpuDecoderLayerScratch {
    pub cpu: DecoderLayerScratch,
    pub attn: AttentionScratch,
    pub attn_out: Vec<f32>,
    pub residual: Vec<f32>,
    pub normed: Vec<f32>,
    pub gate: Vec<f32>,
    pub mlp_hidden: Vec<f32>,
    pub dense_out: Vec<f32>,
    pub moe_branch: Vec<f32>,
    pub moe_input: Vec<f32>,
    pub moe_token_indices: Vec<u32>,
    pub moe_batch_out: Vec<f32>,
    pub route_indices: Vec<u32>,
    pub route_weights: Vec<f32>,
}

impl GpuDecoderLayerScratch {
    pub fn with_kv_len(
        seq_len: usize,
        cfg: &TextConfig,
        layer: usize,
        kv_cache_len: usize,
    ) -> Result<Self, Error> {
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let attn_params = AttentionParams::for_layer(cfg, layer)?;
        Ok(Self {
            cpu: DecoderLayerScratch::with_kv_len(seq_len, cfg, layer, kv_cache_len)?,
            attn: AttentionScratch::with_kv_len_gpu(seq_len, hidden, &attn_params, kv_cache_len),
            attn_out: vec![0.0; seq_len * hidden],
            residual: vec![0.0; seq_len * hidden],
            normed: vec![0.0; seq_len * hidden],
            gate: vec![0.0; seq_len * inter],
            mlp_hidden: vec![0.0; seq_len * inter],
            dense_out: vec![0.0; seq_len * hidden],
            moe_branch: vec![0.0; seq_len * hidden],
            moe_input: vec![0.0; seq_len * hidden],
            moe_token_indices: Vec::new(),
            moe_batch_out: Vec::new(),
            route_indices: vec![0; seq_len * cfg.top_k_experts],
            route_weights: vec![0.0; seq_len * cfg.top_k_experts],
        })
    }
}

pub fn forward_decoder(
    out: &mut [f32],
    hidden_states: &[f32],
    weights: Option<&DecoderLayerWeights<'_>>,
    cached: &GpuLayerWeightCache,
    expert_cache: &GpuDecoderWeightCache,
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    positions: &[i64],
    kv: LayerKvView<'_>,
    mask: Option<&DecoderAttnMask>,
    scratch: &mut GpuDecoderLayerScratch,
    engine: &mut GpuDecoderEngine,
    gpu_kv: Option<&GpuKvCache>,
) -> Result<(), Error> {
    let _hidden = cfg.hidden_size;

    forward_decoder_attention(
        &mut scratch.attn_out,
        hidden_states,
        cached,
        cfg,
        layer,
        seq_len,
        positions,
        kv,
        mask,
        &mut scratch.attn,
        engine,
        gpu_kv,
        engine.use_mps_q4(),
    )?;

    forward_layer_ff(
        out,
        hidden_states,
        weights,
        cached,
        expert_cache,
        cfg,
        layer,
        seq_len,
        scratch,
        engine,
    )
}

fn forward_layer_ff(
    out: &mut [f32],
    hidden_states: &[f32],
    weights: Option<&DecoderLayerWeights<'_>>,
    cached: &GpuLayerWeightCache,
    expert_cache: &GpuDecoderWeightCache,
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    scratch: &mut GpuDecoderLayerScratch,
    engine: &mut GpuDecoderEngine,
) -> Result<(), Error> {
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps as f32;

    scratch.residual.copy_from_slice(hidden_states);

    if expert_cache.is_dgq() {
        forward_layer_ff_dgq_gpu(
            out,
            cached,
            expert_cache,
            cfg,
            layer,
            seq_len,
            hidden,
            eps,
            scratch,
            engine,
        )?;
    } else {
        forward_layer_ff_bf16(
            out,
            hidden_states,
            weights,
            cached,
            expert_cache,
            cfg,
            layer,
            seq_len,
            hidden,
            eps,
            scratch,
            engine,
        )?;
    }
    for v in out.iter_mut() {
        *v *= cached.layer_scalar;
    }

    Ok(())
}

/// `.dgq` path: one batch per layer — dense FF, GPU router, grouped MoE, final combine.
/// Skips ~5.6 MB/layer residual+dense readback; routes read back via mid-batch flush (~32 KB).
fn forward_layer_ff_dgq_gpu(
    out: &mut [f32],
    cached: &GpuLayerWeightCache,
    expert_cache: &GpuDecoderWeightCache,
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    hidden: usize,
    eps: f32,
    scratch: &mut GpuDecoderLayerScratch,
    engine: &mut GpuDecoderEngine,
) -> Result<(), Error> {
    let len = seq_len * hidden;
    let experts = cfg.num_experts;
    let top_k = cfg.top_k_experts;
    let use_mps_q4 = engine.use_mps_q4();

    let telemetry = engine.batch_telemetry();
    let mut batch = begin_engine_batch(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        telemetry.clone(),
    )?;

    let buf_res = batch.alloc_f32(&scratch.residual)?;
    let buf_attn = batch.alloc_f32(&scratch.attn_out)?;
    let buf_stream = bk::rms_norm_rows_gpu_buf(
        &mut batch,
        &engine.kernels,
        &buf_attn,
        cached.post_attn_norm.as_slice(),
        seq_len,
        hidden,
        eps,
    )?;
    bk::vec_add_gpu_bufs(&mut batch, &engine.kernels, &buf_stream, &buf_res, len)?;
    let buf_normed = bk::rms_norm_rows_gpu_buf(
        &mut batch,
        &engine.kernels,
        &buf_stream,
        cached.pre_ff_norm.as_slice(),
        seq_len,
        hidden,
        eps,
    )?;
    let buf_gate = linear_cached_batched_in_buf(
        &mut batch,
        &engine.f32_bf16_linear_pipeline,
        &engine.f32_q4_linear_pipeline,
        if use_mps_q4 {
            Some((&mut engine.mps_matmul, &engine.dequant_q4_matrix_pipeline))
        } else {
            None
        },
        &buf_normed,
        &cached.mlp_gate,
        seq_len,
    )?;
    let buf_up = linear_cached_batched_in_buf(
        &mut batch,
        &engine.f32_bf16_linear_pipeline,
        &engine.f32_q4_linear_pipeline,
        if use_mps_q4 {
            Some((&mut engine.mps_matmul, &engine.dequant_q4_matrix_pipeline))
        } else {
            None
        },
        &buf_normed,
        &cached.mlp_up,
        seq_len,
    )?;
    let act_len = seq_len * cached.mlp_gate.out_dim();
    bk::gelu_pytorch_tanh_gpu_buf(&mut batch, &engine.kernels, &buf_gate, act_len)?;
    bk::swiglu_mul_gpu_bufs(&mut batch, &engine.kernels, &buf_gate, &buf_up, act_len)?;
    let buf_down = linear_cached_batched_in_buf(
        &mut batch,
        &engine.f32_bf16_linear_pipeline,
        &engine.f32_q4_linear_pipeline,
        if use_mps_q4 {
            Some((&mut engine.mps_matmul, &engine.dequant_q4_matrix_pipeline))
        } else {
            None
        },
        &buf_gate,
        &cached.mlp_down,
        seq_len,
    )?;
    let buf_ff = bk::rms_norm_rows_gpu_buf(
        &mut batch,
        &engine.kernels,
        &buf_down,
        cached.post_ff_norm_1.as_slice(),
        seq_len,
        hidden,
        eps,
    )?;

    let mut route_scratch = GpuRouteScratch {
        indices: std::mem::take(&mut scratch.route_indices),
        weights: std::mem::take(&mut scratch.route_weights),
    };
    route_gpu_in_batch(
        &mut batch,
        &engine.kernels,
        &engine.f32_f32_linear_pipeline.pipeline,
        &buf_stream,
        cached,
        cfg,
        seq_len,
        &mut route_scratch,
    )?;
    batch.flush_reads()?;

    let routes = pack_routes(
        &route_scratch.indices,
        &route_scratch.weights,
        seq_len,
        top_k,
    );
    scratch.route_indices = route_scratch.indices;
    scratch.route_weights = route_scratch.weights;

    let jobs = build_expert_jobs(&routes, experts);
    if let Some(cell) = &telemetry {
        cell.borrow_mut()
            .record_expert_layer(layer, jobs.len(), cfg);
    }

    let mut jobs = jobs;
    let mut out_len = 0usize;
    for job in &mut jobs {
        let batch_size = job.tokens.len();
        job.gate_up_off = 0;
        job.gate_act_off = 0;
        job.out_off = out_len;
        out_len += batch_size * hidden;
    }
    scratch.moe_batch_out.resize(out_len, 0.0);

    if !jobs.is_empty() {
        experts_forward_gpu_grouped_in_batch(
        &mut batch,
        &mut scratch.moe_batch_out,
        &buf_stream,
        cached.pre_ff_norm_2.as_slice(),
        eps,
        expert_cache,
        layer,
        cfg,
        seq_len,
        &jobs,
        &mut scratch.moe_token_indices,
        &engine.kernels,
        &engine.f32_q4_linear_grouped_pipeline,
        )?;
        batch.flush_reads()?;
    }

    scatter_weighted_expert_outputs(
        &mut scratch.moe_branch,
        &scratch.moe_batch_out,
        &jobs,
        hidden,
    );

    let buf_moe = batch.alloc_f32(&scratch.moe_branch)?;
    let buf_moe_n = bk::rms_norm_rows_gpu_buf(
        &mut batch,
        &engine.kernels,
        &buf_moe,
        cached.post_ff_norm_2.as_slice(),
        seq_len,
        hidden,
        eps,
    )?;
    bk::vec_add_gpu_bufs(&mut batch, &engine.kernels, &buf_ff, &buf_moe_n, len)?;
    let buf_out = bk::rms_norm_rows_gpu_buf(
        &mut batch,
        &engine.kernels,
        &buf_ff,
        cached.post_ff_norm.as_slice(),
        seq_len,
        hidden,
        eps,
    )?;
    bk::vec_add_gpu_bufs(&mut batch, &engine.kernels, &buf_out, &buf_stream, len)?;
    batch.register_read(buf_out, out);
    batch.end()?;
    Ok(())
}

fn forward_layer_ff_bf16(
    out: &mut [f32],
    _hidden_states: &[f32],
    weights: Option<&DecoderLayerWeights<'_>>,
    cached: &GpuLayerWeightCache,
    expert_cache: &GpuDecoderWeightCache,
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    hidden: usize,
    eps: f32,
    scratch: &mut GpuDecoderLayerScratch,
    engine: &mut GpuDecoderEngine,
) -> Result<(), Error> {
    let telemetry = engine.batch_telemetry();
    {
        let mut batch = begin_engine_batch(
            &engine.ctx.queue,
            &mut engine.pool,
            &engine.ctx.device,
            telemetry,
        )?;
        let len = seq_len * hidden;
        let buf_res = batch.alloc_f32(&scratch.residual)?;
        let buf_attn = batch.alloc_f32(&scratch.attn_out)?;
        let buf_stream = bk::rms_norm_rows_gpu_buf(
            &mut batch,
            &engine.kernels,
            &buf_attn,
            cached.post_attn_norm.as_slice(),
            seq_len,
            hidden,
            eps,
        )?;
        bk::vec_add_gpu_bufs(&mut batch, &engine.kernels, &buf_stream, &buf_res, len)?;
        let buf_normed = bk::rms_norm_rows_gpu_buf(
            &mut batch,
            &engine.kernels,
            &buf_stream,
            cached.pre_ff_norm.as_slice(),
            seq_len,
            hidden,
            eps,
        )?;
        let buf_gate = linear_cached_batched_in_buf(
            &mut batch,
            &engine.f32_bf16_linear_pipeline,
            &engine.f32_q4_linear_pipeline,
            None,
            &buf_normed,
            &cached.mlp_gate,
            seq_len,
        )?;
        let buf_up = linear_cached_batched_in_buf(
            &mut batch,
            &engine.f32_bf16_linear_pipeline,
            &engine.f32_q4_linear_pipeline,
            None,
            &buf_normed,
            &cached.mlp_up,
            seq_len,
        )?;
        let act_len = seq_len * cached.mlp_gate.out_dim();
        bk::gelu_pytorch_tanh_gpu_buf(&mut batch, &engine.kernels, &buf_gate, act_len)?;
        bk::swiglu_mul_gpu_bufs(&mut batch, &engine.kernels, &buf_gate, &buf_up, act_len)?;
        let buf_down = linear_cached_batched_in_buf(
            &mut batch,
            &engine.f32_bf16_linear_pipeline,
            &engine.f32_q4_linear_pipeline,
            None,
            &buf_gate,
            &cached.mlp_down,
            seq_len,
        )?;
        let buf_ff = bk::rms_norm_rows_gpu_buf(
            &mut batch,
            &engine.kernels,
            &buf_down,
            cached.post_ff_norm_1.as_slice(),
            seq_len,
            hidden,
            eps,
        )?;
        batch.register_read(buf_stream, &mut scratch.residual);
        batch.register_read(buf_ff, &mut scratch.normed);
        batch.end()?;
    }

    let routes = crate::model::moe::route_with_cached_weights(
        &scratch.residual,
        cached.router_proj.as_slice(),
        cached.router_scale.as_slice(),
        cached.per_expert_scale.as_slice(),
        cfg,
        seq_len,
        &mut scratch.cpu.moe,
    )?;

    let telemetry = engine.batch_telemetry();
    experts_forward_gpu_batched(
        &mut scratch.moe_branch,
        &scratch.residual,
        cached.pre_ff_norm_2.as_slice(),
        eps,
        weights,
        expert_cache,
        layer,
        cfg,
        seq_len,
        &routes,
        &mut scratch.moe_token_indices,
        &mut scratch.moe_batch_out,
        &engine.ctx,
        &mut engine.pool,
        &engine.kernels,
        &engine.f32_bf16_linear_pipeline,
        &engine.f32_q4_linear_pipeline,
        &engine.f32_q4_linear_grouped_pipeline,
        telemetry,
    )?;

    {
        let telemetry = engine.batch_telemetry();
        let mut batch = begin_engine_batch(
            &engine.ctx.queue,
            &mut engine.pool,
            &engine.ctx.device,
            telemetry,
        )?;
        let len = seq_len * hidden;
        let buf_moe = batch.alloc_f32(&scratch.moe_branch)?;
        let buf_moe_n = bk::rms_norm_rows_gpu_buf(
            &mut batch,
            &engine.kernels,
            &buf_moe,
            cached.post_ff_norm_2.as_slice(),
            seq_len,
            hidden,
            eps,
        )?;
        let buf_sum = batch.alloc_f32(&scratch.normed)?;
        bk::vec_add_gpu_bufs(&mut batch, &engine.kernels, &buf_sum, &buf_moe_n, len)?;
        let buf_out = bk::rms_norm_rows_gpu_buf(
            &mut batch,
            &engine.kernels,
            &buf_sum,
            cached.post_ff_norm.as_slice(),
            seq_len,
            hidden,
            eps,
        )?;
        let buf_res = batch.alloc_f32(&scratch.residual)?;
        bk::vec_add_gpu_bufs(&mut batch, &engine.kernels, &buf_out, &buf_res, len)?;
        batch.register_read(buf_out, out);
        batch.end()?;
    }
    Ok(())
}

pub fn forward_encoder_prefill(
    out: &mut [f32],
    hidden_states: &[f32],
    weights: Option<&DecoderLayerWeights<'_>>,
    cached: &GpuLayerWeightCache,
    expert_cache: &GpuDecoderWeightCache,
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    positions: &[i64],
    scratch: &mut GpuDecoderLayerScratch,
    engine: &mut GpuDecoderEngine,
    gpu_kv: &GpuKvCache,
) -> Result<(), Error> {
    forward_encoder_prefill_attention(
        &mut scratch.attn_out,
        hidden_states,
        cached,
        cfg,
        layer,
        seq_len,
        positions,
        &mut scratch.attn,
        engine,
        gpu_kv,
        engine.use_mps_q4(),
    )?;

    forward_layer_ff(
        out,
        hidden_states,
        weights,
        cached,
        expert_cache,
        cfg,
        layer,
        seq_len,
        scratch,
        engine,
    )
}

pub fn forward_encoder_extend(
    out: &mut [f32],
    hidden_states: &[f32],
    weights: Option<&DecoderLayerWeights<'_>>,
    cached: &GpuLayerWeightCache,
    expert_cache: &GpuDecoderWeightCache,
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    positions: &[i64],
    kv_cache_len: usize,
    scratch: &mut GpuDecoderLayerScratch,
    engine: &mut GpuDecoderEngine,
    gpu_kv: &GpuKvCache,
) -> Result<(), Error> {
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps as f32;

    forward_encoder_extend_attention(
        &mut scratch.attn_out,
        hidden_states,
        cached,
        cfg,
        layer,
        seq_len,
        positions,
        kv_cache_len,
        &mut scratch.attn,
        engine,
        gpu_kv,
        engine.use_mps_q4(),
    )?;

    forward_layer_ff(
        out,
        hidden_states,
        weights,
        cached,
        expert_cache,
        cfg,
        layer,
        seq_len,
        scratch,
        engine,
    )
}

#[allow(dead_code)]
pub fn forward_decoder_cpu(
    out: &mut [f32],
    hidden_states: &[f32],
    weights: &DecoderLayerWeights<'_>,
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    positions: &[i64],
    kv: LayerKvView<'_>,
    mask: Option<&DecoderAttnMask>,
    scratch: &mut GpuDecoderLayerScratch,
    _engine: &mut GpuDecoderEngine,
) -> Result<(), Error> {
    cpu_forward_decoder(
        out,
        hidden_states,
        weights,
        cfg,
        layer,
        seq_len,
        positions,
        kv,
        mask,
        &mut scratch.cpu,
    )
}
