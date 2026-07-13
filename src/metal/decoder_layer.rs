use crate::Error;
use crate::config::TextConfig;
use crate::metal::batch::begin_engine_batch;
use crate::metal::batched_kernels::{self as bk};
use crate::metal::decoder_attention::{
    forward_decoder_attention, forward_encoder_extend_attention, forward_encoder_prefill_attention,
};
use crate::metal::engine::GpuDecoderEngine;
use crate::metal::kv_cache::GpuKvCache;
use crate::metal::linear::linear_cached_batched_in_buf;
use crate::metal::moe::{build_expert_jobs, experts_forward_dgq_cpu, experts_forward_gpu_batched};
use crate::metal::router::{GpuRouteScratch, route_gpu_in_batch};
use crate::metal::weights::{GpuDecoderWeightCache, GpuLayerWeightCache};
use crate::model::attention::{AttentionParams, AttentionScratch};
use crate::model::decoder_layer::{DecoderLayerScratch, forward_decoder as cpu_forward_decoder};
use crate::model::kv_cache::LayerKvView;
use crate::model::layer_weights::DecoderLayerWeights;
use crate::model::mask::DecoderAttnMask;
use crate::model::moe::prepare_expert_input;

pub struct GpuDecoderLayerScratch {
    pub cpu: DecoderLayerScratch,
    pub attn: AttentionScratch,
    pub attn_out: Vec<f32>,
    pub residual: Vec<f32>,
    pub normed: Vec<f32>,
    pub dense_out: Vec<f32>,
    pub moe_branch: Vec<f32>,
    pub moe_input: Vec<f32>,
    pub moe_token_indices: Vec<u32>,
    pub moe_batch_out: Vec<f32>,
}

impl GpuDecoderLayerScratch {
    pub fn with_kv_len(
        seq_len: usize,
        cfg: &TextConfig,
        layer: usize,
        kv_cache_len: usize,
    ) -> Result<Self, Error> {
        let hidden = cfg.hidden_size;
        let attn_params = AttentionParams::for_layer(cfg, layer)?;
        Ok(Self {
            cpu: DecoderLayerScratch::with_kv_len(seq_len, cfg, layer, kv_cache_len)?,
            attn: AttentionScratch::with_kv_len_gpu(seq_len, hidden, &attn_params, kv_cache_len),
            attn_out: vec![0.0; seq_len * hidden],
            residual: vec![0.0; seq_len * hidden],
            normed: vec![0.0; seq_len * hidden],
            dense_out: vec![0.0; seq_len * hidden],
            moe_branch: vec![0.0; seq_len * hidden],
            moe_input: vec![0.0; seq_len * hidden],
            moe_token_indices: Vec::new(),
            moe_batch_out: Vec::new(),
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
            &*cached,
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
            &*cached,
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
    let telemetry = engine.batch_telemetry();
    let mut batch = begin_engine_batch(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        telemetry.clone(),
    )?;

    let buf_res = batch.alloc_f32(&scratch.residual)?;
    let buf_attn = batch.alloc_f32(&scratch.attn_out)?;
    let mut buf_stream = bk::rms_norm_rows_gpu_buf(
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
        &engine.f32_nvfp4_linear_pipeline,
        &engine.f32_q8_linear_pipeline,
        &buf_normed,
        &cached.mlp_gate,
        seq_len,
    )?;
    let buf_up = linear_cached_batched_in_buf(
        &mut batch,
        &engine.f32_bf16_linear_pipeline,
        &engine.f32_q4_linear_pipeline,
        &engine.f32_nvfp4_linear_pipeline,
        &engine.f32_q8_linear_pipeline,
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
        &engine.f32_nvfp4_linear_pipeline,
        &engine.f32_q8_linear_pipeline,
        &buf_gate,
        &cached.mlp_down,
        seq_len,
    )?;
    let mut buf_ff = bk::rms_norm_rows_gpu_buf(
        &mut batch,
        &engine.kernels,
        &buf_down,
        cached.post_ff_norm_1.as_slice(),
        seq_len,
        hidden,
        eps,
    )?;

    // Dense Q4 fused kernels; GPU router + grouped MoE (monolithic encoder always GPU MoE on .dgq).
    scratch.dense_out.resize(len, 0.0);
    scratch.moe_input.resize(len, 0.0);
    let mut route_scratch = GpuRouteScratch::new(seq_len, top_k);
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
    batch.register_read(buf_ff.clone(), &mut scratch.dense_out);
    batch.register_read(buf_stream.clone(), &mut scratch.moe_input);
    batch.end()?;
    let routes = route_scratch.into_routes(seq_len, top_k);

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
    if out_len > 0 {
        scratch.moe_batch_out.fill(0.0);
    }

    if !jobs.is_empty() {
        scratch.moe_branch.resize(len, 0.0);
        scratch.moe_branch.fill(0.0);
        if engine.encoder_gpu_moe() {
            let telemetry = engine.batch_telemetry();
            experts_forward_gpu_batched(
                &mut scratch.moe_branch,
                &scratch.moe_input,
                cached.pre_ff_norm_2.as_slice(),
                eps,
                None,
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
                // q6 experts ride the "q4" slots (selected by expert kind);
                // the nvfp4 slots keep their own kind-based pick downstream.
                if expert_cache.expert_gate_up_q4(layer, 0).quant_kind()
                    == crate::dgq::layout::QuantKind::Q6Block
                {
                    &engine.f32_q6_linear_pipeline
                } else {
                    &engine.f32_q4_linear_pipeline
                },
                &engine.f32_nvfp4_linear_pipeline,
                if expert_cache.expert_gate_up_q4(layer, 0).quant_kind()
                    == crate::dgq::layout::QuantKind::Q6Block
                {
                    &engine.f32_q6_linear_grouped_pipeline
                } else {
                    &engine.f32_q4_linear_grouped_pipeline
                },
                &engine.f32_nvfp4_linear_grouped_pipeline,
                telemetry,
            )?;
        } else {
            prepare_expert_input(
                &mut scratch.normed,
                &scratch.moe_input,
                cached.pre_ff_norm_2.as_slice(),
                seq_len,
                hidden,
                eps,
            );
            experts_forward_dgq_cpu(
                &mut scratch.moe_branch,
                &scratch.normed,
                expert_cache,
                layer,
                cfg,
                seq_len,
                &routes,
                &mut scratch.cpu.moe,
            )?;
        }
    }

    let mut batch = begin_engine_batch(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        telemetry.clone(),
    )?;
    buf_ff = batch.alloc_f32(&scratch.dense_out)?;
    buf_stream = batch.alloc_f32(&scratch.moe_input)?;

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
    engine.pool.trim(0);
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
            &engine.f32_nvfp4_linear_pipeline,
            &engine.f32_q8_linear_pipeline,
            &buf_normed,
            &cached.mlp_gate,
            seq_len,
        )?;
        let buf_up = linear_cached_batched_in_buf(
            &mut batch,
            &engine.f32_bf16_linear_pipeline,
            &engine.f32_q4_linear_pipeline,
            &engine.f32_nvfp4_linear_pipeline,
            &engine.f32_q8_linear_pipeline,
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
            &engine.f32_nvfp4_linear_pipeline,
            &engine.f32_q8_linear_pipeline,
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
        if expert_cache.expert_gate_up_q4(layer, 0).quant_kind()
            == crate::dgq::layout::QuantKind::Q6Block
        {
            &engine.f32_q6_linear_pipeline
        } else {
            &engine.f32_q4_linear_pipeline
        },
        &engine.f32_nvfp4_linear_pipeline,
        if expert_cache.expert_gate_up_q4(layer, 0).quant_kind()
            == crate::dgq::layout::QuantKind::Q6Block
        {
            &engine.f32_q6_linear_grouped_pipeline
        } else {
            &engine.f32_q4_linear_grouped_pipeline
        },
        &engine.f32_nvfp4_linear_grouped_pipeline,
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

/// Persistent GPU buffers for the GPU-resident encoder prefill: allocated once
/// per prefill, alive across batch boundaries (batch-local allocs are recycled
/// by the pool at every `batch.end()`, so cross-batch tensors must live here).
pub struct PrefillResidentBufs {
    /// Layer input/output hidden-state ping-pong, `[seq, hidden]` f32 each.
    pub hidden:
        [objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>; 2],
    /// Post-attention residual stream (the MoE input), `[seq, hidden]`.
    pub stream: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    /// Dense-FF branch after post_ff_norm_1, `[seq, hidden]`.
    pub ff: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    len: usize,
}

impl PrefillResidentBufs {
    pub fn new(
        engine: &mut GpuDecoderEngine,
        seq_len: usize,
        hidden: usize,
    ) -> Result<Self, Error> {
        let len = seq_len * hidden;
        let bytes = len * 4;
        let mut alloc = || {
            engine
                .pool
                .allocate(&engine.ctx.device, bytes)
                .ok_or(Error::Gpu("prefill resident buffer alloc failed"))
        };
        Ok(Self {
            hidden: [alloc()?, alloc()?],
            stream: alloc()?,
            ff: alloc()?,
            len,
        })
    }

    /// Upload the embedding output into the layer-0 input buffer.
    pub fn upload_hidden(&self, idx: usize, data: &[f32]) -> Result<(), Error> {
        if data.len() != self.len {
            return Err(Error::Runtime("prefill resident hidden size mismatch"));
        }
        crate::metal::buffer::BufferPool::write_f32(&self.hidden[idx], data);
        Ok(())
    }

    /// Return the buffers to the pool at end of prefill.
    pub fn release(self, engine: &mut GpuDecoderEngine) {
        let bytes = self.len * 4;
        let [h0, h1] = self.hidden;
        for buf in [h0, h1, self.stream, self.ff] {
            engine.pool.release(bytes, buf);
        }
    }
}

/// GPU-resident causal prefill/extend layer: identical kernels and dispatch
/// order to `forward_encoder_prefill`/`forward_encoder_extend` +
/// `forward_layer_ff_dgq_gpu` (bit-identical KV and hidden), but the hidden
/// state never leaves the GPU. Two syncs per layer — the router flush
/// (~32 KB) and the MoE/combine batch end (no readback) — instead of four
/// syncs with ~3.5 MiB of hidden-state round-trips.
///
/// `kv_cache_len == 0` is a fresh causal prefill (CausalSliding mask);
/// `kv_cache_len > 0` extends an existing prefix (EncoderExtend mask —
/// `gpu_kv.kv_len` must equal it, and `positions` start there).
///
/// Input hidden is `bufs.hidden[in_idx]`; output is written to
/// `bufs.hidden[1 - in_idx]`. `.dgq` + GPU-MoE only (the production configuration).
#[allow(clippy::too_many_arguments)]
pub fn forward_encoder_prefill_resident(
    bufs: &PrefillResidentBufs,
    in_idx: usize,
    cached: &GpuLayerWeightCache,
    expert_cache: &GpuDecoderWeightCache,
    cfg: &TextConfig,
    layer: usize,
    seq_len: usize,
    positions: &[i64],
    kv_cache_len: usize,
    // Tiny reusable scratch (the full GpuDecoderLayerScratch allocates CPU
    // attention/MoE buffers this GPU-resident path never touches).
    rope_freqs: &mut Vec<f32>,
    token_indices: &mut Vec<u32>,
    engine: &mut GpuDecoderEngine,
    gpu_kv: &GpuKvCache,
) -> Result<(), Error> {
    use crate::metal::attention_batch::dispatch_copy_f32_to_buf;
    use crate::metal::decoder_attention::encode_fused_gpu_kv_attention_buf;
    use crate::metal::moe::{build_scatter_lists, experts_forward_gpu_grouped_in_batch_buf};
    use crate::model::attention::GqaMask;
    use crate::shaders::cpu::{compute_rope_freqs, rope_kind_for_layer};

    let layer_entry = std::time::Instant::now();
    let hidden_size = cfg.hidden_size;
    let eps = cfg.rms_norm_eps as f32;
    let len = seq_len * hidden_size;
    let experts = cfg.num_experts;
    let top_k = cfg.top_k_experts;
    let params = AttentionParams::for_layer(cfg, layer)?;

    assert_eq!(gpu_kv.kv_len, kv_cache_len);
    let rope_kind = rope_kind_for_layer(cfg, layer).ok_or(Error::Format("rope kind"))?;
    rope_freqs.clear();
    rope_freqs.resize(seq_len * params.rotary_dim, 0.0);
    compute_rope_freqs(rope_freqs, positions, rope_kind);
    let mask = if kv_cache_len == 0 {
        GqaMask::CausalSliding
    } else {
        GqaMask::EncoderExtend {
            kv_cache_len,
            positions,
        }
    };

    let k_canvas_off = gpu_kv.canvas_k_elem_offset(layer)?;
    let kv_suffix_elems = seq_len * params.n_kv_heads * params.head_dim;
    let kv_suffix_byte_off = k_canvas_off * std::mem::size_of::<f32>();
    let (k_buf, v_buf) = gpu_kv.layer_buffers(layer)?;

    let telemetry = engine.batch_telemetry();

    // ---- Batch A: attention + dense FF + router; readback = routes only ----
    let mut batch = begin_engine_batch(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        telemetry.clone(),
    )?;
    let buf_hidden = &bufs.hidden[in_idx];
    let buf_attn = encode_fused_gpu_kv_attention_buf(
        &mut batch,
        &engine.kernels,
        &engine.attention,
        &engine.sampler_kernels.copy_f32,
        &engine.f32_bf16_linear_pipeline,
        &engine.f32_q4_linear_pipeline,
        &engine.f32_nvfp4_linear_pipeline,
        &engine.f32_q8_linear_pipeline,
        buf_hidden,
        cached,
        seq_len,
        hidden_size,
        eps,
        &params,
        rope_freqs,
        k_buf,
        v_buf,
        k_canvas_off,
        kv_suffix_byte_off,
        kv_suffix_elems,
        kv_cache_len + seq_len,
        mask,
        &cached.o_proj,
    )?;
    let t_attn_encoded = std::time::Instant::now();
    // Dense FF — mirrors forward_layer_ff_dgq_gpu's first batch exactly.
    let buf_stream = bk::rms_norm_rows_gpu_buf(
        &mut batch,
        &engine.kernels,
        &buf_attn,
        cached.post_attn_norm.as_slice(),
        seq_len,
        hidden_size,
        eps,
    )?;
    bk::vec_add_gpu_bufs(&mut batch, &engine.kernels, &buf_stream, buf_hidden, len)?;
    let buf_normed = bk::rms_norm_rows_gpu_buf(
        &mut batch,
        &engine.kernels,
        &buf_stream,
        cached.pre_ff_norm.as_slice(),
        seq_len,
        hidden_size,
        eps,
    )?;
    let buf_gate = linear_cached_batched_in_buf(
        &mut batch,
        &engine.f32_bf16_linear_pipeline,
        &engine.f32_q4_linear_pipeline,
        &engine.f32_nvfp4_linear_pipeline,
        &engine.f32_q8_linear_pipeline,
        &buf_normed,
        &cached.mlp_gate,
        seq_len,
    )?;
    let buf_up = linear_cached_batched_in_buf(
        &mut batch,
        &engine.f32_bf16_linear_pipeline,
        &engine.f32_q4_linear_pipeline,
        &engine.f32_nvfp4_linear_pipeline,
        &engine.f32_q8_linear_pipeline,
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
        &engine.f32_nvfp4_linear_pipeline,
        &engine.f32_q8_linear_pipeline,
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
        hidden_size,
        eps,
    )?;
    let t_dense_encoded = std::time::Instant::now();
    let mut route_scratch = GpuRouteScratch::new(seq_len, top_k);
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
    // Persist the two tensors batch B needs (batch-local bufs are recycled at end()).
    dispatch_copy_f32_to_buf(
        &mut batch,
        &engine.sampler_kernels.copy_f32,
        &buf_stream,
        &bufs.stream,
        0,
        len,
    );
    dispatch_copy_f32_to_buf(
        &mut batch,
        &engine.sampler_kernels.copy_f32,
        &buf_ff,
        &bufs.ff,
        0,
        len,
    );
    let phase_a_started = std::time::Instant::now();
    batch.end()?;
    let phase_a = phase_a_started.elapsed();

    let routes = route_scratch.into_routes(seq_len, top_k);
    let jobs = build_expert_jobs(&routes, experts);
    if let Some(cell) = &telemetry {
        cell.borrow_mut()
            .record_expert_layer(layer, jobs.len(), cfg);
    }

    // ---- Batch B: MoE + scatter + combine + layer_scalar; no readback ----
    let mut batch = begin_engine_batch(
        &engine.ctx.queue,
        &mut engine.pool,
        &engine.ctx.device,
        telemetry,
    )?;
    let buf_arena = experts_forward_gpu_grouped_in_batch_buf(
        &mut batch,
        &bufs.stream,
        cached.pre_ff_norm_2.as_slice(),
        eps,
        expert_cache,
        layer,
        cfg,
        seq_len,
        &jobs,
        token_indices,
        &engine.kernels,
        if expert_cache.expert_gate_up_q4(layer, 0).quant_kind()
            == crate::dgq::layout::QuantKind::Q6Block
        {
            &engine.f32_q6_linear_grouped_pipeline
        } else {
            &engine.f32_q4_linear_grouped_pipeline
        },
        &engine.f32_nvfp4_linear_grouped_pipeline,
    )?;
    let buf_moe = match buf_arena {
        Some(arena) => {
            let (rows, weights) = build_scatter_lists(&jobs, seq_len, top_k);
            bk::scatter_rows_weighted_gpu(
                &mut batch,
                &engine.kernels,
                &arena,
                &rows,
                &weights,
                seq_len,
                hidden_size,
                top_k,
            )?
        }
        // No routed work: the classic path's moe_branch is all-zero.
        None => batch.alloc_f32_out(len)?,
    };
    let buf_moe_n = bk::rms_norm_rows_gpu_buf(
        &mut batch,
        &engine.kernels,
        &buf_moe,
        cached.post_ff_norm_2.as_slice(),
        seq_len,
        hidden_size,
        eps,
    )?;
    bk::vec_add_gpu_bufs(&mut batch, &engine.kernels, &bufs.ff, &buf_moe_n, len)?;
    let buf_out = bk::rms_norm_rows_gpu_buf(
        &mut batch,
        &engine.kernels,
        &bufs.ff,
        cached.post_ff_norm.as_slice(),
        seq_len,
        hidden_size,
        eps,
    )?;
    bk::vec_add_gpu_bufs(&mut batch, &engine.kernels, &buf_out, &bufs.stream, len)?;
    // GPU analog of the CPU `out[i] *= layer_scalar` epilogue (same f32 multiply).
    bk::vec_scale_gpu_buf(
        &mut batch,
        &engine.kernels,
        &buf_out,
        len,
        cached.layer_scalar,
    )?;
    dispatch_copy_f32_to_buf(
        &mut batch,
        &engine.sampler_kernels.copy_f32,
        &buf_out,
        &bufs.hidden[1 - in_idx],
        0,
        len,
    );
    let phase_b_started = std::time::Instant::now();
    batch.end()?;
    if crate::flags::prefill_profile_enabled() {
        eprintln!(
            "  prefill layer {layer}: encAttn={:.1?} encDense={:.1?} encRoute={:.1?} waitA={phase_a:.1?} encB={:.1?} waitB={:.1?} jobs={}",
            t_attn_encoded - layer_entry,
            t_dense_encoded - t_attn_encoded,
            phase_a_started - t_dense_encoded,
            phase_b_started - (phase_a_started + phase_a),
            phase_b_started.elapsed(),
            jobs.len(),
        );
    }
    Ok(())
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
    let _hidden = cfg.hidden_size;
    let _eps = cfg.rms_norm_eps as f32;

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
