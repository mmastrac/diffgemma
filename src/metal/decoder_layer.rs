use crate::config::TextConfig;
use crate::metal::batched_kernels::{self as bk};
use crate::metal::batch::begin_engine_batch;
use crate::metal::engine::GpuDecoderEngine;
use crate::metal::linear::linear_cached_batched_in_buf;
use crate::metal::moe::{
    build_expert_jobs, experts_forward_dgq_cpu, experts_forward_gpu_batched,
    experts_forward_gpu_grouped_in_batch, experts_forward_gpu_per_job,
    scatter_weighted_expert_outputs,
};
use crate::model::moe::{prepare_expert_input, route_with_cached_weights};
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
        if use_mps_q4 {
            Some((
                &mut engine.mps_matmul,
                &engine.dequant_q4_matrix_pipeline,
                &engine.dequant_nvfp4_matrix_pipeline,
            ))
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
        &engine.f32_nvfp4_linear_pipeline,
        if use_mps_q4 {
            Some((
                &mut engine.mps_matmul,
                &engine.dequant_q4_matrix_pipeline,
                &engine.dequant_nvfp4_matrix_pipeline,
            ))
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
        &engine.f32_nvfp4_linear_pipeline,
        if use_mps_q4 {
            Some((
                &mut engine.mps_matmul,
                &engine.dequant_q4_matrix_pipeline,
                &engine.dequant_nvfp4_matrix_pipeline,
            ))
        } else {
            None
        },
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

    // Dense Q4 may use MPS matmul; router stays CPU (small readback). MoE uses grouped GPU
    // by default (`gemm_linear_grouped`); opt out with `DGQ_ENCODER_GPU_MOE=0`.
    scratch.dense_out.resize(len, 0.0);
    scratch.moe_input.resize(len, 0.0);
    batch.register_read(buf_ff.clone(), &mut scratch.dense_out);
    batch.register_read(buf_stream.clone(), &mut scratch.moe_input);
    batch.end()?;
    let routes = route_with_cached_weights(
        &scratch.moe_input,
        cached.router_proj.as_slice(),
        cached.router_scale.as_slice(),
        cached.per_expert_scale.as_slice(),
        cfg,
        seq_len,
        &mut scratch.cpu.moe,
    )?;
    let mut batch = None;

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
                &engine.f32_q4_linear_pipeline,
                &engine.f32_nvfp4_linear_pipeline,
                &engine.f32_q4_linear_grouped_pipeline,
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
        batch = Some(begin_engine_batch(
            &engine.ctx.queue,
            &mut engine.pool,
            &engine.ctx.device,
            telemetry.clone(),
        )?);
        buf_ff = batch.as_mut().expect("combine batch").alloc_f32(&scratch.dense_out)?;
        buf_stream = batch
            .as_mut()
            .expect("combine batch")
            .alloc_f32(&scratch.moe_input)?;
    } else {
        batch = Some(begin_engine_batch(
            &engine.ctx.queue,
            &mut engine.pool,
            &engine.ctx.device,
            telemetry.clone(),
        )?);
        buf_ff = batch.as_mut().expect("combine batch").alloc_f32(&scratch.dense_out)?;
        buf_stream = batch
            .as_mut()
            .expect("combine batch")
            .alloc_f32(&scratch.moe_input)?;
    }

    let mut batch = batch.expect("ff combine batch");

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
            None,
            &buf_normed,
            &cached.mlp_gate,
            seq_len,
        )?;
        let buf_up = linear_cached_batched_in_buf(
            &mut batch,
            &engine.f32_bf16_linear_pipeline,
            &engine.f32_q4_linear_pipeline,
            &engine.f32_nvfp4_linear_pipeline,
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
            &engine.f32_nvfp4_linear_pipeline,
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
        &engine.f32_nvfp4_linear_pipeline,
        &engine.f32_q4_linear_grouped_pipeline,
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

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod determinism_tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::metal::decoder_attention::forward_decoder_attention;
    use crate::metal::decoder::load_weight_cache;
    use crate::metal::router::{pack_routes, route_gpu_in_batch, GpuRouteScratch};
    use crate::metal::batch::begin_engine_batch;
    use crate::metal::batched_kernels as bk;
    use crate::metal::linear::linear_cached_batched_in_buf;
    use crate::metal::encoder_extend::prefill_gpu;
    use crate::metal::GpuDecoderScratch;
    use crate::model::embed::embed_tokens_from_store;
    use crate::model::encoder::EncoderPrefillInput;
    use crate::model::encoder::EncoderScratch;
    use crate::model::kv_cache::{KvCache, LayerKvView};
    use crate::model::mask::DecoderAttnMask;
    use crate::sample::{initialize_canvas, Rng};
    use crate::weights::WeightStore;

    fn dgq_dir() -> Option<std::path::PathBuf> {
        let dir = std::path::Path::new("/tmp/quantized-weights");
        if dir.join("model.dgq.json").exists() {
            Some(dir.to_path_buf())
        } else {
            None
        }
    }

    struct LayerTestCtx {
        store: WeightStore,
        cfg: ModelConfig,
        weights: crate::metal::GpuDecoderWeightCache,
        engine: GpuDecoderEngine,
        dec: GpuDecoderScratch,
        kv: KvCache,
        hidden_in: Vec<f32>,
        canvas_tokens: Vec<u32>,
        mask: DecoderAttnMask,
        positions: Vec<i64>,
    }

    impl LayerTestCtx {
        fn new(prefill_layers: usize) -> Self {
            let dir = dgq_dir().expect("dgq weights");
            let store = WeightStore::open(&dir).expect("open");
            let cfg = ModelConfig::load(&dir).expect("cfg");
            let text = &cfg.text_config;
            let canvas = cfg.canvas_length;
            let hidden = text.hidden_size;
            let max_kv = 1 + 256;
            let mut weights =
                load_weight_cache(&store, text, canvas, max_kv).expect("cache");
            let mut engine = GpuDecoderEngine::new().expect("engine");
            engine.set_use_mps_q4(false);
            let mut enc = EncoderScratch::new(canvas.max(1), &cfg);
            let mut dec = GpuDecoderScratch::new(canvas, &cfg);
            dec.use_gpu_sampler = false;
            let kv = prefill_gpu(
                &store,
                &cfg,
                &EncoderPrefillInput {
                    token_ids: &[9259u32],
                    position_offset: 0,
                },
                &mut enc,
                &mut dec,
                &mut weights,
                &mut engine,
                max_kv,
                canvas,
                Some(prefill_layers),
            )
            .expect("prefill");
            let mut rng = Rng::new(42);
            let canvas_tokens = initialize_canvas(canvas, text.vocab_size, &mut rng);
            let mut hidden_in = vec![0.0f32; canvas * hidden];
            embed_tokens_from_store(
                &store,
                &mut hidden_in,
                &canvas_tokens,
                hidden,
                (hidden as f32).sqrt(),
            )
            .expect("embed");
            let mask = DecoderAttnMask::all_valid(canvas, kv.kv_len);
            let positions: Vec<i64> =
                (kv.kv_len as i64..kv.kv_len as i64 + canvas as i64).collect();
            Self {
                store,
                cfg,
                weights,
                engine,
                dec,
                kv,
                hidden_in,
                canvas_tokens,
                mask,
                positions,
            }
        }

        fn run_layer(&mut self, layer: usize, hidden: &[f32], scratch: &mut GpuDecoderLayerScratch) -> Vec<f32> {
            let text = &self.cfg.text_config;
            let canvas = self.cfg.canvas_length;
            let hidden_dim = text.hidden_size;
            self.weights
                .ensure_layer(
                    &self.store,
                    text,
                    layer,
                    &self.engine.ctx.device,
                    &mut self.engine.pool,
                )
                .expect("layer weights");
            let cached = self.weights.layer_ref(layer);
            let mut out = vec![0.0f32; canvas * hidden_dim];
            let kv_view = LayerKvView::from_layer(
                self.kv.layer(layer).expect("kv layer"),
                self.kv.kv_len,
            );
            forward_decoder(
                &mut out,
                hidden,
                None,
                &cached,
                &self.weights,
                text,
                layer,
                canvas,
                &self.positions,
                kv_view,
                Some(&self.mask),
                scratch,
                &mut self.engine,
                self.dec.gpu_kv.as_ref(),
            )
            .expect("layer forward");
            self.weights.release_layer();
            out
        }
    }

    fn first_f32_diff(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
        a.iter()
            .zip(b.iter())
            .enumerate()
            .find_map(|(i, (&x, &y))| if x.to_bits() != y.to_bits() { Some((i, x, y)) } else { None })
    }

    fn assert_f32_repeatable<F>(label: &str, mut run: F)
    where
        F: FnMut() -> Vec<f32>,
    {
        let a = run();
        let b = run();
        if a != b {
            let detail = first_f32_diff(&a, &b)
                .map(|(i, x, y)| format!("idx {i}: {x} vs {y}"))
                .unwrap_or_default();
            panic!("{label}: {detail}");
        }
    }

    fn run_dense_ff_dgq(
        ctx: &mut LayerTestCtx,
        scratch: &mut GpuDecoderLayerScratch,
        layer: usize,
        canvas_len: usize,
        hidden_dim: usize,
        eps: f32,
    ) -> Vec<f32> {
        let text = &ctx.cfg.text_config;
        ctx.weights
            .ensure_layer(&ctx.store, text, layer, &ctx.engine.ctx.device, &mut ctx.engine.pool)
            .expect("layer weights");
        let cached = ctx.weights.layer_ref(layer);
        let seq_len = canvas_len;
        let hidden = hidden_dim;
        let len = seq_len * hidden;
        let telemetry = ctx.engine.batch_telemetry();
        let mut batch = begin_engine_batch(
            &ctx.engine.ctx.queue,
            &mut ctx.engine.pool,
            &ctx.engine.ctx.device,
            telemetry,
        )
        .expect("batch");
        let buf_res = batch.alloc_f32(&scratch.residual).expect("res");
        let buf_attn = batch.alloc_f32(&scratch.attn_out).expect("attn");
        let buf_stream = bk::rms_norm_rows_gpu_buf(
            &mut batch,
            &ctx.engine.kernels,
            &buf_attn,
            cached.post_attn_norm.as_slice(),
            seq_len,
            hidden,
            eps,
        )
        .expect("post_attn_norm");
        bk::vec_add_gpu_bufs(&mut batch, &ctx.engine.kernels, &buf_stream, &buf_res, len).expect("add");
        let buf_normed = bk::rms_norm_rows_gpu_buf(
            &mut batch,
            &ctx.engine.kernels,
            &buf_stream,
            cached.pre_ff_norm.as_slice(),
            seq_len,
            hidden,
            eps,
        )
        .expect("pre_ff_norm");
        let buf_gate = linear_cached_batched_in_buf(
            &mut batch,
            &ctx.engine.f32_bf16_linear_pipeline,
            &ctx.engine.f32_q4_linear_pipeline,
            &ctx.engine.f32_nvfp4_linear_pipeline,
            None,
            &buf_normed,
            &cached.mlp_gate,
            seq_len,
        )
        .expect("gate");
        let buf_up = linear_cached_batched_in_buf(
            &mut batch,
            &ctx.engine.f32_bf16_linear_pipeline,
            &ctx.engine.f32_q4_linear_pipeline,
            &ctx.engine.f32_nvfp4_linear_pipeline,
            None,
            &buf_normed,
            &cached.mlp_up,
            seq_len,
        )
        .expect("up");
        let act_len = seq_len * cached.mlp_gate.out_dim();
        bk::gelu_pytorch_tanh_gpu_buf(&mut batch, &ctx.engine.kernels, &buf_gate, act_len).expect("gelu");
        bk::swiglu_mul_gpu_bufs(&mut batch, &ctx.engine.kernels, &buf_gate, &buf_up, act_len).expect("swiglu");
        let buf_down = linear_cached_batched_in_buf(
            &mut batch,
            &ctx.engine.f32_bf16_linear_pipeline,
            &ctx.engine.f32_q4_linear_pipeline,
            &ctx.engine.f32_nvfp4_linear_pipeline,
            None,
            &buf_gate,
            &cached.mlp_down,
            seq_len,
        )
        .expect("down");
        let buf_ff = bk::rms_norm_rows_gpu_buf(
            &mut batch,
            &ctx.engine.kernels,
            &buf_down,
            cached.post_ff_norm_1.as_slice(),
            seq_len,
            hidden,
            eps,
        )
        .expect("post_ff_norm_1");
        let mut out = vec![0.0f32; len];
        batch.register_read(buf_ff, &mut out);
        batch.end().expect("end");
        ctx.weights.release_layer();
        out
    }

    #[test]
    fn dgq_layer0_attention_repeatable() {
        if dgq_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let mut ctx = LayerTestCtx::new(2);
        let canvas_len = ctx.cfg.canvas_length;
        let kv_len = ctx.kv.kv_len;
        let hidden_dim = ctx.cfg.text_config.hidden_size;
        let mut scratch = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            0,
            kv_len,
        )
        .expect("scratch");
        let hidden_in = ctx.hidden_in.clone();
        let mut a = vec![0.0f32; canvas_len * hidden_dim];
        let mut b = vec![0.0f32; canvas_len * hidden_dim];
        ctx.weights
            .ensure_layer(&ctx.store, &ctx.cfg.text_config, 0, &ctx.engine.ctx.device, &mut ctx.engine.pool)
            .expect("layer0");
        let cached = ctx.weights.layer_ref(0);
        let kv_view = LayerKvView::from_layer(ctx.kv.layer(0).expect("kv"), kv_len);
        ctx.engine.set_use_mps_q4(false);
        forward_decoder_attention(
            &mut a,
            &hidden_in,
            &cached,
            &ctx.cfg.text_config,
            0,
            canvas_len,
            &ctx.positions,
            kv_view,
            Some(&ctx.mask),
            &mut scratch.attn,
            &mut ctx.engine,
            ctx.dec.gpu_kv.as_ref(),
            false,
        )
        .expect("attn a");
        forward_decoder_attention(
            &mut b,
            &hidden_in,
            &cached,
            &ctx.cfg.text_config,
            0,
            canvas_len,
            &ctx.positions,
            kv_view,
            Some(&ctx.mask),
            &mut scratch.attn,
            &mut ctx.engine,
            ctx.dec.gpu_kv.as_ref(),
            false,
        )
        .expect("attn b");
        ctx.weights.release_layer();
        if a != b {
            let detail = first_f32_diff(&a, &b)
                .map(|(i, x, y)| format!("idx {i}: {x} vs {y}"))
                .unwrap_or_default();
            panic!("layer 0 attention drift: {detail}");
        }
    }

    fn stream_residual_for_ff(
        ctx: &mut LayerTestCtx,
        scratch: &mut GpuDecoderLayerScratch,
        hidden_in: &[f32],
        layer: usize,
    ) -> Result<Vec<f32>, Error> {
        let canvas_len = ctx.cfg.canvas_length;
        let hidden_dim = ctx.cfg.text_config.hidden_size;
        let text = &ctx.cfg.text_config;
        let kv_len = ctx.kv.kv_len;
        ctx.weights
            .ensure_layer(&ctx.store, text, layer, &ctx.engine.ctx.device, &mut ctx.engine.pool)?;
        let cached = ctx.weights.layer_ref(layer);
        let kv_view = LayerKvView::from_layer(ctx.kv.layer(layer).expect("kv"), kv_len);
        forward_decoder_attention(
            &mut scratch.attn_out,
            hidden_in,
            &cached,
            text,
            layer,
            canvas_len,
            &ctx.positions,
            kv_view,
            Some(&ctx.mask),
            &mut scratch.attn,
            &mut ctx.engine,
            ctx.dec.gpu_kv.as_ref(),
            false,
        )?;
        let eps = text.rms_norm_eps as f32;
        let len = canvas_len * hidden_dim;
        let telemetry = ctx.engine.batch_telemetry();
        let mut batch = begin_engine_batch(
            &ctx.engine.ctx.queue,
            &mut ctx.engine.pool,
            &ctx.engine.ctx.device,
            telemetry,
        )?;
        let buf_res = batch.alloc_f32(hidden_in)?;
        let buf_attn = batch.alloc_f32(&scratch.attn_out)?;
        let buf_stream = crate::metal::batched_kernels::rms_norm_rows_gpu_buf(
            &mut batch,
            &ctx.engine.kernels,
            &buf_attn,
            cached.post_attn_norm.as_slice(),
            canvas_len,
            hidden_dim,
            eps,
        )?;
        crate::metal::batched_kernels::vec_add_gpu_bufs(
            &mut batch,
            &ctx.engine.kernels,
            &buf_stream,
            &buf_res,
            len,
        )?;
        let mut stream = vec![0.0f32; len];
        batch.register_read(buf_stream, &mut stream);
        batch.end()?;
        ctx.weights.release_layer();
        Ok(stream)
    }

    #[test]
    fn dgq_layer0_route_repeatable() {
        if dgq_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let mut ctx = LayerTestCtx::new(2);
        let canvas_len = ctx.cfg.canvas_length;
        let top_k = ctx.cfg.text_config.top_k_experts;
        let mut scratch = GpuDecoderLayerScratch::with_kv_len(canvas_len, &ctx.cfg.text_config, 0, ctx.kv.kv_len)
            .expect("scratch");
        let hidden_in = ctx.hidden_in.clone();
        let stream = stream_residual_for_ff(&mut ctx, &mut scratch, &hidden_in, 0).expect("stream");
        let cached = ctx.weights.layer_ref(0);
        let mut first: Option<Vec<crate::model::moe::RouteResult>> = None;
        for _ in 0..2 {
            let telemetry = ctx.engine.batch_telemetry();
            let mut batch = begin_engine_batch(
                &ctx.engine.ctx.queue,
                &mut ctx.engine.pool,
                &ctx.engine.ctx.device,
                telemetry,
            )
            .expect("batch");
            let buf_stream = batch.alloc_f32(&stream).expect("buf");
            let mut route_scratch = GpuRouteScratch::new(canvas_len, top_k);
            route_gpu_in_batch(
                &mut batch,
                &ctx.engine.kernels,
                &ctx.engine.f32_f32_linear_pipeline.pipeline,
                &buf_stream,
                &cached,
                &ctx.cfg.text_config,
                canvas_len,
                &mut route_scratch,
            )
            .expect("route");
            batch.end().expect("end");
            let packed = pack_routes(&route_scratch.indices, &route_scratch.weights, canvas_len, top_k);
            if let Some(ref prev) = first {
                assert_eq!(packed.len(), prev.len(), "layer 0 route count drift");
                for (i, (ra, rb)) in packed.iter().zip(prev.iter()).enumerate() {
                    assert_eq!(ra.indices, rb.indices, "layer 0 route idx drift at {i}");
                    let max_w_err = ra
                        .weights
                        .iter()
                        .zip(rb.weights.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    assert!(
                        max_w_err < 1e-5,
                        "layer 0 route weight drift at {i}: max_err={max_w_err}"
                    );
                }
            } else {
                first = Some(packed);
            }
        }
        ctx.weights.release_layer();
    }

    #[test]
    fn dgq_layer0_dense_ff_repeatable() {
        if dgq_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let mut ctx = LayerTestCtx::new(2);
        let canvas_len = ctx.cfg.canvas_length;
        let hidden_dim = ctx.cfg.text_config.hidden_size;
        let eps = ctx.cfg.text_config.rms_norm_eps as f32;
        let mut scratch = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            0,
            ctx.kv.kv_len,
        )
        .expect("scratch");
        let hidden_in = ctx.hidden_in.clone();
        let kv_view = LayerKvView::from_layer(ctx.kv.layer(0).expect("kv"), ctx.kv.kv_len);
        ctx.weights
            .ensure_layer(&ctx.store, &ctx.cfg.text_config, 0, &ctx.engine.ctx.device, &mut ctx.engine.pool)
            .expect("layer0");
        {
            let cached = ctx.weights.layer_ref(0);
            forward_decoder_attention(
                &mut scratch.attn_out,
                &hidden_in,
                &*cached,
                &ctx.cfg.text_config,
                0,
                canvas_len,
                &ctx.positions,
                kv_view,
                Some(&ctx.mask),
                &mut scratch.attn,
                &mut ctx.engine,
                ctx.dec.gpu_kv.as_ref(),
                false,
            )
            .expect("attn");
        }
        scratch.residual.copy_from_slice(&hidden_in);
        ctx.weights.release_layer();
        assert_f32_repeatable("layer 0 dense FF drift", || {
            run_dense_ff_dgq(&mut ctx, &mut scratch, 0, canvas_len, hidden_dim, eps)
        });
    }

    #[test]
    fn dgq_layer0_ff_repeatable() {
        if dgq_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let mut ctx = LayerTestCtx::new(2);
        let canvas_len = ctx.cfg.canvas_length;
        let kv_len = ctx.kv.kv_len;
        let hidden_dim = ctx.cfg.text_config.hidden_size;
        let eps = ctx.cfg.text_config.rms_norm_eps as f32;
        let mut scratch = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            0,
            kv_len,
        )
        .expect("scratch");
        let hidden_in = ctx.hidden_in.clone();
        let kv_view = LayerKvView::from_layer(ctx.kv.layer(0).expect("kv"), kv_len);
        ctx.engine.set_use_mps_q4(false);
        ctx.weights
            .ensure_layer(
                &ctx.store,
                &ctx.cfg.text_config,
                0,
                &ctx.engine.ctx.device,
                &mut ctx.engine.pool,
            )
            .expect("layer0");
        {
            let cached = ctx.weights.layer_ref(0);
            forward_decoder_attention(
                &mut scratch.attn_out,
                &hidden_in,
                &*cached,
                &ctx.cfg.text_config,
                0,
                canvas_len,
                &ctx.positions,
                kv_view,
                Some(&ctx.mask),
                &mut scratch.attn,
                &mut ctx.engine,
                ctx.dec.gpu_kv.as_ref(),
                false,
            )
            .expect("attn");
        }
        scratch.residual.copy_from_slice(&hidden_in);
        ctx.weights.release_layer();
        assert_f32_repeatable("layer 0 FF drift", || {
            ctx.weights
                .ensure_layer(
                    &ctx.store,
                    &ctx.cfg.text_config,
                    0,
                    &ctx.engine.ctx.device,
                    &mut ctx.engine.pool,
                )
                .expect("layer0");
            let cached = ctx.weights.layer_ref(0);
            let mut out = vec![0.0f32; canvas_len * hidden_dim];
            forward_layer_ff_dgq_gpu(
                &mut out,
                &*cached,
                &ctx.weights,
                &ctx.cfg.text_config,
                0,
                canvas_len,
                hidden_dim,
                eps,
                &mut scratch,
                &mut ctx.engine,
            )
            .expect("ff");
            ctx.weights.release_layer();
            out
        });
    }

    #[test]
    fn dgq_layer1_ff_repeatable() {
        if dgq_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let mut ctx = LayerTestCtx::new(2);
        let canvas_len = ctx.cfg.canvas_length;
        let kv_len = ctx.kv.kv_len;
        let hidden_dim = ctx.cfg.text_config.hidden_size;
        let eps = ctx.cfg.text_config.rms_norm_eps as f32;
        let mut scratch0 = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            0,
            kv_len,
        )
        .expect("scratch0");
        let hidden_in = ctx.hidden_in.clone();
        let h0 = ctx.run_layer(0, &hidden_in, &mut scratch0);
        let mut scratch1 = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            1,
            kv_len,
        )
        .expect("scratch1");
        ctx.engine.set_use_mps_q4(false);
        ctx.weights
            .ensure_layer(
                &ctx.store,
                &ctx.cfg.text_config,
                1,
                &ctx.engine.ctx.device,
                &mut ctx.engine.pool,
            )
            .expect("layer1");
        {
            let cached = ctx.weights.layer_ref(1);
            let kv_view = LayerKvView::from_layer(ctx.kv.layer(1).expect("kv"), kv_len);
            forward_decoder_attention(
                &mut scratch1.attn_out,
                &h0,
                &*cached,
                &ctx.cfg.text_config,
                1,
                canvas_len,
                &ctx.positions,
                kv_view,
                Some(&ctx.mask),
                &mut scratch1.attn,
                &mut ctx.engine,
                ctx.dec.gpu_kv.as_ref(),
                false,
            )
            .expect("attn");
        }
        scratch1.residual.copy_from_slice(&h0);
        let attn_out_fixed = scratch1.attn_out.clone();
        let residual_fixed = scratch1.residual.clone();
        ctx.weights.release_layer();
        ctx.engine.pool.clear();
        assert_f32_repeatable("layer 1 FF drift", || {
            scratch1.attn_out.copy_from_slice(&attn_out_fixed);
            scratch1.residual.copy_from_slice(&residual_fixed);
            ctx.weights
                .ensure_layer(
                    &ctx.store,
                    &ctx.cfg.text_config,
                    1,
                    &ctx.engine.ctx.device,
                    &mut ctx.engine.pool,
                )
                .expect("layer1");
            let cached = ctx.weights.layer_ref(1);
            let mut out = vec![0.0f32; canvas_len * hidden_dim];
            forward_layer_ff_dgq_gpu(
                &mut out,
                &*cached,
                &ctx.weights,
                &ctx.cfg.text_config,
                1,
                canvas_len,
                hidden_dim,
                eps,
                &mut scratch1,
                &mut ctx.engine,
            )
            .expect("ff");
            ctx.weights.release_layer();
            out
        });
    }

    #[test]
    fn dgq_layer2_ff_repeatable() {
        if dgq_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let mut ctx = LayerTestCtx::new(3);
        let canvas_len = ctx.cfg.canvas_length;
        let kv_len = ctx.kv.kv_len;
        let hidden_dim = ctx.cfg.text_config.hidden_size;
        let eps = ctx.cfg.text_config.rms_norm_eps as f32;
        let mut scratch0 = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            0,
            kv_len,
        )
        .expect("scratch0");
        let hidden_in = ctx.hidden_in.clone();
        let h0 = ctx.run_layer(0, &hidden_in, &mut scratch0);
        let mut scratch1 = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            1,
            kv_len,
        )
        .expect("scratch1");
        let h1 = ctx.run_layer(1, &h0, &mut scratch1);
        let mut scratch2 = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            2,
            kv_len,
        )
        .expect("scratch2");
        ctx.engine.set_use_mps_q4(false);
        ctx.weights
            .ensure_layer(
                &ctx.store,
                &ctx.cfg.text_config,
                2,
                &ctx.engine.ctx.device,
                &mut ctx.engine.pool,
            )
            .expect("layer2");
        {
            let cached = ctx.weights.layer_ref(2);
            let kv_view = LayerKvView::from_layer(ctx.kv.layer(2).expect("kv"), kv_len);
            forward_decoder_attention(
                &mut scratch2.attn_out,
                &h1,
                &*cached,
                &ctx.cfg.text_config,
                2,
                canvas_len,
                &ctx.positions,
                kv_view,
                Some(&ctx.mask),
                &mut scratch2.attn,
                &mut ctx.engine,
                ctx.dec.gpu_kv.as_ref(),
                false,
            )
            .expect("attn");
        }
        scratch2.residual.copy_from_slice(&h1);
        let attn_out_fixed = scratch2.attn_out.clone();
        let residual_fixed = scratch2.residual.clone();
        ctx.weights.release_layer();
        ctx.engine.pool.clear();
        assert_f32_repeatable("layer 2 FF drift", || {
            scratch2.attn_out.copy_from_slice(&attn_out_fixed);
            scratch2.residual.copy_from_slice(&residual_fixed);
            ctx.weights
                .ensure_layer(
                    &ctx.store,
                    &ctx.cfg.text_config,
                    2,
                    &ctx.engine.ctx.device,
                    &mut ctx.engine.pool,
                )
                .expect("layer2");
            let cached = ctx.weights.layer_ref(2);
            let mut out = vec![0.0f32; canvas_len * hidden_dim];
            forward_layer_ff_dgq_gpu(
                &mut out,
                &*cached,
                &ctx.weights,
                &ctx.cfg.text_config,
                2,
                canvas_len,
                hidden_dim,
                eps,
                &mut scratch2,
                &mut ctx.engine,
            )
            .expect("ff");
            ctx.weights.release_layer();
            out
        });
    }

    #[test]
    fn dgq_layer1_attention_repeatable() {
        if dgq_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let mut ctx = LayerTestCtx::new(2);
        let canvas_len = ctx.cfg.canvas_length;
        let kv_len = ctx.kv.kv_len;
        let hidden_dim = ctx.cfg.text_config.hidden_size;
        let mut scratch0 = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            0,
            kv_len,
        )
        .expect("scratch0");
        let hidden_in = ctx.hidden_in.clone();
        let h0 = ctx.run_layer(0, &hidden_in, &mut scratch0);
        let h0_fixed = h0.clone();
        let mut scratch = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            1,
            kv_len,
        )
        .expect("scratch");
        ctx.weights
            .ensure_layer(&ctx.store, &ctx.cfg.text_config, 1, &ctx.engine.ctx.device, &mut ctx.engine.pool)
            .expect("layer1");
        let cached = ctx.weights.layer_ref(1);
        let kv_view = LayerKvView::from_layer(ctx.kv.layer(1).expect("kv"), kv_len);
        ctx.engine.set_use_mps_q4(false);
        let mut a = vec![0.0f32; canvas_len * hidden_dim];
        let mut b = vec![0.0f32; canvas_len * hidden_dim];
        forward_decoder_attention(
            &mut a,
            &h0_fixed,
            &cached,
            &ctx.cfg.text_config,
            1,
            canvas_len,
            &ctx.positions,
            kv_view,
            Some(&ctx.mask),
            &mut scratch.attn,
            &mut ctx.engine,
            ctx.dec.gpu_kv.as_ref(),
            false,
        )
        .expect("attn a");
        forward_decoder_attention(
            &mut b,
            &h0_fixed,
            &cached,
            &ctx.cfg.text_config,
            1,
            canvas_len,
            &ctx.positions,
            kv_view,
            Some(&ctx.mask),
            &mut scratch.attn,
            &mut ctx.engine,
            ctx.dec.gpu_kv.as_ref(),
            false,
        )
        .expect("attn b");
        ctx.weights.release_layer();
        if a != b {
            let detail = first_f32_diff(&a, &b)
                .map(|(i, x, y)| format!("idx {i}: {x} vs {y}"))
                .unwrap_or_default();
            panic!("layer 1 attention drift: {detail}");
        }
    }

    #[test]
    fn dgq_layer0_ff_shared_engine_survey() {
        if dgq_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let mut setup = LayerTestCtx::new(3);
        let canvas_len = setup.cfg.canvas_length;
        let hidden_dim = setup.cfg.text_config.hidden_size;
        let eps = setup.cfg.text_config.rms_norm_eps as f32;
        let mut scratch0 = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &setup.cfg.text_config,
            0,
            setup.kv.kv_len,
        )
        .expect("scratch");
        let hidden_in = setup.hidden_in.clone();
        let _ = stream_residual_for_ff(&mut setup, &mut scratch0, &hidden_in, 0).expect("stream");
        let attn_fixed = scratch0.attn_out.clone();
        let res_fixed = scratch0.residual.clone();

        let mut unique = std::collections::HashSet::new();
        for trial in 0..8 {
            setup.engine.pool.clear();
            let mut scratch = GpuDecoderLayerScratch::with_kv_len(
                canvas_len,
                &setup.cfg.text_config,
                0,
                setup.kv.kv_len,
            )
            .expect("scratch");
            scratch.attn_out.copy_from_slice(&attn_fixed);
            scratch.residual.copy_from_slice(&res_fixed);
            setup.weights
                .ensure_layer(
                    &setup.store,
                    &setup.cfg.text_config,
                    0,
                    &setup.engine.ctx.device,
                    &mut setup.engine.pool,
                )
                .expect("layer0");
            let cached = setup.weights.layer_ref(0);
            let mut out = vec![0.0f32; canvas_len * hidden_dim];
            forward_layer_ff_dgq_gpu(
                &mut out,
                &cached,
                &setup.weights,
                &setup.cfg.text_config,
                0,
                canvas_len,
                hidden_dim,
                eps,
                &mut scratch,
                &mut setup.engine,
            )
            .expect("ff");
            setup.weights.release_layer();
            let tag = out.iter().take(4).map(|v| v.to_bits()).collect::<Vec<_>>();
            eprintln!("shared-engine ff trial {trial}: {tag:?}");
            unique.insert(tag);
        }
        assert_eq!(unique.len(), 1, "shared-engine ff drift: {unique:?}");
    }

    #[test]
    fn dgq_layer0_ff_fresh_engine_survey() {
        if dgq_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let mut setup = LayerTestCtx::new(3);
        let canvas_len = setup.cfg.canvas_length;
        let hidden_dim = setup.cfg.text_config.hidden_size;
        let eps = setup.cfg.text_config.rms_norm_eps as f32;
        let mut scratch = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &setup.cfg.text_config,
            0,
            setup.kv.kv_len,
        )
        .expect("scratch");
        let hidden_in = setup.hidden_in.clone();
        let stream = stream_residual_for_ff(&mut setup, &mut scratch, &hidden_in, 0).expect("stream");
        let attn_fixed = scratch.attn_out.clone();
        let res_fixed = scratch.residual.clone();

        let mut unique = std::collections::HashSet::new();
        for trial in 0..8 {
            let mut ctx = LayerTestCtx::new(3);
            let mut scratch = GpuDecoderLayerScratch::with_kv_len(
                canvas_len,
                &ctx.cfg.text_config,
                0,
                ctx.kv.kv_len,
            )
            .expect("scratch");
            scratch.attn_out.copy_from_slice(&attn_fixed);
            scratch.residual.copy_from_slice(&res_fixed);
            ctx.weights
                .ensure_layer(&ctx.store, &ctx.cfg.text_config, 0, &ctx.engine.ctx.device, &mut ctx.engine.pool)
                .expect("layer0");
            let cached = ctx.weights.layer_ref(0);
            let mut out = vec![0.0f32; canvas_len * hidden_dim];
            forward_layer_ff_dgq_gpu(
                &mut out,
                &cached,
                &ctx.weights,
                &ctx.cfg.text_config,
                0,
                canvas_len,
                hidden_dim,
                eps,
                &mut scratch,
                &mut ctx.engine,
            )
            .expect("ff");
            ctx.weights.release_layer();
            let tag = out.iter().take(4).map(|v| v.to_bits()).collect::<Vec<_>>();
            eprintln!("layer0 ff fresh trial {trial}: {tag:?}");
            unique.insert(tag);
        }
        assert_eq!(unique.len(), 1, "layer0 ff fresh-engine drift: {unique:?}");
    }

    #[test]
    fn dgq_layer0_route_fresh_engine_survey() {
        if dgq_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let mut setup = LayerTestCtx::new(3);
        let canvas_len = setup.cfg.canvas_length;
        let top_k = setup.cfg.text_config.top_k_experts;
        let mut scratch = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &setup.cfg.text_config,
            0,
            setup.kv.kv_len,
        )
        .expect("scratch");
        let hidden_in = setup.hidden_in.clone();
        let stream = stream_residual_for_ff(&mut setup, &mut scratch, &hidden_in, 0).expect("stream");

        let mut unique = std::collections::HashSet::new();
        for trial in 0..8 {
            let mut ctx = LayerTestCtx::new(3);
            ctx.weights
                .ensure_layer(&ctx.store, &ctx.cfg.text_config, 0, &ctx.engine.ctx.device, &mut ctx.engine.pool)
                .expect("layer0");
            let cached = ctx.weights.layer_ref(0);
            let telemetry = ctx.engine.batch_telemetry();
            let mut batch = begin_engine_batch(
                &ctx.engine.ctx.queue,
                &mut ctx.engine.pool,
                &ctx.engine.ctx.device,
                telemetry,
            )
            .expect("batch");
            let buf_stream = batch.alloc_f32(&stream).expect("buf");
            let mut route_scratch = GpuRouteScratch::new(canvas_len, top_k);
            route_gpu_in_batch(
                &mut batch,
                &ctx.engine.kernels,
                &ctx.engine.f32_f32_linear_pipeline.pipeline,
                &buf_stream,
                &cached,
                &ctx.cfg.text_config,
                canvas_len,
                &mut route_scratch,
            )
            .expect("route");
            batch.end().expect("end");
            let routes = pack_routes(&route_scratch.indices, &route_scratch.weights, canvas_len, top_k);
            let tag = routes[1].indices.clone();
            eprintln!("layer0 route fresh trial {trial}: experts={tag:?}");
            unique.insert(tag);
            ctx.weights.release_layer();
        }
        assert_eq!(unique.len(), 1, "layer0 route fresh-engine drift: {unique:?}");
    }

    #[test]
    fn dgq_layer0_attn_fresh_engine_survey() {
        if dgq_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let mut unique = std::collections::HashSet::new();
        for trial in 0..8 {
            let mut ctx = LayerTestCtx::new(3);
            let canvas_len = ctx.cfg.canvas_length;
            let hidden_dim = ctx.cfg.text_config.hidden_size;
            let mut scratch = GpuDecoderLayerScratch::with_kv_len(
                canvas_len,
                &ctx.cfg.text_config,
                0,
                ctx.kv.kv_len,
            )
            .expect("scratch");
            ctx.weights
                .ensure_layer(&ctx.store, &ctx.cfg.text_config, 0, &ctx.engine.ctx.device, &mut ctx.engine.pool)
                .expect("layer0");
            let cached = ctx.weights.layer_ref(0);
            let kv_view = LayerKvView::from_layer(ctx.kv.layer(0).expect("kv"), ctx.kv.kv_len);
            ctx.engine.set_use_mps_q4(false);
            let mut out = vec![0.0f32; canvas_len * hidden_dim];
            forward_decoder_attention(
                &mut out,
                &ctx.hidden_in,
                &cached,
                &ctx.cfg.text_config,
                0,
                canvas_len,
                &ctx.positions,
                kv_view,
                Some(&ctx.mask),
                &mut scratch.attn,
                &mut ctx.engine,
                ctx.dec.gpu_kv.as_ref(),
                false,
            )
            .expect("attn");
            ctx.weights.release_layer();
            let tag = out.iter().take(4).map(|v| v.to_bits()).collect::<Vec<_>>();
            eprintln!("layer0 attn fresh trial {trial}: {tag:?}");
            unique.insert(tag);
        }
        assert_eq!(unique.len(), 1, "layer0 attn fresh-engine drift: {unique:?}");
    }

    #[test]
    fn dgq_layer0_fresh_engine_survey() {
        if dgq_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let mut unique = std::collections::HashSet::new();
        for trial in 0..8 {
            let mut ctx = LayerTestCtx::new(3);
            let canvas_len = ctx.cfg.canvas_length;
            let mut scratch = GpuDecoderLayerScratch::with_kv_len(
                canvas_len,
                &ctx.cfg.text_config,
                0,
                ctx.kv.kv_len,
            )
            .expect("scratch");
            let out = ctx.run_layer(0, &ctx.hidden_in.clone(), &mut scratch);
            let tag = out.iter().take(4).map(|v| v.to_bits()).collect::<Vec<_>>();
            eprintln!("layer0 fresh trial {trial}: {tag:?}");
            unique.insert(tag);
        }
        assert_eq!(unique.len(), 1, "layer0 fresh-engine drift: {unique:?}");
    }

    #[test]
    fn dgq_layer0_repeat_with_clear_only() {
        if dgq_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let mut ctx = LayerTestCtx::new(2);
        let canvas_len = ctx.cfg.canvas_length;
        let hidden_in = ctx.hidden_in.clone();
        let mut scratch_a = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            0,
            ctx.kv.kv_len,
        )
        .expect("scratch_a");
        let mut scratch_b = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            0,
            ctx.kv.kv_len,
        )
        .expect("scratch_b");
        if let Some(gpu_kv) = ctx.dec.gpu_kv.as_ref() {
            gpu_kv.clear_canvas_suffix(canvas_len).expect("clear");
        }
        let a = ctx.run_layer(0, &hidden_in, &mut scratch_a);
        if let Some(gpu_kv) = ctx.dec.gpu_kv.as_ref() {
            gpu_kv.clear_canvas_suffix(canvas_len).expect("clear");
        }
        let b = ctx.run_layer(0, &hidden_in, &mut scratch_b);
        if a != b {
            let detail = first_f32_diff(&a, &b)
                .map(|(i, x, y)| format!("idx {i}: {x} vs {y}"))
                .unwrap_or_default();
            panic!("layer 0 clear-only repeat drift: {detail}");
        }
    }

    #[test]
    fn dgq_layer0_repeatable() {
        if dgq_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let mut ctx = LayerTestCtx::new(2);
        let canvas_len = ctx.cfg.canvas_length;
        let hidden_in = ctx.hidden_in.clone();
        let run_once = |ctx: &mut LayerTestCtx, clear_kv: bool| {
            let mut scratch = GpuDecoderLayerScratch::with_kv_len(
                canvas_len,
                &ctx.cfg.text_config,
                0,
                ctx.kv.kv_len,
            )
            .expect("scratch");
            if clear_kv {
                if let Some(gpu_kv) = ctx.dec.gpu_kv.as_ref() {
                    gpu_kv
                        .clear_canvas_suffix(canvas_len)
                        .expect("clear canvas kv");
                }
            }
            ctx.run_layer(0, &hidden_in, &mut scratch)
        };
        let a = run_once(&mut ctx, false);
        let b = run_once(&mut ctx, false);
        if a != b {
            let detail = first_f32_diff(&a, &b)
                .map(|(i, x, y)| format!("idx {i}: {x} vs {y}"))
                .unwrap_or_default();
            panic!("layer 0 repeat (no clear) drift: {detail}");
        }
        let c = run_once(&mut ctx, true);
        let d = run_once(&mut ctx, true);
        if c != d {
            let detail = first_f32_diff(&c, &d)
                .map(|(i, x, y)| format!("idx {i}: {x} vs {y}"))
                .unwrap_or_default();
            panic!("layer 0 repeat (with clear) drift: {detail}");
        }
    }

    #[test]
    fn dgq_layer2_after_full_stack_once() {
        if dgq_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let mut ctx = LayerTestCtx::new(3);
        let canvas_len = ctx.cfg.canvas_length;
        let kv_len = ctx.kv.kv_len;
        let hidden_in = ctx.hidden_in.clone();
        let mut s0 = GpuDecoderLayerScratch::with_kv_len(canvas_len, &ctx.cfg.text_config, 0, kv_len).expect("s0");
        let mut s1 = GpuDecoderLayerScratch::with_kv_len(canvas_len, &ctx.cfg.text_config, 1, kv_len).expect("s1");
        let mut s2 = GpuDecoderLayerScratch::with_kv_len(canvas_len, &ctx.cfg.text_config, 2, kv_len).expect("s2");
        let h0 = ctx.run_layer(0, &hidden_in, &mut s0);
        let h1 = ctx.run_layer(1, &h0, &mut s1);
        let _ = ctx.run_layer(2, &h1, &mut s2);

        let mut s2a = GpuDecoderLayerScratch::with_kv_len(canvas_len, &ctx.cfg.text_config, 2, kv_len).expect("s2a");
        let mut s2b = GpuDecoderLayerScratch::with_kv_len(canvas_len, &ctx.cfg.text_config, 2, kv_len).expect("s2b");
        if let Some(gpu_kv) = ctx.dec.gpu_kv.as_ref() {
            gpu_kv.clear_canvas_suffix(canvas_len).expect("clear kv");
        }
        let a = ctx.run_layer(2, &h1, &mut s2a);
        if let Some(gpu_kv) = ctx.dec.gpu_kv.as_ref() {
            gpu_kv.clear_canvas_suffix(canvas_len).expect("clear kv");
        }
        let b = ctx.run_layer(2, &h1, &mut s2b);
        if a != b {
            let detail = first_f32_diff(&a, &b)
                .map(|(i, x, y)| format!("idx {i}: {x} vs {y}"))
                .unwrap_or_default();
            panic!("layer2 after full stack drift: {detail}");
        }
    }

    #[test]
    fn dgq_three_layer_same_engine_twice() {
        if dgq_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let mut ctx = LayerTestCtx::new(3);
        let canvas_len = ctx.cfg.canvas_length;
        let kv_len = ctx.kv.kv_len;
        let hidden_in = ctx.hidden_in.clone();

        let mut s0a = GpuDecoderLayerScratch::with_kv_len(canvas_len, &ctx.cfg.text_config, 0, kv_len).expect("s0a");
        let mut s1a = GpuDecoderLayerScratch::with_kv_len(canvas_len, &ctx.cfg.text_config, 1, kv_len).expect("s1a");
        let mut s2a = GpuDecoderLayerScratch::with_kv_len(canvas_len, &ctx.cfg.text_config, 2, kv_len).expect("s2a");
        if let Some(gpu_kv) = ctx.dec.gpu_kv.as_ref() {
            gpu_kv.clear_canvas_suffix(canvas_len).expect("clear kv");
        }
        let h0a = ctx.run_layer(0, &hidden_in, &mut s0a);
        let h1a = ctx.run_layer(1, &h0a, &mut s1a);
        let a = ctx.run_layer(2, &h1a, &mut s2a);

        let mut s0b = GpuDecoderLayerScratch::with_kv_len(canvas_len, &ctx.cfg.text_config, 0, kv_len).expect("s0b");
        let mut s1b = GpuDecoderLayerScratch::with_kv_len(canvas_len, &ctx.cfg.text_config, 1, kv_len).expect("s1b");
        let mut s2b = GpuDecoderLayerScratch::with_kv_len(canvas_len, &ctx.cfg.text_config, 2, kv_len).expect("s2b");
        if let Some(gpu_kv) = ctx.dec.gpu_kv.as_ref() {
            gpu_kv.clear_canvas_suffix(canvas_len).expect("clear kv");
        }
        let h0b = ctx.run_layer(0, &hidden_in, &mut s0b);
        let h1b = ctx.run_layer(1, &h0b, &mut s1b);
        let b = ctx.run_layer(2, &h1b, &mut s2b);

        if a != b {
            let detail = first_f32_diff(&a, &b)
                .map(|(i, x, y)| format!("idx {i}: {x} vs {y}"))
                .unwrap_or_default();
            panic!("3-layer same-engine repeat drift: {detail}");
        }
    }

    #[test]
    fn dgq_layer2_endogenous_fresh_engine_survey() {
        if dgq_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let mut unique = std::collections::HashSet::new();
        for trial in 0..8 {
            let mut ctx = LayerTestCtx::new(3);
            let canvas_len = ctx.cfg.canvas_length;
            let kv_len = ctx.kv.kv_len;
            let mut scratch0 = GpuDecoderLayerScratch::with_kv_len(
                canvas_len,
                &ctx.cfg.text_config,
                0,
                kv_len,
            )
            .expect("scratch0");
            let mut scratch1 = GpuDecoderLayerScratch::with_kv_len(
                canvas_len,
                &ctx.cfg.text_config,
                1,
                kv_len,
            )
            .expect("scratch1");
            let mut scratch2 = GpuDecoderLayerScratch::with_kv_len(
                canvas_len,
                &ctx.cfg.text_config,
                2,
                kv_len,
            )
            .expect("scratch2");
            let hidden_in = ctx.hidden_in.clone();
            let h0 = ctx.run_layer(0, &hidden_in, &mut scratch0);
            let h1 = ctx.run_layer(1, &h0, &mut scratch1);
            let h2 = ctx.run_layer(2, &h1, &mut scratch2);
            let nan_n = h2.iter().filter(|v| v.is_nan()).count();
            let tag = h2.iter().take(4).map(|v| v.to_bits()).collect::<Vec<_>>();
            eprintln!("endogenous trial {trial}: nan={nan_n} head={tag:?}");
            unique.insert(tag);
        }
        assert_eq!(
            unique.len(),
            1,
            "layer2 endogenous fresh-engine drift: {unique:?}"
        );
    }

    #[test]
    fn dgq_layer2_fresh_engine_survey() {
        if dgq_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let mut setup = LayerTestCtx::new(3);
        let canvas_len = setup.cfg.canvas_length;
        let kv_len = setup.kv.kv_len;
        let mut scratch0 = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &setup.cfg.text_config,
            0,
            kv_len,
        )
        .expect("scratch0");
        let mut scratch1 = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &setup.cfg.text_config,
            1,
            kv_len,
        )
        .expect("scratch1");
        let hidden_in = setup.hidden_in.clone();
        let h0 = setup.run_layer(0, &hidden_in, &mut scratch0);
        let h1 = setup.run_layer(1, &h0, &mut scratch1);
        let h1_fixed = h1.clone();
        drop(setup);

        let mut unique_hashes = std::collections::HashSet::new();
        for trial in 0..8 {
            let mut ctx = LayerTestCtx::new(3);
            let mut scratch2 = GpuDecoderLayerScratch::with_kv_len(
                canvas_len,
                &ctx.cfg.text_config,
                2,
                kv_len,
            )
            .expect("scratch2");
            let out = ctx.run_layer(2, &h1_fixed, &mut scratch2);
            let nan_n = out.iter().filter(|v| v.is_nan()).count();
            let hash = out.iter().take(16).map(|v| v.to_bits()).collect::<Vec<_>>();
            eprintln!("layer2 fresh-engine trial {trial}: nan={nan_n} head_bits={hash:?}");
            unique_hashes.insert(hash);
        }
        assert_eq!(
            unique_hashes.len(),
            1,
            "layer 2 drift across fresh engines: {} unique",
            unique_hashes.len()
        );
    }

    #[test]
    fn dgq_layer2_repeatable_with_fixed_input() {
        if dgq_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let mut ctx = LayerTestCtx::new(3);
        let canvas_len = ctx.cfg.canvas_length;
        let kv_len = ctx.kv.kv_len;
        let mut scratch0 = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            0,
            kv_len,
        )
        .expect("scratch0");
        let mut scratch1 = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            1,
            kv_len,
        )
        .expect("scratch1");
        let hidden_in = ctx.hidden_in.clone();
        let h0 = ctx.run_layer(0, &hidden_in, &mut scratch0);
        let h1 = ctx.run_layer(1, &h0, &mut scratch1);
        let h1_fixed = h1.clone();

        let mut scratch2a = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            2,
            kv_len,
        )
        .expect("scratch2a");
        let mut scratch2b = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            2,
            kv_len,
        )
        .expect("scratch2b");
        let a = ctx.run_layer(2, &h1_fixed, &mut scratch2a);
        let b = ctx.run_layer(2, &h1_fixed, &mut scratch2b);
        if a != b {
            let detail = first_f32_diff(&a, &b)
                .map(|(i, x, y)| format!("idx {i}: {x} vs {y}"))
                .unwrap_or_default();
            panic!("layer 2 drift with fixed h1: {detail}");
        }
    }

    #[test]
    fn dgq_layer2_attention_repeatable() {
        if dgq_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let mut ctx = LayerTestCtx::new(3);
        let canvas_len = ctx.cfg.canvas_length;
        let kv_len = ctx.kv.kv_len;
        let hidden_dim = ctx.cfg.text_config.hidden_size;
        let mut scratch0 = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            0,
            kv_len,
        )
        .expect("scratch0");
        let mut scratch1 = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            1,
            kv_len,
        )
        .expect("scratch1");
        let hidden_in = ctx.hidden_in.clone();
        let h0 = ctx.run_layer(0, &hidden_in, &mut scratch0);
        let h1 = ctx.run_layer(1, &h0, &mut scratch1);
        let h1_fixed = h1.clone();
        let mut scratch = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            2,
            kv_len,
        )
        .expect("scratch");
        ctx.weights
            .ensure_layer(&ctx.store, &ctx.cfg.text_config, 2, &ctx.engine.ctx.device, &mut ctx.engine.pool)
            .expect("layer2");
        let cached = ctx.weights.layer_ref(2);
        let kv_view = LayerKvView::from_layer(ctx.kv.layer(2).expect("kv"), kv_len);
        ctx.engine.set_use_mps_q4(false);
        if let Some(gpu_kv) = ctx.dec.gpu_kv.as_ref() {
            gpu_kv.clear_canvas_suffix(canvas_len).expect("clear canvas kv");
        }
        let mut a = vec![0.0f32; canvas_len * hidden_dim];
        let mut b = vec![0.0f32; canvas_len * hidden_dim];
        forward_decoder_attention(
            &mut a,
            &h1_fixed,
            &cached,
            &ctx.cfg.text_config,
            2,
            canvas_len,
            &ctx.positions,
            kv_view,
            Some(&ctx.mask),
            &mut scratch.attn,
            &mut ctx.engine,
            ctx.dec.gpu_kv.as_ref(),
            false,
        )
        .expect("attn a");
        let mut scratch_b = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            2,
            kv_len,
        )
        .expect("scratch_b");
        if let Some(gpu_kv) = ctx.dec.gpu_kv.as_ref() {
            gpu_kv.clear_canvas_suffix(canvas_len).expect("clear canvas kv");
        }
        forward_decoder_attention(
            &mut b,
            &h1_fixed,
            &cached,
            &ctx.cfg.text_config,
            2,
            canvas_len,
            &ctx.positions,
            kv_view,
            Some(&ctx.mask),
            &mut scratch_b.attn,
            &mut ctx.engine,
            ctx.dec.gpu_kv.as_ref(),
            false,
        )
        .expect("attn b");
        ctx.weights.release_layer();
        if a != b {
            let detail = first_f32_diff(&a, &b)
                .map(|(i, x, y)| format!("idx {i}: {x} vs {y}"))
                .unwrap_or_default();
            panic!("layer 2 attention drift: {detail}");
        }
    }

    #[test]
    fn dgq_layer1_repeatable_with_fixed_input() {
        if dgq_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let mut ctx = LayerTestCtx::new(2);
        let canvas_len = ctx.cfg.canvas_length;
        let kv_len = ctx.kv.kv_len;
        let mut scratch0 = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            0,
            kv_len,
        )
        .expect("scratch0");
        let hidden_in = ctx.hidden_in.clone();
        let h0 = ctx.run_layer(0, &hidden_in, &mut scratch0);
        let h0_fixed = h0.clone();

        let mut scratch1a = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            1,
            kv_len,
        )
        .expect("scratch1a");
        let mut scratch1b = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            1,
            kv_len,
        )
        .expect("scratch1b");
        let a = ctx.run_layer(1, &h0_fixed, &mut scratch1a);
        let b = ctx.run_layer(1, &h0_fixed, &mut scratch1b);
        if a != b {
            let detail = first_f32_diff(&a, &b)
                .map(|(i, x, y)| format!("idx {i}: {x} vs {y}"))
                .unwrap_or_default();
            panic!("layer 1 drift with fixed h0: {detail}");
        }
    }

    #[test]
    fn dgq_layer0_output_stable_before_layer1() {
        if dgq_dir().is_none() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let mut ctx = LayerTestCtx::new(2);
        let canvas_len = ctx.cfg.canvas_length;
        let kv_len = ctx.kv.kv_len;
        let mut scratch0 = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            0,
            kv_len,
        )
        .expect("scratch0");
        let hidden_in = ctx.hidden_in.clone();
        let h0_a = ctx.run_layer(0, &hidden_in, &mut scratch0);
        let h0_fixed = h0_a.clone();
        let mut scratch1 = GpuDecoderLayerScratch::with_kv_len(
            canvas_len,
            &ctx.cfg.text_config,
            1,
            kv_len,
        )
        .expect("scratch1");
        let _ = ctx.run_layer(1, &h0_fixed, &mut scratch1);
        let h0_b = ctx.run_layer(0, &hidden_in, &mut scratch0);
        if h0_a != h0_b {
            let detail = first_f32_diff(&h0_a, &h0_b)
                .map(|(i, x, y)| format!("idx {i}: {x} vs {y}"))
                .unwrap_or_default();
            panic!("layer 0 output changed after layer 1 ran: {detail}");
        }
    }
}
