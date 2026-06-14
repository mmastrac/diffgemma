//! GPU MoE router: RMSNorm → scale → linear → softmax → top-k.

use crate::config::TextConfig;
use crate::metal::batched_kernels::{self as bk, f32_f32_linear_gpu_bufs};
use crate::metal::batch::{set_bytes, GpuBatch};
use crate::metal::embed::softmax_rows_gpu_buf;
use crate::metal::kernels::GpuKernels;
use crate::metal::weights::GpuLayerWeightCache;
use crate::model::moe::RouteResult;
use crate::safetensors::Error;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};

pub struct GpuRouteScratch {
    pub indices: Vec<u32>,
    pub weights: Vec<f32>,
}

impl GpuRouteScratch {
    pub fn new(seq_len: usize, top_k: usize) -> Self {
        Self {
            indices: vec![0; seq_len * top_k],
            weights: vec![0.0; seq_len * top_k],
        }
    }

    pub fn into_routes(self, seq_len: usize, top_k: usize) -> Vec<RouteResult> {
        pack_routes(&self.indices, &self.weights, seq_len, top_k)
    }
}

pub fn pack_routes(
    indices: &[u32],
    weights: &[f32],
    seq_len: usize,
    top_k: usize,
) -> Vec<RouteResult> {
    let mut routes = Vec::with_capacity(seq_len);
    for s in 0..seq_len {
        let off = s * top_k;
        routes.push(RouteResult {
            indices: indices[off..off + top_k]
                .iter()
                .map(|&i| i as usize)
                .collect(),
            weights: weights[off..off + top_k].to_vec(),
        });
    }
    routes
}

/// Run the Gemma4 router on GPU; schedules readback into `route_scratch`.
pub fn route_gpu_in_batch(
    batch: &mut GpuBatch<'_>,
    kernels: &GpuKernels,
    f32_linear: &ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    residual_buf: &ProtocolObject<dyn MTLBuffer>,
    cached: &GpuLayerWeightCache,
    cfg: &TextConfig,
    seq_len: usize,
    route_scratch: &mut GpuRouteScratch,
) -> Result<(), Error> {
    let hidden = cfg.hidden_size;
    let experts = cfg.num_experts;
    let top_k = cfg.top_k_experts;
    let eps = cfg.rms_norm_eps as f32;
    let root = (hidden as f32).powf(-0.5);

    if route_scratch.indices.len() != seq_len * top_k || route_scratch.weights.len() != seq_len * top_k
    {
        return Err(Error::Format("route scratch size mismatch"));
    }

    let buf_router_in = bk::rms_norm_rows_no_scale_gpu_buf(
        batch,
        kernels,
        residual_buf,
        seq_len,
        hidden,
        eps,
    )?;
    bk::router_scale_rows_gpu_buf(
        batch,
        kernels,
        &buf_router_in,
        cached.router_scale.as_slice(),
        seq_len,
        hidden,
        root,
    )?;

    let buf_w = batch.alloc_f32(cached.router_proj.as_slice())?;
    let buf_logits = f32_f32_linear_gpu_bufs(
        batch,
        f32_linear,
        &buf_router_in,
        &buf_w,
        seq_len,
        hidden,
        experts,
    )?;
    softmax_rows_gpu_buf(batch, kernels, &buf_logits, seq_len, experts);

    let buf_scale = batch.alloc_f32(cached.per_expert_scale.as_slice())?;
    let buf_idx = batch.alloc_u32_out(seq_len * top_k)?;
    let buf_wt = batch.alloc_f32_out(seq_len * top_k)?;
    let params = [seq_len as u32, experts as u32, top_k as u32];
    let buf_dump = batch.alloc_f32_out(1)?;
    batch.dispatch_1d(&kernels.router_top_k.pipeline, seq_len, |enc| {
        crate::kernels::sub::router_top_k_rows::bind_gpu_buffers(
            &enc,
            &buf_logits,
            &buf_scale,
            &buf_idx,
            &buf_wt,
            &buf_dump,
            &params,
        );
    });

    batch.register_read_u32(buf_idx, &mut route_scratch.indices);
    batch.register_read(buf_wt, &mut route_scratch.weights);
    Ok(())
}

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod tests {
    use super::*;
    use crate::metal::batch::GpuBatch;
    use crate::metal::buffer::BufferPool;
        use crate::metal::engine::GpuDecoderEngine;
    use crate::model::moe::{route_with_cached_weights, MoeScratch};
    use crate::weights::WeightStore;

    fn open_model() -> Option<(WeightStore, crate::config::TextConfig)> {
        let dir = std::path::Path::new("model/transformer");
        if !dir.exists() {
            return None;
        }
        let store = WeightStore::open(dir).ok()?;
        let cfg = crate::config::ModelConfig::load(dir).ok()?;
        Some((store, cfg.text_config))
    }

    #[test]
    fn gpu_router_matches_cpu_layer0() {
        use crate::metal::decoder::load_weight_cache;

        let Some((store, text)) = open_model() else {
            eprintln!("skip: model/transformer missing");
            return;
        };
        let engine = GpuDecoderEngine::new().expect("engine");
        let mut pool = BufferPool::new();
        let cache = load_weight_cache(&store, &text, 4, 4).expect("cache");
        cache
            .ensure_layer(&store, &text, 0, &engine.ctx.device, &mut pool)
            .expect("layer0");
        let layer = cache.layer_ref(0);
        let seq_len = 4usize;
        let hidden = text.hidden_size;
        let mut residual = vec![0.0f32; seq_len * hidden];
        for (i, v) in residual.iter_mut().enumerate() {
            *v = ((i as f32 * 0.001) % 0.02) - 0.01;
        }

        let mut moe = MoeScratch::new(seq_len, &text);
        let cpu_routes = route_with_cached_weights(
            &residual,
            layer.router_proj.as_slice(),
            layer.router_scale.as_slice(),
            layer.per_expert_scale.as_slice(),
            &text,
            seq_len,
            &mut moe,
        )
        .expect("cpu route");

        let mut batch = GpuBatch::begin(&engine.ctx.queue, &mut pool, &engine.ctx.device)
            .expect("batch");
        let buf_res = batch.alloc_f32(&residual).expect("buf");
        let mut route_scratch = GpuRouteScratch::new(seq_len, text.top_k_experts);
        route_gpu_in_batch(
            &mut batch,
            &engine.kernels,
            &engine.f32_f32_linear_pipeline.pipeline,
            &buf_res,
            &layer,
            &text,
            seq_len,
            &mut route_scratch,
        )
        .expect("gpu route");
        batch.end().expect("end");

        let gpu_routes = route_scratch.into_routes(seq_len, text.top_k_experts);
        assert_eq!(gpu_routes.len(), cpu_routes.len());
        for (g, c) in gpu_routes.iter().zip(cpu_routes.iter()) {
            assert_eq!(
                g.indices, c.indices,
                "expert indices mismatch (gpu={:?} cpu={:?})",
                g.indices, c.indices
            );
            let max_w_err = g
                .weights
                .iter()
                .zip(c.weights.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_w_err < 0.25,
                "route weight drift too large: max_err={max_w_err} gpu={:?} cpu={:?}",
                g.weights, c.weights
            );
        }
    }
}
