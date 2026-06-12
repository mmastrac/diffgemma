//! GPU decoder weights: PyTorch-layout bf16 linears + f32 norms (upload once per layer).

use crate::buffer::Buffer;
use crate::config::TextConfig;
use crate::fast_slice::{bf16_to_f32_into, FastBf16Slice};
use crate::metal::expert_cache::{ExpertCacheStats, ExpertWeightCache};
use crate::metal::linear::CachedLinear;
use crate::model::layer_weights::DecoderLayerWeights;
use crate::safetensors::Error;
use crate::tensor::Bf16Slice;
use crate::weights::WeightStore;
use std::cell::RefCell;

const NO_LAYER: usize = usize::MAX;

fn bf16_tensor_to_f32_buffer(slice: Bf16Slice<'_>) -> Buffer<f32> {
    let mut out = Buffer::new(slice.len());
    bf16_to_f32_into(FastBf16Slice::from_bf16(slice), out.as_fast_slice_mut());
    out
}


struct LayerSlot {
    loaded_layer: usize,
    cache: Option<GpuLayerWeightCache>,
}

pub struct GpuLayerWeightCache {
    pub input_layernorm: Buffer<f32>,
    pub q_norm: Buffer<f32>,
    pub k_norm: Buffer<f32>,
    pub q_proj: CachedLinear,
    pub k_proj: CachedLinear,
    pub v_proj: Option<CachedLinear>,
    pub o_proj: CachedLinear,
    pub post_attn_norm: Buffer<f32>,
    pub pre_ff_norm: Buffer<f32>,
    pub post_ff_norm: Buffer<f32>,
    pub post_ff_norm_1: Buffer<f32>,
    pub pre_ff_norm_2: Buffer<f32>,
    pub post_ff_norm_2: Buffer<f32>,
    pub mlp_gate: CachedLinear,
    pub mlp_up: CachedLinear,
    pub mlp_down: CachedLinear,
    pub router_proj: Buffer<f32>,
    pub router_scale: Buffer<f32>,
    pub per_expert_scale: Buffer<f32>,
    pub layer_scalar: f32,
}

impl GpuLayerWeightCache {
    pub fn load(weights: &DecoderLayerWeights<'_>, cfg: &TextConfig) -> Result<Self, Error> {
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let shapes = crate::model::layer_weights::DecoderLayerShapes::for_layer(cfg, weights.keys.layer)?;
        let q_out = shapes.q_proj[0] as usize;
        let kv_out = shapes.k_proj[0] as usize;
        Ok(Self {
            input_layernorm: bf16_tensor_to_f32_buffer(weights.input_layernorm.bf16()?),
            q_norm: bf16_tensor_to_f32_buffer(weights.q_norm.bf16()?),
            k_norm: bf16_tensor_to_f32_buffer(weights.k_norm.bf16()?),
            q_proj: CachedLinear::from_bf16(weights.q_proj.bf16()?, q_out, hidden),
            k_proj: CachedLinear::from_bf16(weights.k_proj.bf16()?, kv_out, hidden),
            v_proj: match &weights.v_proj {
                Some(v) => Some(CachedLinear::from_bf16(v.bf16()?, kv_out, hidden)),
                None => None,
            },
            o_proj: CachedLinear::from_bf16(weights.o_proj.bf16()?, hidden, q_out),
            post_attn_norm: bf16_tensor_to_f32_buffer(weights.post_attention_layernorm.bf16()?),
            pre_ff_norm: bf16_tensor_to_f32_buffer(weights.pre_feedforward_layernorm.bf16()?),
            post_ff_norm: bf16_tensor_to_f32_buffer(weights.post_feedforward_layernorm.bf16()?),
            post_ff_norm_1: bf16_tensor_to_f32_buffer(weights.post_feedforward_layernorm_1.bf16()?),
            pre_ff_norm_2: bf16_tensor_to_f32_buffer(weights.pre_feedforward_layernorm_2.bf16()?),
            post_ff_norm_2: bf16_tensor_to_f32_buffer(weights.post_feedforward_layernorm_2.bf16()?),
            mlp_gate: CachedLinear::from_bf16(weights.mlp_gate.bf16()?, inter, hidden),
            mlp_up: CachedLinear::from_bf16(weights.mlp_up.bf16()?, inter, hidden),
            mlp_down: CachedLinear::from_bf16(weights.mlp_down.bf16()?, hidden, inter),
            router_proj: bf16_tensor_to_f32_buffer(weights.router_proj.bf16()?),
            router_scale: bf16_tensor_to_f32_buffer(weights.router_scale.bf16()?),
            per_expert_scale: bf16_tensor_to_f32_buffer(weights.router_per_expert_scale.bf16()?),
            layer_scalar: weights.layer_scalar.bf16_scalar()?,
        })
    }

    pub fn resident_bytes(&self) -> u64 {
        let mut bytes = (self.input_layernorm.len()
            + self.q_norm.len()
            + self.k_norm.len()
            + self.post_attn_norm.len()
            + self.pre_ff_norm.len()
            + self.post_ff_norm.len()
            + self.post_ff_norm_1.len()
            + self.pre_ff_norm_2.len()
            + self.post_ff_norm_2.len()
            + self.router_proj.len()
            + self.router_scale.len()
            + self.per_expert_scale.len()) as u64
            * 4;
        bytes += (self.q_proj.w.len()
            + self.k_proj.w.len()
            + self.o_proj.w.len()
            + self.mlp_gate.w.len()
            + self.mlp_up.w.len()
            + self.mlp_down.w.len()) as u64
            * 2;
        if let Some(v) = &self.v_proj {
            bytes += v.w.len() as u64 * 2;
        }
        bytes
    }
}

/// Final norm plus at most one layer of GPU weights resident at a time.
/// MoE experts use an LRU of GPU buffers (mmap → GPU upload, no transpose).
pub struct GpuDecoderWeightCache {
    pub final_norm: Buffer<f32>,
    layer: RefCell<LayerSlot>,
    expert_cache: RefCell<ExpertWeightCache>,
}

impl GpuDecoderWeightCache {
    pub fn load(
        store: &WeightStore,
        text: &TextConfig,
        expert_budget_bytes: u64,
    ) -> Result<Self, Error> {
        let final_norm =
            bf16_tensor_to_f32_buffer(store.tensor("model.decoder.norm.weight")?.bf16()?);
        Ok(Self {
            final_norm,
            layer: RefCell::new(LayerSlot {
                loaded_layer: NO_LAYER,
                cache: None,
            }),
            expert_cache: RefCell::new(ExpertWeightCache::new(text, expert_budget_bytes)),
        })
    }

    pub fn expert_cache_stats(&self) -> ExpertCacheStats {
        self.expert_cache.borrow().stats()
    }

    pub fn prefetch_expert(
        &self,
        layer: usize,
        gate_up: Bf16Slice<'_>,
        down: Bf16Slice<'_>,
        expert: usize,
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        pool: &mut crate::metal::buffer::BufferPool,
    ) {
        self.expert_cache.borrow_mut().prefetch_expert(
            layer, gate_up, down, expert, device, pool,
        );
    }

    pub fn expert_gate_up_buf(
        &self,
        layer: usize,
        expert: usize,
    ) -> objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>> {
        self.expert_cache.borrow().expert_gate_up_buf(layer, expert)
    }

    pub fn expert_down_buf(
        &self,
        layer: usize,
        expert: usize,
    ) -> objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>> {
        self.expert_cache.borrow().expert_down_buf(layer, expert)
    }

    pub fn with_expert_gate_up_buf<R>(
        &self,
        layer: usize,
        gate_up: Bf16Slice<'_>,
        down: Bf16Slice<'_>,
        expert: usize,
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        pool: &mut crate::metal::buffer::BufferPool,
        f: impl FnOnce(&objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>) -> R,
    ) -> R {
        self.expert_cache.borrow_mut().with_expert_gate_up_buf(
            layer, gate_up, down, expert, device, pool, f,
        )
    }

    pub fn with_expert_down_buf<R>(
        &self,
        layer: usize,
        gate_up: Bf16Slice<'_>,
        down: Bf16Slice<'_>,
        expert: usize,
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        pool: &mut crate::metal::buffer::BufferPool,
        f: impl FnOnce(&objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>) -> R,
    ) -> R {
        self.expert_cache.borrow_mut().with_expert_down_buf(
            layer, gate_up, down, expert, device, pool, f,
        )
    }

    pub fn ensure_layer(
        &self,
        store: &WeightStore,
        text: &TextConfig,
        layer: usize,
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        pool: &mut crate::metal::buffer::BufferPool,
    ) -> Result<(), Error> {
        let mut slot = self.layer.borrow_mut();
        if slot.loaded_layer != layer || slot.cache.is_none() {
            let weights = DecoderLayerWeights::load(store, layer, text)?;
            let cache = GpuLayerWeightCache::load(&weights, text)?;
            slot.cache = Some(cache);
            slot.loaded_layer = layer;
            drop(slot);
            self.warm_layer_gpu(layer, device, pool);
        }
        Ok(())
    }

    fn warm_layer_gpu(
        &self,
        layer: usize,
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        pool: &mut crate::metal::buffer::BufferPool,
    ) {
        let slot = self.layer.borrow();
        let cache = slot.cache.as_ref().expect("layer cache");
        if slot.loaded_layer != layer {
            return;
        }
        let _ = cache.q_proj.gpu_weight(device, pool);
        let _ = cache.k_proj.gpu_weight(device, pool);
        if let Some(v) = &cache.v_proj {
            let _ = v.gpu_weight(device, pool);
        }
        let _ = cache.o_proj.gpu_weight(device, pool);
        let _ = cache.mlp_gate.gpu_weight(device, pool);
        let _ = cache.mlp_up.gpu_weight(device, pool);
        let _ = cache.mlp_down.gpu_weight(device, pool);
    }

    pub fn layer(&self) -> std::cell::Ref<'_, GpuLayerWeightCache> {
        std::cell::Ref::map(self.layer.borrow(), |s| {
            s.cache
                .as_ref()
                .expect("layer cache not loaded; call ensure_layer first")
        })
    }

    pub fn release_layer(&self) {
        let mut slot = self.layer.borrow_mut();
        if let Some(cache) = slot.cache.as_ref() {
            cache.q_proj.clear_gpu();
            cache.k_proj.clear_gpu();
            if let Some(v) = &cache.v_proj {
                v.clear_gpu();
            }
            cache.o_proj.clear_gpu();
            cache.mlp_gate.clear_gpu();
            cache.mlp_up.clear_gpu();
            cache.mlp_down.clear_gpu();
        }
        slot.cache = None;
        slot.loaded_layer = NO_LAYER;
    }

    pub fn resident_bytes(&self) -> u64 {
        self.final_norm.len() as u64 * 4
            + self
                .layer
                .borrow()
                .cache
                .as_ref()
                .map(GpuLayerWeightCache::resident_bytes)
                .unwrap_or(0)
            + self.expert_cache.borrow().resident_bytes()
    }
}
