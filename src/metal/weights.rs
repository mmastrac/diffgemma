//! GPU decoder weights: bf16 paging + expert LRU, or resident `.dgq` blob.

use crate::buffer::Buffer;
use crate::config::TextConfig;
use crate::dgq::DgqStore;
use crate::fast_slice::{bf16_to_f32_into, FastBf16Slice};
use crate::metal::dgq_gpu::{
    load_block_expert_stack, load_block_linear,
    load_q8_linear, load_raw_view, DgqGpuBlob,
    Q4ExpertStackGpu, Q8LinearGpu,
};
use crate::metal::self_conditioning::GpuSelfConditioningWeights;
use crate::metal::expert_cache::{ExpertCacheStats, ExpertWeightCache};
use crate::metal::linear::GpuLinearWeight;
use crate::model::layer_weights::DecoderLayerWeights;
use crate::safetensors::Error;
use crate::tensor::Bf16Slice;
use crate::weights::WeightStore;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice};
use std::cell::RefCell;
use std::sync::Arc;

const NO_LAYER: usize = usize::MAX;

const SC_PRE_NORM: &str = "model.decoder.self_conditioning.pre_norm.weight";
const SC_GATE: &str = "model.decoder.self_conditioning.gate_proj.weight";
const SC_UP: &str = "model.decoder.self_conditioning.up_proj.weight";
const SC_DOWN: &str = "model.decoder.self_conditioning.down_proj.weight";

fn load_self_conditioning_dgq(
    dgq: &DgqStore,
    blob: Arc<DgqGpuBlob>,
) -> Result<GpuSelfConditioningWeights, Error> {
    Ok(GpuSelfConditioningWeights {
        pre_norm: raw_blob_bf16_to_f32(&load_raw_view(
            dgq,
            Arc::clone(&blob),
            SC_PRE_NORM,
        )?)?,
        gate_proj: load_q8_linear(dgq, Arc::clone(&blob), SC_GATE)?,
        up_proj: load_q8_linear(dgq, Arc::clone(&blob), SC_UP)?,
        down_proj: load_q8_linear(dgq, blob, SC_DOWN)?,
    })
}

fn bf16_tensor_to_f32_buffer(slice: Bf16Slice<'_>) -> Buffer<f32> {
    let mut out = Buffer::new(slice.len());
    bf16_to_f32_into(FastBf16Slice::from_bf16(slice), out.as_fast_slice_mut());
    out
}

fn raw_blob_bf16_to_f32(
    view: &crate::metal::dgq_gpu::RawBlobView,
) -> Result<Buffer<f32>, Error> {
    let blob = &view.blob.buffer;
    let ptr = blob.contents().as_ptr() as *const u8;
    let src = unsafe {
        std::slice::from_raw_parts(
            ptr.add(crate::dgq::layout::blob_offset_for_mtl(view.byte_offset)),
            crate::dgq::layout::blob_offset_for_mtl(view.byte_len),
        )
    };
    if src.len() != view.numel * 2 {
        return Err(Error::Format("raw bf16 norm size mismatch"));
    }
    Ok(bf16_tensor_to_f32_buffer(Bf16Slice::from_bytes(src)))
}

fn layer_scalar_from_bytes(src: &[u8]) -> Result<f32, Error> {
    if src.len() == 2 {
        let bits = u16::from_le_bytes([src[0], src[1]]);
        Ok(f32::from_bits((bits as u32) << 16))
    } else if src.len() == 4 {
        Ok(f32::from_le_bytes([src[0], src[1], src[2], src[3]]))
    } else {
        Err(Error::Format("layer_scalar size mismatch"))
    }
}

pub struct GpuLayerWeightCache {
    pub input_layernorm: Buffer<f32>,
    pub q_norm: Buffer<f32>,
    pub k_norm: Buffer<f32>,
    pub q_proj: GpuLinearWeight,
    pub k_proj: GpuLinearWeight,
    pub v_proj: Option<GpuLinearWeight>,
    pub o_proj: GpuLinearWeight,
    pub post_attn_norm: Buffer<f32>,
    pub pre_ff_norm: Buffer<f32>,
    pub post_ff_norm: Buffer<f32>,
    pub post_ff_norm_1: Buffer<f32>,
    pub pre_ff_norm_2: Buffer<f32>,
    pub post_ff_norm_2: Buffer<f32>,
    pub mlp_gate: GpuLinearWeight,
    pub mlp_up: GpuLinearWeight,
    pub mlp_down: GpuLinearWeight,
    pub router_proj: Buffer<f32>,
    pub router_scale: Buffer<f32>,
    pub per_expert_scale: Buffer<f32>,
    pub layer_scalar: f32,
    /// Resident q4 expert stacks (`.dgq` only).
    pub experts_gate_up: Option<Q4ExpertStackGpu>,
    pub experts_down: Option<Q4ExpertStackGpu>,
}

impl GpuLayerWeightCache {
    pub fn load_bf16(weights: &DecoderLayerWeights<'_>, cfg: &TextConfig) -> Result<Self, Error> {
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let shapes =
            crate::model::layer_weights::DecoderLayerShapes::for_layer(cfg, weights.keys.layer)?;
        let q_out = shapes.q_proj[0] as usize;
        let kv_out = shapes.k_proj[0] as usize;
        Ok(Self {
            input_layernorm: bf16_tensor_to_f32_buffer(weights.input_layernorm.bf16()?),
            q_norm: bf16_tensor_to_f32_buffer(weights.q_norm.bf16()?),
            k_norm: bf16_tensor_to_f32_buffer(weights.k_norm.bf16()?),
            q_proj: GpuLinearWeight::bf16(weights.q_proj.bf16()?, q_out, hidden),
            k_proj: GpuLinearWeight::bf16(weights.k_proj.bf16()?, kv_out, hidden),
            v_proj: match &weights.v_proj {
                Some(v) => Some(GpuLinearWeight::bf16(v.bf16()?, kv_out, hidden)),
                None => None,
            },
            o_proj: GpuLinearWeight::bf16(weights.o_proj.bf16()?, hidden, q_out),
            post_attn_norm: bf16_tensor_to_f32_buffer(weights.post_attention_layernorm.bf16()?),
            pre_ff_norm: bf16_tensor_to_f32_buffer(weights.pre_feedforward_layernorm.bf16()?),
            post_ff_norm: bf16_tensor_to_f32_buffer(weights.post_feedforward_layernorm.bf16()?),
            post_ff_norm_1: bf16_tensor_to_f32_buffer(
                weights.post_feedforward_layernorm_1.bf16()?,
            ),
            pre_ff_norm_2: bf16_tensor_to_f32_buffer(
                weights.pre_feedforward_layernorm_2.bf16()?,
            ),
            post_ff_norm_2: bf16_tensor_to_f32_buffer(
                weights.post_feedforward_layernorm_2.bf16()?,
            ),
            mlp_gate: GpuLinearWeight::bf16(weights.mlp_gate.bf16()?, inter, hidden),
            mlp_up: GpuLinearWeight::bf16(weights.mlp_up.bf16()?, inter, hidden),
            mlp_down: GpuLinearWeight::bf16(weights.mlp_down.bf16()?, hidden, inter),
            router_proj: bf16_tensor_to_f32_buffer(weights.router_proj.bf16()?),
            router_scale: bf16_tensor_to_f32_buffer(weights.router_scale.bf16()?),
            per_expert_scale: bf16_tensor_to_f32_buffer(weights.router_per_expert_scale.bf16()?),
            layer_scalar: weights.layer_scalar.bf16_scalar()?,
            experts_gate_up: None,
            experts_down: None,
        })
    }

    pub fn load_dgq(
        store: &DgqStore,
        blob: Arc<DgqGpuBlob>,
        keys: &crate::model::layer_weights::DecoderLayerKeys,
        cfg: &TextConfig,
    ) -> Result<Self, Error> {
        let shapes = crate::model::layer_weights::DecoderLayerShapes::for_layer(cfg, keys.layer)?;
        let scalar_src = store.tensor_bytes(&keys.layer_scalar)?;
        Ok(Self {
            input_layernorm: raw_blob_bf16_to_f32(&load_raw_view(
                store,
                Arc::clone(&blob),
                &keys.input_layernorm,
            )?)?,
            q_norm: raw_blob_bf16_to_f32(&load_raw_view(
                store,
                Arc::clone(&blob),
                &keys.q_norm,
            )?)?,
            k_norm: raw_blob_bf16_to_f32(&load_raw_view(
                store,
                Arc::clone(&blob),
                &keys.k_norm,
            )?)?,
            q_proj: GpuLinearWeight::q4(load_block_linear(
                store,
                Arc::clone(&blob),
                &keys.q_proj,
            )?),
            k_proj: GpuLinearWeight::q4(load_block_linear(
                store,
                Arc::clone(&blob),
                &keys.k_proj,
            )?),
            v_proj: match &shapes.v_proj {
                Some(_) => Some(GpuLinearWeight::q4(load_block_linear(
                    store,
                    Arc::clone(&blob),
                    &keys.v_proj,
                )?)),
                None => None,
            },
            o_proj: GpuLinearWeight::q4(load_block_linear(
                store,
                Arc::clone(&blob),
                &keys.o_proj,
            )?),
            post_attn_norm: raw_blob_bf16_to_f32(&load_raw_view(
                store,
                Arc::clone(&blob),
                &keys.post_attention_layernorm,
            )?)?,
            pre_ff_norm: raw_blob_bf16_to_f32(&load_raw_view(
                store,
                Arc::clone(&blob),
                &keys.pre_feedforward_layernorm,
            )?)?,
            post_ff_norm: raw_blob_bf16_to_f32(&load_raw_view(
                store,
                Arc::clone(&blob),
                &keys.post_feedforward_layernorm,
            )?)?,
            post_ff_norm_1: raw_blob_bf16_to_f32(&load_raw_view(
                store,
                Arc::clone(&blob),
                &keys.post_feedforward_layernorm_1,
            )?)?,
            pre_ff_norm_2: raw_blob_bf16_to_f32(&load_raw_view(
                store,
                Arc::clone(&blob),
                &keys.pre_feedforward_layernorm_2,
            )?)?,
            post_ff_norm_2: raw_blob_bf16_to_f32(&load_raw_view(
                store,
                Arc::clone(&blob),
                &keys.post_feedforward_layernorm_2,
            )?)?,
            mlp_gate: GpuLinearWeight::q4(load_block_linear(
                store,
                Arc::clone(&blob),
                &keys.mlp_gate,
            )?),
            mlp_up: GpuLinearWeight::q4(load_block_linear(
                store,
                Arc::clone(&blob),
                &keys.mlp_up,
            )?),
            mlp_down: GpuLinearWeight::q4(load_block_linear(
                store,
                Arc::clone(&blob),
                &keys.mlp_down,
            )?),
            router_proj: raw_blob_bf16_to_f32(&load_raw_view(
                store,
                Arc::clone(&blob),
                &keys.router_proj,
            )?)?,
            router_scale: raw_blob_bf16_to_f32(&load_raw_view(
                store,
                Arc::clone(&blob),
                &keys.router_scale,
            )?)?,
            per_expert_scale: raw_blob_bf16_to_f32(&load_raw_view(
                store,
                Arc::clone(&blob),
                &keys.router_per_expert_scale,
            )?)?,
            layer_scalar: layer_scalar_from_bytes(scalar_src)?,
            experts_gate_up: Some(load_block_expert_stack(
                store,
                Arc::clone(&blob),
                &keys.experts_gate_up,
            )?),
            experts_down: Some(load_block_expert_stack(
                store,
                blob,
                &keys.experts_down,
            )?),
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
        for w in [
            &self.q_proj,
            &self.k_proj,
            &self.o_proj,
            &self.mlp_gate,
            &self.mlp_up,
            &self.mlp_down,
        ] {
            if let GpuLinearWeight::Bf16(c) = w {
                bytes += c.w.len() as u64 * 2;
            }
        }
        if let Some(v) = &self.v_proj {
            if let GpuLinearWeight::Bf16(c) = v {
                bytes += c.w.len() as u64 * 2;
            }
        }
        bytes
    }

    fn clear_bf16_gpu(&self) {
        for w in [
            &self.q_proj,
            &self.k_proj,
            &self.o_proj,
            &self.mlp_gate,
            &self.mlp_up,
            &self.mlp_down,
        ] {
            if let GpuLinearWeight::Bf16(c) = w {
                c.clear_gpu();
            }
        }
        if let Some(GpuLinearWeight::Bf16(c)) = &self.v_proj {
            c.clear_gpu();
        }
    }
}

struct Bf16LayerSlot {
    loaded_layer: usize,
    cache: Option<GpuLayerWeightCache>,
}

pub struct Bf16Paged {
    final_norm: Buffer<f32>,
    layer: RefCell<Bf16LayerSlot>,
    expert_cache: RefCell<ExpertWeightCache>,
}

pub struct DgqResident {
    blob: Arc<DgqGpuBlob>,
    final_norm: Buffer<f32>,
    embed_q8: Option<Q8LinearGpu>,
    self_conditioning: GpuSelfConditioningWeights,
    layers: Vec<GpuLayerWeightCache>,
}

/// GPU weight cache: paged bf16 (safetensors) or fully resident `.dgq`.
pub enum GpuDecoderWeightCache {
    Bf16(Bf16Paged),
    Dgq(DgqResident),
}

impl GpuDecoderWeightCache {
    pub fn load(
        store: &WeightStore,
        text: &TextConfig,
        expert_budget_bytes: u64,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Result<Self, Error> {
        Self::load_opt(store, text, expert_budget_bytes, device, None)
    }

    /// `.dgq` path with an existing GPU blob (avoids duplicate 15 GiB resident copy).
    pub fn load_with_dgq_blob(
        store: &WeightStore,
        text: &TextConfig,
        device: &ProtocolObject<dyn MTLDevice>,
        blob: Arc<DgqGpuBlob>,
    ) -> Result<Self, Error> {
        Self::load_opt(store, text, 0, device, Some(blob))
    }

    fn load_opt(
        store: &WeightStore,
        text: &TextConfig,
        expert_budget_bytes: u64,
        device: &ProtocolObject<dyn MTLDevice>,
        shared_dgq_blob: Option<Arc<DgqGpuBlob>>,
    ) -> Result<Self, Error> {
        match store {
            WeightStore::Dgq(dgq) => {
                let blob = match shared_dgq_blob {
                    Some(b) => b,
                    None => DgqGpuBlob::from_store(dgq, device)?,
                };
                Self::load_dgq_layers(dgq, text, blob)
            }
            _ => {
                let final_norm =
                    bf16_tensor_to_f32_buffer(store.tensor("model.decoder.norm.weight")?.bf16()?);
                Ok(Self::Bf16(Bf16Paged {
                    final_norm,
                    layer: RefCell::new(Bf16LayerSlot {
                        loaded_layer: NO_LAYER,
                        cache: None,
                    }),
                    expert_cache: RefCell::new(ExpertWeightCache::new(text, expert_budget_bytes)),
                }))
            }
        }
    }

    fn load_dgq_layers(
        dgq: &DgqStore,
        text: &TextConfig,
        blob: Arc<DgqGpuBlob>,
    ) -> Result<Self, Error> {
        let final_norm = raw_blob_bf16_to_f32(&load_raw_view(
            dgq,
            Arc::clone(&blob),
            "model.decoder.norm.weight",
        )?)?;
        let mut layers = Vec::with_capacity(text.num_hidden_layers);
        for layer in 0..text.num_hidden_layers {
            let keys = crate::model::layer_weights::DecoderLayerKeys::new(layer);
            layers.push(GpuLayerWeightCache::load_dgq(
                dgq,
                Arc::clone(&blob),
                &keys,
                text,
            )?);
        }
        let embed_q8 = load_q8_linear(
            dgq,
            Arc::clone(&blob),
            "model.decoder.embed_tokens.weight",
        )
        .ok();
        let self_conditioning = load_self_conditioning_dgq(dgq, Arc::clone(&blob))?;
        Ok(Self::Dgq(DgqResident {
            blob,
            final_norm,
            embed_q8,
            self_conditioning,
            layers,
        }))
    }

    pub fn is_dgq(&self) -> bool {
        matches!(self, Self::Dgq(_))
    }

    pub fn dgq_blob(&self) -> Option<&ProtocolObject<dyn MTLBuffer>> {
        match self {
            Self::Dgq(d) => Some(&d.blob.buffer),
            Self::Bf16(_) => None,
        }
    }

    pub fn embed_q8(&self) -> Option<&Q8LinearGpu> {
        match self {
            Self::Dgq(d) => d.embed_q8.as_ref(),
            Self::Bf16(_) => None,
        }
    }

    pub fn self_conditioning(&self) -> Option<&GpuSelfConditioningWeights> {
        match self {
            Self::Dgq(d) => Some(&d.self_conditioning),
            Self::Bf16(_) => None,
        }
    }

    pub fn expert_cache_stats(&self) -> ExpertCacheStats {
        match self {
            Self::Bf16(b) => b.expert_cache.borrow().stats(),
            Self::Dgq(_) => ExpertCacheStats::default(),
        }
    }

    pub fn prefetch_expert(
        &self,
        layer: usize,
        gate_up: Bf16Slice<'_>,
        down: Bf16Slice<'_>,
        expert: usize,
        device: &ProtocolObject<dyn MTLDevice>,
        pool: &mut crate::metal::buffer::BufferPool,
    ) {
        if let Self::Bf16(b) = self {
            b.expert_cache.borrow_mut().prefetch_expert(
                layer, gate_up, down, expert, device, pool,
            );
        }
    }

    pub fn expert_gate_up_buf(
        &self,
        layer: usize,
        expert: usize,
    ) -> objc2::rc::Retained<ProtocolObject<dyn MTLBuffer>> {
        match self {
            Self::Bf16(b) => b.expert_cache.borrow().expert_gate_up_buf(layer, expert),
            Self::Dgq(_) => panic!("expert_gate_up_buf: use q4 expert view for .dgq"),
        }
    }

    pub fn expert_down_buf(
        &self,
        layer: usize,
        expert: usize,
    ) -> objc2::rc::Retained<ProtocolObject<dyn MTLBuffer>> {
        match self {
            Self::Bf16(b) => b.expert_cache.borrow().expert_down_buf(layer, expert),
            Self::Dgq(_) => panic!("expert_down_buf: use q4 expert view for .dgq"),
        }
    }

    pub fn expert_gate_up_q4(&self, layer: usize, expert: usize) -> crate::metal::dgq_gpu::Q4LinearGpu {
        match self {
            Self::Dgq(d) => d.layers[layer]
                .experts_gate_up
                .as_ref()
                .expect("dgq experts")
                .expert_linear(expert),
            Self::Bf16(_) => panic!("expert_gate_up_q4: bf16 path uses LRU buffers"),
        }
    }

    pub fn expert_down_q4(&self, layer: usize, expert: usize) -> crate::metal::dgq_gpu::Q4LinearGpu {
        match self {
            Self::Dgq(d) => d.layers[layer]
                .experts_down
                .as_ref()
                .expect("dgq experts")
                .expert_linear(expert),
            Self::Bf16(_) => panic!("expert_down_q4: bf16 path uses LRU buffers"),
        }
    }

    pub fn ensure_layer(
        &self,
        store: &WeightStore,
        text: &TextConfig,
        layer: usize,
        device: &ProtocolObject<dyn MTLDevice>,
        pool: &mut crate::metal::buffer::BufferPool,
    ) -> Result<(), Error> {
        match self {
            Self::Dgq(_) => Ok(()),
            Self::Bf16(b) => {
                let mut slot = b.layer.borrow_mut();
                if slot.loaded_layer != layer || slot.cache.is_none() {
                    let weights = DecoderLayerWeights::load(store, layer, text)?;
                    let cache = GpuLayerWeightCache::load_bf16(&weights, text)?;
                    slot.cache = Some(cache);
                    slot.loaded_layer = layer;
                    drop(slot);
                    self.warm_layer_gpu(layer, device, pool);
                }
                Ok(())
            }
        }
    }

    fn warm_layer_gpu(
        &self,
        layer: usize,
        device: &ProtocolObject<dyn MTLDevice>,
        pool: &mut crate::metal::buffer::BufferPool,
    ) {
        let Self::Bf16(b) = self else {
            return;
        };
        let slot = b.layer.borrow();
        let cache = slot.cache.as_ref().expect("layer cache");
        if slot.loaded_layer != layer {
            return;
        }
        for w in [
            &cache.q_proj,
            &cache.k_proj,
            &cache.o_proj,
            &cache.mlp_gate,
            &cache.mlp_up,
            &cache.mlp_down,
        ] {
            if let GpuLinearWeight::Bf16(c) = w {
                let _ = c.gpu_weight(device, pool);
            }
        }
        if let Some(GpuLinearWeight::Bf16(c)) = &cache.v_proj {
            let _ = c.gpu_weight(device, pool);
        }
    }

    pub fn layer_ref(&self, layer: usize) -> LayerWeightRef<'_> {
        match self {
            Self::Dgq(d) => LayerWeightRef::Dgq(&d.layers[layer]),
            Self::Bf16(b) => LayerWeightRef::Bf16(std::cell::Ref::map(b.layer.borrow(), |s| {
                assert_eq!(
                    s.loaded_layer,
                    layer,
                    "ensure_layer({layer}) before layer_ref"
                );
                s.cache
                    .as_ref()
                    .expect("layer cache not loaded; call ensure_layer first")
            })),
        }
    }

    pub fn final_norm(&self) -> &[f32] {
        match self {
            Self::Dgq(d) => d.final_norm.as_slice(),
            Self::Bf16(b) => b.final_norm.as_slice(),
        }
    }

    /// Drop any `layer_ref` before calling (bf16 paged path reuses one RefCell slot).
    pub fn release_layer(&self) {
        if let Self::Bf16(b) = self {
            let mut slot = b.layer.borrow_mut();
            if let Some(cache) = slot.cache.as_ref() {
                cache.clear_bf16_gpu();
            }
            slot.cache = None;
            slot.loaded_layer = NO_LAYER;
        }
    }

    pub fn resident_bytes(&self) -> u64 {
        match self {
            Self::Dgq(d) => d.blob.len as u64 + d.final_norm.len() as u64 * 4,
            Self::Bf16(b) => {
                b.final_norm.len() as u64 * 4
                    + b.layer
                        .borrow()
                        .cache
                        .as_ref()
                        .map(GpuLayerWeightCache::resident_bytes)
                        .unwrap_or(0)
                    + b.expert_cache.borrow().resident_bytes()
            }
        }
    }
}

pub enum LayerWeightRef<'a> {
    Dgq(&'a GpuLayerWeightCache),
    Bf16(std::cell::Ref<'a, GpuLayerWeightCache>),
}

impl std::ops::Deref for LayerWeightRef<'_> {
    type Target = GpuLayerWeightCache;

    fn deref(&self) -> &GpuLayerWeightCache {
        match self {
            Self::Dgq(c) => c,
            Self::Bf16(r) => r,
        }
    }
}
