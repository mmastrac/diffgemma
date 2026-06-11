//! Pre-transposed bf16 weights and f32 norm vectors (load once, reuse every forward).

use crate::buffer::Buffer;
use crate::config::TextConfig;
use crate::fast_slice::{bf16_to_f32_into, transpose_bf16_weight_into, FastBf16Slice, FastSlice};
use crate::metal::linear::CachedLinear;
use crate::model::layer_weights::DecoderLayerWeights;
use crate::safetensors::Error;
use crate::tensor::Bf16Slice;
use crate::weights::WeightStore;
use std::cell::RefCell;

fn bf16_tensor_to_f32_buffer(slice: Bf16Slice<'_>) -> Buffer<f32> {
    let mut out = Buffer::new(slice.len());
    bf16_to_f32_into(FastBf16Slice::from_bf16(slice), out.as_fast_slice_mut());
    out
}

pub struct GpuLayerWeightCache {
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
    expert_gate_up: RefCell<Vec<Option<Buffer<u16>>>>,
    expert_down: RefCell<Vec<Option<Buffer<u16>>>>,
    gate_up_stride: usize,
    down_stride: usize,
    moe_inter: usize,
    hidden: usize,
}

impl GpuLayerWeightCache {
    pub fn load(weights: &DecoderLayerWeights<'_>, cfg: &TextConfig) -> Result<Self, Error> {
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let moe_inter = cfg.moe_intermediate_size;
        let experts = cfg.num_experts;
        Ok(Self {
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
            expert_gate_up: RefCell::new((0..experts).map(|_| None).collect()),
            expert_down: RefCell::new((0..experts).map(|_| None).collect()),
            gate_up_stride: moe_inter * 2 * hidden,
            down_stride: hidden * moe_inter,
            moe_inter,
            hidden,
        })
    }

    fn ensure_expert_gate_up(&self, gate_up: Bf16Slice<'_>, expert: usize) {
        let mut cache = self.expert_gate_up.borrow_mut();
        if cache[expert].is_none() {
            let out_dim = self.moe_inter * 2;
            let elem_count = out_dim * self.hidden;
            let mut buf = Buffer::new(elem_count);
            let src = FastBf16Slice::from_bf16(gate_up);
            let src = unsafe { src.slice_unchecked(expert * self.gate_up_stride, elem_count) };
            transpose_bf16_weight_into(src, buf.as_fast_slice_mut(), out_dim, self.hidden);
            cache[expert] = Some(buf);
        }
    }

    fn ensure_expert_down(&self, down: Bf16Slice<'_>, expert: usize) {
        let mut cache = self.expert_down.borrow_mut();
        if cache[expert].is_none() {
            let elem_count = self.hidden * self.moe_inter;
            let mut buf = Buffer::new(elem_count);
            let src = FastBf16Slice::from_bf16(down);
            let src = unsafe { src.slice_unchecked(expert * self.down_stride, elem_count) };
            transpose_bf16_weight_into(src, buf.as_fast_slice_mut(), self.hidden, self.moe_inter);
            cache[expert] = Some(buf);
        }
    }

    pub fn with_expert_gate_up_t<R>(
        &self,
        gate_up: Bf16Slice<'_>,
        expert: usize,
        f: impl FnOnce(FastSlice<'_, u16>) -> R,
    ) -> R {
        self.ensure_expert_gate_up(gate_up, expert);
        let cache = self.expert_gate_up.borrow();
        f(cache[expert].as_ref().expect("expert gate_up").as_fast_slice())
    }

    pub fn with_expert_down_t<R>(
        &self,
        down: Bf16Slice<'_>,
        expert: usize,
        f: impl FnOnce(FastSlice<'_, u16>) -> R,
    ) -> R {
        self.ensure_expert_down(down, expert);
        let cache = self.expert_down.borrow();
        f(cache[expert].as_ref().expect("expert down").as_fast_slice())
    }

    /// Drop lazily transposed expert weights (call after each layer forward).
    pub fn clear_expert_cache(&self) {
        let n = self.expert_gate_up.borrow().len();
        *self.expert_gate_up.borrow_mut() = (0..n).map(|_| None).collect();
        *self.expert_down.borrow_mut() = (0..n).map(|_| None).collect();
    }

    pub fn resident_bytes(&self) -> u64 {
        let mut bytes = (self.post_attn_norm.len()
            + self.pre_ff_norm.len()
            + self.post_ff_norm.len()
            + self.post_ff_norm_1.len()
            + self.pre_ff_norm_2.len()
            + self.post_ff_norm_2.len()
            + self.router_proj.len()
            + self.router_scale.len()
            + self.per_expert_scale.len()) as u64
            * 4;
        bytes += (self.mlp_gate.w_t.len() + self.mlp_up.w_t.len() + self.mlp_down.w_t.len()) as u64 * 2;
        for slot in self.expert_gate_up.borrow().iter().chain(self.expert_down.borrow().iter()) {
            if let Some(w) = slot {
                bytes += w.len() as u64 * 2;
            }
        }
        bytes
    }
}

pub struct GpuDecoderWeightCache {
    pub layers: Vec<GpuLayerWeightCache>,
    pub final_norm: Buffer<f32>,
}

impl GpuDecoderWeightCache {
    pub fn load(store: &WeightStore, text: &TextConfig) -> Result<Self, Error> {
        let mut layers = Vec::with_capacity(text.num_hidden_layers);
        for layer in 0..text.num_hidden_layers {
            let weights = DecoderLayerWeights::load(store, layer, text)?;
            layers.push(GpuLayerWeightCache::load(&weights, text)?);
        }
        let final_norm =
            bf16_tensor_to_f32_buffer(store.tensor("model.decoder.norm.weight")?.bf16()?);
        Ok(Self { layers, final_norm })
    }
}
