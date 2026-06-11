//! Pre-transposed bf16 weights and f32 norm vectors (load once, reuse every forward).

use crate::config::TextConfig;
use crate::metal::linear::{transpose_weight_bf16, CachedLinear};
use crate::model::layer_weights::DecoderLayerWeights;
use crate::safetensors::Error;
use crate::tensor::Bf16Slice;
use crate::weights::WeightStore;
use std::cell::RefCell;

pub struct GpuLayerWeightCache {
    pub post_attn_norm: Vec<f32>,
    pub pre_ff_norm: Vec<f32>,
    pub post_ff_norm: Vec<f32>,
    pub post_ff_norm_1: Vec<f32>,
    pub pre_ff_norm_2: Vec<f32>,
    pub post_ff_norm_2: Vec<f32>,
    pub mlp_gate: CachedLinear,
    pub mlp_up: CachedLinear,
    pub mlp_down: CachedLinear,
    pub router_proj: Vec<f32>,
    pub router_scale: Vec<f32>,
    pub per_expert_scale: Vec<f32>,
    pub layer_scalar: f32,
    expert_gate_up: RefCell<Vec<Option<Vec<u16>>>>,
    expert_down: RefCell<Vec<Option<Vec<u16>>>>,
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
            post_attn_norm: weights.post_attention_layernorm.bf16()?.to_f32_vec(),
            pre_ff_norm: weights.pre_feedforward_layernorm.bf16()?.to_f32_vec(),
            post_ff_norm: weights.post_feedforward_layernorm.bf16()?.to_f32_vec(),
            post_ff_norm_1: weights.post_feedforward_layernorm_1.bf16()?.to_f32_vec(),
            pre_ff_norm_2: weights.pre_feedforward_layernorm_2.bf16()?.to_f32_vec(),
            post_ff_norm_2: weights.post_feedforward_layernorm_2.bf16()?.to_f32_vec(),
            mlp_gate: CachedLinear::from_bf16(weights.mlp_gate.bf16()?, inter, hidden),
            mlp_up: CachedLinear::from_bf16(weights.mlp_up.bf16()?, inter, hidden),
            mlp_down: CachedLinear::from_bf16(weights.mlp_down.bf16()?, hidden, inter),
            router_proj: weights.router_proj.bf16()?.to_f32_vec(),
            router_scale: weights.router_scale.bf16()?.to_f32_vec(),
            per_expert_scale: weights.router_per_expert_scale.bf16()?.to_f32_vec(),
            layer_scalar: weights.layer_scalar.bf16_scalar()?,
            expert_gate_up: RefCell::new(vec![None; experts]),
            expert_down: RefCell::new(vec![None; experts]),
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
            let w_off = expert * self.gate_up_stride;
            let w_raw: Vec<u16> = (0..out_dim * self.hidden)
                .map(|i| gate_up.get(w_off + i))
                .collect();
            cache[expert] = Some(transpose_weight_bf16(&w_raw, out_dim, self.hidden));
        }
    }

    fn ensure_expert_down(&self, down: Bf16Slice<'_>, expert: usize) {
        let mut cache = self.expert_down.borrow_mut();
        if cache[expert].is_none() {
            let w_off = expert * self.down_stride;
            let w_raw: Vec<u16> = (0..self.hidden * self.moe_inter)
                .map(|i| down.get(w_off + i))
                .collect();
            cache[expert] = Some(transpose_weight_bf16(&w_raw, self.hidden, self.moe_inter));
        }
    }

    pub fn with_expert_gate_up_t<R>(
        &self,
        gate_up: Bf16Slice<'_>,
        expert: usize,
        f: impl FnOnce(&[u16]) -> R,
    ) -> R {
        self.ensure_expert_gate_up(gate_up, expert);
        let cache = self.expert_gate_up.borrow();
        f(cache[expert].as_ref().unwrap())
    }

    pub fn with_expert_down_t<R>(
        &self,
        down: Bf16Slice<'_>,
        expert: usize,
        f: impl FnOnce(&[u16]) -> R,
    ) -> R {
        self.ensure_expert_down(down, expert);
        let cache = self.expert_down.borrow();
        f(cache[expert].as_ref().unwrap())
    }

    /// Drop lazily transposed expert weights (call after each layer forward).
    pub fn clear_expert_cache(&self) {
        let n = self.expert_gate_up.borrow().len();
        *self.expert_gate_up.borrow_mut() = vec![None; n];
        *self.expert_down.borrow_mut() = vec![None; n];
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
    pub final_norm: Vec<f32>,
}

impl GpuDecoderWeightCache {
    pub fn load(store: &WeightStore, text: &TextConfig) -> Result<Self, Error> {
        let mut layers = Vec::with_capacity(text.num_hidden_layers);
        for layer in 0..text.num_hidden_layers {
            let weights = DecoderLayerWeights::load(store, layer, text)?;
            layers.push(GpuLayerWeightCache::load(&weights, text)?);
        }
        let final_norm = store.tensor("model.decoder.norm.weight")?.bf16()?.to_f32_vec();
        Ok(Self { layers, final_norm })
    }
}
