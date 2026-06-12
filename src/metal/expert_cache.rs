//! LRU cache for transposed MoE expert weights, bounded by GPU memory budget.

use crate::buffer::Buffer;
use crate::config::TextConfig;
use crate::fast_slice::{transpose_bf16_weight_into, FastBf16Slice, FastSlice};
use crate::tensor::Bf16Slice;
use std::collections::VecDeque;

/// Target max fraction of Metal `recommendedMaxWorkingSetSize` for all resident data.
pub const GPU_RESIDENT_FRACTION: f64 = 0.80;

/// Conservative allowance for transient `BufferPool` allocations not in forward estimates.
pub const BUFFER_POOL_FUDGE_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub struct ExpertCacheStats {
    pub budget_bytes: u64,
    pub used_bytes: u64,
    pub entries: usize,
    pub evictions: u64,
}

/// Bytes for one expert's transposed gate_up + down (bf16).
pub fn expert_entry_bytes(text: &TextConfig) -> u64 {
    let moe_inter = text.moe_intermediate_size as u64;
    let hidden = text.hidden_size as u64;
    let gate_up = moe_inter * 2 * hidden * 2;
    let down = hidden * moe_inter * 2;
    gate_up + down
}

struct LayerExpertWeightCache {
    gate_up: Vec<Option<Buffer<u16>>>,
    down: Vec<Option<Buffer<u16>>>,
}

struct ExpertLruCache {
    budget_bytes: u64,
    used_bytes: u64,
    evictions: u64,
    touch_order: VecDeque<(usize, usize)>,
    layers: Vec<LayerExpertWeightCache>,
    entry_bytes: u64,
    moe_inter: usize,
    hidden: usize,
    gate_up_stride: usize,
    down_stride: usize,
}

impl ExpertLruCache {
    fn new(text: &TextConfig, budget_bytes: u64) -> Self {
        let experts = text.num_experts;
        let entry_bytes = expert_entry_bytes(text);
        let budget_bytes = budget_bytes.max(entry_bytes);
        Self {
            budget_bytes,
            used_bytes: 0,
            evictions: 0,
            touch_order: VecDeque::new(),
            layers: (0..text.num_hidden_layers)
                .map(|_| LayerExpertWeightCache {
                    gate_up: (0..experts).map(|_| None).collect(),
                    down: (0..experts).map(|_| None).collect(),
                })
                .collect(),
            entry_bytes,
            moe_inter: text.moe_intermediate_size,
            hidden: text.hidden_size,
            gate_up_stride: text.moe_intermediate_size * 2 * text.hidden_size,
            down_stride: text.hidden_size * text.moe_intermediate_size,
        }
    }

    fn stats(&self) -> ExpertCacheStats {
        let entries = self
            .layers
            .iter()
            .flat_map(|l| l.gate_up.iter())
            .filter(|e| e.is_some())
            .count();
        ExpertCacheStats {
            budget_bytes: self.budget_bytes,
            used_bytes: self.used_bytes,
            entries,
            evictions: self.evictions,
        }
    }

    fn touch(&mut self, layer: usize, expert: usize) {
        if let Some(pos) = self
            .touch_order
            .iter()
            .position(|&(l, e)| l == layer && e == expert)
        {
            self.touch_order.remove(pos);
        }
        self.touch_order.push_back((layer, expert));
    }

    fn evict_one(&mut self) {
        let Some((layer, expert)) = self.touch_order.pop_front() else {
            return;
        };
        let slot = &mut self.layers[layer];
        if let Some(buf) = slot.gate_up[expert].take() {
            self.used_bytes = self.used_bytes.saturating_sub(buf.len() as u64 * 2);
        }
        if let Some(buf) = slot.down[expert].take() {
            self.used_bytes = self.used_bytes.saturating_sub(buf.len() as u64 * 2);
        }
        self.evictions += 1;
    }

    fn make_room(&mut self, need: u64) {
        while self.used_bytes + need > self.budget_bytes {
            if self.touch_order.is_empty() {
                break;
            }
            self.evict_one();
        }
    }

    fn ensure_expert_weights(
        &mut self,
        layer: usize,
        gate_up: Bf16Slice<'_>,
        down: Bf16Slice<'_>,
        expert: usize,
    ) {
        if self.layers[layer].gate_up[expert].is_some() {
            self.touch(layer, expert);
            return;
        }

        self.make_room(self.entry_bytes);

        let out_dim = self.moe_inter * 2;
        let gate_elems = out_dim * self.hidden;
        let mut gate_buf = Buffer::new(gate_elems);
        let src = FastBf16Slice::from_bf16(gate_up);
        let src = unsafe { src.slice_unchecked(expert * self.gate_up_stride, gate_elems) };
        transpose_bf16_weight_into(src, gate_buf.as_fast_slice_mut(), out_dim, self.hidden);

        let down_elems = self.hidden * self.moe_inter;
        let mut down_buf = Buffer::new(down_elems);
        let src = FastBf16Slice::from_bf16(down);
        let src = unsafe { src.slice_unchecked(expert * self.down_stride, down_elems) };
        transpose_bf16_weight_into(src, down_buf.as_fast_slice_mut(), self.hidden, self.moe_inter);

        self.used_bytes += self.entry_bytes;
        let slot = &mut self.layers[layer];
        slot.gate_up[expert] = Some(gate_buf);
        slot.down[expert] = Some(down_buf);
        self.touch(layer, expert);
    }

    fn with_gate_up<R>(
        &mut self,
        layer: usize,
        gate_up: Bf16Slice<'_>,
        down: Bf16Slice<'_>,
        expert: usize,
        f: impl FnOnce(FastSlice<'_, u16>) -> R,
    ) -> R {
        self.ensure_expert_weights(layer, gate_up, down, expert);
        let layers = &self.layers;
        f(layers[layer].gate_up[expert]
            .as_ref()
            .expect("expert gate_up")
            .as_fast_slice())
    }

    fn with_down<R>(
        &mut self,
        layer: usize,
        gate_up: Bf16Slice<'_>,
        down: Bf16Slice<'_>,
        expert: usize,
        f: impl FnOnce(FastSlice<'_, u16>) -> R,
    ) -> R {
        self.ensure_expert_weights(layer, gate_up, down, expert);
        let layers = &self.layers;
        f(layers[layer].down[expert]
            .as_ref()
            .expect("expert down")
            .as_fast_slice())
    }

    fn resident_bytes(&self) -> u64 {
        self.used_bytes
    }
}

pub struct ExpertWeightCache {
    inner: ExpertLruCache,
}

impl ExpertWeightCache {
    pub fn new(text: &TextConfig, budget_bytes: u64) -> Self {
        Self {
            inner: ExpertLruCache::new(text, budget_bytes),
        }
    }

    pub fn stats(&self) -> ExpertCacheStats {
        self.inner.stats()
    }

    pub fn resident_bytes(&self) -> u64 {
        self.inner.resident_bytes()
    }

    pub fn with_expert_gate_up_t<R>(
        &mut self,
        layer: usize,
        gate_up: Bf16Slice<'_>,
        down: Bf16Slice<'_>,
        expert: usize,
        f: impl FnOnce(FastSlice<'_, u16>) -> R,
    ) -> R {
        self.inner.with_gate_up(layer, gate_up, down, expert, f)
    }

    pub fn with_expert_down_t<R>(
        &mut self,
        layer: usize,
        gate_up: Bf16Slice<'_>,
        down: Bf16Slice<'_>,
        expert: usize,
        f: impl FnOnce(FastSlice<'_, u16>) -> R,
    ) -> R {
        self.inner.with_down(layer, gate_up, down, expert, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru_evicts_oldest_when_over_budget() {
        let text = test_text_config();
        let entry = expert_entry_bytes(&text);
        let mut cache = ExpertLruCache::new(&text, entry);
        cache.insert_dummy(0, 0);
        cache.insert_dummy(0, 1);
        assert!(cache.layers[0].gate_up[0].is_none());
        assert!(cache.layers[0].gate_up[1].is_some());
        assert_eq!(cache.evictions, 1);
    }

    fn test_text_config() -> crate::config::TextConfig {
        use crate::config::{LayerType, RopeConfig, RopeParameters, TextConfig};
        TextConfig {
            hidden_size: 4,
            intermediate_size: 8,
            moe_intermediate_size: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            num_global_key_value_heads: 1,
            num_hidden_layers: 1,
            num_experts: 3,
            top_k_experts: 2,
            head_dim: 2,
            global_head_dim: 2,
            vocab_size: 16,
            max_position_embeddings: 32,
            sliding_window: 16,
            rms_norm_eps: 1e-6,
            final_logit_softcapping: 30.0,
            layer_types: vec![LayerType::SlidingAttention],
            rope_parameters: RopeParameters {
                full_attention: RopeConfig {
                    rope_theta: 10000.0,
                    rope_type: "default".into(),
                    partial_rotary_factor: None,
                },
                sliding_attention: RopeConfig {
                    rope_theta: 10000.0,
                    rope_type: "default".into(),
                    partial_rotary_factor: None,
                },
            },
        }
    }

    #[cfg(test)]
    impl ExpertLruCache {
        fn insert_dummy(&mut self, layer: usize, expert: usize) {
            if self.layers[layer].gate_up[expert].is_some() {
                self.touch(layer, expert);
                return;
            }
            self.make_room(self.entry_bytes);
            self.layers[layer].gate_up[expert] = Some(Buffer::new(4));
            self.layers[layer].down[expert] = Some(Buffer::new(4));
            self.used_bytes += self.entry_bytes;
            self.touch(layer, expert);
        }
    }
}
