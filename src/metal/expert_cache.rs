//! LRU cache for MoE expert bf16 weights on GPU (PyTorch layout, no transpose).

use crate::metal::buffer::BufferPool;
use crate::config::TextConfig;
use crate::tensor::Bf16Slice;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice};
use std::collections::VecDeque;

/// Target max fraction of Metal `recommendedMaxWorkingSetSize` for all resident data.
pub const GPU_RESIDENT_FRACTION: f64 = 0.80;

/// Expert LRU may use at most this fraction of the overall resident cap (not the remainder).
pub const EXPERT_MAX_FRACTION_OF_RESIDENT_CAP: f64 = 0.30;

/// Conservative allowance for transient `BufferPool` allocations (small-seq runs).
pub const BUFFER_POOL_FUDGE_BYTES: u64 = 64 * 1024 * 1024;

/// Buffer pool peak allowance for full-canvas denoise (seq=256).
pub const BUFFER_POOL_LARGE_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub struct ExpertCacheStats {
    pub budget_bytes: u64,
    pub used_bytes: u64,
    pub entries: usize,
    pub evictions: u64,
    pub hits: u64,
    pub misses: u64,
    pub upload_bytes: u64,
}

/// Bytes for one expert's gate_up + down (bf16, PyTorch layout).
pub fn expert_entry_bytes(text: &TextConfig) -> u64 {
    let moe_inter = text.moe_intermediate_size as u64;
    let hidden = text.hidden_size as u64;
    let gate_up = moe_inter * 2 * hidden * 2;
    let down = hidden * moe_inter * 2;
    gate_up + down
}

struct LayerExpertWeightCache {
    gate_up: Vec<Option<Retained<ProtocolObject<dyn MTLBuffer>>>>,
    down: Vec<Option<Retained<ProtocolObject<dyn MTLBuffer>>>>,
}

struct ExpertLruCache {
    budget_bytes: u64,
    used_bytes: u64,
    evictions: u64,
    hits: u64,
    misses: u64,
    upload_bytes: u64,
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
            hits: 0,
            misses: 0,
            upload_bytes: 0,
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
            hits: self.hits,
            misses: self.misses,
            upload_bytes: self.upload_bytes,
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
            self.used_bytes = self.used_bytes.saturating_sub(buf.length() as u64);
        }
        if let Some(buf) = slot.down[expert].take() {
            self.used_bytes = self.used_bytes.saturating_sub(buf.length() as u64);
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

    fn upload_expert_bf16(
        device: &ProtocolObject<dyn MTLDevice>,
        pool: &mut BufferPool,
        src: Bf16Slice<'_>,
        elem_off: usize,
        elem_len: usize,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let bytes = elem_len * 2;
        let buf = pool
            .allocate(device, bytes)
            .expect("expert weight buffer alloc failed");
        let src_bytes = &src.as_bytes()[elem_off * 2..elem_off * 2 + bytes];
        BufferPool::write_bytes(&buf, src_bytes);
        buf
    }

    fn ensure_expert_weights(
        &mut self,
        layer: usize,
        gate_up: Bf16Slice<'_>,
        down: Bf16Slice<'_>,
        expert: usize,
        device: &ProtocolObject<dyn MTLDevice>,
        pool: &mut BufferPool,
    ) {
        if self.layers[layer].gate_up[expert].is_some() {
            self.touch(layer, expert);
            self.hits += 1;
            return;
        }

        self.misses += 1;
        self.make_room(self.entry_bytes);

        let gate_elems = self.moe_inter * 2 * self.hidden;
        let down_elems = self.hidden * self.moe_inter;
        let gate_off = expert * self.gate_up_stride;
        let down_off = expert * self.down_stride;

        let gate_buf = Self::upload_expert_bf16(device, pool, gate_up, gate_off, gate_elems);
        let down_buf = Self::upload_expert_bf16(device, pool, down, down_off, down_elems);

        let uploaded = gate_buf.length() as u64 + down_buf.length() as u64;
        self.used_bytes += uploaded;
        self.upload_bytes += uploaded;
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
        device: &ProtocolObject<dyn MTLDevice>,
        pool: &mut BufferPool,
        f: impl FnOnce(&ProtocolObject<dyn MTLBuffer>) -> R,
    ) -> R {
        self.ensure_expert_weights(layer, gate_up, down, expert, device, pool);
        let buf = self.layers[layer].gate_up[expert]
            .as_ref()
            .expect("expert gate_up");
        f(buf)
    }

    fn with_down<R>(
        &mut self,
        layer: usize,
        gate_up: Bf16Slice<'_>,
        down: Bf16Slice<'_>,
        expert: usize,
        device: &ProtocolObject<dyn MTLDevice>,
        pool: &mut BufferPool,
        f: impl FnOnce(&ProtocolObject<dyn MTLBuffer>) -> R,
    ) -> R {
        self.ensure_expert_weights(layer, gate_up, down, expert, device, pool);
        let buf = self.layers[layer].down[expert]
            .as_ref()
            .expect("expert down");
        f(buf)
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

    pub fn with_expert_gate_up_buf<R>(
        &mut self,
        layer: usize,
        gate_up: Bf16Slice<'_>,
        down: Bf16Slice<'_>,
        expert: usize,
        device: &ProtocolObject<dyn MTLDevice>,
        pool: &mut BufferPool,
        f: impl FnOnce(&ProtocolObject<dyn MTLBuffer>) -> R,
    ) -> R {
        self.inner
            .with_gate_up(layer, gate_up, down, expert, device, pool, f)
    }

    pub fn with_expert_down_buf<R>(
        &mut self,
        layer: usize,
        gate_up: Bf16Slice<'_>,
        down: Bf16Slice<'_>,
        expert: usize,
        device: &ProtocolObject<dyn MTLDevice>,
        pool: &mut BufferPool,
        f: impl FnOnce(&ProtocolObject<dyn MTLBuffer>) -> R,
    ) -> R {
        self.inner
            .with_down(layer, gate_up, down, expert, device, pool, f)
    }

    pub fn prefetch_expert(
        &mut self,
        layer: usize,
        gate_up: Bf16Slice<'_>,
        down: Bf16Slice<'_>,
        expert: usize,
        device: &ProtocolObject<dyn MTLDevice>,
        pool: &mut BufferPool,
    ) {
        self.inner
            .ensure_expert_weights(layer, gate_up, down, expert, device, pool);
    }

    pub fn expert_gate_up_buf(
        &self,
        layer: usize,
        expert: usize,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        self.inner.layers[layer].gate_up[expert]
            .as_ref()
            .expect("expert gate_up not prefetched")
            .clone()
    }

    pub fn expert_down_buf(
        &self,
        layer: usize,
        expert: usize,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        self.inner.layers[layer].down[expert]
            .as_ref()
            .expect("expert down not prefetched")
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_entry_bytes_positive() {
        let text = test_text_config();
        assert!(expert_entry_bytes(&text) > 0);
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
}
