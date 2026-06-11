//! Rough resident-memory estimates for planning GPU decoder runs.

use crate::config::TextConfig;
use crate::metal::weights::GpuDecoderWeightCache;

#[derive(Debug, Clone, Copy)]
pub struct MemoryEstimate {
    pub weight_cache_bytes: u64,
    pub layer_scratch_bytes: u64,
    pub kv_cache_bytes: u64,
    pub activations_bytes: u64,
    pub logits_bytes: u64,
}

impl MemoryEstimate {
    pub fn total_bytes(&self) -> u64 {
        self.weight_cache_bytes
            + self.layer_scratch_bytes
            + self.kv_cache_bytes
            + self.activations_bytes
            + self.logits_bytes
    }

    pub fn print_summary(&self, label: &str) {
        println!("{label} memory estimate (resident, approximate):");
        println!("  weight cache:   {:.1} MiB", self.weight_cache_bytes as f64 / (1024.0 * 1024.0));
        println!("  layer scratch:  {:.1} MiB", self.layer_scratch_bytes as f64 / (1024.0 * 1024.0));
        println!("  kv cache:       {:.1} MiB", self.kv_cache_bytes as f64 / (1024.0 * 1024.0));
        println!("  activations:    {:.1} MiB", self.activations_bytes as f64 / (1024.0 * 1024.0));
        println!("  logits:         {:.1} MiB", self.logits_bytes as f64 / (1024.0 * 1024.0));
        println!("  total:          {:.1} MiB", self.total_bytes() as f64 / (1024.0 * 1024.0));
    }
}

pub fn estimate_decoder_forward(text: &TextConfig, seq_len: usize, kv_len: usize) -> MemoryEstimate {
    let hidden = text.hidden_size as u64;
    let inter = text.intermediate_size as u64;
    let moe_inter = text.moe_intermediate_size as u64;
    let experts = text.num_experts as u64;
    let layers = text.num_hidden_layers as u64;
    let vocab = text.vocab_size as u64;
    let seq = seq_len as u64;
    let kv = kv_len as u64;
    let total_kv = kv + seq;

    let weight_cache_bytes = estimate_paged_layer_bytes(text) + hidden * 4;

    let layer_scratch_bytes = layer_attn_scratch_bytes_gpu(
        text, seq, total_kv, hidden, inter, moe_inter, experts,
    ) * 2;

    // Per-layer KV tensors in the dummy cache (sliding layers dominate size).
    let kv_dim = text.num_key_value_heads as u64 * text.head_dim as u64;
    let kv_cache_bytes = layers * kv * kv_dim * 2 * 4;

    // Embed + hidden ping-pong in GpuDecoderScratch.cpu.
    let activations_bytes = seq * hidden * 4 * 4;

    let logits_bytes = seq * vocab * 4;

    MemoryEstimate {
        weight_cache_bytes,
        layer_scratch_bytes,
        kv_cache_bytes,
        activations_bytes,
        logits_bytes,
    }
}

fn layer_attn_scratch_bytes_gpu(
    text: &TextConfig,
    seq: u64,
    total_kv: u64,
    hidden: u64,
    inter: u64,
    moe_inter: u64,
    experts: u64,
) -> u64 {
    let sliding_q = text.num_attention_heads as u64 * text.head_dim as u64;
    let full_q = text.num_attention_heads as u64 * text.global_head_dim as u64;
    let sliding_kv = text.num_key_value_heads as u64 * text.head_dim as u64;
    let full_kv = text.num_global_key_value_heads as u64 * text.global_head_dim as u64;
    let q_dim = sliding_q.max(full_q);
    let kv_dim = sliding_kv.max(full_kv);
    // GPU path: no k_full/v_full/scores (KV on Metal buffers).
    let attn_bufs = (seq * hidden * 2 + seq * q_dim * 2 + seq * kv_dim * 4
        + seq * text.global_head_dim.max(text.head_dim) as u64 * 4)
        * 4;
    let ff_bufs = seq * (hidden * 5 + inter * 2 + moe_inter * 3) * 4;
    let moe_router = seq * (hidden + experts) * 4;
    attn_bufs + ff_bufs + moe_router
}

#[allow(dead_code)]
fn layer_attn_scratch_bytes_cpu(
    text: &TextConfig,
    seq: u64,
    total_kv: u64,
    hidden: u64,
    inter: u64,
    moe_inter: u64,
    experts: u64,
) -> u64 {
    let sliding_q = text.num_attention_heads as u64 * text.head_dim as u64;
    let full_q = text.num_attention_heads as u64 * text.global_head_dim as u64;
    let sliding_kv = text.num_key_value_heads as u64 * text.head_dim as u64;
    let full_kv = text.num_global_key_value_heads as u64 * text.global_head_dim as u64;
    let q_dim = sliding_q.max(full_q);
    let kv_dim = sliding_kv.max(full_kv);
    let scores = seq * text.num_attention_heads as u64 * total_kv * 4;
    let attn_bufs = (seq * hidden * 2 + seq * q_dim * 2 + seq * kv_dim * 4 + total_kv * kv_dim * 8
        + seq * text.global_head_dim.max(text.head_dim) as u64 * 4)
        * 4;
    let attn_weights = (hidden * q_dim + hidden * kv_dim + q_dim * hidden + hidden * kv_dim) * 4;
    let ff_bufs = seq * (hidden * 5 + inter * 2 + moe_inter * 3) * 4;
    let moe_router = seq * (hidden + experts) * 4;
    attn_bufs + scores + attn_weights + ff_bufs + moe_router
}

pub fn estimate_weight_cache(cache: &GpuDecoderWeightCache) -> u64 {
    cache.resident_bytes()
}

pub fn estimate_paged_layer_bytes(text: &TextConfig) -> u64 {
    let hidden = text.hidden_size as u64;
    let inter = text.intermediate_size as u64;
    let per_layer_dense = 2 * (inter * hidden + hidden * inter) + inter * hidden;
    // Attention + MLP transposed bf16 (q/k/v/o + gate/up/down); norms in f32.
    let attn_dense = (hidden * hidden * 4) as u64 * 2; // rough upper bound q+k+v+o
    let per_layer_f32 = hidden * 9
        + text.num_experts as u64 * hidden
        + hidden
        + text.num_experts as u64;
    per_layer_dense * 2 + attn_dense + per_layer_f32 * 4
}
