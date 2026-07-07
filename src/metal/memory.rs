//! Rough resident-memory estimates for planning GPU decoder runs.

use crate::config::TextConfig;
use crate::metal::expert_cache::{
    BUFFER_POOL_FUDGE_BYTES, BUFFER_POOL_LARGE_BYTES, EXPERT_MAX_FRACTION_OF_RESIDENT_CAP,
    ExpertCacheStats, GPU_RESIDENT_FRACTION, expert_entry_bytes,
};
use crate::metal::weights::GpuDecoderWeightCache;

#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2::runtime::ProtocolObject;
#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal::MTLDevice;

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
        println!(
            "  weight cache:   {:.1} MiB",
            self.weight_cache_bytes as f64 / (1024.0 * 1024.0)
        );
        println!(
            "  layer scratch:  {:.1} MiB",
            self.layer_scratch_bytes as f64 / (1024.0 * 1024.0)
        );
        println!(
            "  kv cache:       {:.1} MiB",
            self.kv_cache_bytes as f64 / (1024.0 * 1024.0)
        );
        println!(
            "  activations:    {:.1} MiB",
            self.activations_bytes as f64 / (1024.0 * 1024.0)
        );
        println!(
            "  logits:         {:.1} MiB",
            self.logits_bytes as f64 / (1024.0 * 1024.0)
        );
        println!(
            "  total:          {:.1} MiB",
            self.total_bytes() as f64 / (1024.0 * 1024.0)
        );
    }
}

pub fn estimate_decoder_forward(
    text: &TextConfig,
    seq_len: usize,
    kv_len: usize,
) -> MemoryEstimate {
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

    let layer_scratch_bytes =
        layer_attn_scratch_bytes_gpu(text, seq, total_kv, hidden, inter, moe_inter, experts) * 2;

    // GPU KV is sized for encoder prefix capacity + canvas (see `GpuKvCache::new`).
    let kv_dim = max_kv_dim(text);
    let kv_cache_bytes = layers * (kv + seq) * kv_dim * 2 * 4;

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
    _total_kv: u64,
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
    let attn_bufs = (seq * hidden * 2
        + seq * q_dim * 2
        + seq * kv_dim * 4
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
    let attn_bufs = (seq * hidden * 2
        + seq * q_dim * 2
        + seq * kv_dim * 4
        + total_kv * kv_dim * 8
        + seq * text.global_head_dim.max(text.head_dim) as u64 * 4)
        * 4;
    let attn_weights = (hidden * q_dim + hidden * kv_dim + q_dim * hidden + hidden * kv_dim) * 4;
    let ff_bufs = seq * (hidden * 5 + inter * 2 + moe_inter * 3) * 4;
    let moe_router = seq * (hidden + experts) * 4;
    attn_bufs + scores + attn_weights + ff_bufs + moe_router
}

fn max_kv_dim(text: &TextConfig) -> u64 {
    let sliding = text.num_key_value_heads as u64 * text.head_dim as u64;
    let full = text.num_global_key_value_heads as u64 * text.global_head_dim as u64;
    sliding.max(full)
}

/// Resident bytes for a generate/bench session (decoder forward + denoise side buffers).
pub fn estimate_session_resident_bytes(
    text: &TextConfig,
    canvas_len: usize,
    max_encoder_kv: usize,
) -> u64 {
    let est = estimate_decoder_forward(text, canvas_len, max_encoder_kv);
    let mut bytes = est.total_bytes();

    // Denoise keeps processed + sample + self-conditioning logit tensors.
    bytes += est.logits_bytes * 2;

    // Self-conditioning embed signal + probability scratch.
    let hidden = text.hidden_size as u64;
    let canvas = canvas_len as u64;
    bytes += canvas * hidden * 4 * 2;

    // Encoder prefill/extend scratch (embed + hidden ping-pong).
    let enc_seq = max_encoder_kv.max(canvas_len) as u64;
    bytes += enc_seq * hidden * 4 * 3;

    // Transient Metal buffer pool peaks with canvas width.
    bytes += if canvas_len >= 256 {
        BUFFER_POOL_LARGE_BYTES
    } else {
        BUFFER_POOL_FUDGE_BYTES
    };

    bytes
}

/// Metal-reported working set limit (unified memory on Apple Silicon).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu_working_set_cap_bytes(device: &ProtocolObject<dyn MTLDevice>) -> u64 {
    let recommended = device.recommendedMaxWorkingSetSize();
    if recommended > 0 {
        return recommended;
    }
    // Fallback if the driver does not report a limit.
    16_u64 * 1024 * 1024 * 1024
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn gpu_working_set_cap_bytes() -> u64 {
    16_u64 * 1024 * 1024 * 1024
}

/// LRU budget for transposed MoE experts: 80% of working set minus estimated forward resident.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn expert_cache_budget_bytes(
    device: &ProtocolObject<dyn MTLDevice>,
    text: &TextConfig,
    seq_len: usize,
    kv_len: usize,
) -> u64 {
    let cap = (gpu_working_set_cap_bytes(device) as f64 * GPU_RESIDENT_FRACTION) as u64;
    let forward = estimate_session_resident_bytes(text, seq_len, kv_len);
    let entry = expert_entry_bytes(text);
    let expert_cap = (cap as f64 * EXPERT_MAX_FRACTION_OF_RESIDENT_CAP) as u64;
    cap.saturating_sub(forward).min(expert_cap).max(entry)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn log_gpu_memory_plan(
    device: &ProtocolObject<dyn MTLDevice>,
    text: &TextConfig,
    seq_len: usize,
    kv_len: usize,
    expert_budget: u64,
) {
    let working_set = gpu_working_set_cap_bytes(device);
    let cap = (working_set as f64 * GPU_RESIDENT_FRACTION) as u64;
    let forward = estimate_session_resident_bytes(text, seq_len, kv_len);
    let expert_cap = (cap as f64 * EXPERT_MAX_FRACTION_OF_RESIDENT_CAP) as u64;
    let entry = expert_entry_bytes(text);
    let max_experts = expert_budget / entry.max(1);
    let unified = if device.hasUnifiedMemory() {
        "yes"
    } else {
        "no"
    };
    eprintln!(
        "  GPU memory: working_set={:.1} GiB, unified={unified}, resident cap={:.0}% ({:.1} GiB)",
        working_set as f64 / (1024.0 * 1024.0 * 1024.0),
        GPU_RESIDENT_FRACTION * 100.0,
        cap as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    eprintln!(
        "  session reserve (canvas={seq_len}, encoder_kv={kv_len}): {:.1} MiB",
        forward as f64 / (1024.0 * 1024.0),
    );
    eprintln!(
        "  expert LRU budget: {:.1} MiB (~{max_experts} experts @ {:.1} MiB each, max {:.1} MiB = {:.0}% of resident)",
        expert_budget as f64 / (1024.0 * 1024.0),
        entry as f64 / (1024.0 * 1024.0),
        expert_cap as f64 / (1024.0 * 1024.0),
        EXPERT_MAX_FRACTION_OF_RESIDENT_CAP * 100.0,
    );
}

pub fn estimate_paged_layer_bytes(text: &TextConfig) -> u64 {
    let hidden = text.hidden_size as u64;
    let inter = text.intermediate_size as u64;
    let per_layer_dense = 2 * (inter * hidden + hidden * inter) + inter * hidden;
    // Attention + MLP transposed bf16 (q/k/v/o + gate/up/down); norms in f32.
    let attn_dense = (hidden * hidden * 4) as u64 * 2; // rough upper bound q+k+v+o
    let per_layer_f32 =
        hidden * 9 + text.num_experts as u64 * hidden + hidden + text.num_experts as u64;
    per_layer_dense * 2 + attn_dense + per_layer_f32 * 4
}
