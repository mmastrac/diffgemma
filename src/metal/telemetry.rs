//! Per-forward and per-step counters for Q0 calibration (PLAN2).

use crate::metal::expert_cache::expert_entry_bytes;
use crate::config::TextConfig;

/// Counters accumulated during one GPU decoder forward.
#[derive(Debug, Default, Clone)]
pub struct ForwardTelemetry {
    pub gpu_syncs: u64,
    pub gpu_readback_bytes: u64,
    /// Unique routed experts per layer (index = layer id).
    pub expert_unique_per_layer: Vec<u32>,
    /// `expert_entry_bytes × unique experts` summed over layers (bf16 working set touched).
    pub expert_weight_bytes_touched: u64,
    pub expert_hits: u64,
    pub expert_misses: u64,
    pub expert_upload_bytes: u64,
    pub dense_gpu_upload_bytes: u64,
    /// `canvas × vocab × 4` when lm_head runs on CPU.
    pub lm_head_logits_bytes: u64,
}

impl ForwardTelemetry {
    /// One monolithic denoise step: single `waitUntilCompleted`, host readback tracked separately.
    pub fn monolithic_gpu_step(host_readback_bytes: u64) -> Self {
        Self {
            gpu_syncs: 1,
            gpu_readback_bytes: host_readback_bytes,
            expert_weight_bytes_touched: 0,
            expert_hits: 0,
            expert_misses: 0,
            expert_upload_bytes: 0,
            dense_gpu_upload_bytes: 0,
            lm_head_logits_bytes: 0,
            ..Self::default()
        }
    }

    /// Backward-compatible alias (assumes P2.1 hot-path readback budget).
    pub fn monolithic_gpu_step_default() -> Self {
        Self::monolithic_gpu_step(0)
    }

    pub fn record_expert_layer(
        &mut self,
        layer: usize,
        unique_experts: usize,
        text: &TextConfig,
    ) {
        if self.expert_unique_per_layer.len() <= layer {
            self.expert_unique_per_layer.resize(layer + 1, 0);
        }
        self.expert_unique_per_layer[layer] = unique_experts as u32;
        self.expert_weight_bytes_touched +=
            unique_experts as u64 * expert_entry_bytes(text);
    }

    /// Monolithic grouped MoE: per-layer unique counts + `.dgq` blob bytes/expert.
    pub fn record_expert_layers_grouped(
        &mut self,
        counts: &[u32],
        weight_bytes_per_expert: u64,
    ) {
        self.expert_unique_per_layer = counts.to_vec();
        if weight_bytes_per_expert > 0 {
            let total_unique: u64 = counts.iter().map(|&u| u as u64).sum();
            self.expert_weight_bytes_touched = total_unique * weight_bytes_per_expert;
        }
    }

    pub fn expert_unique_mean(&self) -> f64 {
        if self.expert_unique_per_layer.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.expert_unique_per_layer.iter().map(|&n| n as u64).sum();
        sum as f64 / self.expert_unique_per_layer.len() as f64
    }

    pub fn expert_unique_min_max(&self) -> (u32, u32) {
        let mut min = u32::MAX;
        let mut max = 0u32;
        for &n in &self.expert_unique_per_layer {
            min = min.min(n);
            max = max.max(n);
        }
        if self.expert_unique_per_layer.is_empty() {
            (0, 0)
        } else {
            (min, max)
        }
    }

    pub fn print_summary(&self, label: &str) {
        let (emin, emax) = self.expert_unique_min_max();
        println!("{label}");
        println!("  gpu_syncs:              {}", self.gpu_syncs);
        println!(
            "  gpu_readback:           {:.2} MiB",
            self.gpu_readback_bytes as f64 / (1024.0 * 1024.0)
        );
        println!(
            "  expert unique/layer:    mean {:.1}, min {emin}, max {emax} ({} layers)",
            self.expert_unique_mean(),
            self.expert_unique_per_layer.len()
        );
        println!(
            "  expert weight touched:  {:.2} GiB/step",
            self.expert_weight_bytes_touched as f64 / (1024.0_f64.powi(3))
        );
        println!(
            "  expert LRU:             {} hits, {} misses, {:.2} MiB uploaded",
            self.expert_hits,
            self.expert_misses,
            self.expert_upload_bytes as f64 / (1024.0 * 1024.0)
        );
        println!(
            "  dense GPU upload:       {:.2} MiB",
            self.dense_gpu_upload_bytes as f64 / (1024.0 * 1024.0)
        );
        println!(
            "  lm_head logits (CPU):   {:.2} MiB",
            self.lm_head_logits_bytes as f64 / (1024.0 * 1024.0)
        );
    }
}

/// Per denoise-step phase split (generate-gpu).
#[derive(Debug, Default, Clone)]
pub struct StepPhaseTelemetry {
    pub decoder_ms: f64,
    pub sampler_ms: f64,
    pub forward: ForwardTelemetry,
}

#[derive(Debug, Default, Clone)]
pub struct SessionTelemetry {
    pub steps: Vec<StepPhaseTelemetry>,
}

impl SessionTelemetry {
    pub fn aggregate_forward(&self) -> ForwardTelemetry {
        let mut agg = ForwardTelemetry::default();
        for step in &self.steps {
            let f = &step.forward;
            agg.gpu_syncs += f.gpu_syncs;
            agg.gpu_readback_bytes += f.gpu_readback_bytes;
            agg.expert_weight_bytes_touched += f.expert_weight_bytes_touched;
            agg.expert_hits += f.expert_hits;
            agg.expert_misses += f.expert_misses;
            agg.expert_upload_bytes += f.expert_upload_bytes;
            agg.dense_gpu_upload_bytes += f.dense_gpu_upload_bytes;
            agg.lm_head_logits_bytes += f.lm_head_logits_bytes;
            let n = f.expert_unique_per_layer.len();
            if agg.expert_unique_per_layer.len() < n {
                agg.expert_unique_per_layer.resize(n, 0);
            }
            for (i, &u) in f.expert_unique_per_layer.iter().enumerate() {
                agg.expert_unique_per_layer[i] += u;
            }
        }
        let step_count = self.steps.len().max(1) as f64;
        for u in &mut agg.expert_unique_per_layer {
            *u = ((*u as f64) / step_count).round() as u32;
        }
        agg
    }

    pub fn mean_decoder_ms(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        self.steps.iter().map(|s| s.decoder_ms).sum::<f64>() / self.steps.len() as f64
    }

    pub fn mean_sampler_ms(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        self.steps.iter().map(|s| s.sampler_ms).sum::<f64>() / self.steps.len() as f64
    }

    pub fn print_summary(&self, label: &str) {
        println!("{label} ({} steps)", self.steps.len());
        println!(
            "  decoder: {:.1} ms/step (mean)",
            self.mean_decoder_ms()
        );
        println!(
            "  sampler: {:.1} ms/step (mean)",
            self.mean_sampler_ms()
        );
        self.aggregate_forward().print_summary("  per-step forward (mean fields):");
    }
}
