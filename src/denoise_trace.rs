//! Compact per-step denoise traces for P1.6 parity (Rust vs HuggingFace).

use crate::sample::StepEntropyStats;
use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DenoiseStepTrace {
    /// 1-based block index within the generation run.
    pub block: u32,
    /// 1-based denoise step index within this block.
    pub step_index: u32,
    /// HuggingFace-style countdown (`max_denoising_steps` .. 1).
    pub cur_step: u32,
    pub accept_count: u32,
    pub low_entropy_positions: u32,
    pub min_entropy: f32,
    pub mean_entropy: f32,
    pub max_entropy: f32,
    /// First 16 argmax token ids after this step.
    pub argmax_prefix: Vec<u32>,
    /// First 16 per-position entropies (nats) when `DGQ_TRACE_ENTROPY=1`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entropy_prefix: Option<Vec<f32>>,
    pub early_stop: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DenoiseTrace {
    pub schema_version: u32,
    pub source: String,
    pub prompt: String,
    pub prompt_token_ids: Vec<u32>,
    pub seed: u64,
    pub max_denoise_steps: usize,
    pub layers: usize,
    pub max_new_tokens: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights_profile: Option<String>,
    pub entropy_bound: f32,
    pub step_traces: Vec<DenoiseStepTrace>,
    pub denoise_steps_run: usize,
    pub blocks_committed: usize,
    pub output_token_ids: Vec<u32>,
    /// Canvas token ids at block start (before first denoise step).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_canvas_ids: Option<Vec<u32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canvas_rng: Option<String>,
}

impl DenoiseTrace {
    pub fn write(&self, path: &std::path::Path) -> Result<(), crate::safetensors::Error> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(crate::safetensors::Error::Io)?;
        }
        let text = serde_json::to_string_pretty(self).map_err(crate::safetensors::Error::Json)?;
        std::fs::write(path, text).map_err(crate::safetensors::Error::Io)
    }
}

pub fn step_trace_from_stats(
    block: u32,
    step_index: u32,
    max_denoise_steps: usize,
    stats: &StepEntropyStats,
    argmax: &[u32],
    entropies: Option<&[f32]>,
    early_stop: bool,
) -> DenoiseStepTrace {
    let full_trace = std::env::var("DGQ_TRACE_ENTROPY")
        .map(|v| v.eq_ignore_ascii_case("full"))
        .unwrap_or(false);
    let prefix_len = if full_trace {
        argmax.len()
    } else {
        argmax.len().min(16)
    };
    let entropy_prefix = entropies.map(|e| {
        let n = if full_trace { e.len() } else { e.len().min(16) };
        e[..n].to_vec()
    });
    DenoiseStepTrace {
        block,
        step_index,
        cur_step: max_denoise_steps
            .saturating_sub(step_index as usize)
            .saturating_add(1) as u32,
        accept_count: stats.accept_count,
        low_entropy_positions: stats.low_entropy_positions,
        min_entropy: stats.min_entropy,
        mean_entropy: stats.mean_entropy,
        max_entropy: stats.max_entropy,
        argmax_prefix: argmax[..prefix_len].to_vec(),
        entropy_prefix,
        early_stop,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::step_entropy_stats;

    #[test]
    fn cur_step_matches_hf_countdown() {
        let ent = vec![0.05f32; 256];
        let accept = vec![1u32; 256];
        let stats = step_entropy_stats(&ent, &accept);
        let row = step_trace_from_stats(1, 48, 48, &stats, &[1, 2, 3], None, false);
        assert_eq!(row.cur_step, 1);
        let row = step_trace_from_stats(1, 1, 48, &stats, &[1, 2, 3], None, false);
        assert_eq!(row.cur_step, 48);
    }
}
