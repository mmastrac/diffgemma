//! Validate `config.json` against monolithic shader compile-time constants (M4.5).

use crate::Error;
use crate::config::{LayerType, ModelConfig};
use crate::metal::step_kernel::{
    CANVAS, DENSE_FF, FULL_LAYERS, HID, MOE_FF, N_EXPERTS, N_LAYERS, TOP_K, VOCAB,
};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ValidatedStepModel {
    pub num_layers: usize,
    pub canvas_length: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub dims: ModelDims,
}

/// Model geometry derived from `config.json`, threaded through the step
/// runtime in place of the compile-time constants. While the constants
/// exist, `validate_step_model` guarantees the two agree; the constants are
/// removed consumer by consumer.
#[derive(Debug, Clone)]
pub struct ModelDims {
    pub hid: usize,
    pub vocab: usize,
    pub canvas: usize,
    pub n_layers: usize,
    pub dense_ff: usize,
    pub moe_ff: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub n_q_heads: usize,
    pub sliding_kv_heads: usize,
    pub sliding_head_dim: usize,
    pub full_kv_heads: usize,
    pub full_head_dim: usize,
    /// Layer indices with full (global) attention; the rest slide.
    pub full_layers: Vec<usize>,
    pub rms_eps: f64,
    /// Causal sub-chunks batched into one prefill forward (see PREFILL_SUBS).
    pub prefill_subs: usize,
}

impl ModelDims {
    /// The reference checkpoint's geometry, for test and audit helpers that
    /// run without a `config.json`. Production paths derive dims from config.
    pub fn reference() -> Self {
        use crate::metal::step_kernel as sk;
        Self {
            hid: sk::HID,
            vocab: sk::VOCAB,
            canvas: sk::CANVAS,
            n_layers: sk::N_LAYERS,
            dense_ff: sk::DENSE_FF as usize,
            moe_ff: sk::MOE_FF as usize,
            n_experts: sk::N_EXPERTS,
            top_k: sk::TOP_K,
            n_q_heads: 16,
            sliding_kv_heads: 8,
            sliding_head_dim: 256,
            full_kv_heads: 2,
            full_head_dim: 512,
            full_layers: sk::FULL_LAYERS.to_vec(),
            rms_eps: 1e-6,
            prefill_subs: sk::PREFILL_SUBS,
        }
    }

    pub fn from_config(cfg: &ModelConfig) -> Self {
        let t = &cfg.text_config;
        Self {
            hid: t.hidden_size,
            vocab: t.vocab_size,
            canvas: cfg.canvas_length,
            n_layers: t.num_hidden_layers,
            dense_ff: t.intermediate_size,
            moe_ff: t.moe_intermediate_size,
            n_experts: t.num_experts,
            top_k: t.top_k_experts,
            n_q_heads: t.num_attention_heads,
            sliding_kv_heads: t.num_key_value_heads,
            sliding_head_dim: t.head_dim,
            full_kv_heads: t.num_global_key_value_heads,
            full_head_dim: t.global_head_dim,
            full_layers: full_layers_from_config(&t.layer_types),
            rms_eps: t.rms_norm_eps,
            prefill_subs: crate::metal::step_kernel::PREFILL_SUBS,
        }
    }

    pub fn is_full_layer(&self, layer: usize) -> bool {
        self.full_layers.contains(&layer)
    }

    pub fn head_dim(&self, full: bool) -> usize {
        if full {
            self.full_head_dim
        } else {
            self.sliding_head_dim
        }
    }

    pub fn kv_heads(&self, full: bool) -> usize {
        if full {
            self.full_kv_heads
        } else {
            self.sliding_kv_heads
        }
    }

    /// Q-projection output width: every query head at this layer kind's dim.
    pub fn q_cols(&self, full: bool) -> usize {
        self.n_q_heads * self.head_dim(full)
    }

    /// K (= V) projection output width.
    pub fn kv_cols(&self, full: bool) -> usize {
        self.kv_heads(full) * self.head_dim(full)
    }

    /// Rows per batched-prefill super-chunk forward.
    pub fn prefill_m(&self) -> usize {
        self.canvas * self.prefill_subs
    }

    /// u32 words in the sampler's per-canvas accept bitmask.
    pub fn frozen_words(&self) -> usize {
        self.canvas / 32
    }
}

fn mismatch(field: &str, config: usize, shader: usize) -> Error {
    Error::NotFound(format!(
        "step-kernel {field} mismatch: config.json={config}, shader={shader}"
    ))
}

fn full_layers_from_config(layer_types: &[LayerType]) -> Vec<usize> {
    layer_types
        .iter()
        .enumerate()
        .filter_map(|(i, t)| (*t == LayerType::FullAttention).then_some(i))
        .collect()
}

/// Load and validate model config against shader ABI. Fails fast on unsupported models.
pub fn validate_step_model(model_dir: &Path) -> Result<ValidatedStepModel, Error> {
    let cfg = ModelConfig::load(model_dir)?;
    let t = &cfg.text_config;

    if cfg.canvas_length != CANVAS {
        return Err(mismatch("canvas_length", cfg.canvas_length, CANVAS));
    }
    if t.vocab_size != VOCAB {
        return Err(mismatch("vocab_size", t.vocab_size, VOCAB));
    }
    if t.hidden_size != HID {
        return Err(mismatch("hidden_size", t.hidden_size, HID));
    }
    if t.num_hidden_layers > N_LAYERS {
        return Err(Error::NotFound(format!(
            "step-kernel supports at most {N_LAYERS} layers; config has {}",
            t.num_hidden_layers
        )));
    }
    if t.num_experts != N_EXPERTS {
        return Err(mismatch("num_experts", t.num_experts, N_EXPERTS));
    }
    if t.top_k_experts != TOP_K {
        return Err(mismatch("top_k_experts", t.top_k_experts, TOP_K));
    }
    if t.intermediate_size != DENSE_FF as usize {
        return Err(mismatch(
            "intermediate_size",
            t.intermediate_size,
            DENSE_FF as usize,
        ));
    }
    if t.moe_intermediate_size != MOE_FF as usize {
        return Err(mismatch(
            "moe_intermediate_size",
            t.moe_intermediate_size,
            MOE_FF as usize,
        ));
    }
    if t.num_attention_heads != 16 {
        return Err(mismatch("num_attention_heads", t.num_attention_heads, 16));
    }
    if t.num_key_value_heads != 8 {
        return Err(mismatch("num_key_value_heads", t.num_key_value_heads, 8));
    }
    if t.head_dim != 256 {
        return Err(mismatch("head_dim", t.head_dim, 256));
    }
    if t.num_global_key_value_heads != 2 {
        return Err(mismatch(
            "num_global_key_value_heads",
            t.num_global_key_value_heads,
            2,
        ));
    }
    if t.global_head_dim != 512 {
        return Err(mismatch("global_head_dim", t.global_head_dim, 512));
    }
    if (t.rms_norm_eps - 1e-6).abs() > 1e-9 {
        return Err(Error::NotFound(format!(
            "step-kernel rms_norm_eps mismatch: config={} shader=1e-6",
            t.rms_norm_eps
        )));
    }

    let full = full_layers_from_config(&t.layer_types);
    if full.len() != FULL_LAYERS.len() {
        return Err(Error::NotFound(format!(
            "step-kernel full_attention layer count mismatch: config={} shader={}",
            full.len(),
            FULL_LAYERS.len()
        )));
    }
    for (i, &layer) in FULL_LAYERS.iter().enumerate() {
        if full.get(i) != Some(&layer) {
            return Err(Error::NotFound(format!(
                "step-kernel full_attention indices mismatch at {i}: config={full:?} shader={FULL_LAYERS:?}"
            )));
        }
    }

    Ok(ValidatedStepModel {
        num_layers: t.num_hidden_layers,
        canvas_length: cfg.canvas_length,
        vocab_size: t.vocab_size,
        hidden_size: t.hidden_size,
        dims: ModelDims::from_config(&cfg),
    })
}

pub fn log_validated_step_model(v: &ValidatedStepModel) {
    if crate::flags::progress_enabled() {
        eprintln!(
            "step-kernel model config ok (canvas={}, vocab={}, hidden={}, layers={})",
            v.canvas_length, v.vocab_size, v.hidden_size, v.num_layers
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_default_transformer_config() {
        let dir = Path::new("model/transformer");
        if !dir.join("config.json").exists() {
            return;
        }
        let v = validate_step_model(dir).expect("default config should match shader");
        assert_eq!(v.num_layers, 30);
        assert_eq!(v.canvas_length, 256);
    }

    /// The derived geometry must reproduce every value the step kernel
    /// hardcodes today; consumers switch from the constants to these.
    #[test]
    fn dims_reproduce_shader_constants() {
        use crate::metal::step_kernel as sk;
        let dir = Path::new("model/transformer");
        if !dir.join("config.json").exists() {
            return;
        }
        let d = validate_step_model(dir).expect("reference config").dims;
        assert_eq!(d.hid, sk::HID);
        assert_eq!(d.vocab, sk::VOCAB);
        assert_eq!(d.canvas, sk::CANVAS);
        assert_eq!(d.n_layers, sk::N_LAYERS);
        assert_eq!(d.dense_ff, sk::DENSE_FF as usize);
        assert_eq!(d.moe_ff, sk::MOE_FF as usize);
        assert_eq!(d.n_experts, sk::N_EXPERTS);
        assert_eq!(d.top_k, sk::TOP_K);
        assert_eq!(d.full_layers, sk::FULL_LAYERS);
        assert_eq!(d.prefill_m(), sk::PREFILL_M);
        assert_eq!(d.frozen_words(), sk::FROZEN_WORDS);
        // The head-geometry literals in enc.rs and the pipeline shape list.
        assert_eq!(d.q_cols(false), 4096);
        assert_eq!(d.q_cols(true), 8192);
        assert_eq!(d.kv_cols(false), 2048);
        assert_eq!(d.kv_cols(true), 1024);
        assert_eq!((d.head_dim(true), d.kv_heads(true)), (512, 2));
        assert_eq!((d.head_dim(false), d.kv_heads(false)), (256, 8));
    }
}
