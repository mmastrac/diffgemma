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
}
