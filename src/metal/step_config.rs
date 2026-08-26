//! Model geometry from `config.json`: `ModelDims` (threaded through the step
//! runtime) and the load-time envelope check (`validate_step_model`).

use crate::Error;
use crate::config::{LayerType, ModelConfig};
use crate::metal::step_kernel::{CANVAS, HID, MOE_FF, N_EXPERTS, N_LAYERS, TOP_K, VOCAB};
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
/// runtime. The remaining compile-time constants are capacity bounds
/// (`validate_step_model` checks dims fit them), except canvas/vocab/eps,
/// which stay exact-match until their consumers are dims-driven.
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
    pub rope_theta_sliding: f64,
    pub rope_theta_full: f64,
    /// Rotated dims per layer kind (sliding: the whole head; full: the
    /// partial_rotary_factor slice of it).
    pub rot_sliding: usize,
    pub rot_full: usize,
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
            rope_theta_sliding: 1.0e4,
            rope_theta_full: 1.0e6,
            rot_sliding: 256,
            rot_full: 128,
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
            rope_theta_sliding: t.rope_parameters.sliding_attention.rope_theta,
            rope_theta_full: t.rope_parameters.full_attention.rope_theta,
            rot_sliding: t.head_dim,
            rot_full: ((t.global_head_dim as f64)
                * t.rope_parameters
                    .full_attention
                    .partial_rotary_factor
                    .unwrap_or(0.25))
            .round() as usize,
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

    /// Stable hash over every field, for cache keys that must distinguish
    /// models (the process-global step-pipeline cache).
    pub fn fingerprint(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        let mut eat = |v: u64| {
            for b in v.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        };
        for v in [
            self.hid,
            self.vocab,
            self.canvas,
            self.n_layers,
            self.dense_ff,
            self.moe_ff,
            self.n_experts,
            self.top_k,
            self.n_q_heads,
            self.sliding_kv_heads,
            self.sliding_head_dim,
            self.full_kv_heads,
            self.full_head_dim,
            self.prefill_subs,
        ] {
            eat(v as u64);
        }
        for &l in &self.full_layers {
            eat(l as u64);
        }
        eat(self.rot_sliding as u64);
        eat(self.rot_full as u64);
        eat(self.rms_eps.to_bits());
        eat(self.rope_theta_sliding.to_bits());
        eat(self.rope_theta_full.to_bits());
        h
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

/// Load model config and check it fits the step kernel's envelope.
///
/// Dims flow from config at runtime (`ModelDims`), so most checks are caps
/// and divisibility, not equality: a smaller model of the same architecture
/// loads; a larger one fails naming the cap to raise. Three values stay
/// exact-match until their consumers are dims-driven: `canvas_length` and
/// `vocab_size` (the CPU sampler's `SAMPLER_CANVAS` / `FILLER_TOKEN_ID` and
/// the `CanvasState` ABI), and `rms_norm_eps` (`ATTN_RMS_EPS` baked into
/// attention_device.metal).
pub fn validate_step_model(model_dir: &Path) -> Result<ValidatedStepModel, Error> {
    let cfg = ModelConfig::load(model_dir)?;
    let t = &cfg.text_config;

    let cap = |field: &str, value: usize, cap: usize, holder: &str| {
        if value > cap {
            Err(Error::NotFound(format!(
                "step-kernel {field}={value} exceeds the {holder} cap ({cap}); \
                 raise the cap and recompile to load this model"
            )))
        } else {
            Ok(())
        }
    };
    let multiple = |field: &str, value: usize, of: usize, why: &str| {
        if !value.is_multiple_of(of) {
            Err(Error::NotFound(format!(
                "step-kernel {field}={value} must be a multiple of {of} ({why})"
            )))
        } else {
            Ok(())
        }
    };

    if cfg.canvas_length != CANVAS {
        return Err(mismatch("canvas_length", cfg.canvas_length, CANVAS));
    }
    if t.vocab_size != VOCAB {
        return Err(mismatch("vocab_size", t.vocab_size, VOCAB));
    }
    if (t.rms_norm_eps - 1e-6).abs() > 1e-9 {
        return Err(Error::NotFound(format!(
            "step-kernel rms_norm_eps mismatch: config={} shader=1e-6 (ATTN_RMS_EPS)",
            t.rms_norm_eps
        )));
    }

    cap(
        "num_hidden_layers",
        t.num_hidden_layers,
        N_LAYERS,
        "ModelLayout.layers",
    )?;
    cap(
        "num_experts",
        t.num_experts,
        N_EXPERTS,
        "RouteScratch/MOE_MAX_EXPERTS",
    )?;
    cap(
        "top_k_experts",
        t.top_k_experts,
        TOP_K,
        "RouteScratch/MOE_MAX_TOP_K",
    )?;
    if t.top_k_experts > t.num_experts {
        return Err(Error::NotFound(format!(
            "step-kernel top_k_experts={} exceeds num_experts={}",
            t.top_k_experts, t.num_experts
        )));
    }
    cap(
        "moe_intermediate_size",
        t.moe_intermediate_size,
        MOE_FF as usize,
        "MOE_MAX_FF",
    )?;
    cap("hidden_size", t.hidden_size, HID, "MOE_MAX_HIDDEN")?;
    multiple("hidden_size", t.hidden_size, 64, "GEMM K-tile loads")?;
    multiple(
        "intermediate_size",
        t.intermediate_size,
        32,
        "quant group size",
    )?;
    multiple(
        "moe_intermediate_size",
        t.moe_intermediate_size,
        32,
        "quant group size",
    )?;

    // The MMA attention kernels assume the reference GQA group shapes:
    // attention_mma2 is written for group 2 (sliding), attention_mma_full and
    // the top-k pipeline for group 8 (full).
    if t.num_attention_heads != t.num_key_value_heads * 2 {
        return Err(Error::NotFound(format!(
            "step-kernel sliding GQA group must be 2 (mma2): q={} kv={}",
            t.num_attention_heads, t.num_key_value_heads
        )));
    }
    if t.num_attention_heads != t.num_global_key_value_heads * 8 {
        return Err(Error::NotFound(format!(
            "step-kernel full GQA group must be 8 (mma_full): q={} kv={}",
            t.num_attention_heads, t.num_global_key_value_heads
        )));
    }
    cap("head_dim", t.head_dim, 256, "attention_mma2 HD_MAX")?;
    cap(
        "global_head_dim",
        t.global_head_dim,
        512,
        "qk_rope_kv head buffer",
    )?;
    multiple("head_dim", t.head_dim, 8, "simdgroup 8x8 MMA tiles")?;
    multiple(
        "global_head_dim",
        t.global_head_dim,
        8,
        "simdgroup 8x8 MMA tiles",
    )?;
    cap(
        "q columns (heads * global_head_dim)",
        t.num_attention_heads * t.global_head_dim,
        crate::metal::step_kernel::MAX_ATTN_Q_COLS,
        "arena attnq plane",
    )?;
    cap(
        "kv columns (kv heads * head_dim)",
        t.num_key_value_heads * t.head_dim,
        crate::metal::step_kernel::MAX_ATTN_KV_COLS,
        "arena attnk/v planes",
    )?;

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
        assert_eq!(d.canvas / 32, sk::FROZEN_WORDS);
        // The head-geometry literals in enc.rs and the pipeline shape list.
        assert_eq!(d.q_cols(false), 4096);
        assert_eq!(d.q_cols(true), 8192);
        assert_eq!(d.kv_cols(false), 2048);
        assert_eq!(d.kv_cols(true), 1024);
        assert_eq!((d.head_dim(true), d.kv_heads(true)), (512, 2));
        assert_eq!((d.head_dim(false), d.kv_heads(false)), (256, 8));
    }
}
