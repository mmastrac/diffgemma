//! The subset of `config.json` the forward pass and sampler need.

use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParams {
    pub rope_theta: f64,
    pub rope_type: String,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_global_key_value_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub num_experts: usize,
    pub top_k_experts: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub final_logit_softcapping: f64,
    pub sliding_window: usize,
    pub layer_types: Vec<LayerType>,
    pub rope_parameters: std::collections::HashMap<String, RopeParams>,
    #[serde(default)]
    pub canvas_length: Option<usize>,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
}

impl TextConfig {
    pub fn layer_type(&self, layer: usize) -> LayerType {
        self.layer_types
            .get(layer)
            .copied()
            .unwrap_or(LayerType::SlidingAttention)
    }

    pub fn is_full(&self, layer: usize) -> bool {
        self.layer_type(layer) == LayerType::FullAttention
    }

    /// (n_kv_heads, head_dim, rotary_dim, theta, sliding_window)
    pub fn attn_geometry(&self, layer: usize) -> (usize, usize, usize, f32, Option<usize>) {
        match self.layer_type(layer) {
            LayerType::SlidingAttention => {
                let rope = self.rope_parameters.get("sliding_attention");
                let theta = rope.map(|r| r.rope_theta as f32).unwrap_or(10_000.0);
                (
                    self.num_key_value_heads,
                    self.head_dim,
                    self.head_dim,
                    theta,
                    Some(self.sliding_window),
                )
            }
            LayerType::FullAttention => {
                let rope = self.rope_parameters.get("full_attention");
                let theta = rope.map(|r| r.rope_theta as f32).unwrap_or(1_000_000.0);
                let factor = rope.and_then(|r| r.partial_rotary_factor).unwrap_or(0.25);
                let rotary_dim = (self.global_head_dim as f64 * factor) as usize;
                (
                    self.num_global_key_value_heads,
                    self.global_head_dim,
                    rotary_dim,
                    theta,
                    None,
                )
            }
        }
    }

    pub fn eos_token_ids(&self) -> Vec<u32> {
        match &self.eos_token_id {
            Some(serde_json::Value::Number(n)) => vec![n.as_u64().unwrap_or(1) as u32],
            Some(serde_json::Value::Array(a)) => a
                .iter()
                .filter_map(|v| v.as_u64())
                .map(|v| v as u32)
                .collect(),
            _ => vec![1],
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub canvas_length: Option<usize>,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    pub text_config: TextConfig,
}

impl ModelConfig {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, Error> {
        let json = std::fs::read_to_string(dir.as_ref().join("config.json"))?;
        let mut cfg: ModelConfig = serde_json::from_str(&json)?;
        if cfg.text_config.canvas_length.is_none() {
            cfg.text_config.canvas_length = cfg.canvas_length;
        }
        if cfg.text_config.eos_token_id.is_none() {
            cfg.text_config.eos_token_id = cfg.eos_token_id.clone();
        }
        Ok(cfg)
    }
}

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Json(serde_json::Error),
    Pack(dgops::dgq::Error),
    Gpu(gpukit::Error),
    Msg(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Json(e) => write!(f, "config: {e}"),
            Error::Pack(e) => write!(f, "pack: {e}"),
            Error::Gpu(e) => write!(f, "gpu: {e}"),
            Error::Msg(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Json(e)
    }
}
impl From<dgops::dgq::Error> for Error {
    fn from(e: dgops::dgq::Error) -> Self {
        Error::Pack(e)
    }
}
impl From<gpukit::Error> for Error {
    fn from(e: gpukit::Error) -> Self {
        Error::Gpu(e)
    }
}
