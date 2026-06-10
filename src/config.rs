//! Model configuration from `config.json`. Fields are consumed across later phases.
#![allow(dead_code)]

use crate::safetensors::Error;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub canvas_length: usize,
    pub boi_token_id: u32,
    pub eoi_token_id: u32,
    pub eos_token_id: serde_json::Value,
    pub image_token_id: u32,
    pub vision_soft_tokens_per_image: usize,
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_global_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub num_experts: usize,
    pub top_k_experts: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub rms_norm_eps: f64,
    pub final_logit_softcapping: f64,
    pub layer_types: Vec<LayerType>,
    pub rope_parameters: RopeParameters,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

impl std::fmt::Display for LayerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SlidingAttention => write!(f, "sliding_attention"),
            Self::FullAttention => write!(f, "full_attention"),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub full_attention: RopeConfig,
    pub sliding_attention: RopeConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeConfig {
    pub rope_theta: f64,
    pub rope_type: String,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub default_output_length: usize,
}

impl ModelConfig {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, Error> {
        let path = model_dir.as_ref().join("config.json");
        let json = std::fs::read_to_string(&path)?;
        Ok(serde_json::from_str(&json)?)
    }

    pub fn q_dim(&self) -> usize {
        self.text_config.num_attention_heads * self.text_config.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.text_config.num_key_value_heads * self.text_config.head_dim
    }

    pub fn global_kv_dim(&self) -> usize {
        self.text_config.num_global_key_value_heads * self.text_config.global_head_dim
    }
}

impl TextConfig {
    /// Rotary dimension for a layer (full `head_dim` sliding, partial on `global_head_dim`).
    pub fn rotary_dim_for_layer(&self, layer: usize) -> Option<usize> {
        let kind = self.layer_types.get(layer)?;
        Some(match kind {
            LayerType::SlidingAttention => self.head_dim,
            LayerType::FullAttention => {
                let factor = self
                    .rope_parameters
                    .full_attention
                    .partial_rotary_factor
                    .unwrap_or(0.25);
                ((self.global_head_dim as f64) * factor).round() as usize
            }
        })
    }

    pub fn full_head_dim_for_layer(&self, layer: usize) -> Option<usize> {
        let kind = self.layer_types.get(layer)?;
        Some(match kind {
            LayerType::SlidingAttention => self.head_dim,
            LayerType::FullAttention => self.global_head_dim,
        })
    }

    pub fn rope_theta_for_layer(&self, layer: usize) -> Option<f32> {
        let kind = self.layer_types.get(layer)?;
        Some(match kind {
            LayerType::SlidingAttention => self.rope_parameters.sliding_attention.rope_theta as f32,
            LayerType::FullAttention => self.rope_parameters.full_attention.rope_theta as f32,
        })
    }
}

impl ModelConfig {
    pub fn print_summary(&self) {
        let t = &self.text_config;
        println!("DiffusionGemma config");
        println!("  architecture:       {}", self.architectures.join(", "));
        println!("  model_type:         {}", self.model_type);
        println!("  canvas_length:      {}", self.canvas_length);
        println!("  vocab_size:         {}", t.vocab_size);
        println!("  hidden_size:        {}", t.hidden_size);
        println!("  num_hidden_layers:  {}", t.num_hidden_layers);
        println!("  num_experts:        {}", t.num_experts);
        println!("  top_k_experts:      {}", t.top_k_experts);
        println!("  attention heads:    {} Q / {} KV ({} global KV)", t.num_attention_heads, t.num_key_value_heads, t.num_global_key_value_heads);
        println!("  head_dim:           {} (global {})", t.head_dim, t.global_head_dim);
        println!("  q_dim / kv_dim:     {} / {}", self.q_dim(), self.kv_dim());
        println!("  sliding_window:     {}", t.sliding_window);
        println!("  moe intermediate:   {}", t.moe_intermediate_size);
        println!("  shared intermediate:{}", t.intermediate_size);
        println!("  vision layers:      {}", self.vision_config.num_hidden_layers);
        println!("  vision soft tokens: {}", self.vision_soft_tokens_per_image);
        println!("\n  layer_types:");
        for (i, kind) in t.layer_types.iter().enumerate() {
            println!("    layer {i:2}: {kind}");
        }
    }
}
