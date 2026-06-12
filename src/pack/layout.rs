//! On-disk layout tags for `iris.pack` weights.

use serde::{Deserialize, Serialize};

pub const MANIFEST_FILE: &str = "iris.pack.json";
pub const BLOB_FILE: &str = "iris.pack.bin";
pub const PACK_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorLayout {
    /// Byte-identical to source safetensors payload.
    Raw,
    /// 2-D linear weight transposed from PyTorch `[out, in]` to GEMM `[in, out]` bf16.
    GemmBf16T,
    /// MoE `experts.gate_up_proj`: per-expert `[hidden, moe*2]` bf16, experts concatenated.
    ExpertGateUpT,
    /// MoE `experts.down_proj`: per-expert `[moe, hidden]` bf16, experts concatenated.
    ExpertDownT,
}

impl TensorLayout {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::GemmBf16T => "gemm_bf16_t",
            Self::ExpertGateUpT => "expert_gate_up_t",
            Self::ExpertDownT => "expert_down_t",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "raw" => Some(Self::Raw),
            "gemm_bf16_t" => Some(Self::GemmBf16T),
            "expert_gate_up_t" => Some(Self::ExpertGateUpT),
            "expert_down_t" => Some(Self::ExpertDownT),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackTensorMeta {
    pub layout: String,
    pub dtype: String,
    pub shape: Vec<i64>,
    pub offset: u64,
    pub byte_len: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackManifest {
    pub version: u32,
    pub source_model: String,
    pub blob_file: String,
    pub tensors: Vec<PackTensorEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackTensorEntry {
    pub name: String,
    #[serde(flatten)]
    pub meta: PackTensorMeta,
}

pub fn align_offset(offset: u64) -> u64 {
    (offset + 63) & !63
}

/// Classify a safetensors tensor for inference-oriented packing.
pub fn classify_tensor(name: &str, shape: &[i64], dtype: crate::safetensors::DType) -> TensorLayout {
    if name.ends_with(".experts.gate_up_proj") && shape.len() == 3 {
        return TensorLayout::ExpertGateUpT;
    }
    if name.ends_with(".experts.down_proj") && shape.len() == 3 {
        return TensorLayout::ExpertDownT;
    }
    if dtype == crate::safetensors::DType::BF16 && shape.len() == 2 && is_gemm_linear(name) {
        return TensorLayout::GemmBf16T;
    }
    TensorLayout::Raw
}

/// Only weights consumed by Metal `CachedLinear` / f32@bf16 GEMM.
fn is_gemm_linear(name: &str) -> bool {
    if !name.ends_with(".weight") {
        return false;
    }
    // Decoder attention + shared MLP (GPU). Exclude router + self-conditioning (CPU linear).
    (name.contains(".self_attn.")
        && (name.contains(".q_proj.")
            || name.contains(".k_proj.")
            || name.contains(".v_proj.")
            || name.contains(".o_proj.")))
        || name.contains(".mlp.gate_proj.")
        || name.contains(".mlp.up_proj.")
        || name.contains(".mlp.down_proj.")
}
