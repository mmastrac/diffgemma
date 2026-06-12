//! On-disk layout for `.dgq` quantized weights.

use serde::{Deserialize, Serialize};

pub const MANIFEST_FILE: &str = "model.dgq.json";
pub const BLOB_FILE: &str = "model.dgq.bin";
pub const DGQ_VERSION: u32 = 1;
pub const GROUP_SIZE: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantProfile {
    Q4,
    Q5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantKind {
    /// Affine int4 blocks along K (Q4_1-style: fp16 scale + fp16 min + nibbles).
    Q4Block,
    /// Per-row int8 + fp16 scale (embed / self-conditioning).
    Q8Row,
    /// Byte-identical bf16/f16/f32 payload.
    Raw,
}

impl QuantKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Q4Block => "q4_block",
            Self::Q8Row => "q8_row",
            Self::Raw => "raw",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DgqTensorMeta {
    pub kind: String,
    pub dtype: String,
    pub shape: Vec<i64>,
    pub offset: u64,
    pub byte_len: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DgqManifest {
    pub version: u32,
    pub profile: QuantProfile,
    pub source_model: String,
    pub blob_file: String,
    pub tensors: Vec<DgqTensorEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DgqTensorEntry {
    pub name: String,
    #[serde(flatten)]
    pub meta: DgqTensorMeta,
}

pub fn align_offset(offset: u64) -> u64 {
    (offset + 63) & !63
}

/// Bytes for one Q4 block covering `GROUP_SIZE` weights along K.
pub const Q4_BLOCK_BYTES: usize = 4 + GROUP_SIZE / 2; // fp16 scale + fp16 min + 16 nibbles

pub fn q4_row_bytes(in_dim: usize) -> usize {
    let groups = in_dim.div_ceil(GROUP_SIZE);
    groups * Q4_BLOCK_BYTES
}

pub fn q4_matrix_bytes(out_dim: usize, in_dim: usize) -> usize {
    out_dim * q4_row_bytes(in_dim)
}

pub fn q8_row_bytes(in_dim: usize) -> usize {
    2 + in_dim // fp16 scale + int8 weights
}

pub fn q8_matrix_bytes(out_dim: usize, in_dim: usize) -> usize {
    out_dim * q8_row_bytes(in_dim)
}

/// Mixed-precision mapping for a safetensors tensor name.
pub fn classify_tensor(name: &str, shape: &[i64], profile: QuantProfile) -> QuantKind {
    if name.contains(".router.") {
        return QuantKind::Raw;
    }
    if name.contains("_norm") || name.ends_with(".scale") || name.contains("layer_scalar") {
        return QuantKind::Raw;
    }
    if name.contains("embed_tokens") || name.contains("self_conditioning") {
        return QuantKind::Q8Row;
    }
    if name.contains(".experts.") && shape.len() == 3 {
        return QuantKind::Q4Block;
    }
    if shape.len() == 2 && is_gemm_linear(name) {
        return match profile {
            QuantProfile::Q4 | QuantProfile::Q5 => QuantKind::Q4Block,
        };
    }
    QuantKind::Raw
}

fn is_gemm_linear(name: &str) -> bool {
    if !name.ends_with(".weight") {
        return false;
    }
    (name.contains(".self_attn.")
        && (name.contains(".q_proj.")
            || name.contains(".k_proj.")
            || name.contains(".v_proj.")
            || name.contains(".o_proj.")))
        || name.contains(".mlp.gate_proj.")
        || name.contains(".mlp.up_proj.")
        || name.contains(".mlp.down_proj.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_experts_and_router() {
        assert_eq!(
            classify_tensor(
                "model.decoder.layers.0.experts.gate_up_proj",
                &[128, 352, 2816],
                QuantProfile::Q4,
            ),
            QuantKind::Q4Block,
        );
        assert_eq!(
            classify_tensor(
                "model.decoder.layers.0.router.proj.weight",
                &[128, 2816],
                QuantProfile::Q4,
            ),
            QuantKind::Raw,
        );
        assert_eq!(
            classify_tensor(
                "model.decoder.embed_tokens.weight",
                &[262144, 2816],
                QuantProfile::Q4,
            ),
            QuantKind::Q8Row,
        );
    }
}
