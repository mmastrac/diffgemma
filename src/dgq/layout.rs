//! On-disk layout for `.dgq` quantized weights.

use crate::safetensors::Error;
use serde::{Deserialize, Serialize};

pub const MANIFEST_FILE: &str = "model.dgq.json";
pub const BLOB_FILE: &str = "model.dgq.bin";
/// Version 1: affine Q4 (`q4_block`). Version 2: adds NVFP4 (`nvfp4_block`).
pub const DGQ_VERSION_AFFINE: u32 = 1;
pub const DGQ_VERSION_NVFP4: u32 = 2;

pub fn dgq_version_for_profile(profile: QuantProfile) -> u32 {
    match profile {
        QuantProfile::Nvfp4 => DGQ_VERSION_NVFP4,
        QuantProfile::Q4 | QuantProfile::Q5 => DGQ_VERSION_AFFINE,
    }
}

pub fn dgq_version_supported(version: u32) -> bool {
    version == DGQ_VERSION_AFFINE || version == DGQ_VERSION_NVFP4
}
/// Affine int4 group size (legacy `q4_block`).
pub const GROUP_SIZE: usize = 32;
/// NVFP4 micro-block size (MLX / NVIDIA nvfp4).
pub const NVFP4_GROUP_SIZE: usize = 16;
/// Per-tensor FP32 global scale prefix on `nvfp4_block` payloads (1.0 for 2-tier MLX quant).
pub const NVFP4_HEADER_BYTES: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantProfile {
    Q4,
    Q5,
    Nvfp4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantKind {
    /// Affine int4 blocks along K (Q4_1-style: fp16 scale + fp16 min + nibbles).
    Q4Block,
    /// NVFP4 blocks: E2M1 nibbles + FP8 E4M3 scale per 16 weights (MLX-compatible 2-tier).
    Nvfp4Block,
    /// Per-row int8 + fp16 scale (embed / self-conditioning).
    Q8Row,
    /// Byte-identical bf16/f16/f32 payload.
    Raw,
}

impl QuantKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Q4Block => "q4_block",
            Self::Nvfp4Block => "nvfp4_block",
            Self::Q8Row => "q8_row",
            Self::Raw => "raw",
        }
    }
}

pub fn parse_quant_kind(s: &str) -> Result<QuantKind, Error> {
    match s {
        "q4_block" => Ok(QuantKind::Q4Block),
        "nvfp4_block" => Ok(QuantKind::Nvfp4Block),
        "q8_row" => Ok(QuantKind::Q8Row),
        "raw" => Ok(QuantKind::Raw),
        _ => Err(Error::Format("unknown dgq tensor kind")),
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

/// Convert a `.dgq` blob byte offset for host pointer / MTL buffer slicing.
/// NVFP4 blobs can exceed `u32::MAX`; never truncate to u32 before this call.
pub fn blob_offset_usize(off: u64) -> Result<usize, Error> {
    usize::try_from(off).map_err(|_| Error::Format("dgq blob offset exceeds host address space"))
}

/// `(start, end)` byte indices into a blob slice, with bounds checks.
pub fn blob_slice_range(off: u64, len: u64, blob_len: u64) -> Result<(usize, usize), Error> {
    let start = blob_offset_usize(off)?;
    let len_usize = blob_offset_usize(len)?;
    let end = start
        .checked_add(len_usize)
        .ok_or(Error::Format("dgq tensor slice overflow"))?;
    let blob_end = blob_offset_usize(blob_len)?;
    if end > blob_end {
        return Err(Error::Format("dgq tensor extends past blob"));
    }
    Ok((start, end))
}

/// Hot GPU dispatch path: offsets are validated when the layout is built.
#[inline]
pub fn blob_offset_for_mtl(off: u64) -> usize {
    blob_offset_usize(off).expect("dgq blob offset exceeds host address space")
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

/// Packed E2M1 nibbles per row (2 codes per byte, low nibble first).
pub fn nvfp4_data_row_bytes(in_dim: usize) -> usize {
    in_dim.div_ceil(2)
}

/// FP8 E4M3 block scales per row (one byte per 16 weights along K).
pub fn nvfp4_scales_row_bytes(in_dim: usize) -> usize {
    in_dim.div_ceil(NVFP4_GROUP_SIZE)
}

pub fn nvfp4_row_bytes(in_dim: usize) -> usize {
    nvfp4_data_row_bytes(in_dim) + nvfp4_scales_row_bytes(in_dim)
}

pub fn nvfp4_matrix_bytes(out_dim: usize, in_dim: usize) -> usize {
    NVFP4_HEADER_BYTES + out_dim * nvfp4_row_bytes(in_dim)
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
    if shape.len() == 2 && is_gemm_linear(name) {
        // Attention (q/k/v/o_proj) and the dense FFN (gate/up/down) keep 8-bit,
        // matching MLX's mixed-precision 4-bit checkpoints. These tensors are ~8x
        // more accurate at q8 than q4 and only ~2x larger; only the bulky MoE
        // experts go to 4-bit. nvfp4 keeps its uniform block format.
        return match profile {
            QuantProfile::Q4 | QuantProfile::Q5 => QuantKind::Q8Row,
            QuantProfile::Nvfp4 => QuantKind::Nvfp4Block,
        };
    }
    if name.contains(".experts.") && shape.len() == 3 {
        return match profile {
            QuantProfile::Q4 | QuantProfile::Q5 => QuantKind::Q4Block,
            QuantProfile::Nvfp4 => QuantKind::Nvfp4Block,
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
                QuantProfile::Nvfp4,
            ),
            QuantKind::Nvfp4Block,
        );
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
