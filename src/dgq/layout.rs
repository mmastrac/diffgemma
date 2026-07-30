//! On-disk layout for `.dgq` quantized weights.

use crate::Error;
use serde::{Deserialize, Serialize};

pub const MANIFEST_FILE: &str = "model.dgq.json";
pub const BLOB_FILE: &str = "model.dgq.bin";
/// Version 1: affine Q4 (`q4_block`). Version 2: adds NVFP4 (`nvfp4_block`).
/// Version 3: adds layered/overlay packs (`DgqTensorMeta::source`,
/// `DgqManifest::external_files`/`base_model`) — a deliberate compatibility
/// gate so a pre-layering binary refuses a layered manifest outright
/// (`dgq_version_supported` returns false) instead of silently misreading
/// `offset` as a local-blob position for an externally-sourced tensor.
pub const DGQ_VERSION_AFFINE: u32 = 1;
pub const DGQ_VERSION_NVFP4: u32 = 2;
pub const DGQ_VERSION_LAYERED: u32 = 3;

pub fn dgq_version_for_profile(profile: QuantProfile) -> u32 {
    match profile {
        QuantProfile::Nvfp4 | QuantProfile::Nvfp4Experts => DGQ_VERSION_NVFP4,
        QuantProfile::Q4 | QuantProfile::Q5 | QuantProfile::Q6 => DGQ_VERSION_AFFINE,
    }
}

pub fn dgq_version_supported(version: u32) -> bool {
    version == DGQ_VERSION_AFFINE || version == DGQ_VERSION_NVFP4 || version == DGQ_VERSION_LAYERED
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
    Q6,
    Nvfp4,
    /// Perf-isolation variant (not a shipped profile): experts go
    /// `nvfp4_block` exactly like `Nvfp4`, but every other tensor is
    /// classified exactly as `Q4` classifies it (attention/dense FFN/vision
    /// linears + embed stay `Raw`). Isolates the expert-format variable from
    /// the `Nvfp4` profile's confound of also quantizing those other
    /// tensors, for an apples-to-apples decode/prefill perf A/B against q4.
    Nvfp4Experts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantKind {
    /// Affine int4 blocks along K (Q4_1-style: fp16 scale + fp16 min + nibbles).
    Q4Block,
    /// Affine int6 blocks along K: bf16 scale + bf16 min + 24B of 6-bit codes
    /// (24-bit LE words, 4 codes each). Experts-only; ~2% rel-RMS vs q4's ~8%.
    Q6Block,
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
            Self::Q6Block => "q6_block",
            Self::Nvfp4Block => "nvfp4_block",
            Self::Q8Row => "q8_row",
            Self::Raw => "raw",
        }
    }
}

pub fn parse_quant_kind(s: &str) -> Result<QuantKind, Error> {
    match s {
        "q4_block" => Ok(QuantKind::Q4Block),
        "q6_block" => Ok(QuantKind::Q6Block),
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
    /// Byte offset in the CANONICAL unified blob address space — the value
    /// every downstream consumer (GPU `w_off` constants via
    /// `build_offsets_from_store`, `DgqGpuBlob::buffer_for`, split-blob
    /// region math) treats as authoritative. For a self-contained pack this
    /// is also where the bytes live in `blob_file`. For a layered pack it is
    /// where the loader MATERIALIZES the tensor's bytes at load time; the
    /// actual byte source is `source` (absent = same as `offset`, in
    /// `blob_file`, exactly today's semantics).
    pub offset: u64,
    pub byte_len: u64,
    /// Where the bytes actually live on disk. `None` = this pack's own
    /// `blob_file` at `offset` (every pack before layering, and every
    /// non-external tensor in a layered pack once `local_offset` is folded
    /// in — see `TensorSource::Local`). Only present in a version-3
    /// (`DGQ_VERSION_LAYERED`) manifest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<TensorSource>,
}

/// Byte source for a layered-pack tensor entry. Kept separate from the
/// canonical `offset` because a layered pack's own blob is COMPACT (only the
/// tensors that actually live in it — experts, q8 — packed with no gaps for
/// the externally-sourced tensors), so a local tensor's on-disk position
/// generally differs from its canonical address-space position too.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TensorSource {
    /// Bytes live in this pack's own `blob_file`, at `local_offset` (may
    /// differ from the entry's canonical `offset`).
    Local { local_offset: u64 },
    /// Bytes live in an external file — a key into
    /// `DgqManifest::external_files` — at `offset` within THAT file.
    External { file: String, offset: u64 },
}

/// One file a layered pack's tensors may be sourced from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalRole {
    /// A safetensors shard inside the pinned HF base-model snapshot.
    HfSafetensors,
    /// Another `.dgq` pack's blob file (e.g. verification / migration).
    PackBin,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalFile {
    pub role: ExternalRole,
    /// `HfSafetensors`: shard filename inside the resolved snapshot dir
    /// (e.g. `"model-00007-of-00011.safetensors"`). `PackBin`: filesystem
    /// path to the referenced pack's blob — absolute, or relative to this
    /// pack's own directory.
    pub path: String,
    pub expected_size: u64,
    /// `HfSafetensors` only: hex SHA-256 of `(8-byte LE header length ||
    /// header JSON bytes)` — pins shard identity/layout without hashing the
    /// multi-GiB payload. `None` for `PackBin`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub header_sha256: Option<String>,
}

/// Pinned HF base-model identity a layered pack's `HfSafetensors` refs
/// resolve against.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseModelRef {
    /// `"org/name"`, e.g. `"google/diffusiongemma-26B-A4B-it"`.
    pub repo: String,
    /// Pinned snapshot revision (commit hash) — resolution never floats to
    /// "latest" so an overlay's meaning can't shift under a cache refresh.
    pub revision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DgqManifest {
    pub version: u32,
    pub profile: QuantProfile,
    pub source_model: String,
    pub blob_file: String,
    /// Page-aligned byte offset where the expert-tensor region begins (experts
    /// are written LAST so blobs above the device max single-buffer length can
    /// be wrapped as two no-copy MTLBuffers). None on pre-split manifests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_split: Option<u64>,
    /// Layered packs only: page-aligned byte offset in `blob_file` where the
    /// expert region begins, written such that `blob_file[local_expert_split..]`
    /// is byte-for-byte identical (same relative offsets) to the canonical
    /// `[expert_split, total_len)` range. When both this and `expert_split`
    /// are set, the loader can wrap the (large) expert tail as a direct
    /// file-backed no-copy region instead of gather-copying it into the
    /// materialized mapping — the tail never needs to be resident twice.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_expert_split: Option<u64>,
    /// Layered packs only: the HF base every `HfSafetensors` external ref
    /// resolves against.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_model: Option<BaseModelRef>,
    /// Layered packs only: files referenced by `DgqTensorMeta::source`, keyed
    /// by the string used in `TensorSource::External::file`.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub external_files: std::collections::BTreeMap<String, ExternalFile>,
    pub tensors: Vec<DgqTensorEntry>,
}

impl DgqManifest {
    /// A layered pack has at least one externally- or locally-redirected
    /// tensor. Self-contained packs (all `source: None`) take the identity
    /// fast path everywhere — zero behavior change from pre-layering code.
    pub fn is_layered(&self) -> bool {
        self.tensors.iter().any(|t| t.meta.source.is_some())
    }
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
    usize::try_from(off).map_err(|_| Error::Runtime("dgq blob offset exceeds host address space"))
}

/// `(start, end)` byte indices into a blob slice, with bounds checks.
pub fn blob_slice_range(off: u64, len: u64, blob_len: u64) -> Result<(usize, usize), Error> {
    let start = blob_offset_usize(off)?;
    let len_usize = blob_offset_usize(len)?;
    let end = start
        .checked_add(len_usize)
        .ok_or(Error::Runtime("dgq tensor slice overflow"))?;
    let blob_end = blob_offset_usize(blob_len)?;
    if end > blob_end {
        return Err(Error::Runtime("dgq tensor extends past blob"));
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

/// Bytes for one Q6 block covering `GROUP_SIZE` weights along K.
pub const Q6_BLOCK_BYTES: usize = 4 + GROUP_SIZE * 6 / 8; // bf16 scale + bf16 min + 24B codes

pub fn q6_row_bytes(in_dim: usize) -> usize {
    let groups = in_dim.div_ceil(GROUP_SIZE);
    groups * Q6_BLOCK_BYTES
}

pub fn q6_matrix_bytes(out_dim: usize, in_dim: usize) -> usize {
    out_dim * q6_row_bytes(in_dim)
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
    if name.contains("embed_tokens") {
        // Tied to the lm_head (logits = hidden @ embed^T) and the SC soft-embed.
        // bf16 (Raw, lossless) keeps logits sharp on hard tail tokens — q8 per-row
        // is ~1.76x coarser than MLX's group-64 affine and stalls tail convergence.
        // Only ~+0.74 GiB (one tensor), unlike the bulky experts.
        return QuantKind::Raw;
    }
    if name.contains("self_conditioning") {
        return QuantKind::Q8Row;
    }
    if shape.len() == 2 && is_gemm_linear(name) {
        // Attention (q/k/v/o_proj) and the dense FFN (gate/up/down) stay bf16
        // (Raw): they are only ~2x larger than q8 but lossless into the half GEMM
        // tiles (the source weights are bf16), beating MLX's q8 on these tensors.
        // Only the bulky MoE experts go to 4-bit. nvfp4 keeps its block format.
        return match profile {
            QuantProfile::Q4 | QuantProfile::Q5 | QuantProfile::Q6 | QuantProfile::Nvfp4Experts => {
                QuantKind::Raw
            }
            QuantProfile::Nvfp4 => QuantKind::Nvfp4Block,
        };
    }
    if name.contains(".experts.") && shape.len() == 3 {
        return match profile {
            QuantProfile::Q4 | QuantProfile::Q5 => QuantKind::Q4Block,
            QuantProfile::Q6 => QuantKind::Q6Block,
            QuantProfile::Nvfp4 | QuantProfile::Nvfp4Experts => QuantKind::Nvfp4Block,
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
        // embed_tokens is bf16 (Raw): tied lm_head + SC need accurate logits.
        assert_eq!(
            classify_tensor(
                "model.decoder.embed_tokens.weight",
                &[262144, 2816],
                QuantProfile::Q4,
            ),
            QuantKind::Raw,
        );
        // self_conditioning stays q8 per-row.
        assert_eq!(
            classify_tensor(
                "model.decoder.self_conditioning.gate_proj.weight",
                &[2112, 2816],
                QuantProfile::Q4,
            ),
            QuantKind::Q8Row,
        );
    }

    fn sample_manifest(external: bool) -> DgqManifest {
        let mut external_files = std::collections::BTreeMap::new();
        let mut tensors = vec![DgqTensorEntry {
            name: "model.decoder.layers.0.experts.gate_up_proj".to_string(),
            meta: DgqTensorMeta {
                kind: "q4_block".to_string(),
                dtype: "BF16".to_string(),
                shape: vec![128, 352, 2816],
                offset: 0,
                byte_len: 1024,
                source: if external {
                    Some(TensorSource::Local { local_offset: 0 })
                } else {
                    None
                },
            },
        }];
        if external {
            external_files.insert(
                "model-00007-of-00011.safetensors".to_string(),
                ExternalFile {
                    role: ExternalRole::HfSafetensors,
                    path: "model-00007-of-00011.safetensors".to_string(),
                    expected_size: 4_800_000_000,
                    header_sha256: Some("a".repeat(64)),
                },
            );
            tensors.push(DgqTensorEntry {
                name: "model.decoder.embed_tokens.weight".to_string(),
                meta: DgqTensorMeta {
                    kind: "raw".to_string(),
                    dtype: "BF16".to_string(),
                    shape: vec![262144, 2816],
                    offset: 2048,
                    byte_len: 1_478_656_000,
                    source: Some(TensorSource::External {
                        file: "model-00007-of-00011.safetensors".to_string(),
                        offset: 123_456,
                    }),
                },
            });
        }
        DgqManifest {
            version: if external {
                DGQ_VERSION_LAYERED
            } else {
                DGQ_VERSION_AFFINE
            },
            profile: QuantProfile::Q4,
            source_model: "model/transformer".to_string(),
            blob_file: BLOB_FILE.to_string(),
            expert_split: Some(0),
            local_expert_split: if external { Some(0) } else { None },
            base_model: if external {
                Some(BaseModelRef {
                    repo: "google/diffusiongemma-26B-A4B-it".to_string(),
                    revision: "0f28bc42f588fbd8f71e08102b1c3960298a1358".to_string(),
                })
            } else {
                None
            },
            external_files,
            tensors,
        }
    }

    #[test]
    fn manifest_round_trip_layered() {
        let m = sample_manifest(true);
        let json = serde_json::to_string_pretty(&m).expect("serialize");
        let back: DgqManifest = serde_json::from_str(&json).expect("deserialize");
        assert!(back.is_layered());
        assert_eq!(back.base_model.as_ref().unwrap().repo, m.base_model.unwrap().repo);
        assert_eq!(back.external_files.len(), 1);
        match &back.tensors[1].meta.source {
            Some(TensorSource::External { file, offset }) => {
                assert_eq!(file, "model-00007-of-00011.safetensors");
                assert_eq!(*offset, 123_456);
            }
            other => panic!("expected External source, got {other:?}"),
        }
        match &back.tensors[0].meta.source {
            Some(TensorSource::Local { local_offset }) => assert_eq!(*local_offset, 0),
            other => panic!("expected Local source, got {other:?}"),
        }
    }

    #[test]
    fn manifest_round_trip_self_contained() {
        let m = sample_manifest(false);
        let json = serde_json::to_string_pretty(&m).expect("serialize");
        let back: DgqManifest = serde_json::from_str(&json).expect("deserialize");
        assert!(!back.is_layered());
        assert!(back.base_model.is_none());
        assert!(back.external_files.is_empty());
        assert!(back.tensors[0].meta.source.is_none());
    }

    /// A manifest written before layering existed has neither `source` on
    /// entries nor `base_model`/`external_files` at the top level. It must
    /// still parse and behave exactly as before (back-compat).
    #[test]
    fn parse_pre_layering_manifest() {
        let json = r#"{
            "version": 1,
            "profile": "q4",
            "source_model": "model/transformer",
            "blob_file": "model.dgq.bin",
            "expert_split": 5953781760,
            "tensors": [
                {
                    "name": "model.decoder.embed_tokens.weight",
                    "kind": "raw",
                    "dtype": "BF16",
                    "shape": [262144, 2816],
                    "offset": 0,
                    "byte_len": 1478656000
                }
            ]
        }"#;
        let m: DgqManifest = serde_json::from_str(json).expect("parse legacy manifest");
        assert!(!m.is_layered());
        assert!(m.base_model.is_none());
        assert!(m.external_files.is_empty());
        assert!(m.tensors[0].meta.source.is_none());
        assert!(dgq_version_supported(m.version));
    }

    #[test]
    fn layered_version_gate() {
        assert!(dgq_version_supported(DGQ_VERSION_LAYERED));
        assert!(!dgq_version_supported(4));
    }
}
