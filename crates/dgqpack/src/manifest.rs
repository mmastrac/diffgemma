//! The on-disk `.dgq` pack format: the manifest types, the version gate, and
//! the byte arithmetic every reader shares.

use crate::Error;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const MANIFEST_FILE: &str = "model.dgq.json";
pub const BLOB_FILE: &str = "model.dgq.bin";
/// Version 1: affine Q4 (`q4_block`). Version 2: adds NVFP4 (`nvfp4_block`).
/// Version 3: adds layered/overlay packs (`DgqTensorMeta::source`,
/// `DgqManifest::external_files`/`base_model`), a deliberate compatibility
/// gate so a pre-layering binary refuses a layered manifest outright
/// (`dgq_version_supported` returns false) instead of silently misreading
/// `offset` as a local-blob position for an externally-sourced tensor.
pub const DGQ_VERSION_AFFINE: u32 = 1;
pub const DGQ_VERSION_NVFP4: u32 = 2;
pub const DGQ_VERSION_LAYERED: u32 = 3;
/// Any pack with a non-empty `custom_classes` map. Binaries that predate
/// kind-driven dispatch derived kernel formats from `profile` and would
/// silently mis-dispatch a class that diverges from it, so they must refuse.
pub const DGQ_VERSION_CUSTOM: u32 = 4;

/// Held at offset 0 of a blob while `diffgemma download` assembles it, and
/// overwritten by the real first chunk once every other byte is on disk and
/// its length checks out. A blob that still carries this is a transfer that
/// did not finish, whatever its name or size says.
pub const INCOMPLETE_SENTINEL: &[u8; 32] = b"dgq-incomplete-download-00000000";

pub fn dgq_version_for_profile(profile: QuantProfile) -> u32 {
    match profile {
        QuantProfile::Nvfp4 | QuantProfile::Nvfp4Experts => DGQ_VERSION_NVFP4,
        QuantProfile::Q4 | QuantProfile::Q5 | QuantProfile::Q6 => DGQ_VERSION_AFFINE,
    }
}

pub fn dgq_version_supported(version: u32) -> bool {
    version == DGQ_VERSION_AFFINE
        || version == DGQ_VERSION_NVFP4
        || version == DGQ_VERSION_LAYERED
        || version == DGQ_VERSION_CUSTOM
}

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
    /// Byte offset in the canonical unified blob address space, the value
    /// every downstream consumer (GPU `w_off` constants via
    /// `build_offsets_from_store`, `DgqGpuBlob::buffer_for`, split-blob
    /// region math) treats as authoritative. For a self-contained pack this
    /// is also where the bytes live in `blob_file`. For a layered pack it is
    /// where the loader materializes the tensor's bytes at load time, and
    /// `source` says where they are read from (absent means `blob_file` at
    /// this same offset).
    pub offset: u64,
    pub byte_len: u64,
    /// Where the bytes actually live on disk. `None` = this pack's own
    /// `blob_file` at `offset` (every pack before layering, and every
    /// non-external tensor in a layered pack once `local_offset` is folded
    /// in, see `TensorSource::Local`). Only present in a version-3
    /// (`DGQ_VERSION_LAYERED`) manifest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<TensorSource>,
}

/// Byte source for a layered-pack tensor entry. Kept separate from the
/// canonical `offset` because a layered pack's own blob is compact (only the
/// tensors that actually live in it, experts and q8, packed with no gaps for
/// the externally-sourced tensors), so a local tensor's on-disk position
/// generally differs from its canonical address-space position too.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TensorSource {
    /// Bytes live in this pack's own `blob_file`, at `local_offset` (may
    /// differ from the entry's canonical `offset`).
    Local { local_offset: u64 },
    /// Bytes live in an external file, a key into
    /// `DgqManifest::external_files`, at `offset` within that file.
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
    /// path to the referenced pack's blob, absolute or relative to this
    /// pack's own directory.
    pub path: String,
    pub expected_size: u64,
    /// `HfSafetensors` only: hex SHA-256 of `(8-byte LE header length ||
    /// header JSON bytes)`, which pins shard identity and layout while
    /// leaving the multi-GiB payload unhashed. `None` for `PackBin`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub header_sha256: Option<String>,
}

/// Pinned HF base-model identity a layered pack's `HfSafetensors` refs
/// resolve against.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseModelRef {
    /// `"org/name"`, e.g. `"google/diffusiongemma-26B-A4B-it"`.
    pub repo: String,
    /// Pinned snapshot revision (commit hash). Resolution stays on this
    /// exact commit so an overlay's meaning holds across a cache refresh.
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
    /// materialized mapping, so the tail is resident once.
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
    /// `quantize --set class=format` overrides actually applied, keyed by
    /// `TensorClass::as_str()` (e.g. `"attn"`, `"experts.gate_up"`) and
    /// valued by the resolved `QuantKind::as_str()` (e.g. `"nvfp4_block"`).
    /// Empty for a plain base-profile pack. Purely descriptive: the loader
    /// dispatches on each tensor's own `kind` (see
    /// `src/metal/step_quant.rs`), so a reader that predates this field
    /// ignores it and still dispatches correctly per-tensor.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub custom_classes: std::collections::BTreeMap<String, String>,
    pub tensors: Vec<DgqTensorEntry>,
}

impl DgqManifest {
    /// A layered pack has at least one externally- or locally-redirected
    /// tensor. Self-contained packs (all `source: None`) take the identity
    /// fast path everywhere, which is what pre-layering code already did.
    pub fn is_layered(&self) -> bool {
        self.tensors.iter().any(|t| t.meta.source.is_some())
    }

    /// Highest byte offset any tensor reaches in the CANONICAL address space
    /// that `offset` and every `w_off` index. For a self-contained pack this
    /// equals the blob file's length. For a layered pack it is the size of
    /// the mapping the loader materializes, which is larger than the compact
    /// local blob on disk.
    pub fn canonical_extent(&self) -> u64 {
        self.tensors
            .iter()
            .map(|t| t.meta.offset + t.meta.byte_len)
            .max()
            .unwrap_or(0)
    }

    /// Byte length this pack's own `blob_file` must reach for every tensor
    /// sourced from it to be readable. A self-contained entry (`source:
    /// None`) lives at its canonical `offset`, a `Local` entry at its
    /// `local_offset`, and an `External` entry's bytes are in another file,
    /// so they contribute nothing here.
    pub fn local_blob_extent(&self) -> u64 {
        self.tensors
            .iter()
            .map(|t| match &t.meta.source {
                None => t.meta.offset + t.meta.byte_len,
                Some(TensorSource::Local { local_offset }) => local_offset + t.meta.byte_len,
                Some(TensorSource::External { .. }) => 0,
            })
            .max()
            .unwrap_or(0)
    }

    /// Reject a `blob_file` shorter than the manifest says it is.
    ///
    /// A short blob is the download failure that survives every other check.
    /// The GPU wraps the truncated mmap as an `MTLBuffer`, kernels read
    /// `blob + w_off` past its end, and out-of-range reads come back as
    /// zeros. Zero weights give exactly uniform logits (entropy `ln(vocab)`)
    /// and an all-`<pad>` canvas, which reads as a broken model rather than
    /// a broken file. Nothing on the GPU path bounds-checks `w_off`, so this
    /// is the only place the shortfall is visible.
    pub fn check_local_blob_len(&self, blob_len: u64, blob_path: &Path) -> Result<(), Error> {
        let need = self.local_blob_extent();
        if blob_len >= need {
            return Ok(());
        }
        Err(Error::Pack(format!(
            "{}: truncated. {blob_len} bytes on disk, manifest needs {need}.\n\
             \x20 The pack is incomplete or corrupt. An interrupted or out-of-disk\n\
             \x20 download is the usual cause. Re-fetch it:\n\
             \x20   diffgemma download --force",
            blob_path.display()
        )))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DgqTensorEntry {
    pub name: String,
    #[serde(flatten)]
    pub meta: DgqTensorMeta,
}

impl DgqTensorEntry {
    pub fn numel(&self) -> usize {
        self.meta.shape.iter().product::<i64>() as usize
    }

    /// A byte-identical bf16/f16/f32 payload, as opposed to one of the
    /// quantized block formats.
    pub fn is_raw(&self) -> bool {
        self.meta.kind == "raw"
    }
}

pub fn align_offset(offset: u64) -> u64 {
    (offset + 63) & !63
}

/// Convert a `.dgq` blob byte offset for host pointer / MTL buffer slicing.
/// NVFP4 blobs can exceed `u32::MAX`, so the caller keeps `u64` up to here.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn entry(offset: u64, byte_len: u64, source: Option<TensorSource>) -> DgqTensorEntry {
        DgqTensorEntry {
            name: format!("t{offset}"),
            meta: DgqTensorMeta {
                kind: "raw".to_string(),
                dtype: "bf16".to_string(),
                shape: vec![1],
                offset,
                byte_len,
                source,
            },
        }
    }

    fn manifest(tensors: Vec<DgqTensorEntry>) -> DgqManifest {
        DgqManifest {
            version: DGQ_VERSION_AFFINE,
            profile: QuantProfile::Q4,
            source_model: "test".to_string(),
            blob_file: BLOB_FILE.to_string(),
            expert_split: None,
            local_expert_split: None,
            base_model: None,
            external_files: BTreeMap::new(),
            custom_classes: BTreeMap::new(),
            tensors,
        }
    }

    /// The published q4 pack is 20227522560 bytes. A copy 4096 bytes short of
    /// that loaded without complaint and then generated nothing but `<pad>`
    /// at entropy ln(vocab), so one byte short has to fail here.
    #[test]
    fn short_blob_is_rejected() {
        let m = manifest(vec![entry(0, 64, None), entry(64, 20227522496, None)]);
        assert_eq!(m.local_blob_extent(), 20227522560);
        let path = Path::new("model.dgq.bin");
        assert!(m.check_local_blob_len(20227522560, path).is_ok());
        assert!(m.check_local_blob_len(20227522561, path).is_ok());
        assert!(m.check_local_blob_len(20227522559, path).is_err());
    }

    /// A layered pack's own blob holds only its `Local` tensors, so the
    /// extent it must reach is the largest `local_offset` end. An `External`
    /// tensor's canonical `offset` is an address in the HF base.
    #[test]
    fn local_blob_extent_ignores_external_bytes() {
        let m = manifest(vec![
            entry(
                0,
                4096,
                Some(TensorSource::External {
                    file: "shard".to_string(),
                    offset: 0,
                }),
            ),
            entry(
                1 << 30,
                512,
                Some(TensorSource::Local { local_offset: 128 }),
            ),
        ]);
        assert_eq!(m.local_blob_extent(), 640);
        assert!(m.check_local_blob_len(640, Path::new("x.bin")).is_ok());
        assert!(m.check_local_blob_len(639, Path::new("x.bin")).is_err());
    }
}
