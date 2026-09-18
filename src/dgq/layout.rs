//! Quantization policy for `.dgq` packs: which format each tensor class gets,
//! and the dimension rules a format imposes.
//!
//! The on-disk format itself lives in `dgqpack`, below the backends. These
//! re-exports keep the engine's `crate::dgq::layout::` paths pointing at it.

use crate::Error;
use std::collections::BTreeMap;

/// The engine is a binary, so a re-export has no downstream consumer to
/// silence the unused-import lint.
#[allow(unused_imports)]
pub use dgqpack::manifest::{
    BLOB_FILE, BaseModelRef, DGQ_VERSION_AFFINE, DGQ_VERSION_CUSTOM, DGQ_VERSION_LAYERED,
    DGQ_VERSION_NVFP4, DgqManifest, DgqTensorEntry, DgqTensorMeta, ExternalFile, ExternalRole,
    INCOMPLETE_SENTINEL, MANIFEST_FILE, QuantKind, QuantProfile, TensorSource, align_offset,
    blob_offset_for_mtl, blob_offset_usize, blob_slice_range, dgq_version_for_profile,
    dgq_version_supported,
};
pub use dgqpack::pack::PackFile;

/// Block-size arithmetic lives in `dgemm`. These names keep their engine
/// paths by re-exporting it. (`Q4_BLOCK_BYTES`/`Q6_BLOCK_BYTES` have no engine
/// caller of their own and are re-exported for path stability.)
#[allow(unused_imports)]
pub use dgemm::format::layout::{
    GROUP_SIZE, NVFP4_GROUP_SIZE, NVFP4_HEADER_BYTES, Q4_BLOCK_BYTES, Q6_BLOCK_BYTES,
    nvfp4_data_row_bytes, nvfp4_matrix_bytes, nvfp4_row_bytes, nvfp4_scales_row_bytes,
    q4_matrix_bytes, q4_row_bytes, q6_matrix_bytes, q6_row_bytes, q8_matrix_bytes, q8_row_bytes,
};

pub fn parse_quant_kind(s: &str) -> Result<QuantKind, Error> {
    Ok(dgqpack::manifest::parse_quant_kind(s)?)
}

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

// ---------------------------------------------------------------------------
// Custom quantization classes (`quantize --set class=format`).
//
// `classify_tensor` above is the fixed per-profile mapping. The functions
// below let the CLI override it per tensor CLASS (not per tensor) on top of
// a base profile, without hand-adding a new `QuantProfile` variant for every
// mixed-precision experiment arm (that mechanism — `QuantProfile::Nvfp4Experts`
// — is what this generalizes; see AGENTS.md and commit ce3c7c7).
// ---------------------------------------------------------------------------

/// A knob-addressable tensor bucket, independent of any base profile.
/// `classify_tensor` answers "what format does profile P give this tensor";
/// `tensor_class` answers "which CLI knob, if any, controls it". Locked
/// tensors (router, norms/scalars, embed_tokens) are not a `TensorClass` at
/// all — `tensor_class` returns `None` for them, so no override can ever
/// reach them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TensorClass {
    ExpertsGateUp,
    ExpertsDown,
    Attn,
    Dense,
    Vision,
    Sc,
}

impl TensorClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ExpertsGateUp => "experts.gate_up",
            Self::ExpertsDown => "experts.down",
            Self::Attn => "attn",
            Self::Dense => "dense",
            Self::Vision => "vision",
            Self::Sc => "sc",
        }
    }

    /// Formats the step-kernel runtime can actually dispatch this class
    /// through today. Quantize-time validation is fatal outside this set —
    /// see `src/metal/step_quant.rs` (`DenseWeightFormat`, `StepBlockProfile`):
    /// a format with no compiled kernel for a class would otherwise
    /// mis-dispatch silently rather than fail loud (AGENTS.md §1), so it is
    /// rejected here, before anything is written.
    pub fn supported_formats(self) -> &'static [QuantKind] {
        match self {
            // Block-sparse expert GEMM: q4/q6/nvfp4 all have a grouped-GEMM
            // kernel. Raw/q8 experts would fall back to the scalar per-expert
            // kernel, which AGENTS.md documents as a probe/oracle surface
            // only — not a production path — so they are excluded here.
            Self::ExpertsGateUp | Self::ExpertsDown => &[
                QuantKind::Q4Block,
                QuantKind::Q6Block,
                QuantKind::Nvfp4Block,
            ],
            // Dense-linear GEMM: raw/q8/q4/nvfp4 all have a compiled kernel
            // at these (n,k) shapes. q6 does not (block-sparse/experts-only
            // dequant, never wired into the dense-linear kernel body).
            Self::Attn | Self::Dense | Self::Sc => &[
                QuantKind::Raw,
                QuantKind::Q8Row,
                QuantKind::Q4Block,
                QuantKind::Nvfp4Block,
            ],
            // The vision tower is not wired into the step-kernel forward
            // pass at all (no source file references vision_tower /
            // embed_vision) — any codec is safe here; this is a pure
            // disk-size lever until a vision forward path exists.
            Self::Vision => &[
                QuantKind::Raw,
                QuantKind::Q4Block,
                QuantKind::Q6Block,
                QuantKind::Q8Row,
                QuantKind::Nvfp4Block,
            ],
        }
    }
}

/// CLI class name -> `TensorClass`(es). `"experts"` expands to BOTH
/// substacks (gate_up and down currently must share one format — the
/// step-kernel's batched grouped-GEMM dispatch keys on a single value; see
/// `build_step_runtime`'s gate_up/down match assertion); `"experts.gate_up"`
/// / `"experts.down"` address one substack alone. Locked classes (embed,
/// router, norms) resolve to a fatal `Error::Config` naming the lock reason,
/// never silently ignored.
pub fn parse_tensor_class(s: &str) -> Result<Vec<TensorClass>, Error> {
    match s {
        "experts" => Ok(vec![TensorClass::ExpertsGateUp, TensorClass::ExpertsDown]),
        "experts.gate_up" => Ok(vec![TensorClass::ExpertsGateUp]),
        "experts.down" => Ok(vec![TensorClass::ExpertsDown]),
        "attn" => Ok(vec![TensorClass::Attn]),
        "dense" => Ok(vec![TensorClass::Dense]),
        "vision" => Ok(vec![TensorClass::Vision]),
        "sc" => Ok(vec![TensorClass::Sc]),
        "embed" | "embed_tokens" => Err(Error::Config(format!(
            "--set: class '{s}' is locked — embed_tokens stays raw bf16 always. It is tied to \
             the lm_head (logits = hidden @ embed^T) and the SC soft-embed; anything less than \
             bf16 measurably stalls tail-token convergence (see classify_tensor's embed_tokens \
             comment)."
        ))),
        "router" => Err(Error::Config(
            "--set: class 'router' is locked — the MoE router stays raw bf16 always. Routing \
             decisions are precision-sensitive and the router tensors are tiny (not a \
             meaningful disk-size lever)."
                .to_string(),
        )),
        "norms" | "norm" | "scalars" | "scalar" | "layer_scalar" => Err(Error::Config(format!(
            "--set: class '{s}' is locked — norm/scale/layer_scalar vectors stay raw always. \
             They are tiny (not a meaningful disk-size lever) and precision-sensitive."
        ))),
        other => Err(Error::Config(format!(
            "--set: unknown class '{other}' (known classes: experts, experts.gate_up, \
             experts.down, attn, dense, vision, sc)"
        ))),
    }
}

/// Short CLI format name -> `QuantKind`. Distinct from `parse_quant_kind`,
/// which parses the on-disk `kind` string (e.g. `"q4_block"`) — this parses
/// the terser `--set class=FORMAT` spelling (`"q4"`).
pub fn parse_custom_format(s: &str) -> Result<QuantKind, Error> {
    match s {
        "raw" => Ok(QuantKind::Raw),
        "q4" => Ok(QuantKind::Q4Block),
        "q6" => Ok(QuantKind::Q6Block),
        "q8" => Ok(QuantKind::Q8Row),
        "nvfp4" => Ok(QuantKind::Nvfp4Block),
        other => Err(Error::Config(format!(
            "--set: unknown format '{other}' (known formats: raw, q4, q6, q8, nvfp4)"
        ))),
    }
}

/// Router / norms-scalars / embed_tokens: always raw, never a `TensorClass`.
fn is_locked_tensor(name: &str) -> bool {
    name.contains(".router.")
        || name.contains("_norm")
        || name.contains("layernorm")
        || name.ends_with(".scale")
        || name.contains("layer_scalar")
        || name.contains("embed_tokens")
}

fn is_attn_linear(name: &str) -> bool {
    name.ends_with(".weight")
        && name.contains(".self_attn.")
        && (name.contains(".q_proj.")
            || name.contains(".k_proj.")
            || name.contains(".v_proj.")
            || name.contains(".o_proj."))
}

fn is_dense_linear(name: &str) -> bool {
    name.ends_with(".weight")
        && (name.contains(".mlp.gate_proj.")
            || name.contains(".mlp.up_proj.")
            || name.contains(".mlp.down_proj."))
}

fn is_vision_linear(name: &str) -> bool {
    name == "model.encoder.embed_vision.embedding_projection.weight"
        || (name.ends_with(".linear.weight")
            && (name.contains(".self_attn.") || name.contains(".mlp.")))
}

/// A tensor's knob-addressable class from name/shape alone (profile-
/// independent). `None` = locked, or otherwise not overridable — falls
/// through to `classify_tensor`'s catch-all `Raw`, same as today.
pub fn tensor_class(name: &str, shape: &[i64]) -> Option<TensorClass> {
    if is_locked_tensor(name) {
        return None;
    }
    if name.contains(".experts.") && shape.len() == 3 {
        if name.ends_with("gate_up_proj") {
            return Some(TensorClass::ExpertsGateUp);
        }
        if name.ends_with("down_proj") {
            return Some(TensorClass::ExpertsDown);
        }
        return None;
    }
    if shape.len() == 2 && name.contains("self_conditioning") && name.ends_with(".weight") {
        return Some(TensorClass::Sc);
    }
    // Vision tensors are checked before the decoder attn/dense matchers:
    // `is_attn_linear`/`is_dense_linear` key on `.self_attn.`/`.mlp.`
    // substrings that vision-tower names also contain.
    if name.contains(".vision_tower.") || name.contains("embed_vision") {
        if shape.len() == 2 && is_vision_linear(name) {
            return Some(TensorClass::Vision);
        }
        return None;
    }
    if shape.len() == 2 && is_attn_linear(name) {
        return Some(TensorClass::Attn);
    }
    if shape.len() == 2 && is_dense_linear(name) {
        return Some(TensorClass::Dense);
    }
    None
}

/// Per-tensor dimension constraint for a resolved format, checked BEFORE any
/// bytes are written (an invalid combo must never partially write a pack):
/// q4/q6 need K (last dim) a multiple of `GROUP_SIZE`; nvfp4 needs a
/// multiple of `NVFP4_GROUP_SIZE`; q8 is a per-row scale (rank 2 only, no
/// block-size constraint); raw has no constraint.
pub fn validate_format_dims(name: &str, shape: &[i64], kind: QuantKind) -> Result<(), Error> {
    match kind {
        QuantKind::Raw => Ok(()),
        QuantKind::Q8Row => {
            if shape.len() != 2 {
                return Err(Error::Config(format!(
                    "{name}: q8 requires a rank-2 (row-major) tensor, got shape {shape:?}"
                )));
            }
            Ok(())
        }
        QuantKind::Q4Block | QuantKind::Q6Block => {
            let k = *shape.last().ok_or_else(|| {
                Error::Config(format!("{name}: q4/q6 requires a non-scalar tensor"))
            })?;
            if k % GROUP_SIZE as i64 != 0 {
                return Err(Error::Config(format!(
                    "{name}: {} requires K (last dim, {k}) to be a multiple of {GROUP_SIZE}",
                    kind.as_str()
                )));
            }
            Ok(())
        }
        QuantKind::Nvfp4Block => {
            let k = *shape.last().ok_or_else(|| {
                Error::Config(format!("{name}: nvfp4 requires a non-scalar tensor"))
            })?;
            if k % NVFP4_GROUP_SIZE as i64 != 0 {
                return Err(Error::Config(format!(
                    "{name}: nvfp4 requires K (last dim, {k}) to be a multiple of {NVFP4_GROUP_SIZE}"
                )));
            }
            Ok(())
        }
    }
}

/// Mixed-precision mapping for a safetensors tensor name, with `--set
/// class=format` overrides layered on top of `profile`. Zero overrides is
/// IDENTICAL to `classify_tensor(name, shape, profile)` by construction:
/// `overrides` is only ever consulted for a tensor `tensor_class` maps to a
/// class, and every other tensor (including every locked one) falls through
/// to the base-profile function unchanged.
pub fn classify_tensor_custom(
    name: &str,
    shape: &[i64],
    profile: QuantProfile,
    overrides: &BTreeMap<TensorClass, QuantKind>,
) -> QuantKind {
    if let Some(class) = tensor_class(name, shape)
        && let Some(&kind) = overrides.get(&class)
    {
        return kind;
    }
    classify_tensor(name, shape, profile)
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
            custom_classes: BTreeMap::new(),
            tensors,
        }
    }

    #[test]
    fn manifest_round_trip_layered() {
        let m = sample_manifest(true);
        let json = serde_json::to_string_pretty(&m).expect("serialize");
        let back: DgqManifest = serde_json::from_str(&json).expect("deserialize");
        assert!(back.is_layered());
        assert_eq!(
            back.base_model.as_ref().unwrap().repo,
            m.base_model.unwrap().repo
        );
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
        assert!(dgq_version_supported(DGQ_VERSION_CUSTOM));
        assert!(!dgq_version_supported(5));
    }

    // -- Custom quantization classes (`quantize --set class=format`) --------

    /// Gate (a): zero overrides classifies EVERY tensor in the real manifest
    /// identically to the base profile. `classify_tensor_custom` with an
    /// empty override map is the same function as `classify_tensor` by
    /// construction (it only ever consults `overrides`), but this proves it
    /// against the full real ~1047-tensor list, not just the hand-picked
    /// cases in `classify_experts_and_router` above.
    #[test]
    fn custom_zero_override_matches_base_profile_on_real_manifest() {
        let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
            eprintln!("skip: no dgq model dir (model/diffgemma-26b-a4b-it-q4 missing)");
            return;
        };
        let json = std::fs::read_to_string(dir.join(MANIFEST_FILE)).expect("read manifest");
        let manifest: DgqManifest = serde_json::from_str(&json).expect("parse manifest");
        let overrides = BTreeMap::new();
        for t in &manifest.tensors {
            let base = classify_tensor(&t.name, &t.meta.shape, manifest.profile);
            let custom =
                classify_tensor_custom(&t.name, &t.meta.shape, manifest.profile, &overrides);
            assert_eq!(
                base, custom,
                "{}: zero-override custom classification diverged from base profile",
                t.name
            );
        }
        assert!(
            manifest.tensors.len() > 500,
            "expected the full real manifest (~1047 tensors), got {}",
            manifest.tensors.len()
        );
    }

    /// Gate (b): kind-histogram for a few override combos against the real
    /// tensor name/shape list. Purely name/shape classification — no blob
    /// read, so this is still Tier-1 fast (AGENTS.md §5).
    #[test]
    fn custom_override_kind_histograms() {
        let Some(dir) = crate::shaders::test_util::dgq_model_dir() else {
            eprintln!("skip: no dgq model dir (model/diffgemma-26b-a4b-it-q4 missing)");
            return;
        };
        let json = std::fs::read_to_string(dir.join(MANIFEST_FILE)).expect("read manifest");
        let manifest: DgqManifest = serde_json::from_str(&json).expect("parse manifest");

        let histogram = |overrides: &BTreeMap<TensorClass, QuantKind>| -> Vec<(QuantKind, usize)> {
            let mut h: Vec<(QuantKind, usize)> = Vec::new();
            for t in &manifest.tensors {
                let kind =
                    classify_tensor_custom(&t.name, &t.meta.shape, manifest.profile, overrides);
                match h.iter_mut().find(|(k, _)| *k == kind) {
                    Some((_, n)) => *n += 1,
                    None => h.push((kind, 1)),
                }
            }
            h
        };
        let count = |h: &[(QuantKind, usize)], k: QuantKind| {
            h.iter()
                .find(|(kk, _)| *kk == k)
                .map(|(_, n)| *n)
                .unwrap_or(0)
        };

        // Base q4 (no overrides): 60 expert stacks q4_block (30 gate_up + 30
        // down, one stacked 3D tensor each per layer), 3 SC q8_row, no
        // nvfp4/q6 anywhere — matches the documented production composition.
        let base = histogram(&BTreeMap::new());
        assert_eq!(count(&base, QuantKind::Q4Block), 60);
        assert_eq!(count(&base, QuantKind::Q8Row), 3);
        assert_eq!(count(&base, QuantKind::Nvfp4Block), 0);
        assert_eq!(count(&base, QuantKind::Q6Block), 0);

        // --set experts=nvfp4 (both substacks): all 60 move q4_block -> nvfp4_block.
        let mut experts_nvfp4 = BTreeMap::new();
        experts_nvfp4.insert(TensorClass::ExpertsGateUp, QuantKind::Nvfp4Block);
        experts_nvfp4.insert(TensorClass::ExpertsDown, QuantKind::Nvfp4Block);
        let h = histogram(&experts_nvfp4);
        assert_eq!(count(&h, QuantKind::Nvfp4Block), 60);
        assert_eq!(count(&h, QuantKind::Q4Block), 0);
        assert_eq!(count(&h, QuantKind::Q8Row), 3, "sc untouched");

        // --set experts=q6 (both substacks): all 60 move q4_block -> q6_block.
        let mut experts_q6 = BTreeMap::new();
        experts_q6.insert(TensorClass::ExpertsGateUp, QuantKind::Q6Block);
        experts_q6.insert(TensorClass::ExpertsDown, QuantKind::Q6Block);
        let h = histogram(&experts_q6);
        assert_eq!(count(&h, QuantKind::Q6Block), 60);
        assert_eq!(count(&h, QuantKind::Q4Block), 0);

        // --set attn=nvfp4: 115 attn q/k/v/o_proj tensors move raw ->
        // nvfp4_block; experts (60, q4_block) and SC (3, q8_row) untouched.
        let mut attn_nvfp4 = BTreeMap::new();
        attn_nvfp4.insert(TensorClass::Attn, QuantKind::Nvfp4Block);
        let h = histogram(&attn_nvfp4);
        assert_eq!(count(&h, QuantKind::Nvfp4Block), 115);
        assert_eq!(count(&h, QuantKind::Q4Block), 60, "experts untouched");
        assert_eq!(count(&h, QuantKind::Q8Row), 3, "sc untouched");
        assert_eq!(
            count(&base, QuantKind::Raw) - count(&h, QuantKind::Raw),
            115,
            "exactly the 115 attn tensors left the raw bucket"
        );
    }

    /// Gate (c): locked classes are rejected with the lock reason in the
    /// message, never silently ignored; unknown classes are also rejected.
    #[test]
    fn locked_classes_rejected_with_reason() {
        for locked in [
            "embed",
            "embed_tokens",
            "router",
            "norms",
            "norm",
            "layer_scalar",
        ] {
            let err = parse_tensor_class(locked).expect_err("locked class must be rejected");
            let msg = err.to_string();
            assert!(
                msg.contains("locked"),
                "{locked}: expected a lock reason in the error, got: {msg}"
            );
        }
        assert!(parse_tensor_class("bogus").is_err());
        assert!(parse_tensor_class("attn").is_ok());
    }

    /// Gate (d): per-tensor dimension constraints are rejected BEFORE
    /// anything would be written — q4/q6 need K % 32 == 0, nvfp4 needs
    /// K % 16 == 0, q8 needs rank 2.
    #[test]
    fn dimension_constraints_rejected() {
        assert!(validate_format_dims("t", &[64, 33], QuantKind::Q4Block).is_err());
        assert!(validate_format_dims("t", &[64, 32], QuantKind::Q4Block).is_ok());
        assert!(validate_format_dims("t", &[64, 48], QuantKind::Q6Block).is_err());
        assert!(validate_format_dims("t", &[64, 64], QuantKind::Q6Block).is_ok());
        assert!(validate_format_dims("t", &[64, 17], QuantKind::Nvfp4Block).is_err());
        assert!(validate_format_dims("t", &[64, 16], QuantKind::Nvfp4Block).is_ok());
        assert!(validate_format_dims("t", &[128, 64, 32], QuantKind::Q8Row).is_err());
        assert!(validate_format_dims("t", &[128, 64], QuantKind::Q8Row).is_ok());
        assert!(validate_format_dims("t", &[1], QuantKind::Raw).is_ok());
    }

    /// The runtime-support matrix behind the fatal "unsupported for class"
    /// CLI error (`model_ops::parse_custom_overrides`): q6 has no
    /// dense-linear kernel, so attn/dense/sc must exclude it while the
    /// block-sparse expert classes accept it.
    #[test]
    fn class_supported_formats_excludes_unsupported_combos() {
        assert!(
            !TensorClass::Attn
                .supported_formats()
                .contains(&QuantKind::Q6Block)
        );
        assert!(
            !TensorClass::Dense
                .supported_formats()
                .contains(&QuantKind::Q6Block)
        );
        assert!(
            !TensorClass::Sc
                .supported_formats()
                .contains(&QuantKind::Q6Block)
        );
        assert!(
            TensorClass::ExpertsGateUp
                .supported_formats()
                .contains(&QuantKind::Q6Block)
        );
        assert!(
            !TensorClass::ExpertsGateUp
                .supported_formats()
                .contains(&QuantKind::Raw),
            "raw experts would fall back to the scalar per-expert kernel (probe/oracle only)"
        );
        assert!(
            TensorClass::Vision
                .supported_formats()
                .contains(&QuantKind::Q6Block)
        );
    }
}
