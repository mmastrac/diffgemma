//! `repack --overlay`: split a self-contained `.dgq` pack into an
//! experts-only overlay whose Raw (verbatim bf16) tensors become external
//! refs into the HF base's safetensors shards, byte-copying the quantized
//! (transformed) tensors into a small local blob. No requantization.

use crate::Error;
use crate::dgq::convert::copy_sidecar_files;
use crate::dgq::hf_resolve::{hash_safetensors_header, resolve_snapshot_dir};
use crate::dgq::layout::{
    BLOB_FILE, BaseModelRef, DGQ_VERSION_LAYERED, DgqManifest, DgqTensorEntry, ExternalFile,
    ExternalRole, MANIFEST_FILE, QuantKind, TensorSource, align_offset, parse_quant_kind,
};
use crate::dgq::store::DgqStore;
use crate::weights::SafetensorStore;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

pub struct RepackOverlayOptions {
    pub pack_dir: PathBuf,
    pub output_dir: PathBuf,
    /// HF safetensors source dir to verify against. Defaults to the pack
    /// manifest's `source_model` (the dir it was originally quantized from).
    pub hf_source_dir: Option<PathBuf>,
    /// Override auto-detected `(repo, revision)` when `hf_source_dir` isn't
    /// itself inside a recognizable `models--org--name/snapshots/<rev>` HF
    /// cache layout (e.g. a plain local checkout).
    pub hf_repo_override: Option<String>,
    pub hf_revision_override: Option<String>,
}

pub struct RepackOverlaySummary {
    pub total_tensors: usize,
    pub external_tensors: usize,
    pub local_tensors: usize,
    /// Raw-kind tensors that did NOT match the HF base byte-for-byte (kept
    /// local instead of externalized) — should be empty; a non-empty list
    /// means the pack's raw region isn't verbatim HF bytes for that tensor.
    pub verbatim_mismatches: Vec<String>,
    pub local_blob_bytes: u64,
    pub base_model: BaseModelRef,
    pub shard_count: usize,
}

pub fn repack_overlay(opts: RepackOverlayOptions) -> Result<RepackOverlaySummary, Error> {
    let store = DgqStore::open(&opts.pack_dir)?;
    if store.is_layered() {
        return Err(Error::Layered(
            "repack --overlay expects a SELF-CONTAINED source pack (already layered)".to_string(),
        ));
    }
    let hf_dir = opts
        .hf_source_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(source_model_of(&opts.pack_dir)));
    let hf_store = SafetensorStore::open(&hf_dir)?;

    let base_model = auto_or_override_base_model(
        &hf_dir,
        &hf_store,
        opts.hf_repo_override.clone(),
        opts.hf_revision_override.clone(),
    )?;
    // Confirm the pinned snapshot the pack will reference actually resolves
    // (fails loud here rather than producing an overlay nobody can load).
    resolve_snapshot_dir(&base_model)?;

    fs::create_dir_all(&opts.output_dir)?;
    let blob_path = opts.output_dir.join(BLOB_FILE);
    let mut blob = BufWriter::new(File::create(&blob_path)?);

    let mut local_offset = 0u64;
    let mut entries = Vec::with_capacity(store.tensor_entries().len());
    let mut external_files: BTreeMap<String, ExternalFile> = BTreeMap::new();
    let mut shard_size_cache: BTreeMap<String, (u64, String)> = BTreeMap::new();
    let mut external_tensors = 0usize;
    let mut local_tensors = 0usize;
    let mut verbatim_mismatches = Vec::new();
    // Page-aligned local expert-region start, set once the first expert
    // tensor is written locally. Experts are always local (never Raw, so
    // never externalized) and the source pack already orders them last, so
    // this makes `blob_file[local_expert_split..]` byte-identical (same
    // relative offsets) to canonical `[expert_split, total)` — see
    // `DgqManifest::local_expert_split`.
    let mut local_expert_split: Option<u64> = None;
    const EXPERT_SPLIT_ALIGN: u64 = 16384;
    let is_expert_entry =
        |e: &DgqTensorEntry| e.name.contains(".experts.") && e.meta.shape.len() == 3;

    let total = store.tensor_entries().len();
    for (i, entry) in store.tensor_entries().iter().enumerate() {
        let kind = parse_quant_kind(&entry.meta.kind)?;
        let pack_bytes = store.tensor_bytes(&entry.name)?;

        let externalized = if kind == QuantKind::Raw {
            match hf_store.get(&entry.name) {
                Some((shard, info)) if shard.data(info) == pack_bytes => {
                    let shard_name = hf_store
                        .weight_map
                        .get(&entry.name)
                        .cloned()
                        .ok_or_else(|| Error::NotFound(entry.name.clone()))?;
                    let (size, hash) = match shard_size_cache.get(&shard_name) {
                        Some(pin) => pin.clone(),
                        None => {
                            let path = hf_dir.join(&shard_name);
                            let (hash, size) = hash_safetensors_header(&path)?;
                            shard_size_cache.insert(shard_name.clone(), (size, hash.clone()));
                            (size, hash)
                        }
                    };
                    external_files.entry(shard_name.clone()).or_insert(ExternalFile {
                        role: ExternalRole::HfSafetensors,
                        path: shard_name.clone(),
                        expected_size: size,
                        header_sha256: Some(hash),
                    });
                    Some(TensorSource::External {
                        file: shard_name,
                        offset: shard.absolute_data_offset(info),
                    })
                }
                Some(_) => {
                    verbatim_mismatches.push(entry.name.clone());
                    None
                }
                None => {
                    verbatim_mismatches.push(entry.name.clone());
                    None
                }
            }
        } else {
            None
        };

        let source = match externalized {
            Some(src) => {
                external_tensors += 1;
                src
            }
            None => {
                local_tensors += 1;
                if local_expert_split.is_none() && is_expert_entry(entry) {
                    let aligned = (local_offset + EXPERT_SPLIT_ALIGN - 1) & !(EXPERT_SPLIT_ALIGN - 1);
                    blob.write_all(&vec![0u8; (aligned - local_offset) as usize])?;
                    local_offset = aligned;
                    local_expert_split = Some(aligned);
                }
                let start = align_offset(local_offset);
                if start > local_offset {
                    blob.write_all(&vec![0u8; (start - local_offset) as usize])?;
                }
                blob.write_all(pack_bytes)?;
                local_offset = start + pack_bytes.len() as u64;
                TensorSource::Local {
                    local_offset: start,
                }
            }
        };

        entries.push(DgqTensorEntry {
            name: entry.name.clone(),
            meta: crate::dgq::layout::DgqTensorMeta {
                kind: entry.meta.kind.clone(),
                dtype: entry.meta.dtype.clone(),
                shape: entry.meta.shape.clone(),
                offset: entry.meta.offset,
                byte_len: entry.meta.byte_len,
                source: Some(source),
            },
        });

        if (i + 1) % 100 == 0 || i + 1 == total {
            eprintln!(
                "  repacked {}/{total} tensors ({external_tensors} external, {local_tensors} local)...",
                i + 1
            );
        }
    }
    blob.flush()?;

    if !verbatim_mismatches.is_empty() {
        eprintln!(
            "warning: {} raw tensor(s) did not match the HF base byte-for-byte and were kept \
             LOCAL instead of externalized: {:?}",
            verbatim_mismatches.len(),
            verbatim_mismatches
        );
    }

    let manifest = DgqManifest {
        version: DGQ_VERSION_LAYERED,
        profile: profile_of(&opts.pack_dir)?,
        source_model: hf_dir.display().to_string(),
        blob_file: BLOB_FILE.to_string(),
        expert_split: expert_split_of(&opts.pack_dir)?,
        local_expert_split,
        base_model: Some(base_model.clone()),
        external_files: external_files.clone(),
        tensors: entries,
    };
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    fs::write(opts.output_dir.join(MANIFEST_FILE), manifest_json)?;
    copy_sidecar_files(&opts.pack_dir, &opts.output_dir)?;

    Ok(RepackOverlaySummary {
        total_tensors: total,
        external_tensors,
        local_tensors,
        verbatim_mismatches,
        local_blob_bytes: local_offset,
        base_model,
        shard_count: external_files.len(),
    })
}

fn source_model_of(pack_dir: &Path) -> String {
    manifest_of(pack_dir)
        .map(|m| m.source_model)
        .unwrap_or_default()
}

fn profile_of(pack_dir: &Path) -> Result<crate::dgq::layout::QuantProfile, Error> {
    Ok(manifest_of(pack_dir)?.profile)
}

fn expert_split_of(pack_dir: &Path) -> Result<Option<u64>, Error> {
    Ok(manifest_of(pack_dir)?.expert_split)
}

fn manifest_of(pack_dir: &Path) -> Result<DgqManifest, Error> {
    let json = fs::read_to_string(pack_dir.join(MANIFEST_FILE))?;
    Ok(serde_json::from_str(&json)?)
}

/// Auto-detect `(repo, revision)` from one symlink hop past any HF shard,
/// matching the `models--org--name/snapshots/<rev>/` cache layout. Explicit
/// overrides win (either both must be given, or neither).
pub fn auto_or_override_base_model(
    hf_dir: &Path,
    hf_store: &SafetensorStore,
    repo_override: Option<String>,
    revision_override: Option<String>,
) -> Result<BaseModelRef, Error> {
    if let (Some(repo), Some(revision)) = (&repo_override, &revision_override) {
        return Ok(BaseModelRef {
            repo: repo.clone(),
            revision: revision.clone(),
        });
    }
    let any_shard = hf_store
        .weight_map
        .values()
        .next()
        .ok_or(Error::Format("HF source has no tensors"))?;
    // A single symlink hop (NOT `fs::canonicalize`, which follows the HF
    // cache's SECOND symlink layer — snapshots/<rev>/file -> blobs/<hash> —
    // and loses the snapshots/<rev> path component we need). `model/transformer`
    // here is a symlink farm straight into the cache snapshot dir; a `-m`
    // pointed directly at a snapshot dir needs no hop at all (the path already
    // contains `snapshots/<rev>` as ancestor components).
    let joined = hf_dir.join(any_shard);
    let probe = fs::read_link(&joined).unwrap_or_else(|_| joined.clone());
    let probe = if probe.is_absolute() {
        probe
    } else {
        joined.parent().unwrap_or(Path::new(".")).join(&probe)
    };
    detect_base_model(&probe).ok_or_else(|| {
        Error::Layered(format!(
            "could not auto-detect (repo, revision) from {} — it isn't inside a \
             models--org--name/snapshots/<rev> HF cache layout. Pass --hf-repo/--hf-revision \
             explicitly.",
            probe.display()
        ))
    })
}

fn detect_base_model(abs_shard_path: &Path) -> Option<BaseModelRef> {
    let comps: Vec<String> = abs_shard_path
        .components()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect();
    let snap_idx = comps.iter().position(|c| c == "snapshots")?;
    let revision = comps.get(snap_idx + 1)?.clone();
    let models_dir = comps.get(snap_idx.checked_sub(1)?)?;
    let rest = models_dir.strip_prefix("models--")?;
    let (org, name) = rest.split_once("--")?;
    Some(BaseModelRef {
        repo: format!("{org}/{name}"),
        revision,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_base_model_parses_hf_cache_layout() {
        let path = Path::new(
            "/Users/x/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/abc123/model-00001-of-00011.safetensors",
        );
        let base = detect_base_model(path).expect("detect");
        assert_eq!(base.repo, "google/diffusiongemma-26B-A4B-it");
        assert_eq!(base.revision, "abc123");
    }

    #[test]
    fn detect_base_model_rejects_non_cache_path() {
        let path = Path::new("/Users/x/model/transformer-local/model-00001-of-00011.safetensors");
        assert!(detect_base_model(path).is_none());
    }
}
