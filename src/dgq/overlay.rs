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

    let mut external_files: BTreeMap<String, ExternalFile> = BTreeMap::new();
    let mut shard_size_cache: BTreeMap<String, (u64, String)> = BTreeMap::new();
    let mut verbatim_mismatches = Vec::new();
    let is_expert_entry =
        |e: &DgqTensorEntry| e.name.contains(".experts.") && e.meta.shape.len() == 3;

    // Pass 1: decide External vs Local per tensor (unchanged verbatim-match
    // logic) WITHOUT assigning any offsets yet — the canonical layout is
    // computed fresh in pass 2 (see below), not carried over from the
    // self-contained source pack's (alphabetical, non-spliceable) offsets.
    struct Decision<'a> {
        entry: &'a DgqTensorEntry,
        /// `Some((shard_file, file_byte_offset))` when this tensor verified
        /// byte-identical to the HF base and will be externalized.
        external: Option<(String, u64)>,
    }
    let mut decisions: Vec<Decision> = Vec::with_capacity(store.tensor_entries().len());
    for entry in store.tensor_entries() {
        let kind = parse_quant_kind(&entry.meta.kind)?;
        let pack_bytes = store.tensor_bytes(&entry.name)?;
        let external = if kind == QuantKind::Raw {
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
                    external_files
                        .entry(shard_name.clone())
                        .or_insert(ExternalFile {
                            role: ExternalRole::HfSafetensors,
                            path: shard_name.clone(),
                            expected_size: size,
                            header_sha256: Some(hash),
                        });
                    Some((shard_name, shard.absolute_data_offset(info)))
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
        decisions.push(Decision { entry, external });
    }

    // Pass 2: visiting order — name-sorted, experts always last (the
    // expert-tail no-copy wrap depends on experts occupying the blob tail).
    let mut order: Vec<usize> = (0..decisions.len()).collect();
    order.sort_by_key(|&i| {
        (
            is_expert_entry(decisions[i].entry),
            decisions[i].entry.name.clone(),
        )
    });

    // Pass 3: write the local blob and assign fresh canonical offsets in
    // `order`, every tensor 64-byte aligned (GPU typed-pointer invariant —
    // see the matching comment in convert.rs's `quantize_model`).
    let mut offset = 0u64;
    let mut local_offset = 0u64;
    let mut expert_split: Option<u64> = None;
    let mut local_expert_split: Option<u64> = None;
    const EXPERT_SPLIT_ALIGN: u64 = 16384;
    let mut external_tensors = 0usize;
    let mut local_tensors = 0usize;
    let mut entries = Vec::with_capacity(decisions.len());

    let total = order.len();
    for (progress, &i) in order.iter().enumerate() {
        let d = &decisions[i];
        let entry = d.entry;
        offset = align_offset(offset);

        if expert_split.is_none() && is_expert_entry(entry) {
            offset = (offset + EXPERT_SPLIT_ALIGN - 1) & !(EXPERT_SPLIT_ALIGN - 1);
            expert_split = Some(offset);
            let aligned = (local_offset + EXPERT_SPLIT_ALIGN - 1) & !(EXPERT_SPLIT_ALIGN - 1);
            blob.write_all(&vec![0u8; (aligned - local_offset) as usize])?;
            local_offset = aligned;
            local_expert_split = Some(aligned);
        }
        let canonical_start = offset;

        let source = match &d.external {
            Some((shard_name, file_off)) => {
                external_tensors += 1;
                TensorSource::External {
                    file: shard_name.clone(),
                    offset: *file_off,
                }
            }
            None => {
                local_tensors += 1;
                let start = align_offset(local_offset);
                if start > local_offset {
                    blob.write_all(&vec![0u8; (start - local_offset) as usize])?;
                }
                let pack_bytes = store.tensor_bytes(&entry.name)?;
                blob.write_all(pack_bytes)?;
                local_offset = start + pack_bytes.len() as u64;
                TensorSource::Local {
                    local_offset: start,
                }
            }
        };
        offset += entry.meta.byte_len;

        entries.push(DgqTensorEntry {
            name: entry.name.clone(),
            meta: crate::dgq::layout::DgqTensorMeta {
                kind: entry.meta.kind.clone(),
                dtype: entry.meta.dtype.clone(),
                shape: entry.meta.shape.clone(),
                offset: canonical_start,
                byte_len: entry.meta.byte_len,
                source: Some(source),
            },
        });

        if (progress + 1) % 100 == 0 || progress + 1 == total {
            eprintln!(
                "  repacked {}/{total} tensors ({external_tensors} external, {local_tensors} local)...",
                progress + 1
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
        expert_split,
        local_expert_split,
        base_model: Some(base_model.clone()),
        external_files: external_files.clone(),
        // Repacking only moves bytes (external ref vs local blob) — it never
        // requantizes, so any custom-class overrides the source pack was
        // built with still describe it exactly.
        custom_classes: manifest_of(&opts.pack_dir)?.custom_classes,
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

pub struct RepackMonolithicOptions {
    pub pack_dir: PathBuf,
    pub output_dir: PathBuf,
}

pub struct RepackMonolithicSummary {
    pub tensor_count: usize,
    pub blob_bytes: u64,
    pub kind_histogram: BTreeMap<String, usize>,
}

/// `repack --monolithic`: the dual of `repack --overlay` — flatten a layered
/// pack back into a self-contained one. Every tensor's bytes (wherever they
/// actually live: this pack's own compact blob, or a resolved external HF
/// shard — `DgqStore` already resolved every `external_files` entry at
/// `open()`) are streamed to the output blob at the manifest's CANONICAL
/// offset, so the output is byte-identical to what a plain (non-overlay)
/// `quantize` of the same source would have produced: no requantization,
/// only a byte-copy driven entirely by the existing manifest offsets. Memory
/// stays bounded to one `BufWriter` chunk regardless of pack size — each
/// tensor's bytes come from an mmap'd slice, never a bulk in-memory copy.
pub fn repack_monolithic(opts: RepackMonolithicOptions) -> Result<RepackMonolithicSummary, Error> {
    let store = DgqStore::open(&opts.pack_dir)?;
    if !store.is_layered() {
        return Err(Error::Layered(
            "repack --monolithic expects a LAYERED source pack (this one is already \
             self-contained — nothing to flatten)"
                .to_string(),
        ));
    }

    fs::create_dir_all(&opts.output_dir)?;
    let blob_path = opts.output_dir.join(BLOB_FILE);
    let mut blob = BufWriter::new(File::create(&blob_path)?);

    // Visit tensors in CANONICAL offset order so the writer's cursor only
    // ever advances forward (the layered writer already produces this order,
    // but a monolithic repack must not assume it — the manifest's `offset`
    // is the single source of truth for where each tensor lands).
    let mut entries: Vec<&DgqTensorEntry> = store.tensor_entries().iter().collect();
    entries.sort_by_key(|e| e.meta.offset);

    let mut cursor = 0u64;
    let mut kind_histogram: BTreeMap<String, usize> = BTreeMap::new();
    let total = entries.len();
    let mut out_entries = Vec::with_capacity(total);
    for (i, entry) in entries.iter().enumerate() {
        let start = entry.meta.offset;
        if start < cursor {
            return Err(Error::Layered(format!(
                "repack --monolithic: tensor {} canonical offset {start} overlaps the previous \
                 tensor's end {cursor} — manifest is internally inconsistent",
                entry.name
            )));
        }
        if start > cursor {
            blob.write_all(&vec![0u8; (start - cursor) as usize])?;
        }
        let bytes = store.tensor_bytes(&entry.name)?;
        if bytes.len() as u64 != entry.meta.byte_len {
            return Err(Error::Layered(format!(
                "repack --monolithic: tensor {} resolved {} bytes, manifest claims {}",
                entry.name,
                bytes.len(),
                entry.meta.byte_len
            )));
        }
        blob.write_all(bytes)?;
        cursor = start + entry.meta.byte_len;

        *kind_histogram.entry(entry.meta.kind.clone()).or_insert(0) += 1;
        out_entries.push(DgqTensorEntry {
            name: entry.name.clone(),
            meta: crate::dgq::layout::DgqTensorMeta {
                kind: entry.meta.kind.clone(),
                dtype: entry.meta.dtype.clone(),
                shape: entry.meta.shape.clone(),
                offset: entry.meta.offset,
                byte_len: entry.meta.byte_len,
                source: None,
            },
        });

        if (i + 1) % 100 == 0 || i + 1 == total {
            eprintln!(
                "  flattened {}/{total} tensors ({:.2} GiB)...",
                i + 1,
                cursor as f64 / (1024.0_f64.powi(3))
            );
        }
    }
    blob.flush()?;

    let src_manifest = manifest_of(&opts.pack_dir)?;
    let has_nvfp4 = out_entries.iter().any(|e| e.meta.kind == "nvfp4_block");
    let base_version = crate::dgq::layout::dgq_version_for_profile(src_manifest.profile);
    let version = if !src_manifest.custom_classes.is_empty() {
        crate::dgq::layout::DGQ_VERSION_CUSTOM
    } else if has_nvfp4 {
        base_version.max(crate::dgq::layout::DGQ_VERSION_NVFP4)
    } else {
        base_version
    };

    let manifest = DgqManifest {
        version,
        profile: src_manifest.profile,
        source_model: src_manifest.source_model,
        blob_file: BLOB_FILE.to_string(),
        expert_split: src_manifest.expert_split,
        local_expert_split: None,
        base_model: None,
        external_files: BTreeMap::new(),
        custom_classes: src_manifest.custom_classes,
        tensors: out_entries,
    };
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    fs::write(opts.output_dir.join(MANIFEST_FILE), manifest_json)?;
    copy_sidecar_files(&opts.pack_dir, &opts.output_dir)?;

    Ok(RepackMonolithicSummary {
        tensor_count: total,
        blob_bytes: cursor,
        kind_histogram,
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

/// Round-trip: `quantize --overlay` -> `repack --monolithic` reproduces the
/// same tensor bytes and kind histogram a plain (non-overlay) `quantize` of
/// the identical source would have — synthetic fixture (a few hand-built
/// safetensors tensors covering Raw/Q8Row/Q4Block), never the real 19 GiB
/// model, so this runs in milliseconds like any other Tier-1 test.
#[cfg(test)]
mod monolithic_roundtrip_tests {
    use super::*;
    use crate::dgq::layout::QuantProfile;
    use crate::dgq::store::DgqStore;
    use crate::dgq::test_fixtures::{bf16_payload, write_index, write_shard};
    use crate::dgq::{QuantizeOptions, quantize_model};

    /// Build a fixture HF snapshot dir under `hf_home/hub/models--acme--
    /// widgets/snapshots/<rev>/`: one shard with a Raw-classified attn
    /// tensor, a Q8Row-classified SC tensor, and a Q4Block-classified expert
    /// stack — one representative of each `TensorSource` a real overlay
    /// pack actually uses (External for Raw, Local for everything else).
    fn build_fixture_snapshot(hf_home: &Path) -> (PathBuf, BaseModelRef) {
        let snapshot_dir = hf_home
            .join("hub")
            .join("models--acme--widgets")
            .join("snapshots")
            .join("cafef00d");
        std::fs::create_dir_all(&snapshot_dir).expect("mkdir snapshot");

        let attn = (
            "model.decoder.layers.0.self_attn.q_proj.weight",
            vec![4, 8],
            bf16_payload(32, 1),
        );
        let sc = (
            "model.decoder.self_conditioning.down_proj.weight",
            vec![4, 8],
            bf16_payload(32, 2),
        );
        let experts = (
            "model.decoder.layers.0.experts.gate_up_proj",
            vec![2, 4, 32],
            bf16_payload(2 * 4 * 32, 3),
        );
        let names: Vec<&str> = vec![attn.0, sc.0, experts.0];
        write_shard(
            &snapshot_dir.join("model-00001-of-00001.safetensors"),
            &[attn, sc, experts],
        );
        write_index(&snapshot_dir, "model-00001-of-00001.safetensors", &names);

        (
            snapshot_dir,
            BaseModelRef {
                repo: "acme/widgets".to_string(),
                revision: "cafef00d".to_string(),
            },
        )
    }

    #[test]
    fn overlay_then_monolithic_matches_plain_quantize() {
        let root =
            std::env::temp_dir().join(format!("dgq-monolithic-roundtrip-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let hf_home = root.join("hf_home");
        let (snapshot_dir, base) = build_fixture_snapshot(&hf_home);

        let cfg = crate::flags::RuntimeConfig::from_pairs(&[(
            "DGQ_HF_HOME".to_string(),
            hf_home.display().to_string(),
        )])
        .0;
        let _guard = crate::flags::install_for_test(cfg);

        // Plain (non-overlay) quantize — the ground truth this round trip
        // must reproduce byte-for-byte.
        let baseline_dir = root.join("baseline");
        quantize_model(QuantizeOptions {
            source_dir: snapshot_dir.clone(),
            output_prefix: baseline_dir.clone(),
            profile: QuantProfile::Q4,
            overlay_base: None,
            custom_overrides: Default::default(),
        })
        .expect("baseline quantize");

        // Overlay quantize from the SAME source, pinned at the fixture base.
        let overlay_dir = root.join("overlay");
        quantize_model(QuantizeOptions {
            source_dir: snapshot_dir.clone(),
            output_prefix: overlay_dir.clone(),
            profile: QuantProfile::Q4,
            overlay_base: Some(base),
            custom_overrides: Default::default(),
        })
        .expect("overlay quantize");
        let overlay_store = DgqStore::open(&overlay_dir).expect("open overlay");
        assert!(overlay_store.is_layered());

        // repack --monolithic: flatten the overlay back to self-contained.
        let mono_dir = root.join("monolithic");
        let summary = repack_monolithic(RepackMonolithicOptions {
            pack_dir: overlay_dir.clone(),
            output_dir: mono_dir.clone(),
        })
        .expect("repack --monolithic");

        let baseline_store = DgqStore::open(&baseline_dir).expect("open baseline");
        let mono_store = DgqStore::open(&mono_dir).expect("open monolithic");
        assert!(!mono_store.is_layered());

        // Sample-compare every tensor's bytes against the baseline.
        for entry in baseline_store.tensor_entries() {
            let want = baseline_store
                .tensor_bytes(&entry.name)
                .expect("baseline bytes");
            let got = mono_store
                .tensor_bytes(&entry.name)
                .expect("monolithic bytes");
            assert_eq!(
                want, got,
                "tensor {} diverged after overlay -> monolithic round trip",
                entry.name
            );
        }

        // Manifest kind histogram identical between baseline and the
        // round-tripped monolithic pack.
        let mut baseline_kinds: BTreeMap<&str, usize> = BTreeMap::new();
        for e in baseline_store.tensor_entries() {
            *baseline_kinds.entry(e.meta.kind.as_str()).or_insert(0) += 1;
        }
        let mut mono_kinds: BTreeMap<&str, usize> = BTreeMap::new();
        for e in mono_store.tensor_entries() {
            *mono_kinds.entry(e.meta.kind.as_str()).or_insert(0) += 1;
        }
        assert_eq!(baseline_kinds, mono_kinds);
        assert_eq!(summary.tensor_count, baseline_store.tensor_entries().len());
        assert_eq!(summary.kind_histogram.len(), baseline_kinds.len());

        let _ = std::fs::remove_dir_all(&root);
    }
}
