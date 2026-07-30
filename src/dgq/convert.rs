//! Safetensors → `.dgq` offline quantizer (always from raw bf16 weights).

use crate::Error;
use crate::dgq::block::{
    quantize_bf16_matrix_q4, quantize_bf16_matrix_q6, quantize_bf16_matrix_q8,
    quantize_expert_stack_q4, quantize_expert_stack_q6,
};
use crate::dgq::layout::{
    BLOB_FILE, BaseModelRef, DGQ_VERSION_LAYERED, DGQ_VERSION_NVFP4, DgqManifest, DgqTensorEntry,
    DgqTensorMeta, ExternalFile, ExternalRole, MANIFEST_FILE, QuantKind, QuantProfile,
    TensorClass, TensorSource, align_offset, classify_tensor_custom, dgq_version_for_profile,
    tensor_class, validate_format_dims,
};
use crate::dgq::hf_resolve::hash_safetensors_header;
use crate::dgq::nvfp4::{quantize_bf16_matrix_nvfp4, quantize_expert_stack_nvfp4};
use crate::weights::SafetensorStore;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

pub struct QuantizeOptions {
    pub source_dir: PathBuf,
    pub output_prefix: PathBuf,
    pub profile: QuantProfile,
    /// When `Some`, emit an EXPERTS-ONLY OVERLAY: Raw-kind tensors (embed,
    /// attention, dense FFN, norms, router — everything `classify_tensor`
    /// leaves bf16) become external refs into this HF base's safetensors
    /// shards instead of being copied into this pack's own blob. Only the
    /// quantized (transformed) tensors are written locally. Requires
    /// `source_dir` to be (symlinks into) a resolvable HF cache snapshot —
    /// see `dgq::overlay::auto_or_override_base_model`.
    pub overlay_base: Option<BaseModelRef>,
    /// `quantize --set class=format` overrides layered on `profile`
    /// (`classify_tensor_custom`). Empty = identical to the base profile.
    pub custom_overrides: BTreeMap<TensorClass, QuantKind>,
}

pub struct QuantizeSummary {
    pub tensor_count: usize,
    /// Canonical (virtual) blob size — identical to `local_blob_bytes` unless
    /// `overlay_base` was set, in which case it also counts externalized
    /// (not locally stored) Raw tensors.
    pub blob_bytes: u64,
    /// Bytes actually written to this pack's own blob file.
    pub local_blob_bytes: u64,
    pub q4_tensors: usize,
    pub q6_tensors: usize,
    pub nvfp4_tensors: usize,
    pub q8_tensors: usize,
    pub raw_tensors: usize,
    /// Resolved class -> format actually applied (mirrors the manifest's
    /// `custom_classes`; empty for a plain base-profile pack).
    pub custom_classes: BTreeMap<String, String>,
    /// Per-class byte totals (canonical, i.e. `blob_bytes`-space) for the
    /// disk-math printout: `TensorClass::as_str()` for knob-addressable
    /// tensors, `"locked/other"` for everything `tensor_class` doesn't map
    /// (router, norms, embed_tokens, and anything else outside the five
    /// classes).
    pub class_bytes: BTreeMap<String, u64>,
}

const LOCKED_BUCKET: &str = "locked/other";

pub fn quantize_model(opts: QuantizeOptions) -> Result<QuantizeSummary, Error> {
    let out_dir = opts.output_prefix;
    fs::create_dir_all(&out_dir)?;
    copy_sidecar_files(&opts.source_dir, &out_dir)?;

    let store = SafetensorStore::open(&opts.source_dir)?;

    // Validate every tensor's resolved (class, format) combination BEFORE
    // writing a single byte — an invalid combo must never leave a partially
    // written pack on disk.
    for name in store.weight_map.keys() {
        let (_, info) = store.get(name).ok_or_else(|| Error::NotFound(name.clone()))?;
        let kind = classify_tensor_custom(name, &info.shape, opts.profile, &opts.custom_overrides);
        validate_format_dims(name, &info.shape, kind)?;
    }

    let blob_path = out_dir.join(BLOB_FILE);
    let mut blob = BufWriter::new(File::create(&blob_path)?);

    // `offset` is the CANONICAL (virtual) cursor: every tensor advances it,
    // including externalized ones in overlay mode, so `w_off` addressing
    // stays a single consistent space regardless of where bytes live.
    // `local_bytes_written` is this pack's own blob FILE cursor: only
    // tensors actually persisted here advance it. In non-overlay mode the
    // two are identical (today's behavior, unchanged).
    let mut offset = 0u64;
    let mut local_bytes_written = 0u64;
    let mut entries = Vec::with_capacity(store.weight_map.len());
    let mut q4_tensors = 0usize;
    let mut q6_tensors = 0usize;
    let mut nvfp4_tensors = 0usize;
    let mut q8_tensors = 0usize;
    let mut raw_tensors = 0usize;
    let mut class_bytes: BTreeMap<String, u64> = BTreeMap::new();
    let mut external_files: BTreeMap<String, ExternalFile> = BTreeMap::new();
    let mut shard_pin_cache: BTreeMap<String, (u64, String)> = BTreeMap::new();

    let mut names: Vec<String> = store.weight_map.keys().cloned().collect();
    names.sort();
    // Experts LAST: lets the loader wrap [0, split) + [split, end) as two
    // no-copy MTLBuffers when the blob exceeds the device's max single-buffer
    // length (M3 Pro/36GB: 20.25 GiB; q6 blob: ~24 GiB).
    let is_expert = |n: &str| -> bool {
        store
            .get(n)
            .map(|(_, info)| n.contains(".experts.") && info.shape.len() == 3)
            .unwrap_or(false)
    };
    names.sort_by_key(|n| (is_expert(n), n.clone()));
    let mut expert_split: Option<u64> = None;
    // Overlay mode only: the local blob's own expert-region start, page-
    // aligned the same way as `expert_split`. Because `names` visits experts
    // LAST and every expert is always written locally, the two cursors
    // accumulate the identical sequence of (byte_len, align_offset padding)
    // once both start page-aligned — so `blob_file[local_expert_split..]` ends
    // up byte-identical to canonical `[expert_split, total)`, letting the
    // loader wrap the (large) expert tail straight off this pack's own blob
    // instead of gather-copying it (see `DgqManifest::local_expert_split`).
    let mut local_expert_split: Option<u64> = None;
    const EXPERT_SPLIT_ALIGN: u64 = 16384; // page alignment for no-copy regions
    for (i, name) in names.iter().enumerate() {
        let (shard, info) = store
            .get(name)
            .ok_or_else(|| Error::NotFound(name.clone()))?;
        let src = shard.data(info);
        let kind = classify_tensor_custom(name, &info.shape, opts.profile, &opts.custom_overrides);
        let class_bucket = tensor_class(name, &info.shape)
            .map(TensorClass::as_str)
            .unwrap_or(LOCKED_BUCKET);

        offset = align_offset(offset);
        if expert_split.is_none() && is_expert(name) {
            offset = (offset + EXPERT_SPLIT_ALIGN - 1) & !(EXPERT_SPLIT_ALIGN - 1);
            expert_split = Some(offset);
            if opts.overlay_base.is_some() {
                let aligned = (local_bytes_written + EXPERT_SPLIT_ALIGN - 1) & !(EXPERT_SPLIT_ALIGN - 1);
                let pad = (aligned - local_bytes_written) as usize;
                blob.write_all(&vec![0u8; pad])?;
                local_bytes_written = aligned;
                local_expert_split = Some(aligned);
            }
        }
        let canonical_start = offset;

        let (byte_len, source) = if kind == QuantKind::Raw && opts.overlay_base.is_some() {
            raw_tensors += 1;
            let shard_name = store
                .weight_map
                .get(name)
                .cloned()
                .ok_or_else(|| Error::NotFound(name.clone()))?;
            let (size, hash) = match shard_pin_cache.get(&shard_name) {
                Some(pin) => pin.clone(),
                None => {
                    let path = opts.source_dir.join(&shard_name);
                    let (hash, size) = hash_safetensors_header(&path)?;
                    shard_pin_cache.insert(shard_name.clone(), (size, hash.clone()));
                    (size, hash)
                }
            };
            external_files.entry(shard_name.clone()).or_insert(ExternalFile {
                role: ExternalRole::HfSafetensors,
                path: shard_name.clone(),
                expected_size: size,
                header_sha256: Some(hash),
            });
            (
                src.len() as u64,
                Some(TensorSource::External {
                    file: shard_name,
                    offset: shard.absolute_data_offset(info),
                }),
            )
        } else {
            let local_start = align_offset(local_bytes_written);
            if local_start > local_bytes_written {
                let pad = (local_start - local_bytes_written) as usize;
                blob.write_all(&vec![0u8; pad])?;
                local_bytes_written += pad as u64;
            }
            let byte_len = match kind {
                QuantKind::Raw => {
                    raw_tensors += 1;
                    blob.write_all(src)?;
                    src.len() as u64
                }
                QuantKind::Q4Block => {
                    q4_tensors += 1;
                    write_q4_tensor(&mut blob, src, &info.shape)?
                }
                QuantKind::Q6Block => {
                    q6_tensors += 1;
                    write_q6_tensor(&mut blob, src, &info.shape)?
                }
                QuantKind::Nvfp4Block => {
                    nvfp4_tensors += 1;
                    write_nvfp4_tensor(&mut blob, src, &info.shape)?
                }
                QuantKind::Q8Row => {
                    q8_tensors += 1;
                    write_q8_tensor(&mut blob, src, &info.shape)?
                }
            };
            local_bytes_written += byte_len;
            let source = opts
                .overlay_base
                .is_some()
                .then_some(TensorSource::Local {
                    local_offset: local_start,
                });
            (byte_len, source)
        };
        offset += byte_len;
        *class_bytes.entry(class_bucket.to_string()).or_insert(0) += byte_len;

        entries.push(DgqTensorEntry {
            name: name.clone(),
            meta: DgqTensorMeta {
                kind: kind.as_str().to_string(),
                dtype: info.dtype.as_str().to_string(),
                shape: info.shape.clone(),
                offset: canonical_start,
                byte_len,
                source,
            },
        });

        if (i + 1) % 50 == 0 || i + 1 == names.len() {
            eprintln!(
                "  quantized {}/{} tensors ({:.2} GiB canonical, {:.2} GiB local blob)...",
                i + 1,
                names.len(),
                offset as f64 / (1024.0_f64.powi(3)),
                local_bytes_written as f64 / (1024.0_f64.powi(3))
            );
        }
    }
    blob.flush()?;

    // Resolved class -> format map actually applied (only the classes the
    // caller explicitly overrode — a plain base-profile pack has none).
    let custom_classes: BTreeMap<String, String> = opts
        .custom_overrides
        .iter()
        .map(|(&class, &kind)| (class.as_str().to_string(), kind.as_str().to_string()))
        .collect();

    // Version gates readers that predate a format's introduction (see
    // `DGQ_VERSION_NVFP4` doc comment): a custom pack can introduce nvfp4
    // tensors even when `profile` alone wouldn't (e.g. `--profile q4 --set
    // experts=nvfp4`), so the bump is keyed on tensors ACTUALLY present, not
    // on `profile`.
    let base_version = dgq_version_for_profile(opts.profile);
    let has_nvfp4 = entries.iter().any(|e| e.meta.kind == "nvfp4_block");
    let version = if opts.overlay_base.is_some() {
        DGQ_VERSION_LAYERED
    } else if has_nvfp4 {
        base_version.max(DGQ_VERSION_NVFP4)
    } else {
        base_version
    };

    let manifest = DgqManifest {
        version,
        profile: opts.profile,
        expert_split,
        local_expert_split,
        source_model: opts.source_dir.display().to_string(),
        blob_file: BLOB_FILE.to_string(),
        base_model: opts.overlay_base.clone(),
        external_files,
        custom_classes: custom_classes.clone(),
        tensors: entries,
    };
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    fs::write(out_dir.join(MANIFEST_FILE), manifest_json)?;

    Ok(QuantizeSummary {
        tensor_count: names.len(),
        blob_bytes: offset,
        local_blob_bytes: local_bytes_written,
        q4_tensors,
        q6_tensors,
        nvfp4_tensors,
        q8_tensors,
        raw_tensors,
        custom_classes,
        class_bytes,
    })
}

fn write_q6_tensor(out: &mut impl Write, src: &[u8], shape: &[i64]) -> Result<u64, Error> {
    match shape.len() {
        2 => {
            let out_dim = shape[0] as usize;
            let in_dim = shape[1] as usize;
            let need = crate::dgq::layout::q6_matrix_bytes(out_dim, in_dim);
            let mut buf = vec![0u8; need];
            quantize_bf16_matrix_q6(src, out_dim, in_dim, &mut buf);
            out.write_all(&buf)?;
            Ok(need as u64)
        }
        3 => {
            let experts = shape[0] as usize;
            let out_dim = shape[1] as usize;
            let in_dim = shape[2] as usize;
            let need = experts * crate::dgq::layout::q6_matrix_bytes(out_dim, in_dim);
            let mut buf = vec![0u8; need];
            quantize_expert_stack_q6(src, experts, out_dim, in_dim, &mut buf)?;
            out.write_all(&buf)?;
            Ok(need as u64)
        }
        _ => Err(Error::Format("q6 tensor must be 2D or 3D")),
    }
}

fn write_q4_tensor(out: &mut impl Write, src: &[u8], shape: &[i64]) -> Result<u64, Error> {
    match shape.len() {
        2 => {
            let out_dim = shape[0] as usize;
            let in_dim = shape[1] as usize;
            let need = crate::dgq::layout::q4_matrix_bytes(out_dim, in_dim);
            let mut buf = vec![0u8; need];
            quantize_bf16_matrix_q4(src, out_dim, in_dim, &mut buf);
            out.write_all(&buf)?;
            Ok(need as u64)
        }
        3 => {
            let experts = shape[0] as usize;
            let out_dim = shape[1] as usize;
            let in_dim = shape[2] as usize;
            let need = experts * crate::dgq::layout::q4_matrix_bytes(out_dim, in_dim);
            let mut buf = vec![0u8; need];
            quantize_expert_stack_q4(src, experts, out_dim, in_dim, &mut buf)?;
            out.write_all(&buf)?;
            Ok(need as u64)
        }
        _ => Err(Error::Format("q4 unsupported rank")),
    }
}

fn write_nvfp4_tensor(out: &mut impl Write, src: &[u8], shape: &[i64]) -> Result<u64, Error> {
    match shape.len() {
        2 => {
            let out_dim = shape[0] as usize;
            let in_dim = shape[1] as usize;
            let need = crate::dgq::layout::nvfp4_matrix_bytes(out_dim, in_dim);
            let mut buf = vec![0u8; need];
            quantize_bf16_matrix_nvfp4(src, out_dim, in_dim, &mut buf);
            out.write_all(&buf)?;
            Ok(need as u64)
        }
        3 => {
            let experts = shape[0] as usize;
            let out_dim = shape[1] as usize;
            let in_dim = shape[2] as usize;
            let need = experts * crate::dgq::layout::nvfp4_matrix_bytes(out_dim, in_dim);
            let mut buf = vec![0u8; need];
            quantize_expert_stack_nvfp4(src, experts, out_dim, in_dim, &mut buf)?;
            out.write_all(&buf)?;
            Ok(need as u64)
        }
        _ => Err(Error::Format("nvfp4 unsupported rank")),
    }
}

fn write_q8_tensor(out: &mut impl Write, src: &[u8], shape: &[i64]) -> Result<u64, Error> {
    if shape.len() != 2 {
        return Err(Error::Format("q8 expects rank 2"));
    }
    let out_dim = shape[0] as usize;
    let in_dim = shape[1] as usize;
    let need = crate::dgq::layout::q8_matrix_bytes(out_dim, in_dim);
    let mut buf = vec![0u8; need];
    quantize_bf16_matrix_q8(src, out_dim, in_dim, &mut buf);
    out.write_all(&buf)?;
    Ok(need as u64)
}

pub(crate) fn copy_sidecar_files(source: &Path, dest: &Path) -> Result<(), Error> {
    for name in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ] {
        let src = source.join(name);
        if src.is_file() {
            fs::copy(&src, dest.join(name))?;
        }
    }
    Ok(())
}
