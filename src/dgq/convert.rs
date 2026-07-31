//! Safetensors → `.dgq` offline quantizer (always from raw bf16 weights).

use crate::Error;
use crate::dgq::block::{
    quantize_bf16_matrix_q4, quantize_bf16_matrix_q6, quantize_bf16_matrix_q8,
    quantize_expert_stack_q4, quantize_expert_stack_q6,
};
use crate::dgq::layout::{
    BLOB_FILE, BaseModelRef, DGQ_VERSION_LAYERED, DGQ_VERSION_NVFP4, DgqManifest, DgqTensorEntry,
    DgqTensorMeta, ExternalFile, ExternalRole, MANIFEST_FILE, QuantKind, QuantProfile,
    DGQ_VERSION_CUSTOM, TensorClass, TensorSource, align_offset, classify_tensor_custom,
    dgq_version_for_profile,
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

    // Experts LAST: lets the loader wrap [0, split) + [split, end) as two
    // no-copy MTLBuffers when the blob exceeds the device's max single-buffer
    // length (M3 Pro/36GB: 20.25 GiB; q6 blob: ~24 GiB).
    let is_expert = |n: &str| -> bool {
        store
            .get(n)
            .map(|(_, info)| n.contains(".experts.") && info.shape.len() == 3)
            .unwrap_or(false)
    };
    let names: Vec<String> = if opts.overlay_base.is_some() {
        // Overlay mode: lay the head out for VA-splice (ARCHITECTURE.md
        // §8.1) — every externalizable Raw tensor grouped by source shard,
        // in that shard's own byte order, so the loader's `plan_head_splice`
        // can recognize each shard's raw tensors as ONE contiguous run.
        let classified: Vec<(String, QuantKind)> = store
            .weight_map
            .keys()
            .map(|n| {
                let (_, info) = store.get(n).expect("weight_map key exists");
                (
                    n.clone(),
                    classify_tensor_custom(n, &info.shape, opts.profile, &opts.custom_overrides),
                )
            })
            .collect();
        build_overlay_tensor_order(&store, &classified, &is_expert)
    } else {
        let mut ns: Vec<String> = store.weight_map.keys().cloned().collect();
        ns.sort();
        ns.sort_by_key(|n| (is_expert(n), n.clone()));
        ns
    };
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
    // Overlay mode only: `(shard, next expected file byte offset)` while
    // we're inside a head run, so consecutive tensors advance the canonical
    // cursor by EXACTLY their byte_len (zero gap, mirroring the shard's own
    // layout) — but ONLY when they are ACTUALLY file-contiguous. Same-shard
    // alone is NOT enough: two Raw tensors from the same HF shard can sit
    // gigabytes apart in the file (e.g. `embed_tokens` then, gigabytes of
    // interleaved expert tensors later, `layers.0.input_layernorm`) — an
    // earlier version of this code merged them into one canonical zero-gap
    // "run" on shard-name alone, which then hit `plan_head_splice`'s own
    // (correct, file-contiguity-checked) grouping as two SEPARATE runs with
    // NO page separation between them — the writer had skipped the run-
    // boundary page bump because it never saw a boundary. Caught live
    // running the real model regen through `golden`, not by the (same-
    // shard, small, actually-contiguous) synthetic test fixture.
    let mut head_run: Option<(String, u64)> = None;
    for (i, name) in names.iter().enumerate() {
        let (shard, info) = store
            .get(name)
            .ok_or_else(|| Error::NotFound(name.clone()))?;
        let src = shard.data(info);
        let kind = classify_tensor_custom(name, &info.shape, opts.profile, &opts.custom_overrides);
        let class_bucket = tensor_class(name, &info.shape)
            .map(TensorClass::as_str)
            .unwrap_or(LOCKED_BUCKET);

        let is_head_external =
            opts.overlay_base.is_some() && kind == QuantKind::Raw && !is_expert(name);
        let file_start = is_head_external.then(|| shard.absolute_data_offset(info));
        let continues = match (&head_run, file_start) {
            (Some((shard_name, next_file_off)), Some(fs)) => {
                *shard_name == store.weight_map[name] && *next_file_off == fs
            }
            _ => false,
        };

        if continues {
            // Nominally "continuing" the shard-run in FILE terms — but every
            // tensor's OWN canonical start must STILL be 64-byte aligned
            // regardless: `gemm_rowk`-class kernels reinterpret
            // `blob + w_off` as `device const ushort *` and index it, which
            // is only well-defined/correct for a sufficiently aligned
            // `w_off` (confirmed empirically: the working pack has all 1047
            // canonical offsets 64-byte aligned via the old unconditional
            // `align_offset`; an earlier version of this writer skipped
            // alignment entirely inside a run and produced a pack whose
            // GPU-read weights were silently wrong from layer 1 onward
            // despite byte-identical tensor CONTENT — golden caught it,
            // `gpu_buffer_matches_store_for_every_tensor` did not, because
            // that diagnostic reads via `host_ptr`, which has no alignment
            // requirement). `align_offset` is a NO-OP whenever the previous
            // tensor's byte length was already a 64-byte multiple — true for
            // essentially every real weight matrix here (hidden_size=2816 ⇒
            // row strides are multiples of 64) — so in the common case this
            // does not disturb the zero-gap mirroring the splice plan needs;
            // on a rare non-64-multiple tensor (e.g. `layer_scalar`, 2
            // bytes) it inserts padding, which correctly ENDS the run there
            // (the loader's independent canonical-contiguity check agrees)
            // rather than producing a misaligned-but-nominally-contiguous
            // run.
            offset = align_offset(offset);
        } else {
            // A run boundary on EITHER side (closing a shard-run, opening
            // one, or both back-to-back) needs a FULL page of separation,
            // not just congruence: `plan_head_splice`'s MAP_FIXED range for
            // a run is rounded to page boundaries, so a run's own trailing
            // (or leading) fringe can otherwise land on a neighboring
            // tensor's true bytes — caught by
            // `writer_plan_integration_tests`. `round_up_page` also implies
            // 64-byte alignment (16384 is a multiple of 64), so this keeps
            // the same GPU-correctness invariant as the `continues` branch.
            // An ordinary local-to-local transition (neither side a run)
            // stays on the cheap 64-byte `align_offset`.
            if head_run.is_some() || is_head_external {
                offset = round_up_page(offset, EXPERT_SPLIT_ALIGN);
            } else {
                offset = align_offset(offset);
            }
            if let Some(fs) = file_start
                && fs.is_multiple_of(64)
            {
                // Starting a NEW shard-run WHOSE FILE OFFSET IS ITSELF
                // 64-byte aligned: nudge forward (< 16 KiB more) to also
                // make this run's canonical start page-congruent with its
                // file offset. The nudge amount is guaranteed to stay a
                // multiple of 64 too, because both `offset` (just rounded
                // to a 16384-byte, hence 64-byte, boundary) and `fs` are
                // multiples of 64: `congruent_pad` only ever adds a
                // difference/sum of two such multiples. When `fs` is NOT
                // 64-aligned, congruence and 64-byte alignment are
                // mutually exclusive for this tensor (rounding both to the
                // same page necessarily preserves their mod-64 remainder,
                // which is what `gemm_rowk`-class kernels depend on) — so
                // this tensor simply stays aligned-but-not-congruent,
                // which `plan_head_splice` correctly reads as "not
                // spliceable" and anon-fills instead. Correctness first;
                // splice coverage follows from real tensor byte lengths
                // already being 64-byte multiples for this model.
                offset = congruent_pad(offset, fs, EXPERT_SPLIT_ALIGN);
            }
        }
        head_run = file_start.map(|fs| (store.weight_map[name].clone(), fs + src.len() as u64));

        if expert_split.is_none() && is_expert(name) {
            offset = (offset + EXPERT_SPLIT_ALIGN - 1) & !(EXPERT_SPLIT_ALIGN - 1);
            expert_split = Some(offset);
            // Bump the LOCAL blob cursor to the same 16 KiB boundary
            // UNCONDITIONALLY — not just in overlay mode. A self-contained
            // pack's `source` is always `None`, so every reader (including
            // `wrap_mmap`'s expert-region split) treats `meta.offset`
            // (canonical) as the tensor's ACTUAL position in `blob_file`.
            // Bumping only the canonical cursor here (the old behavior) let
            // it diverge from `local_bytes_written` by up to 16383 bytes
            // from the first expert tensor onward, so every expert (and
            // everything after) would be written at one file position while
            // the manifest claimed a different one — silently wrong weights
            // for any self-contained multi-expert pack. No test exercised
            // `overlay_base: None` before the round-trip test that caught
            // this (`dgq::overlay::monolithic_roundtrip_tests`).
            let aligned = (local_bytes_written + EXPERT_SPLIT_ALIGN - 1) & !(EXPERT_SPLIT_ALIGN - 1);
            let pad = (aligned - local_bytes_written) as usize;
            blob.write_all(&vec![0u8; pad])?;
            local_bytes_written = aligned;
            if opts.overlay_base.is_some() {
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
    // Custom class overrides outrank everything: pre-knob binaries derive
    // kernel formats from `profile` and would silently mis-dispatch any class
    // that diverges from it (layered-era binaries accept v3, nvfp4-era v2) —
    // v4 makes every one of them refuse loudly instead.
    let version = if !custom_classes.is_empty() {
        DGQ_VERSION_CUSTOM
    } else if opts.overlay_base.is_some() {
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

/// Overlay-mode tensor visiting order: every externalizable Raw tensor
/// first, grouped by source HF shard and sorted WITHIN each shard by the
/// tensor's own byte offset there (mirrors the shard's internal layout
/// exactly — no gaps this loop doesn't already know about) — the property
/// `src/metal/dgq_gpu.rs::plan_head_splice` needs to recognize a shard's raw
/// tensors as one contiguous, page-congruent run. Shards are visited in
/// filename order (deterministic, otherwise arbitrary — repack --overlay
/// mirrors this same order independently). Every other non-expert tensor
/// (Local: quantized, or a custom-class override that isn't Raw) follows,
/// name-sorted; experts are always last (unchanged tail-split invariant).
fn build_overlay_tensor_order(
    store: &SafetensorStore,
    classified: &[(String, QuantKind)],
    is_expert: &impl Fn(&str) -> bool,
) -> Vec<String> {
    let mut shard_groups: BTreeMap<String, Vec<(u64, String)>> = BTreeMap::new();
    let mut local_head: Vec<String> = Vec::new();
    let mut experts: Vec<String> = Vec::new();

    for (name, kind) in classified {
        if is_expert(name) {
            experts.push(name.clone());
            continue;
        }
        if *kind == QuantKind::Raw {
            let (shard, info) = store.get(name).expect("classified name is in weight_map");
            let shard_name = store.weight_map[name].clone();
            shard_groups
                .entry(shard_name)
                .or_default()
                .push((shard.absolute_data_offset(info), name.clone()));
        } else {
            local_head.push(name.clone());
        }
    }

    let mut order = Vec::with_capacity(classified.len());
    for (_shard, mut tensors) in shard_groups {
        tensors.sort_by_key(|(file_off, _)| *file_off);
        order.extend(tensors.into_iter().map(|(_, name)| name));
    }
    local_head.sort();
    order.extend(local_head);
    experts.sort();
    order.extend(experts);
    order
}

/// Smallest value `>= cursor` that is congruent with `file_off` mod `page`
/// (`0 <= result - cursor < page`) — the padding a new head-splice run needs
/// so its canonical start lines up with its own source-file start once both
/// are rounded down to a page boundary (see `plan_head_splice`'s doc
/// comment for why that's the exact property splicing needs).
pub(crate) fn congruent_pad(cursor: u64, file_off: u64, page: u64) -> u64 {
    let target = file_off % page;
    let cur = cursor % page;
    let add = if target >= cur {
        target - cur
    } else {
        page - cur + target
    };
    cursor + add
}

/// Smallest page-aligned value `>= cursor`.
pub(crate) fn round_up_page(cursor: u64, page: u64) -> u64 {
    cursor.div_ceil(page) * page
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
