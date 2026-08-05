//! `download` subcommand: fetch the quantized `.dgq` model from HuggingFace into
//! a local model directory ready for `ask`/`chat`/`serve` (`-m <dir>`).
//!
//! Transfers go through the `shell-download` crate, which drives whatever
//! download tool the host already has (curl, wget, PowerShell, python3, or a
//! built-in TLS tunnel) behind a small Rust API, so no HTTP/TLS crate enters
//! the build.
//!
//! HF `resolve/` URLs advertise `Accept-Ranges: bytes`, so the large blob is
//! fetched as parallel byte-range chunks. Chunking buys parallel throughput
//! and resume: each chunk is a separate `shell-download` target, so a re-run
//! skips any chunk already complete on disk. (HF also exposes a Xet
//! content-defined-chunk protocol via the response `Link:` header for
//! dedup-aware fetches; that needs a full Xet client, and plain HTTP range
//! covers our needs.)
//!
//! The HF hub cache is honored: a file already present under
//! `~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<sha>/` (e.g. from
//! a prior `huggingface-cli download`) is symlinked in instead of re-fetched.

use super::*;
use shell_download::{Quiet, RequestBuilder, RequestHandle};
use std::path::{Path, PathBuf};

const HF_ENDPOINT: &str = "https://huggingface.co";
/// Byte-range chunk size for the large blob. 256 MiB keeps the part count and
/// per-chunk retry cost both modest (a 20 GiB blob is ~76 chunks).
const CHUNK_SIZE: u64 = 256 * 1024 * 1024;

/// One entry from the HF repo tree listing.
struct RepoFile {
    path: String,
    size: u64,
}

pub(crate) fn run_download(
    repo: &str,
    revision: &str,
    dest: &Path,
    force: bool,
    jobs: usize,
) -> ExitCode {
    let jobs = jobs.max(1);

    let files = match list_repo_files(repo, revision) {
        Ok(f) => f,
        Err(msg) => {
            eprintln!("error: listing {repo}@{revision}: {msg}");
            return ExitCode::FAILURE;
        }
    };
    if files.is_empty() {
        eprintln!("error: {repo}@{revision} lists no downloadable files");
        return ExitCode::FAILURE;
    }

    if let Err(e) = std::fs::create_dir_all(dest) {
        eprintln!("error: creating {}: {e}", dest.display());
        return ExitCode::FAILURE;
    }

    let total: u64 = files.iter().map(|f| f.size).sum();
    eprintln!(
        "download: {repo}@{revision} -> {} ({} files, {:.2} GiB, {jobs} job(s))",
        dest.display(),
        files.len(),
        total as f64 / GIB,
    );

    for f in &files {
        let dest_path = dest.join(&f.path);
        if let Some(parent) = dest_path.parent()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            eprintln!("error: creating {}: {e}", parent.display());
            return ExitCode::FAILURE;
        }

        // Present and unchanged: leave it be unless forced.
        if !force && file_len(&dest_path) == Some(f.size) {
            eprintln!("  ok   {} (present, {} bytes)", f.path, f.size);
            continue;
        }

        // Reuse the HF hub cache if the file is already sitting there.
        if !force
            && let Some(cached) = hf_cache_file(repo, revision, &f.path)
            && file_len(&cached) == Some(f.size)
        {
            match link_or_copy(&cached, &dest_path) {
                Ok(how) => {
                    eprintln!("  {how} {} (from HF cache)", f.path);
                    continue;
                }
                Err(e) => {
                    eprintln!("  warn {}: cache reuse failed ({e}), downloading", f.path);
                }
            }
        }

        eprintln!(
            "  get  {} ({:.2} GiB)",
            f.path,
            f.size as f64 / GIB.max(1.0)
        );
        let url = format!("{HF_ENDPOINT}/{repo}/resolve/{revision}/{}", f.path);
        if let Err(msg) = download_file(&url, &dest_path, f.size, force, jobs) {
            eprintln!("error: downloading {}: {msg}", f.path);
            return ExitCode::FAILURE;
        }
        match file_len(&dest_path) {
            Some(got) if got == f.size => {}
            Some(got) => {
                eprintln!(
                    "error: {} size mismatch: got {got}, expected {}",
                    f.path, f.size
                );
                return ExitCode::FAILURE;
            }
            None => {
                eprintln!("error: {} missing after download", f.path);
                return ExitCode::FAILURE;
            }
        }
    }

    match verify_downloaded_pack(dest) {
        Ok(summary) => {
            eprintln!("  pack: {summary}");
        }
        Err(msg) => {
            eprintln!("error: pack verification failed: {msg}");
            return ExitCode::FAILURE;
        }
    }

    eprintln!("download ok: {}", dest.display());
    eprintln!("  run: diffgemma -m {} chat", dest.display());
    ExitCode::SUCCESS
}

/// Sanity-check a freshly downloaded `.dgq` pack and print a one-line
/// summary: parses the manifest, confirms this binary's `dgq_version_supported`
/// accepts it, confirms the local blob file is at least as large as the
/// manifest's own maximum LOCAL tensor extent (a truncated/corrupt transfer
/// is the most likely download failure mode and would show up here as a
/// short file), and — for a layered pack — reports whether the pinned HF
/// base is already resolvable locally or prints the exact `hf download`
/// remedy. Returns `Err` only for problems that mean the pack itself is
/// broken (bad manifest, unsupported version, short blob); a missing HF base
/// for a layered pack is reported, not failed, since fetching it is a
/// separate, expected step.
fn verify_downloaded_pack(dest: &Path) -> Result<String, String> {
    use crate::dgq::layout::{DgqManifest, MANIFEST_FILE, TensorSource, dgq_version_supported};

    let manifest_path = dest.join(MANIFEST_FILE);
    let manifest_json = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("reading {}: {e}", manifest_path.display()))?;
    let manifest: DgqManifest = serde_json::from_str(&manifest_json)
        .map_err(|e| format!("parsing {}: {e}", manifest_path.display()))?;

    if !dgq_version_supported(manifest.version) {
        return Err(format!(
            "manifest version {} is not supported by this binary (dgq_version_supported \
             rejects it) — you likely need a newer diffgemma build",
            manifest.version
        ));
    }

    let mut canonical_bytes = 0u64;
    let mut max_local_extent = 0u64;
    for t in &manifest.tensors {
        canonical_bytes = canonical_bytes.max(t.meta.offset + t.meta.byte_len);
        let local_end = match &t.meta.source {
            None => t.meta.offset + t.meta.byte_len,
            Some(TensorSource::Local { local_offset }) => local_offset + t.meta.byte_len,
            Some(TensorSource::External { .. }) => 0,
        };
        max_local_extent = max_local_extent.max(local_end);
    }

    let blob_path = dest.join(&manifest.blob_file);
    let blob_len = std::fs::metadata(&blob_path)
        .map_err(|e| format!("stat {}: {e}", blob_path.display()))?
        .len();
    if blob_len < max_local_extent {
        return Err(format!(
            "{} is {blob_len} bytes, but the manifest's local tensors extend to \
             {max_local_extent} bytes — the transfer looks truncated or corrupt; \
             re-run with --force",
            blob_path.display()
        ));
    }

    let layered = manifest.is_layered();
    let mut base_line = String::new();
    if layered {
        base_line = match &manifest.base_model {
            Some(base) => match crate::dgq::hf_resolve::resolve_snapshot_dir(base) {
                Ok(dir) => format!(
                    "\n  base model {}@{} resolved: {}",
                    base.repo,
                    base.revision,
                    dir.display()
                ),
                Err(err) => format!("\n{err}"),
            },
            None => "\n  warning: layered pack has no base_model pin in its manifest".to_string(),
        };
    }

    let custom_classes = if manifest.custom_classes.is_empty() {
        "none".to_string()
    } else {
        let mut pairs: Vec<String> = manifest
            .custom_classes
            .iter()
            .map(|(c, k)| format!("{c}={k}"))
            .collect();
        pairs.sort();
        pairs.join(",")
    };

    Ok(format!(
        "profile={:?} custom_classes={custom_classes} {} canonical={:.2} GiB local_blob={:.2} GiB{base_line}",
        manifest.profile,
        if layered { "layered" } else { "monolithic" },
        canonical_bytes as f64 / GIB,
        blob_len as f64 / GIB,
    ))
}

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn file_len(path: &Path) -> Option<u64> {
    std::fs::metadata(path).ok().map(|m| m.len())
}

/// Fetch the repo file tree from the HF API and return the regular files worth
/// downloading (skips `.gitattributes`). Sizes come back resolved for LFS blobs.
fn list_repo_files(repo: &str, revision: &str) -> Result<Vec<RepoFile>, String> {
    let url = format!("{HF_ENDPOINT}/api/models/{repo}/tree/{revision}?recursive=true");
    let bytes = RequestBuilder::new(url)
        .follow_redirects(true)
        .quiet(Quiet::OnSuccess)
        .fetch_bytes()
        .map_err(|e| format!("{e:?}"))?;
    let entries: Vec<serde_json::Value> =
        serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;

    let mut files = Vec::new();
    for e in entries {
        if e.get("type").and_then(|t| t.as_str()) != Some("file") {
            continue;
        }
        let Some(path) = e.get("path").and_then(|p| p.as_str()) else {
            continue;
        };
        if path == ".gitattributes" {
            continue;
        }
        // For LFS files the real byte count is under `lfs.size`; the top-level
        // `size` is the pointer stub. Prefer the LFS size when present.
        let plain = e.get("size").and_then(|s| s.as_u64());
        let lfs = e
            .get("lfs")
            .and_then(|l| l.get("size"))
            .and_then(|s| s.as_u64());
        let size = lfs.or(plain).unwrap_or(0);
        files.push(RepoFile {
            path: path.to_string(),
            size,
        });
    }
    Ok(files)
}

/// Download one file to `dest`. Files that fit in a single chunk are fetched
/// straight to `dest`. Larger files split into byte-range chunks fetched `jobs`
/// at a time to `.partNNN` siblings, then concatenated; a re-run resumes by
/// skipping any part already fully on disk. A truncated file/part fails its
/// size check and is refetched.
fn download_file(
    url: &str,
    dest: &Path,
    size: u64,
    force: bool,
    jobs: usize,
) -> Result<(), String> {
    if force {
        let _ = std::fs::remove_file(dest);
    }

    let n_chunks = size.div_ceil(CHUNK_SIZE).max(1);
    if n_chunks == 1 {
        let resp = start_fetch(url, dest, None)?
            .join()
            .map_err(|e| format!("{e:?}"))?;
        return match resp.status_code {
            200 => Ok(()),
            code => Err(format!("HTTP {code}")),
        };
    }

    // Chunk plan: [start, end] inclusive byte ranges.
    let mut part_paths = Vec::with_capacity(n_chunks as usize);
    let mut pending = Vec::new(); // (index, part_path, expected_len)
    for i in 0..n_chunks {
        let start = i * CHUNK_SIZE;
        let end = ((i + 1) * CHUNK_SIZE).min(size) - 1;
        let expected = end - start + 1;
        let part = part_path(dest, i);
        if !force && file_len(&part) == Some(expected) {
            // Already fetched on a prior run: resume past it.
        } else {
            let _ = std::fs::remove_file(&part);
            pending.push((i, part.clone(), start, end, expected));
        }
        part_paths.push((part, expected));
    }

    let done = n_chunks as usize - pending.len();
    if done > 0 {
        eprintln!("       resuming: {done}/{n_chunks} chunks already present");
    }

    // Fetch pending chunks `jobs` at a time.
    let mut completed = done;
    for batch in pending.chunks(jobs) {
        let mut running = Vec::new();
        for (idx, part, start, end, expected) in batch {
            let handle = start_fetch(url, part, Some((*start, *end)));
            running.push((*idx, part.clone(), *expected, handle));
        }
        for (idx, part, expected, handle) in running {
            let handle = handle?;
            let resp = handle.join().map_err(|e| format!("chunk {idx}: {e:?}"))?;
            // 206 = partial content (range honored); 200 means the server sent
            // the whole file for a ranged request, which breaks the chunk plan.
            if resp.status_code != 206 {
                return Err(format!(
                    "chunk {idx}: server returned HTTP {} for a range request (expected 206)",
                    resp.status_code
                ));
            }
            match file_len(&part) {
                Some(got) if got == expected => {}
                Some(got) => {
                    return Err(format!("chunk {idx}: got {got} bytes, expected {expected}"));
                }
                None => return Err(format!("chunk {idx}: part missing after fetch")),
            }
            completed += 1;
            eprintln!("       chunk {completed}/{n_chunks} ok");
        }
    }

    // Stitch parts into the final file in order, then drop the parts.
    concat_parts(dest, &part_paths)
}

/// Start a background fetch of `url` to `dest`, optionally for one inclusive
/// byte range. `quiet(OnSuccess)` keeps parallel chunk fetches from interleaving
/// progress bars while still surfacing a failed child's output.
fn start_fetch(url: &str, dest: &Path, range: Option<(u64, u64)>) -> Result<RequestHandle, String> {
    let mut req = RequestBuilder::new(url)
        .follow_redirects(true)
        .quiet(Quiet::OnSuccess);
    if let Some((start, end)) = range {
        req = req.header("Range", format!("bytes={start}-{end}"));
    }
    req.start(dest).map_err(|e| format!("{e:?}"))
}

fn part_path(dest: &Path, i: u64) -> PathBuf {
    let mut name = dest.as_os_str().to_os_string();
    name.push(format!(".part{i:04}"));
    PathBuf::from(name)
}

/// Concatenate `parts` (in order) into `dest`, deleting each part as it is
/// consumed. Parts are 256 MiB; a plain buffered copy is fine.
fn concat_parts(dest: &Path, parts: &[(PathBuf, u64)]) -> Result<(), String> {
    use std::io::{BufWriter, Write};
    eprintln!(
        "       assembling {} chunks -> {}",
        parts.len(),
        dest.display()
    );
    let out = std::fs::File::create(dest).map_err(|e| format!("create {}: {e}", dest.display()))?;
    let mut writer = BufWriter::with_capacity(8 << 20, out);
    for (part, _) in parts {
        let mut r =
            std::fs::File::open(part).map_err(|e| format!("open {}: {e}", part.display()))?;
        std::io::copy(&mut r, &mut writer).map_err(|e| format!("copy {}: {e}", part.display()))?;
    }
    writer.flush().map_err(|e| e.to_string())?;
    drop(writer);
    for (part, _) in parts {
        let _ = std::fs::remove_file(part);
    }
    Ok(())
}

/// Resolve a file inside the local HF hub snapshot for `repo`@`revision`, if the
/// user has already pulled it there. Returns the concrete path only when it
/// exists on disk.
fn hf_cache_file(repo: &str, revision: &str, rel: &str) -> Option<PathBuf> {
    let cache_root = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface")))?
        .join("hub");
    let repo_dir = cache_root.join(format!("models--{}", repo.replace('/', "--")));

    // A revision may be a branch/tag (resolve via refs/) or a bare commit sha.
    let commit = std::fs::read_to_string(repo_dir.join("refs").join(revision))
        .ok()
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| revision.to_string());

    let candidate = repo_dir.join("snapshots").join(commit).join(rel);
    candidate.exists().then_some(candidate)
}

/// Prefer a symlink (no second 20 GiB copy); fall back to a hard copy if the
/// filesystem refuses. Returns a 4-char verb for the progress line.
fn link_or_copy(src: &Path, dst: &Path) -> std::io::Result<&'static str> {
    if dst.exists() {
        std::fs::remove_file(dst)?;
    }
    match std::os::unix::fs::symlink(src, dst) {
        Ok(()) => Ok("link"),
        Err(_) => {
            std::fs::copy(src, dst)?;
            Ok("copy")
        }
    }
}

#[cfg(test)]
mod verify_pack_tests {
    use super::*;
    use dgq::layout::{
        BLOB_FILE, BaseModelRef, DGQ_VERSION_AFFINE, DGQ_VERSION_LAYERED, DgqManifest,
        DgqTensorEntry, DgqTensorMeta, MANIFEST_FILE, QuantProfile, TensorSource,
    };

    fn scratch_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("dgq-download-verify-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    fn write_manifest(dir: &Path, manifest: &DgqManifest) {
        std::fs::write(
            dir.join(MANIFEST_FILE),
            serde_json::to_string_pretty(manifest).unwrap(),
        )
        .expect("write manifest");
    }

    fn base_entry(offset: u64, byte_len: u64, source: Option<TensorSource>) -> DgqTensorEntry {
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

    #[test]
    fn self_contained_pack_verifies_ok() {
        let dir = scratch_dir("selfcontained");
        let manifest = DgqManifest {
            version: DGQ_VERSION_AFFINE,
            profile: QuantProfile::Q4,
            source_model: "src".to_string(),
            blob_file: BLOB_FILE.to_string(),
            expert_split: None,
            local_expert_split: None,
            base_model: None,
            external_files: Default::default(),
            custom_classes: Default::default(),
            tensors: vec![base_entry(0, 100, None)],
        };
        write_manifest(&dir, &manifest);
        std::fs::write(dir.join(BLOB_FILE), vec![0u8; 100]).expect("write blob");

        let summary = verify_downloaded_pack(&dir).expect("verifies ok");
        assert!(summary.contains("monolithic"), "{summary}");
        assert!(summary.contains("custom_classes=none"), "{summary}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unsupported_version_is_rejected() {
        let dir = scratch_dir("badversion");
        let manifest = DgqManifest {
            version: 999,
            profile: QuantProfile::Q4,
            source_model: "src".to_string(),
            blob_file: BLOB_FILE.to_string(),
            expert_split: None,
            local_expert_split: None,
            base_model: None,
            external_files: Default::default(),
            custom_classes: Default::default(),
            tensors: vec![base_entry(0, 8, None)],
        };
        write_manifest(&dir, &manifest);
        std::fs::write(dir.join(BLOB_FILE), vec![0u8; 8]).expect("write blob");

        let err = verify_downloaded_pack(&dir).expect_err("must reject unsupported version");
        assert!(err.contains("version 999"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn truncated_blob_is_rejected() {
        let dir = scratch_dir("truncated");
        let manifest = DgqManifest {
            version: DGQ_VERSION_AFFINE,
            profile: QuantProfile::Q4,
            source_model: "src".to_string(),
            blob_file: BLOB_FILE.to_string(),
            expert_split: None,
            local_expert_split: None,
            base_model: None,
            external_files: Default::default(),
            custom_classes: Default::default(),
            tensors: vec![base_entry(0, 100, None)],
        };
        write_manifest(&dir, &manifest);
        // Blob is short: only 50 of the claimed 100 bytes made it to disk.
        std::fs::write(dir.join(BLOB_FILE), vec![0u8; 50]).expect("write blob");

        let err = verify_downloaded_pack(&dir).expect_err("must reject truncated blob");
        assert!(err.contains("truncated or corrupt"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn layered_pack_missing_base_reports_remedy_without_failing() {
        let dir = scratch_dir("layered");
        let hf_home = std::env::temp_dir().join(format!(
            "dgq-download-verify-layered-hfhome-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&hf_home);
        std::fs::create_dir_all(&hf_home).expect("mkdir hf home");
        let cfg = crate::flags::RuntimeConfig::from_pairs(&[(
            "DGQ_HF_HOME".to_string(),
            hf_home.display().to_string(),
        )])
        .0;
        let _guard = crate::flags::install_for_test(cfg);

        let manifest = DgqManifest {
            version: DGQ_VERSION_LAYERED,
            profile: QuantProfile::Q4,
            source_model: "src".to_string(),
            blob_file: BLOB_FILE.to_string(),
            expert_split: Some(0),
            local_expert_split: Some(0),
            base_model: Some(BaseModelRef {
                repo: "acme/widgets".to_string(),
                revision: "deadbeef".to_string(),
            }),
            external_files: Default::default(),
            custom_classes: Default::default(),
            tensors: vec![base_entry(
                0,
                40,
                Some(TensorSource::Local { local_offset: 0 }),
            )],
        };
        write_manifest(&dir, &manifest);
        std::fs::write(dir.join(BLOB_FILE), vec![0u8; 40]).expect("write blob");

        let summary = verify_downloaded_pack(&dir).expect("layered pack still verifies");
        assert!(summary.contains("layered"), "{summary}");
        assert!(summary.contains("hf download acme/widgets"), "{summary}");

        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&hf_home);
    }
}
