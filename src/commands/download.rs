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
use crate::dgq::layout::INCOMPLETE_SENTINEL;
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

    for (i, f) in files.iter().enumerate() {
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
            report_unfetched(&files, i);
            return ExitCode::FAILURE;
        }
        match file_len(&dest_path) {
            Some(got) if got == f.size => {}
            Some(got) => {
                eprintln!(
                    "error: {} size mismatch: got {got}, expected {}",
                    f.path, f.size
                );
                report_unfetched(&files, i);
                return ExitCode::FAILURE;
            }
            None => {
                eprintln!("error: {} missing after download", f.path);
                report_unfetched(&files, i);
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
    use crate::dgq::layout::{DgqManifest, MANIFEST_FILE, dgq_version_supported};

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

    let canonical_bytes = manifest
        .tensors
        .iter()
        .map(|t| t.meta.offset + t.meta.byte_len)
        .max()
        .unwrap_or(0);

    let blob_path = dest.join(&manifest.blob_file);
    let blob_len = std::fs::metadata(&blob_path)
        .map_err(|e| format!("stat {}: {e}", blob_path.display()))?
        .len();
    // Same check the loader runs, so a pack that passes here cannot fail there.
    manifest
        .check_local_blob_len(blob_len, &blob_path)
        .map_err(|e| e.to_string())?;

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

/// Spell out what the aborted run never got to. The blob is listed before
/// the manifest and the tokenizer, so a blob failure leaves a directory
/// holding the one huge file and missing the small ones, which reads as "it
/// skipped them" unless the abort says otherwise.
fn report_unfetched(files: &[RepoFile], failed_at: usize) {
    let rest: Vec<&str> = files[failed_at + 1..]
        .iter()
        .map(|f| f.path.as_str())
        .collect();
    eprintln!("download aborted: {} is incomplete", files[failed_at].path);
    if !rest.is_empty() {
        eprintln!("  not fetched: {}", rest.join(", "));
    }
    eprintln!("  the model directory is not usable yet. Re-run `diffgemma download` to resume.");
}

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
        let _ = std::fs::remove_file(tmp_path(dest));
    }

    let n_chunks = size.div_ceil(CHUNK_SIZE).max(1);
    if n_chunks == 1 {
        // Fetch beside the destination and rename, so a killed transfer
        // leaves no half file under the name the loader looks for.
        let tmp = tmp_path(dest);
        let resp = start_fetch(url, &tmp, None)?
            .join()
            .map_err(|e| format!("{e:?}"))?;
        if resp.status_code != 200 {
            let _ = std::fs::remove_file(&tmp);
            return Err(format!("HTTP {}", resp.status_code));
        }
        return std::fs::rename(&tmp, dest)
            .map_err(|e| format!("rename {} -> {}: {e}", tmp.display(), dest.display()));
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

/// Assemble `parts` into `dest`, in order, so that `dest` never exists in a
/// state a loader would accept.
///
/// Three things make that true. The bytes land in a `.tmp` sibling and only
/// become `dest` on a successful rename, so an interrupted run leaves a file
/// under a name nothing loads. The first chunk is written LAST, after every
/// other byte is on disk and the assembled length checks out, and until then
/// offset 0 holds `INCOMPLETE_SENTINEL`: a blob carrying it is unfinished
/// whatever its length says, which is the one thing a length check cannot
/// see. And each part is deleted the moment its bytes are flushed, because
/// holding all of them to the end puts the parts and the assembled file on
/// disk at once and a 19 GiB pack would need 38 GiB free to land.
///
/// The cost of freeing as we go: a failure part-way through re-fetches the
/// parts it already consumed.
fn concat_parts(dest: &Path, parts: &[(PathBuf, u64)]) -> Result<(), String> {
    let tmp = tmp_path(dest);
    match concat_parts_inner(dest, &tmp, parts) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

fn concat_parts_inner(dest: &Path, tmp: &Path, parts: &[(PathBuf, u64)]) -> Result<(), String> {
    use std::io::{Seek, SeekFrom, Write};
    eprintln!(
        "       assembling {} chunks -> {}",
        parts.len(),
        dest.display()
    );
    let Some(((_, head_len), tail_parts)) = parts.split_first() else {
        return Err("no chunks to assemble".to_string());
    };

    let mut out =
        std::fs::File::create(tmp).map_err(|e| format!("create {}: {e}", tmp.display()))?;
    // Only when the head is long enough to hold it: a shorter one would leave
    // the sentinel's tail sticking out past the bytes the head will overwrite.
    if *head_len >= INCOMPLETE_SENTINEL.len() as u64 {
        out.write_all(INCOMPLETE_SENTINEL)
            .map_err(|e| format!("write sentinel to {}: {e}", tmp.display()))?;
    }

    // The head's bytes arrive last, so skip its span and start with chunk 1.
    out.seek(SeekFrom::Start(*head_len))
        .map_err(|e| format!("seek {}: {e}", tmp.display()))?;
    copy_parts_into(&mut out, tail_parts, tmp)?;

    let want: u64 = parts.iter().map(|(_, len)| len).sum();
    let got = out
        .metadata()
        .map_err(|e| format!("stat {}: {e}", tmp.display()))?
        .len();
    if got != want {
        return Err(format!(
            "assembled {} bytes from chunks 1..{}, expected {want}",
            got,
            parts.len()
        ));
    }

    out.seek(SeekFrom::Start(0))
        .map_err(|e| format!("seek {}: {e}", tmp.display()))?;
    copy_parts_into(&mut out, &parts[..1], tmp)?;
    out.sync_all()
        .map_err(|e| format!("sync {}: {e}", tmp.display()))?;
    drop(out);

    std::fs::rename(tmp, dest)
        .map_err(|e| format!("rename {} -> {}: {e}", tmp.display(), dest.display()))
}

/// Copy each part into `out` at the current position and delete it once its
/// bytes are flushed.
fn copy_parts_into(
    out: &mut std::fs::File,
    parts: &[(PathBuf, u64)],
    tmp: &Path,
) -> Result<(), String> {
    use std::io::{BufWriter, Write};
    let mut writer = BufWriter::with_capacity(8 << 20, out);
    for (part, _) in parts {
        let mut r =
            std::fs::File::open(part).map_err(|e| format!("open {}: {e}", part.display()))?;
        std::io::copy(&mut r, &mut writer).map_err(|e| format!("copy {}: {e}", part.display()))?;
        writer
            .flush()
            .map_err(|e| format!("flush {}: {e}", tmp.display()))?;
        let _ = std::fs::remove_file(part);
    }
    Ok(())
}

/// The sibling a transfer writes into before it earns `dest`'s name.
fn tmp_path(dest: &Path) -> PathBuf {
    let mut name = dest.as_os_str().to_os_string();
    name.push(".tmp");
    PathBuf::from(name)
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
        assert!(err.contains("truncated"), "{err}");
        assert!(err.contains("50 bytes on disk"), "{err}");
        assert!(err.contains("needs 100"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Chunks wide enough that the head covers the sentinel, as a real
    /// 256 MiB chunk does.
    fn write_parts(dest: &Path, n: u64) -> Vec<(PathBuf, u64)> {
        (0..n)
            .map(|i| {
                let p = part_path(dest, i);
                std::fs::write(&p, vec![b'a' + i as u8; 64]).expect("write part");
                (p, 64)
            })
            .collect()
    }

    /// Peak disk during assembly is what decides whether a 19 GiB pack fits
    /// on a machine with 25 GiB free.
    #[test]
    fn concat_frees_each_part_as_it_goes() {
        let dir = scratch_dir("concat");
        let dest = dir.join("out.bin");
        let parts = write_parts(&dest, 3);

        concat_parts(&dest, &parts).expect("concat");
        let got = std::fs::read(&dest).expect("read dest");
        let mut want = vec![b'a'; 64];
        want.extend(std::iter::repeat_n(b'b', 64));
        want.extend(std::iter::repeat_n(b'c', 64));
        assert_eq!(got, want);
        for (p, _) in &parts {
            assert!(!p.exists(), "{} survived assembly", p.display());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The head lands last, so the finished file must not keep the marker
    /// the loader rejects packs for.
    #[test]
    fn assembled_blob_does_not_keep_the_sentinel() {
        let dir = scratch_dir("concat-sentinel");
        let dest = dir.join("out.bin");
        let parts = write_parts(&dest, 2);

        concat_parts(&dest, &parts).expect("concat");
        let got = std::fs::read(&dest).expect("read dest");
        assert_ne!(&got[..INCOMPLETE_SENTINEL.len()], INCOMPLETE_SENTINEL);
        assert!(!tmp_path(&dest).exists(), "tmp survived a good assembly");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A partial `model.dgq.bin` is worse than no file: it loads, reads past
    /// its own end as zeros, and generates `<pad>` forever. The bytes go to a
    /// `.tmp` that never earns the name, and a failure removes even that.
    #[test]
    fn failed_concat_leaves_nothing_behind() {
        let dir = scratch_dir("concat-fail");
        let dest = dir.join("out.bin");
        let good = part_path(&dest, 0);
        std::fs::write(&good, vec![b'a'; 64]).expect("write part");
        let missing = part_path(&dest, 1);

        concat_parts(&dest, &[(good, 64), (missing, 64)]).expect_err("must fail on missing part");
        assert!(
            !dest.exists(),
            "a failed assembly produced {}",
            dest.display()
        );
        assert!(!tmp_path(&dest).exists(), "a failed assembly left its tmp");
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
