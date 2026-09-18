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

/// Download one file to `dest`.
///
/// A file that fits in one chunk is fetched beside `dest` and renamed. A
/// larger one is fetched as byte ranges, `jobs` at a time, and each chunk is
/// written into its final position in a sparse `.tmp` as soon as it lands.
/// Staging on arrival is what keeps the transfer inside the pack's own size:
/// the only bytes on disk twice are the chunks currently in flight, and there
/// is no assembly pass reading 19 GiB back to write it out again.
///
/// A re-run resumes from `.stage`, which records the SHA-256 of every chunk
/// already staged. Those hashes are re-checked against the file before any of
/// it is trusted, so a chunk that a crash left half written is refetched
/// rather than inherited.
fn download_file(
    url: &str,
    dest: &Path,
    size: u64,
    force: bool,
    jobs: usize,
) -> Result<(), String> {
    let tmp = tmp_path(dest);
    if force {
        let _ = std::fs::remove_file(dest);
        let _ = std::fs::remove_file(&tmp);
        let _ = std::fs::remove_file(stage_log_path(dest));
    }

    let n_chunks = size.div_ceil(CHUNK_SIZE).max(1);
    if n_chunks == 1 {
        // Fetch beside the destination and rename, so a killed transfer
        // leaves no half file under the name the loader looks for.
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

    match stage_chunks(url, dest, &tmp, size, n_chunks, jobs) {
        Ok(()) => {
            let _ = std::fs::remove_file(stage_log_path(dest));
            Ok(())
        }
        // The `.tmp` and its log stay put: between them they are the resume
        // point, and neither can be mistaken for a finished download.
        Err(e) => Err(e),
    }
}

fn stage_chunks(
    url: &str,
    dest: &Path,
    tmp: &Path,
    size: u64,
    n_chunks: u64,
    jobs: usize,
) -> Result<(), String> {
    use std::os::unix::fs::FileExt;

    let out = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(tmp)
        .map_err(|e| format!("open {}: {e}", tmp.display()))?;
    // Full length up front, as a hole. Chunks land at their real offsets, so
    // the file is the right size from the start and only the staged ranges
    // occupy blocks.
    out.set_len(size)
        .map_err(|e| format!("size {}: {e}", tmp.display()))?;

    let head_len = CHUNK_SIZE.min(size);
    if head_len >= INCOMPLETE_SENTINEL.len() as u64 {
        out.write_all_at(INCOMPLETE_SENTINEL, 0)
            .map_err(|e| format!("write sentinel to {}: {e}", tmp.display()))?;
    }

    let mut log =
        StageLog::load(dest, size, CHUNK_SIZE).unwrap_or_else(|| StageLog::new(size, CHUNK_SIZE));
    let staged = log.retain_verified(&out, n_chunks)?;
    if staged > 0 {
        eprintln!("       resuming: {staged}/{n_chunks} chunks already staged and verified");
    }

    // Chunk 0 goes last and is never recorded, so it is always refetched: it
    // carries the sentinel's span, and the file must not hold real head bytes
    // until everything behind them is down.
    let pending: Vec<u64> = (1..n_chunks)
        .filter(|i| !log.has(*i))
        .chain(std::iter::once(0))
        .collect();

    let mut completed = staged;
    for batch in pending.chunks(jobs.max(1)) {
        let mut running = Vec::new();
        for &idx in batch {
            let start = idx * CHUNK_SIZE;
            let end = ((idx + 1) * CHUNK_SIZE).min(size) - 1;
            let part = part_path(dest, idx);
            let _ = std::fs::remove_file(&part);
            running.push((
                idx,
                part.clone(),
                start,
                end - start + 1,
                start_fetch(url, &part, Some((start, end))),
            ));
        }
        for (idx, part, start, expected, handle) in running {
            let staged_hash = stage_one(&out, &part, idx, start, expected, handle);
            let _ = std::fs::remove_file(&part);
            let hash = staged_hash?;
            if idx != 0 {
                log.record(idx, hash);
                log.save(dest)?;
            }
            completed += 1;
            eprintln!("       chunk {completed}/{n_chunks} staged");
        }
    }

    let got = out
        .metadata()
        .map_err(|e| format!("stat {}: {e}", tmp.display()))?
        .len();
    if got != size {
        return Err(format!("staged {got} bytes, expected {size}"));
    }
    out.sync_all()
        .map_err(|e| format!("sync {}: {e}", tmp.display()))?;
    drop(out);

    std::fs::rename(tmp, dest)
        .map_err(|e| format!("rename {} -> {}: {e}", tmp.display(), dest.display()))
}

/// Wait for one chunk's fetch, then stage it. The part file is the caller's
/// to delete either way.
fn stage_one(
    out: &std::fs::File,
    part: &Path,
    idx: u64,
    start: u64,
    expected: u64,
    handle: Result<RequestHandle, String>,
) -> Result<String, String> {
    let resp = handle?.join().map_err(|e| format!("chunk {idx}: {e:?}"))?;
    // 206 = partial content (range honored). 200 means the server sent the
    // whole file for a ranged request, which breaks the chunk plan.
    if resp.status_code != 206 {
        return Err(format!(
            "chunk {idx}: server returned HTTP {} for a range request (expected 206)",
            resp.status_code
        ));
    }
    stage_part(out, part, idx, start, expected)
}

/// Copy a fetched chunk into `out` at `start` and return its SHA-256. One
/// pass: the bytes are hashed as they are written, so the hash the resume log
/// stores is of exactly what landed.
fn stage_part(
    out: &std::fs::File,
    part: &Path,
    idx: u64,
    start: u64,
    expected: u64,
) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    use std::os::unix::fs::FileExt;

    match file_len(part) {
        Some(got) if got == expected => {}
        Some(got) => return Err(format!("chunk {idx}: got {got} bytes, expected {expected}")),
        None => return Err(format!("chunk {idx}: part missing after fetch")),
    }

    let mut src = std::fs::File::open(part).map_err(|e| format!("chunk {idx}: open: {e}"))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    let mut at = start;
    loop {
        let n = src
            .read(&mut buf)
            .map_err(|e| format!("chunk {idx}: read: {e}"))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        out.write_all_at(&buf[..n], at)
            .map_err(|e| format!("chunk {idx}: stage at {at}: {e}"))?;
        at += n as u64;
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// What a `.tmp` already holds, so a resumed run refetches only what it must.
///
/// Sizes cannot answer that once chunks are staged in place: the file is full
/// length from the first write. The log records each staged chunk's SHA-256
/// and `retain_verified` re-reads them, so a range a crash left half written
/// is dropped and refetched instead of being trusted for its position alone.
#[derive(serde::Serialize, serde::Deserialize)]
struct StageLog {
    size: u64,
    chunk_size: u64,
    /// Chunk index to hex SHA-256 of the bytes staged for it.
    chunks: std::collections::BTreeMap<u64, String>,
}

impl StageLog {
    fn new(size: u64, chunk_size: u64) -> Self {
        Self {
            size,
            chunk_size,
            chunks: std::collections::BTreeMap::new(),
        }
    }

    /// The log for `dest`, if one is there and describes this same transfer.
    /// A different size or chunking is a different download, so its hashes
    /// say nothing about these offsets.
    fn load(dest: &Path, size: u64, chunk_size: u64) -> Option<Self> {
        let raw = std::fs::read_to_string(stage_log_path(dest)).ok()?;
        let log: Self = serde_json::from_str(&raw).ok()?;
        (log.size == size && log.chunk_size == chunk_size).then_some(log)
    }

    fn save(&self, dest: &Path) -> Result<(), String> {
        let path = stage_log_path(dest);
        let json = serde_json::to_string(self).map_err(|e| e.to_string())?;
        std::fs::write(&path, json).map_err(|e| format!("write {}: {e}", path.display()))
    }

    fn has(&self, idx: u64) -> bool {
        self.chunks.contains_key(&idx)
    }

    fn record(&mut self, idx: u64, hash: String) {
        self.chunks.insert(idx, hash);
    }

    /// Re-hash every chunk the log claims and keep only the ones that still
    /// match. Returns how many survived.
    fn retain_verified(&mut self, out: &std::fs::File, n_chunks: u64) -> Result<usize, String> {
        if self.chunks.is_empty() {
            return Ok(0);
        }
        eprintln!(
            "       verifying {} staged chunk(s) before resuming",
            self.chunks.len()
        );
        let mut good = std::collections::BTreeMap::new();
        for (&idx, want) in &self.chunks {
            if idx == 0 || idx >= n_chunks {
                continue;
            }
            let start = idx * self.chunk_size;
            let len = self.chunk_size.min(self.size.saturating_sub(start));
            if len == 0 {
                continue;
            }
            if &chunk_hash(out, start, len)? == want {
                good.insert(idx, want.clone());
            } else {
                eprintln!("       chunk {idx} does not match its hash, refetching");
            }
        }
        self.chunks = good;
        Ok(self.chunks.len())
    }
}

fn chunk_hash(out: &std::fs::File, start: u64, len: u64) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    use std::os::unix::fs::FileExt;

    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    let mut at = start;
    let end = start + len;
    while at < end {
        let want = buf.len().min((end - at) as usize);
        out.read_exact_at(&mut buf[..want], at)
            .map_err(|e| format!("read staged bytes at {at}: {e}"))?;
        hasher.update(&buf[..want]);
        at += want as u64;
    }
    Ok(hex(&hasher.finalize()))
}

/// Where the resume log for `dest` lives.
fn stage_log_path(dest: &Path) -> PathBuf {
    let mut name = dest.as_os_str().to_os_string();
    name.push(".stage");
    PathBuf::from(name)
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

    /// Stage `n` chunks of `len` bytes into a sparse file the way the fetch
    /// loop does, and return the file plus what the log ends up holding.
    fn stage(dir: &Path, n: u64, len: u64) -> (PathBuf, StageLog) {
        let tmp = dir.join("out.bin.tmp");
        let out = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&tmp)
            .expect("open tmp");
        out.set_len(n * len).expect("set_len");

        let mut log = StageLog::new(n * len, len);
        for i in 0..n {
            let part = dir.join(format!("part{i}"));
            std::fs::write(&part, vec![b'a' + i as u8; len as usize]).expect("write part");
            let hash = stage_part(&out, &part, i, i * len, len).expect("stage");
            if i != 0 {
                log.record(i, hash);
            }
        }
        (tmp, log)
    }

    /// Staging writes each chunk where it belongs, so the file is correct
    /// with no assembly pass and no second copy of the data on disk.
    #[test]
    fn chunks_land_at_their_own_offsets() {
        let dir = scratch_dir("stage");
        let (tmp, _) = stage(&dir, 3, 64);
        let got = std::fs::read(&tmp).expect("read tmp");
        let mut want = vec![b'a'; 64];
        want.extend(std::iter::repeat_n(b'b', 64));
        want.extend(std::iter::repeat_n(b'c', 64));
        assert_eq!(got, want);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The log's whole job is to say which staged ranges can be trusted, and
    /// a crash mid-write leaves a range that is the right length and the
    /// wrong bytes. Only the hash can tell those apart.
    #[test]
    fn resume_drops_a_chunk_whose_bytes_changed() {
        let dir = scratch_dir("stage-resume");
        let (tmp, mut log) = stage(&dir, 3, 64);
        assert_eq!(log.chunks.len(), 2);

        let out = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&tmp)
            .expect("reopen");
        std::os::unix::fs::FileExt::write_all_at(&out, b"xx", 64).expect("corrupt chunk 1");

        let kept = log.retain_verified(&out, 3).expect("verify");
        assert_eq!(kept, 1, "the corrupted chunk must not survive");
        assert!(!log.has(1));
        assert!(log.has(2));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A log written for a different transfer says nothing about these
    /// offsets, so it is ignored rather than trusted.
    #[test]
    fn resume_log_for_a_different_size_is_ignored() {
        let dir = scratch_dir("stage-mismatch");
        let dest = dir.join("out.bin");
        StageLog::new(4096, CHUNK_SIZE).save(&dest).expect("save");
        assert!(StageLog::load(&dest, 4096, CHUNK_SIZE).is_some());
        assert!(StageLog::load(&dest, 8192, CHUNK_SIZE).is_none());
        assert!(StageLog::load(&dest, 4096, CHUNK_SIZE / 2).is_none());
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
