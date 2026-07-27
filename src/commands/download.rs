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

    eprintln!("download ok: {}", dest.display());
    eprintln!("  run: diffgemma-mps -m {} chat", dest.display());
    ExitCode::SUCCESS
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
                    return Err(format!(
                        "chunk {idx}: got {got} bytes, expected {expected}"
                    ));
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
fn start_fetch(
    url: &str,
    dest: &Path,
    range: Option<(u64, u64)>,
) -> Result<RequestHandle, String> {
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
    eprintln!("       assembling {} chunks -> {}", parts.len(), dest.display());
    let out = std::fs::File::create(dest).map_err(|e| format!("create {}: {e}", dest.display()))?;
    let mut writer = BufWriter::with_capacity(8 << 20, out);
    for (part, _) in parts {
        let mut r = std::fs::File::open(part).map_err(|e| format!("open {}: {e}", part.display()))?;
        std::io::copy(&mut r, &mut writer)
            .map_err(|e| format!("copy {}: {e}", part.display()))?;
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
