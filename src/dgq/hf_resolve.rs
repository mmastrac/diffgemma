//! Resolves layered-pack external refs against the local HuggingFace cache.
//!
//! A layered pack's `HfSafetensors` external files name a shard inside a
//! pinned HF snapshot; this module finds that snapshot on disk and pins its
//! identity (size + header hash) so a stale or wrong cache fails loud instead
//! of silently reading garbage into the model.

use crate::Error;
use crate::dgq::layout::{BaseModelRef, ExternalFile, ExternalRole};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

/// The HF cache root (`$HF_HOME`, i.e. the dir CONTAINING `hub/`), resolved
/// `DGQ_HF_HOME` override -> `HF_HOME` env -> `~/.cache/huggingface`
/// (the `huggingface_hub` default).
pub fn hf_home() -> PathBuf {
    if let Some(dir) = crate::flags::hf_home_override() {
        return dir;
    }
    if let Ok(dir) = std::env::var("HF_HOME")
        && !dir.is_empty()
    {
        return PathBuf::from(dir);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache").join("huggingface")
}

/// `org/name` -> the on-disk `models--org--name` directory name HF's cache
/// layout uses (see `huggingface_hub.constants.repo_folder_name`).
fn repo_folder_name(repo: &str) -> String {
    format!("models--{}", repo.replace('/', "--"))
}

/// `-m` value resolution result: the directory to actually open, plus an
/// exact `(repo, revision)` pin when the directory was found by resolving a
/// bare HF repo id (the snapshot dir name IS the revision — no guessing).
#[derive(Debug)]
pub struct ResolvedModelSource {
    pub dir: PathBuf,
    pub repo_pin: Option<BaseModelRef>,
}

/// `org/name` shape: exactly one `/`, both segments non-empty and made of
/// characters HF repo ids/usernames actually use. Deliberately conservative
/// — a relative path with a slash (`model/transformer`) also matches this
/// shape, but `resolve_model_source` only ever consults it once the
/// existing-directory check has already failed, so a real local dir never
/// gets reinterpreted as a repo id.
pub fn looks_like_hf_repo_id(s: &str) -> bool {
    let parts: Vec<&str> = s.split('/').collect();
    if parts.len() != 2 {
        return false;
    }
    // `model/` and `models/` are the conventional LOCAL weights directories,
    // not HF orgs. Treat them as local relative paths so a missing
    // `-m model/transformer` reports a "no such directory" error instead of a
    // nonsensical `hf download model/transformer` remedy.
    if matches!(parts[0], "model" | "models") {
        return false;
    }
    let ok_segment = |seg: &str| {
        !seg.is_empty()
            && seg != "."
            && seg != ".."
            && seg
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    };
    ok_segment(parts[0]) && ok_segment(parts[1])
}

/// Resolve `-m` to a directory: use it as-is if it already exists; else, if
/// it looks like an HF repo id, resolve the newest local HF-cache snapshot
/// for that repo (fails with the exact `hf download` remedy if none is
/// cached). This is the ONE place `-m` resolution happens; every command
/// that takes `-m` should route through it (see `commands::dispatch`).
pub fn resolve_model_source(spec: &Path) -> Result<ResolvedModelSource, Error> {
    if spec.is_dir() {
        return Ok(ResolvedModelSource {
            dir: spec.to_path_buf(),
            repo_pin: None,
        });
    }
    let s = spec.to_string_lossy().into_owned();
    if looks_like_hf_repo_id(&s) {
        let (dir, revision) = newest_local_snapshot(&s)?;
        return Ok(ResolvedModelSource {
            dir,
            repo_pin: Some(BaseModelRef { repo: s, revision }),
        });
    }
    Err(Error::Layered(format!(
        "-m {}: not a local directory, and it doesn't look like an HF repo id \
         (expected org/name) either — pass an existing directory or a repo id.",
        spec.display()
    )))
}

/// The newest (by mtime) locally cached snapshot dir for `repo`, plus its
/// revision (the snapshot dir's own name — always a commit hash, so this is
/// exact, unlike the single-symlink-hop auto-detect `overlay` uses when the
/// caller only has a shard path). Fails loud with the `hf download` remedy
/// when the repo has no cached snapshots at all.
pub fn newest_local_snapshot(repo: &str) -> Result<(PathBuf, String), Error> {
    let snapshots_dir = hf_home().join("hub").join(repo_folder_name(repo)).join("snapshots");
    let remedy = || {
        Error::Layered(format!(
            "-m {repo}: no local directory by that name, and no cached HF snapshot found \
             (checked {} under HF cache root {}).\n\
             Fetch it with:\n\n    hf download {repo}\n\n\
             or set DGQ_HF_HOME / HF_HOME if it's cached somewhere else, or pass an existing \
             local directory instead.",
            snapshots_dir.display(),
            hf_home().display(),
        ))
    };
    let read = std::fs::read_dir(&snapshots_dir).map_err(|_| remedy())?;
    let mut best: Option<(std::time::SystemTime, PathBuf, String)> = None;
    for entry in read {
        let Ok(entry) = entry else { continue };
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_dir() && !file_type.is_symlink() {
            continue;
        }
        let mtime = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        let revision = entry.file_name().to_string_lossy().into_owned();
        if best.as_ref().is_none_or(|(t, ..)| mtime > *t) {
            best = Some((mtime, entry.path(), revision));
        }
    }
    best.map(|(_, path, rev)| (path, rev)).ok_or_else(remedy)
}

/// The canonical published pack: the `download` default, and the pack the
/// default `-m` resolver prefers locally and in the HF cache.
pub const DEFAULT_MODEL_REPO: &str = "mmastrac/diffgemma-26b-a4b-it-q4";
/// The bare model name of [`DEFAULT_MODEL_REPO`] (its local `model/<name>` dir).
pub const DEFAULT_MODEL_NAME: &str = "diffgemma-26b-a4b-it-q4";

/// Resolve the model source when `-m` was NOT given. Walks a fallback chain and
/// returns the first hit:
///
///   1. `model/transformer`              (the bf16 base-checkpoint convention)
///   2. `model/diffgemma-26b-a4b-it-q4`  (the canonical local pack)
///   3. `model/diffgemma-*`              (any other local diffgemma pack)
///   4. HF-cached `mmastrac/diffgemma-26b-a4b-it-q4`
///   5. any HF-cached `*/diffgemma-*` pack (newest snapshot wins)
///
/// Fails with a `download`-pointing remedy when nothing is found. Unlike a bare
/// `-m` value this never treats the names as HF repo ids, so it can't emit a
/// bogus `hf download model/transformer`.
pub fn resolve_default_model_source() -> Result<ResolvedModelSource, Error> {
    resolve_default_model_source_in(Path::new("model"))
}

/// [`resolve_default_model_source`] against an explicit local `model/` root, so
/// the chain is testable without depending on the process's cwd.
fn resolve_default_model_source_in(model_root: &Path) -> Result<ResolvedModelSource, Error> {
    // 1 + 2: named local dirs, in preference order.
    for name in ["transformer", DEFAULT_MODEL_NAME] {
        let dir = model_root.join(name);
        if dir.is_dir() {
            return Ok(ResolvedModelSource { dir, repo_pin: None });
        }
    }

    // 3: any other local `model/diffgemma-*` pack (name-sorted, deterministic).
    if let Some(dir) = first_local_pack(model_root, "diffgemma-") {
        return Ok(ResolvedModelSource { dir, repo_pin: None });
    }

    // 4: the canonical pack in the HF cache.
    if let Some((dir, revision)) = newest_pack_snapshot(DEFAULT_MODEL_REPO) {
        return Ok(ResolvedModelSource {
            dir,
            repo_pin: Some(BaseModelRef {
                repo: DEFAULT_MODEL_REPO.to_string(),
                revision,
            }),
        });
    }

    // 5: any `*/diffgemma-*` pack in the HF cache (newest snapshot wins).
    if let Some((repo, dir, revision)) = newest_cached_diffgemma_pack() {
        return Ok(ResolvedModelSource {
            dir,
            repo_pin: Some(BaseModelRef { repo, revision }),
        });
    }

    Err(Error::Layered(format!(
        "no model given (-m) and no default model found.\n\
         Looked for local dirs: {root}/transformer, {root}/{DEFAULT_MODEL_NAME}, {root}/diffgemma-*\n\
         and HF-cached packs: {DEFAULT_MODEL_REPO}, any */diffgemma-* under {hub}.\n\n\
         Fetch the default pack with:\n\n    diffgemma-mps download\n\n\
         or pass an existing model with -m <dir | org/name>.",
        root = model_root.display(),
        hub = hf_home().join("hub").display(),
    )))
}

/// Name-sorted first subdir of `root` whose name starts with `prefix` and that
/// contains a `.dgq` manifest (i.e. a real pack, not an arbitrary dir).
fn first_local_pack(root: &Path, prefix: &str) -> Option<PathBuf> {
    let mut hits: Vec<PathBuf> = std::fs::read_dir(root)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(prefix))
                && p.join(crate::dgq::layout::MANIFEST_FILE).is_file()
        })
        .collect();
    hits.sort();
    hits.into_iter().next()
}

/// Newest cache snapshot of `repo` that actually contains a `.dgq` manifest
/// (so a metadata-only or truncated cache entry is skipped, not selected).
fn newest_pack_snapshot(repo: &str) -> Option<(PathBuf, String)> {
    let (dir, revision) = newest_local_snapshot(repo).ok()?;
    dir.join(crate::dgq::layout::MANIFEST_FILE)
        .is_file()
        .then_some((dir, revision))
}

/// Scan the HF cache for `*/diffgemma-*` model repos and return the one whose
/// newest pack snapshot is most recently modified: `(repo_id, snapshot, rev)`.
fn newest_cached_diffgemma_pack() -> Option<(String, PathBuf, String)> {
    let hub = hf_home().join("hub");
    let mut best: Option<(std::time::SystemTime, String, PathBuf, String)> = None;
    for entry in std::fs::read_dir(&hub).ok()?.filter_map(|e| e.ok()) {
        let fname = entry.file_name();
        let fname = fname.to_string_lossy();
        // Cache dir names are `models--{org}--{name}` (org has no `--`).
        let Some(rest) = fname.strip_prefix("models--") else {
            continue;
        };
        let Some((org, name)) = rest.split_once("--") else {
            continue;
        };
        if !name.starts_with("diffgemma-") {
            continue;
        }
        let repo = format!("{org}/{name}");
        let Some((dir, revision)) = newest_pack_snapshot(&repo) else {
            continue;
        };
        let mtime = std::fs::metadata(&dir)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        if best.as_ref().is_none_or(|(t, ..)| mtime > *t) {
            best = Some((mtime, repo, dir, revision));
        }
    }
    best.map(|(_, repo, dir, revision)| (repo, dir, revision))
}

/// Resolve a pinned HF snapshot dir, or fail with the exact command to fetch
/// it.
pub fn resolve_snapshot_dir(base: &BaseModelRef) -> Result<PathBuf, Error> {
    let hub = hf_home().join("hub");
    let dir = hub
        .join(repo_folder_name(&base.repo))
        .join("snapshots")
        .join(&base.revision);
    if !dir.is_dir() {
        return Err(Error::Layered(format!(
            "HF base model not found at {} (checked HF cache root {}).\n\
             This pack's raw tensors are external refs into {}@{} — fetch it with:\n\
             \n    hf download {} --revision {}\n\
             \nor set DGQ_HF_HOME / HF_HOME if it's cached somewhere else.",
            dir.display(),
            hf_home().display(),
            base.repo,
            base.revision,
            base.repo,
            base.revision,
        )));
    }
    Ok(dir)
}

/// SHA-256 of `(8-byte LE header length || header JSON bytes)` plus the file
/// size — cheap enough to run on every load (reads only the header, never
/// the multi-GiB tensor payload) yet pins the shard's tensor layout exactly.
pub fn hash_safetensors_header(path: &Path) -> Result<(String, u64), Error> {
    // Cap before allocating: a corrupt/truncated file's first 8 bytes are
    // attacker- or damage-controlled, and a real safetensors header is at
    // most a few MiB even for this model's 1047-tensor manifest.
    const MAX_HEADER_LEN: u64 = 256 << 20;
    let mut file = File::open(path)?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf);
    if header_len > MAX_HEADER_LEN {
        return Err(Error::Layered(format!(
            "{}: safetensors header length {header_len} exceeds the {MAX_HEADER_LEN}-byte \
             sanity cap — file is corrupt or not a safetensors shard",
            path.display()
        )));
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)?;
    let mut hasher = Sha256::new();
    hasher.update(len_buf);
    hasher.update(&header);
    let digest = hasher.finalize();
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    let size = std::fs::metadata(path)?.len();
    Ok((hex, size))
}

/// Resolve one `external_files` entry to an absolute, verified path.
/// `pack_dir` is the layered pack's own directory (for `PackBin` relative
/// paths); `base_model` must be `Some` when any entry is `HfSafetensors`.
pub fn resolve_external_file(
    pack_dir: &Path,
    base_model: Option<&BaseModelRef>,
    key: &str,
    file: &ExternalFile,
) -> Result<PathBuf, Error> {
    let resolved = match file.role {
        ExternalRole::HfSafetensors => {
            let base = base_model.ok_or_else(|| {
                Error::Layered(format!(
                    "external file '{key}' is role=hf_safetensors but the manifest has no base_model"
                ))
            })?;
            let snapshot = resolve_snapshot_dir(base)?;
            snapshot.join(&file.path)
        }
        ExternalRole::PackBin => {
            let p = PathBuf::from(&file.path);
            if p.is_absolute() { p } else { pack_dir.join(p) }
        }
    };
    if !resolved.is_file() {
        return Err(Error::Layered(format!(
            "external file '{key}' not found at {} (role={:?})",
            resolved.display(),
            file.role
        )));
    }
    let actual_size = std::fs::metadata(&resolved)?.len();
    if actual_size != file.expected_size {
        return Err(Error::Layered(format!(
            "external file '{key}' at {} has size {actual_size} bytes, manifest expects \
             {} bytes — cache contents changed under this pin, re-run `repack --overlay`",
            resolved.display(),
            file.expected_size
        )));
    }
    if let Some(expected_hash) = &file.header_sha256 {
        let (actual_hash, _) = hash_safetensors_header(&resolved)?;
        if &actual_hash != expected_hash {
            return Err(Error::Layered(format!(
                "external file '{key}' at {} has header sha256 {actual_hash}, manifest \
                 pins {expected_hash} — the safetensors shard layout changed under this \
                 pin (different revision or corrupt cache)",
                resolved.display()
            )));
        }
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_folder_name_replaces_slash() {
        assert_eq!(
            repo_folder_name("google/diffusiongemma-26B-A4B-it"),
            "models--google--diffusiongemma-26B-A4B-it"
        );
    }

    #[test]
    fn repo_id_shape_detection() {
        assert!(looks_like_hf_repo_id("google/diffusiongemma-26B-A4B-it"));
        assert!(looks_like_hf_repo_id("mmastrac/diffgemma-26b-a4b-it-q4"));
        // `model/` and `models/` are local weights dirs, never repo ids, so a
        // missing `-m model/transformer` is a path error, not `hf download ...`.
        assert!(!looks_like_hf_repo_id("model/transformer"));
        assert!(!looks_like_hf_repo_id("models/diffgemma-26b-a4b-it-q4"));
        assert!(!looks_like_hf_repo_id("model/transformer/extra"));
        assert!(!looks_like_hf_repo_id("just-a-name"));
        assert!(!looks_like_hf_repo_id("/absolute/two/segs"));
        assert!(!looks_like_hf_repo_id("org/"));
        assert!(!looks_like_hf_repo_id("/name"));
        assert!(!looks_like_hf_repo_id("../escape"));
    }

    #[test]
    fn resolve_model_source_uses_existing_dir_verbatim() {
        let dir = std::env::temp_dir().join(format!("dgq-resolve-existing-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let resolved = resolve_model_source(&dir).expect("existing dir resolves");
        assert_eq!(resolved.dir, dir);
        assert!(resolved.repo_pin.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_model_source_repo_id_picks_newest_snapshot() {
        let hf_home = std::env::temp_dir().join(format!("dgq-resolve-repo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&hf_home);
        let snapshots = hf_home
            .join("hub")
            .join("models--acme--widgets")
            .join("snapshots");
        let old = snapshots.join("aaaaaaa");
        let new = snapshots.join("bbbbbbb");
        std::fs::create_dir_all(&old).expect("mkdir old");
        // Ensure a real mtime gap regardless of filesystem timestamp granularity.
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::create_dir_all(&new).expect("mkdir new");

        let cfg = crate::flags::RuntimeConfig::from_pairs(&[(
            "DGQ_HF_HOME".to_string(),
            hf_home.display().to_string(),
        )])
        .0;
        let _guard = crate::flags::install_for_test(cfg);

        let resolved =
            resolve_model_source(Path::new("acme/widgets")).expect("repo id resolves");
        assert_eq!(resolved.dir, new);
        let pin = resolved.repo_pin.expect("repo pin");
        assert_eq!(pin.repo, "acme/widgets");
        assert_eq!(pin.revision, "bbbbbbb");

        let _ = std::fs::remove_dir_all(&hf_home);
    }

    #[test]
    fn resolve_model_source_repo_id_no_cache_gives_remedy() {
        let hf_home = std::env::temp_dir().join(format!("dgq-resolve-empty-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&hf_home);
        let cfg = crate::flags::RuntimeConfig::from_pairs(&[(
            "DGQ_HF_HOME".to_string(),
            hf_home.display().to_string(),
        )])
        .0;
        let _guard = crate::flags::install_for_test(cfg);

        let err = resolve_model_source(Path::new("acme/nonexistent")).expect_err("must fail");
        let msg = err.to_string();
        assert!(msg.contains("hf download acme/nonexistent"), "{msg}");
        let _ = std::fs::remove_dir_all(&hf_home);
    }

    #[test]
    fn resolve_model_source_neither_dir_nor_repo_shape_fails() {
        let err = resolve_model_source(Path::new("../not/a/real/multi/segment/path"))
            .expect_err("must fail: not a dir, not repo-id shaped");
        assert!(err.to_string().contains("doesn't look like an HF repo id"));
    }

    #[test]
    fn resolve_snapshot_missing_gives_actionable_error() {
        let base = BaseModelRef {
            repo: "google/diffusiongemma-26B-A4B-it".to_string(),
            revision: "definitely-not-a-real-revision".to_string(),
        };
        // Point DGQ_HF_HOME somewhere empty so this doesn't depend on the
        // host's actual cache contents.
        let dir = std::env::temp_dir().join(format!("dgq-hf-resolve-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let cfg = crate::flags::RuntimeConfig::from_pairs(&[(
            "DGQ_HF_HOME".to_string(),
            dir.display().to_string(),
        )])
        .0;
        let _guard = crate::flags::install_for_test(cfg);
        let err = resolve_snapshot_dir(&base).expect_err("must fail: snapshot absent");
        let msg = err.to_string();
        assert!(msg.contains("hf download google/diffusiongemma-26B-A4B-it"));
        assert!(msg.contains("definitely-not-a-real-revision"));
    }

    #[test]
    fn resolve_external_file_missing_pack_bin() {
        let file = ExternalFile {
            role: ExternalRole::PackBin,
            path: "does-not-exist.bin".to_string(),
            expected_size: 123,
            header_sha256: None,
        };
        let err = resolve_external_file(Path::new("/tmp"), None, "k", &file)
            .expect_err("must fail: file absent");
        assert!(err.to_string().contains("does-not-exist.bin"));
    }

    #[test]
    fn resolve_external_file_size_mismatch() {
        let tmp = std::env::temp_dir().join(format!("dgq-hf-resolve-size-{}", std::process::id()));
        std::fs::write(&tmp, b"hello world").expect("write fixture");
        let file = ExternalFile {
            role: ExternalRole::PackBin,
            path: tmp.display().to_string(),
            expected_size: 999,
            header_sha256: None,
        };
        let err = resolve_external_file(Path::new("/tmp"), None, "k", &file)
            .expect_err("must fail: size mismatch");
        assert!(err.to_string().contains("size"));
        let _ = std::fs::remove_file(&tmp);
    }

    fn write_pack(dir: &Path) {
        std::fs::create_dir_all(dir).expect("mkdir pack");
        std::fs::write(dir.join(crate::dgq::layout::MANIFEST_FILE), "{}").expect("write manifest");
    }

    #[test]
    fn default_resolver_prefers_canonical_local_pack() {
        let root = std::env::temp_dir().join(format!("dgq-default-named-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write_pack(&root.join("diffgemma-26b-a4b-it-nvfp4"));
        write_pack(&root.join(DEFAULT_MODEL_NAME));
        let r = resolve_default_model_source_in(&root).expect("resolves");
        assert_eq!(r.dir, root.join(DEFAULT_MODEL_NAME));
        assert!(r.repo_pin.is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn default_resolver_globs_other_local_pack_ignoring_non_packs() {
        let root = std::env::temp_dir().join(format!("dgq-default-glob-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write_pack(&root.join("diffgemma-26b-a4b-it-nvfp4"));
        // Prefixed but not a pack (no manifest); must be skipped.
        std::fs::create_dir_all(root.join("diffgemma-notes")).expect("mkdir");
        let r = resolve_default_model_source_in(&root).expect("resolves");
        assert_eq!(r.dir, root.join("diffgemma-26b-a4b-it-nvfp4"));
        assert!(r.repo_pin.is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn default_resolver_scans_hf_cache_for_any_diffgemma() {
        let root = std::env::temp_dir().join(format!("dgq-default-cache-root-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("empty model root"); // falls through to cache

        let hf_home = std::env::temp_dir().join(format!("dgq-default-cache-hf-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&hf_home);
        let snap = hf_home
            .join("hub")
            .join("models--acme--diffgemma-tiny")
            .join("snapshots")
            .join("abc123");
        write_pack(&snap);
        let cfg = crate::flags::RuntimeConfig::from_pairs(&[(
            "DGQ_HF_HOME".to_string(),
            hf_home.display().to_string(),
        )])
        .0;
        let _guard = crate::flags::install_for_test(cfg);

        let r = resolve_default_model_source_in(&root).expect("resolves from cache");
        assert_eq!(r.dir, snap);
        let pin = r.repo_pin.expect("cache hit pins the repo");
        assert_eq!(pin.repo, "acme/diffgemma-tiny");
        assert_eq!(pin.revision, "abc123");

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&hf_home);
    }

    #[test]
    fn default_resolver_not_found_points_to_download() {
        let root = std::env::temp_dir().join(format!("dgq-default-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("empty model root");
        let hf_home = std::env::temp_dir().join(format!("dgq-default-empty-hf-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&hf_home);
        std::fs::create_dir_all(hf_home.join("hub")).expect("empty hub");
        let cfg = crate::flags::RuntimeConfig::from_pairs(&[(
            "DGQ_HF_HOME".to_string(),
            hf_home.display().to_string(),
        )])
        .0;
        let _guard = crate::flags::install_for_test(cfg);

        let err = resolve_default_model_source_in(&root).expect_err("nothing found");
        let msg = err.to_string();
        assert!(msg.contains("diffgemma-mps download"), "{msg}");
        assert!(msg.contains("no default model found"), "{msg}");

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&hf_home);
    }
}
