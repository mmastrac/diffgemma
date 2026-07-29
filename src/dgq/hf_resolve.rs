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
}
