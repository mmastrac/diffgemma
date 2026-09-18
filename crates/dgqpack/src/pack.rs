//! The one way to open a `.dgq` pack.
//!
//! Every backend reaches a pack's bytes through `PackFile`, so the manifest
//! gates run once, in one place, and a new backend cannot skip them by
//! opening the blob itself. `open` parses and gates the manifest, the `map*`
//! methods hand out the bytes, and what a backend does with them is its own
//! business: Metal wraps the mapping as a no-copy `MTLBuffer`, CUDA uploads
//! from it, the CPU oracle reads it directly.

use crate::error::Error;
use crate::manifest::{
    DgqManifest, INCOMPLETE_SENTINEL, MANIFEST_FILE, blob_offset_usize, dgq_version_supported,
};
use memmap2::{Mmap, MmapOptions};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Canonical-offset alignment every `.dgq` tensor entry has always had.
/// Several kernels reinterpret `blob + w_off` as a typed pointer (e.g.
/// `gemm_rowk.metal`: `device const ushort *w = (device const ushort
/// *)(blob + w_off)`, then indexed), so the read is only correct when
/// `w_off` is sufficiently aligned. 64 bytes is what the writer's
/// unconditional `align_offset` has always produced.
const TENSOR_OFFSET_ALIGN: u64 = 64;

/// A pack whose manifest has been gated and whose blob file is known to be
/// long enough for the tensors it describes.
#[derive(Debug)]
pub struct PackFile {
    manifest: DgqManifest,
    model_dir: PathBuf,
    blob_path: PathBuf,
    file: File,
    len: u64,
}

impl PackFile {
    /// Open `model_dir`'s pack and run every gate: manifest parse, version
    /// support, tensor offset alignment, blob length, and the download
    /// sentinel.
    ///
    /// Returns `Error::Pack` for a blob that is short or unfinished,
    /// `Error::Format` for a manifest this build cannot read, and
    /// `Error::Io`/`Error::Json` for a missing or unparseable file.
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self, Error> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let manifest_json = std::fs::read_to_string(model_dir.join(MANIFEST_FILE))?;
        let manifest: DgqManifest = serde_json::from_str(&manifest_json)?;
        Self::open_with_manifest(model_dir, manifest)
    }

    /// Open the blob for a manifest the caller already parsed. Runs the same
    /// gates as `open`.
    pub fn open_with_manifest(
        model_dir: impl AsRef<Path>,
        manifest: DgqManifest,
    ) -> Result<Self, Error> {
        let model_dir = model_dir.as_ref().to_path_buf();
        if !dgq_version_supported(manifest.version) {
            return Err(Error::Format("unsupported .dgq version"));
        }
        check_tensor_offset_alignment(&manifest)?;

        let blob_path = model_dir.join(&manifest.blob_file);
        let mut file = File::open(&blob_path)?;
        let len = file.metadata()?.len();
        manifest.check_local_blob_len(len, &blob_path)?;
        check_head_sentinel(&mut file, &blob_path)?;

        Ok(Self {
            manifest,
            model_dir,
            blob_path,
            file,
            len,
        })
    }

    pub fn manifest(&self) -> &DgqManifest {
        &self.manifest
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn blob_path(&self) -> &Path {
        &self.blob_path
    }

    /// The blob file's length on disk. For a layered pack this is the
    /// compact local blob, which is smaller than the canonical address space
    /// the manifest's `offset` values index.
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Map the whole blob.
    pub fn map(&self) -> Result<Mmap, Error> {
        Ok(unsafe { Mmap::map(&self.file)? })
    }

    /// Map from `offset` to the end. A layered pack's expert tail is wrapped
    /// this way so the mapping's base address coincides with the region's,
    /// leaving one rebase rule instead of two.
    pub fn map_from(&self, offset: u64) -> Result<Mmap, Error> {
        let offset = blob_offset_usize(offset)?;
        Ok(unsafe { MmapOptions::new().offset(offset as u64).map(&self.file)? })
    }

    /// Consume the pack, yielding its mapping alongside the `File` that backs
    /// it. A no-copy GPU wrap has to keep the file open for as long as the
    /// buffer points into the mapping.
    pub fn into_map_and_file(self) -> Result<(Mmap, File, DgqManifest), Error> {
        let map = unsafe { Mmap::map(&self.file)? };
        Ok((map, self.file, self.manifest))
    }

    /// Give up the blob and keep the manifest, for a layered load that
    /// gathers its bytes from elsewhere.
    pub fn into_manifest(self) -> DgqManifest {
        self.manifest
    }

    /// Hand over the open file, for a caller that took its mappings already
    /// and needs to keep the descriptor for as long as they live.
    pub fn into_file(self) -> File {
        self.file
    }
}

/// Reject a `w_off` that is byte-correct but too coarsely aligned for the
/// typed-pointer reads several kernels do. Cheap (manifest only, no I/O) and
/// unconditional. A byte-content comparison cannot see this class: both
/// sides of such a check read through untyped pointers, and only the GPU's
/// reinterpret-cast cares. It reached generation once as silently wrong
/// output.
fn check_tensor_offset_alignment(manifest: &DgqManifest) -> Result<(), Error> {
    let offenders: Vec<&str> = manifest
        .tensors
        .iter()
        .filter(|t| !t.meta.offset.is_multiple_of(TENSOR_OFFSET_ALIGN))
        .map(|t| t.name.as_str())
        .collect();
    if offenders.is_empty() {
        return Ok(());
    }
    Err(Error::Pack(format!(
        "{} tensor(s) have a canonical offset not aligned to {TENSOR_OFFSET_ALIGN} bytes, so \
         this pack is unsafe to load: GPU kernels read weight bytes through a typed pointer \
         cast at that offset. First few: {:?}",
        offenders.len(),
        &offenders[..offenders.len().min(10)]
    )))
}

/// Reject a blob still carrying the assembly sentinel.
///
/// `diffgemma download` writes the sentinel at offset 0 and overwrites it
/// with the real first chunk only after every other byte is on disk, so a
/// blob holding it is a transfer that did not finish. The length check alone
/// cannot see this: a blob assembled tail-first reaches its full length well
/// before it is complete.
fn check_head_sentinel(file: &mut File, blob_path: &Path) -> Result<(), Error> {
    let mut head = [0u8; INCOMPLETE_SENTINEL.len()];
    if file.read_exact(&mut head).is_err() {
        return Ok(());
    }
    if &head != INCOMPLETE_SENTINEL {
        return Ok(());
    }
    Err(Error::Pack(format!(
        "{}: unfinished download.\n\
         \x20 The first chunk was never written, so this file is missing bytes\n\
         \x20 whatever its size says. Re-fetch it:\n\
         \x20   diffgemma download --force",
        blob_path.display()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{
        BLOB_FILE, DGQ_VERSION_AFFINE, DgqManifest, DgqTensorEntry, DgqTensorMeta, QuantProfile,
    };
    use std::collections::BTreeMap;
    use std::io::Write;

    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("dgqpack-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    /// A pack whose single tensor covers `[0, len)`, written to `dir`.
    fn write_pack(dir: &Path, len: u64, blob: &[u8]) -> DgqManifest {
        let manifest = DgqManifest {
            version: DGQ_VERSION_AFFINE,
            profile: QuantProfile::Q4,
            source_model: "test".to_string(),
            blob_file: BLOB_FILE.to_string(),
            expert_split: None,
            local_expert_split: None,
            base_model: None,
            external_files: BTreeMap::new(),
            custom_classes: BTreeMap::new(),
            tensors: vec![DgqTensorEntry {
                name: "t".to_string(),
                meta: DgqTensorMeta {
                    kind: "raw".to_string(),
                    dtype: "bf16".to_string(),
                    shape: vec![1],
                    offset: 0,
                    byte_len: len,
                    source: None,
                },
            }],
        };
        std::fs::write(
            dir.join(MANIFEST_FILE),
            serde_json::to_string(&manifest).unwrap(),
        )
        .expect("write manifest");
        let mut f = std::fs::File::create(dir.join(BLOB_FILE)).expect("create blob");
        f.write_all(blob).expect("write blob");
        manifest
    }

    #[test]
    fn opens_a_complete_pack() {
        let dir = scratch_dir("ok");
        write_pack(&dir, 64, &[7u8; 64]);
        let pack = PackFile::open(&dir).expect("open");
        assert_eq!(pack.len(), 64);
        assert_eq!(&pack.map().expect("map")[..4], &[7, 7, 7, 7]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The download writes the head last, so a blob can reach its full length
    /// while still missing its first chunk. Length alone cannot see that.
    #[test]
    fn full_length_blob_still_holding_the_sentinel_is_rejected() {
        let dir = scratch_dir("sentinel");
        let mut blob = vec![0u8; 64];
        blob[..INCOMPLETE_SENTINEL.len()].copy_from_slice(INCOMPLETE_SENTINEL);
        write_pack(&dir, 64, &blob);

        let err = PackFile::open(&dir).expect_err("must reject an unfinished blob");
        let msg = err.to_string();
        assert!(msg.contains("unfinished download"), "{msg}");
        assert!(msg.contains("diffgemma download --force"), "{msg}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn short_blob_is_rejected() {
        let dir = scratch_dir("short");
        write_pack(&dir, 64, &[7u8; 63]);
        let err = PackFile::open(&dir).expect_err("must reject a short blob");
        assert!(err.to_string().contains("truncated"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Kernels read `blob + w_off` through a typed pointer cast, so a
    /// correctly-valued but under-aligned offset is wrong on the GPU and
    /// invisible to any byte comparison.
    #[test]
    fn misaligned_tensor_offset_is_rejected() {
        let dir = scratch_dir("align");
        let mut manifest = write_pack(&dir, 128, &[7u8; 128]);
        manifest.tensors[0].meta.offset = 8;
        manifest.tensors[0].meta.byte_len = 120;
        std::fs::write(
            dir.join(MANIFEST_FILE),
            serde_json::to_string(&manifest).unwrap(),
        )
        .expect("rewrite manifest");

        let err = PackFile::open(&dir).expect_err("must reject a misaligned offset");
        assert!(err.to_string().contains("aligned"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
