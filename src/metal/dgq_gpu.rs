//! Zero-copy `.dgq` blob → `MTLBuffer` and per-tensor GPU views.

use crate::Error;
use crate::dgq::DgqStore;
use crate::dgq::layout::{
    DgqManifest, MANIFEST_FILE, QuantKind, TensorSource, blob_offset_for_mtl, blob_offset_usize,
    dgq_version_supported, nvfp4_matrix_bytes, q4_matrix_bytes, q8_row_bytes,
};
use memmap2::{Mmap, MmapMut};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::collections::HashMap;
use std::ffi::c_void;
use std::fs::File;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

/// Keeps mmap + file alive while the GPU buffer references the mapping.
/// `_file`/`_mmap` back region 1 (head); `_tail_file`/`_tail_mmap` are only
/// `Some` when region 2 (experts) is backed by a DIFFERENT mapping than
/// region 1 — the layered split-source path below, where the (large) expert
/// tail is wrapped straight off this pack's own blob file instead of being
/// gather-copied. `_file` is `None` for an anonymous (materialized) mapping.
pub struct DgqGpuBlob {
    _file: Option<File>,
    _mmap: Mmap,
    _tail_file: Option<File>,
    _tail_mmap: Option<Mmap>,
    /// Region 1: [0, expert_split) when split, else the whole blob.
    pub buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Region 2 [expert_split, len) when the blob exceeds max buffer length.
    pub buffer_experts: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// 0 when single-buffer.
    pub expert_split: u64,
    pub len: usize,
}

impl DgqGpuBlob {
    pub fn from_store(
        store: &DgqStore,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Result<Arc<Self>, Error> {
        let model_dir = store.model_dir.clone();
        let manifest_path = model_dir.join(MANIFEST_FILE);
        let manifest_json = std::fs::read_to_string(&manifest_path)?;
        let manifest: DgqManifest = serde_json::from_str(&manifest_json)?;
        if !dgq_version_supported(manifest.version) {
            return Err(Error::Format("unsupported .dgq version"));
        }
        if manifest.is_layered() {
            if let (Some(expert_split), Some(local_expert_split)) =
                (manifest.expert_split, manifest.local_expert_split)
            {
                // Preferred layered path: the writer arranged the local blob
                // so its expert region is byte-identical (same relative
                // offsets) to the canonical expert tail. Materialize only the
                // (much smaller) head into an anonymous mapping and wrap the
                // expert tail as a direct file-backed no-copy region straight
                // off this pack's own blob — the ~13 GiB expert region is
                // never gather-copied, and its pages stay evictable/
                // re-readable from disk under memory pressure (anonymous
                // pages are not).
                return Self::from_store_layered_split(
                    &model_dir,
                    &manifest,
                    expert_split,
                    local_expert_split,
                    device,
                );
            }
            // Fallback: gather-copy every tensor's bytes — from this pack's
            // own (compact) blob or a resolved external file — into one
            // anonymous mapping laid out at the manifest's CANONICAL
            // offsets. Everything downstream (region wrap, w_off addressing,
            // split-blob math) then runs completely unchanged, because it
            // sees the exact byte layout a self-contained pack would have
            // produced. Costs a full-blob memcpy and anonymous (swap-backed,
            // not evictable-and-re-readable) resident pages — see
            // ARCHITECTURE.md's pack-format section.
            let mmap = materialize_layered_blob(&model_dir, &manifest)?;
            return Self::wrap_mmap(None, mmap, manifest.expert_split, device);
        }
        let blob_path = model_dir.join(&manifest.blob_file);
        let file = File::open(&blob_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::wrap_mmap(Some(file), mmap, manifest.expert_split, device)
    }

    /// Layered split-source path: head materialized anonymously, expert tail
    /// wrapped directly off `manifest.blob_file` at `local_expert_split`
    /// (page-aligned, so the pointer arithmetic mirrors the self-contained
    /// split-buffer wrap in [`Self::wrap_mmap`]).
    fn from_store_layered_split(
        model_dir: &Path,
        manifest: &DgqManifest,
        expert_split: u64,
        local_expert_split: u64,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Result<Arc<Self>, Error> {
        let (head_file, head_mmap) = layered_head_mmap(model_dir, manifest, expert_split)?;
        let local_bin_path = model_dir.join(&manifest.blob_file);
        let local_file = File::open(&local_bin_path)?;
        if !local_expert_split.is_multiple_of(16384) {
            return Err(Error::Layered(format!(
                "dgq layered blob: local_expert_split {local_expert_split} is not 16384-aligned — \
                 cannot offset-map / no-copy wrap the expert tail; re-run repack/quantize --overlay"
            )));
        }
        // Map the tail STARTING AT local_expert_split so the mapping's base
        // address and the region-2 MTLBuffer's base address coincide: host_ptr
        // and the GPU wrap then share the single rebase rule
        // (off - expert_split), with no second file-side base to forget.
        let local_mmap = unsafe {
            memmap2::MmapOptions::new()
                .offset(local_expert_split)
                .map(&local_file)?
        };
        let tail_len = local_mmap.len();
        if tail_len == 0 {
            return Err(Error::Runtime(
                "dgq: local_expert_split at or past end of local blob",
            ));
        }

        let head_len = head_mmap.len();
        // Defense in depth: the writer's `local_expert_split` claim (tail is
        // byte-identical, same relative offsets, to canonical
        // [expert_split, total)) is exactly what `w_off` addressing depends
        // on. A mismatched tail length means a writer bug, not a value to
        // silently absorb.
        let canonical_total = manifest
            .tensors
            .iter()
            .map(|t| t.meta.offset + t.meta.byte_len)
            .max()
            .unwrap_or(0);
        let expected_tail_len = blob_offset_usize(canonical_total)?
            .checked_sub(blob_offset_usize(expert_split)?)
            .ok_or(Error::Runtime("dgq: expert_split past canonical total"))?;
        if tail_len != expected_tail_len {
            return Err(Error::Layered(format!(
                "dgq layered blob: local expert-region length {tail_len} != canonical expert \
                 region length {expected_tail_len} — local_expert_split does not describe this \
                 blob_file; re-run repack/quantize --overlay"
            )));
        }
        let max_buf = device.maxBufferLength();
        if head_len > max_buf || tail_len > max_buf {
            return Err(Error::Format(
                "dgq layered blob region exceeds device max buffer length even after expert split",
            ));
        }
        eprintln!(
            "dgq layered blob: head {:.2} GiB materialized, expert tail {:.2} GiB file-backed (no copy)",
            head_len as f64 / 1073741824.0,
            tail_len as f64 / 1073741824.0,
        );
        let head_ptr =
            NonNull::new(head_mmap.as_ptr() as *mut c_void).ok_or(Error::Runtime("dgq mmap null"))?;
        let buffer = unsafe {
            device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    head_ptr,
                    head_len,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
                .ok_or(Error::Gpu("dgq gpu blob region1 alloc failed"))?
        };
        let tail_ptr = NonNull::new(local_mmap.as_ptr() as *mut c_void)
            .ok_or(Error::Runtime("dgq mmap region2 null"))?;
        let buffer_experts = unsafe {
            device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    tail_ptr,
                    tail_len,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
                .ok_or(Error::Gpu("dgq gpu blob region2 alloc failed"))?
        };
        Ok(Arc::new(Self {
            _file: head_file,
            _mmap: head_mmap,
            _tail_file: Some(local_file),
            _tail_mmap: Some(local_mmap),
            buffer,
            buffer_experts: Some(buffer_experts),
            expert_split,
            len: head_len + tail_len,
        }))
    }

    /// Wrap an already-materialized mmap (self-contained file-backed, or a
    /// layered pack's gather-copied anonymous mapping) as one or two no-copy
    /// `MTLBuffer`s, splitting at `expert_split` when the blob exceeds the
    /// device's max single-buffer length.
    fn wrap_mmap(
        file: Option<File>,
        mmap: Mmap,
        expert_split_hint: Option<u64>,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Result<Arc<Self>, Error> {
        let len = mmap.len();
        let ptr =
            NonNull::new(mmap.as_ptr() as *mut c_void).ok_or(Error::Runtime("dgq mmap null"))?;
        let max_buf = device.maxBufferLength();
        if len <= max_buf {
            let buffer = unsafe {
                device
                    .newBufferWithBytesNoCopy_length_options_deallocator(
                        ptr,
                        len,
                        MTLResourceOptions::StorageModeShared,
                        None,
                    )
                    .ok_or(Error::Gpu("dgq gpu blob alloc failed"))?
            };
            return Ok(Arc::new(Self {
                _file: file,
                _mmap: mmap,
                _tail_file: None,
                _tail_mmap: None,
                buffer,
                buffer_experts: None,
                expert_split: 0,
                len,
            }));
        }
        // Blob exceeds the device's max single-buffer length: wrap it as two
        // no-copy regions split at the (page-aligned) expert boundary the
        // converter recorded (experts are written last).
        let split = expert_split_hint.ok_or(Error::Format(
            "dgq blob exceeds device max buffer length and manifest has no expert_split — re-convert with the experts-last converter",
        ))? as usize;
        if !split.is_multiple_of(16384) || split == 0 || split >= len {
            return Err(Error::Runtime("dgq expert_split invalid"));
        }
        if split > max_buf || (len - split) > max_buf {
            return Err(Error::Format(
                "dgq blob region exceeds device max buffer length even after expert split",
            ));
        }
        eprintln!(
            "dgq blob: {:.2} GiB > max buffer {:.2} GiB — split at {:.2} GiB (experts region {:.2} GiB)",
            len as f64 / 1073741824.0,
            max_buf as f64 / 1073741824.0,
            split as f64 / 1073741824.0,
            (len - split) as f64 / 1073741824.0
        );
        let buffer = unsafe {
            device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    ptr,
                    split,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
                .ok_or(Error::Gpu("dgq gpu blob region1 alloc failed"))?
        };
        let ptr2 = NonNull::new(unsafe { (mmap.as_ptr() as *mut u8).add(split) } as *mut c_void)
            .ok_or(Error::Runtime("dgq mmap region2 null"))?;
        let buffer_experts = unsafe {
            device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    ptr2,
                    len - split,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
                .ok_or(Error::Gpu("dgq gpu blob region2 alloc failed"))?
        };
        Ok(Arc::new(Self {
            _file: file,
            _mmap: mmap,
            _tail_file: None,
            _tail_mmap: None,
            buffer,
            buffer_experts: Some(buffer_experts),
            expert_split: split as u64,
            len,
        }))
    }

    /// (buffer, rebased offset) for an absolute blob offset.
    pub fn buffer_for(&self, off: u64) -> (&ProtocolObject<dyn MTLBuffer>, u64) {
        match (
            &self.buffer_experts,
            off >= self.expert_split && self.expert_split > 0,
        ) {
            (Some(b2), true) => (b2, off - self.expert_split),
            _ => (&self.buffer, off),
        }
    }

    /// The buffer holding expert tensors (region 2 when split, else region 1)
    /// plus the base offset to subtract from absolute expert offsets.
    pub fn expert_region(&self) -> (&ProtocolObject<dyn MTLBuffer>, u64) {
        match &self.buffer_experts {
            Some(b2) => (b2, self.expert_split),
            None => (&self.buffer, 0),
        }
    }

    /// Host pointer at an absolute blob offset: reads the tail mapping when
    /// one is present and `off` falls in the expert region, else the head
    /// mapping (which spans the whole blob when there is no separate tail).
    pub fn host_ptr(&self, off: u64) -> *const u8 {
        if let Some(tail) = &self._tail_mmap
            && self.expert_split > 0
            && off >= self.expert_split
        {
            return unsafe { tail.as_ptr().add(blob_offset_for_mtl(off - self.expert_split)) };
        }
        unsafe { self._mmap.as_ptr().add(blob_offset_for_mtl(off)) }
    }
}

/// Open one mmap per file a layered manifest's `external_files` references.
fn resolve_external_mmaps(
    model_dir: &Path,
    manifest: &DgqManifest,
) -> Result<HashMap<String, Mmap>, Error> {
    let mut external = HashMap::with_capacity(manifest.external_files.len());
    for (key, ext_file) in &manifest.external_files {
        let path = crate::dgq::hf_resolve::resolve_external_file(
            model_dir,
            manifest.base_model.as_ref(),
            key,
            ext_file,
        )?;
        let f = File::open(&path)?;
        external.insert(key.clone(), unsafe { Mmap::map(&f)? });
    }
    Ok(external)
}

/// Bounds-checked sub-slice: a manifest entry whose offset+len exceeds its
/// (already size-verified) source file means a corrupt or mismatched pack,
/// not a host process crash.
fn checked_slice<'a>(buf: &'a [u8], start: usize, len: usize, what: &str) -> Result<&'a [u8], Error> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| Error::Layered(format!("{what}: offset overflow")))?;
    buf.get(start..end).ok_or_else(|| {
        Error::Layered(format!(
            "{what}: extends past its source (needs {end} bytes, source has {})",
            buf.len()
        ))
    })
}

/// Resolve one tensor entry's source bytes (own blob, local-redirected, or
/// external), bounds-checked against whichever mmap actually holds them.
fn source_slice<'a>(
    entry: &crate::dgq::layout::DgqTensorEntry,
    local_mmap: &'a [u8],
    external: &'a HashMap<String, Mmap>,
    len: usize,
) -> Result<&'a [u8], Error> {
    match &entry.meta.source {
        None => checked_slice(local_mmap, blob_offset_usize(entry.meta.offset)?, len, &entry.name),
        Some(TensorSource::Local { local_offset }) => {
            checked_slice(local_mmap, blob_offset_usize(*local_offset)?, len, &entry.name)
        }
        Some(TensorSource::External { file, offset }) => {
            let mm = external.get(file).ok_or_else(|| {
                Error::Layered(format!(
                    "tensor {} references external file '{file}' which was not resolved",
                    entry.name
                ))
            })?;
            checked_slice(mm, blob_offset_usize(*offset)?, len, &entry.name)
        }
    }
}

/// Gather-copy a layered pack's tensors into one anonymous mapping at their
/// CANONICAL offsets — i.e. reconstruct exactly the bytes a self-contained
/// pack's blob would contain, from wherever they actually live (this pack's
/// own compact blob, or a resolved external HF-safetensors shard). Bounded
/// memory: each tensor is copied straight from its source mmap (already
/// page-cached, no intermediate heap buffer); only the destination mapping
/// (~18 GiB for the shipped q4 pack) is resident, exactly as a self-contained
/// pack's blob would be once mmap'd. Fallback when the writer didn't record
/// `local_expert_split` — see `materialize_layered_head` for the preferred,
/// much cheaper path.
fn materialize_layered_blob(model_dir: &Path, manifest: &DgqManifest) -> Result<Mmap, Error> {
    let local_bin = File::open(model_dir.join(&manifest.blob_file))?;
    let local_mmap = unsafe { Mmap::map(&local_bin)? };
    let external = resolve_external_mmaps(model_dir, manifest)?;

    let total_len = manifest
        .tensors
        .iter()
        .map(|t| t.meta.offset + t.meta.byte_len)
        .max()
        .unwrap_or(0);
    let total_len = blob_offset_usize(total_len)?;
    let mut anon = MmapMut::map_anon(total_len)?;

    for entry in &manifest.tensors {
        let dst_start = blob_offset_usize(entry.meta.offset)?;
        let len = blob_offset_usize(entry.meta.byte_len)?;
        let dst_end = dst_start
            .checked_add(len)
            .ok_or(Error::Runtime("dgq materialize: offset overflow"))?;
        if dst_end > total_len {
            return Err(Error::Runtime("dgq materialize: tensor extends past blob"));
        }
        let src = source_slice(entry, &local_mmap, &external, len)?;
        anon[dst_start..dst_end].copy_from_slice(src);
    }
    drop(external);
    drop(local_mmap);
    Ok(anon.make_read_only()?)
}

/// Like `materialize_layered_blob` but copies only tensors in the canonical
/// HEAD region `[0, expert_split)` — the (large) expert tail is wrapped
/// directly off the pack's own blob by `DgqGpuBlob::from_store_layered_split`
/// instead of being gather-copied here.
/// Gather-copy every sub-`expert_split` tensor into `dst` at its canonical
/// offset (`dst.len()` must equal the head length).
fn fill_layered_head(
    dst: &mut [u8],
    model_dir: &Path,
    manifest: &DgqManifest,
    expert_split: u64,
) -> Result<(), Error> {
    let local_bin = File::open(model_dir.join(&manifest.blob_file))?;
    let local_mmap = unsafe { Mmap::map(&local_bin)? };
    let external = resolve_external_mmaps(model_dir, manifest)?;

    let head_len = dst.len();
    for entry in &manifest.tensors {
        if entry.meta.offset >= expert_split {
            continue;
        }
        let dst_start = blob_offset_usize(entry.meta.offset)?;
        let len = blob_offset_usize(entry.meta.byte_len)?;
        let dst_end = dst_start
            .checked_add(len)
            .ok_or(Error::Runtime("dgq materialize: offset overflow"))?;
        if dst_end > head_len {
            return Err(Error::Runtime(
                "dgq materialize: head tensor extends past expert_split",
            ));
        }
        let src = source_slice(entry, &local_mmap, &external, len)?;
        dst[dst_start..dst_end].copy_from_slice(src);
    }
    Ok(())
}

const HEAD_CACHE_FILE: &str = "model.dgq.head";
const HEAD_CACHE_META: &str = "model.dgq.head.meta";

/// sha256 hex of the manifest JSON bytes — the head cache's identity. Any
/// manifest change (offsets, sources, external pins) invalidates the cache.
fn manifest_sha256_hex(model_dir: &Path) -> Result<String, Error> {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(model_dir.join(MANIFEST_FILE))?;
    Ok(Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// The materialized head as a FILE-BACKED mapping: `model.dgq.head` next to
/// the pack, built once (manifest-hash-validated) and mmap'd read-only ever
/// after. File-backed clean pages are shared across every concurrent
/// instance/process of the same pack and evict-and-re-read under memory
/// pressure — a private anonymous copy per load does neither (each instance
/// paid ~5.5 GiB of swap-backed dirty pages; N test runtimes made that N-fold
/// and stalled the suite behind fault-bound re-reads). Falls back to the
/// anonymous copy when the pack dir is not writable.
fn layered_head_mmap(
    model_dir: &Path,
    manifest: &DgqManifest,
    expert_split: u64,
) -> Result<(Option<File>, Mmap), Error> {
    let head_len = blob_offset_usize(expert_split)?;
    let meta_want = format!("{} {head_len}\n", manifest_sha256_hex(model_dir)?);
    let head_path = model_dir.join(HEAD_CACHE_FILE);
    let meta_path = model_dir.join(HEAD_CACHE_META);
    if std::fs::read_to_string(&meta_path).is_ok_and(|m| m == meta_want)
        && std::fs::metadata(&head_path).is_ok_and(|md| md.len() == head_len as u64)
    {
        let f = File::open(&head_path)?;
        let mmap = unsafe { Mmap::map(&f)? };
        return Ok((Some(f), mmap));
    }
    match build_head_cache(
        model_dir,
        manifest,
        expert_split,
        head_len,
        &meta_want,
        &head_path,
        &meta_path,
    ) {
        Ok(pair) => Ok(pair),
        Err(err) => {
            eprintln!(
                "dgq layered blob: head cache unavailable ({err}); materializing anonymously"
            );
            let mut anon = MmapMut::map_anon(head_len)?;
            fill_layered_head(&mut anon, model_dir, manifest, expert_split)?;
            Ok((None, anon.make_read_only()?))
        }
    }
}

fn build_head_cache(
    model_dir: &Path,
    manifest: &DgqManifest,
    expert_split: u64,
    head_len: usize,
    meta_want: &str,
    head_path: &Path,
    meta_path: &Path,
) -> Result<(Option<File>, Mmap), Error> {
    let pid = std::process::id();
    let tmp_path = model_dir.join(format!("{HEAD_CACHE_FILE}.tmp.{pid}"));
    let tmp = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp_path)?;
    tmp.set_len(head_len as u64)?;
    let mut map = unsafe { MmapMut::map_mut(&tmp)? };
    if let Err(err) = fill_layered_head(&mut map, model_dir, manifest, expert_split) {
        drop(map);
        let _ = std::fs::remove_file(&tmp_path);
        return Err(err);
    }
    map.flush()?;
    drop(map);
    drop(tmp);
    std::fs::rename(&tmp_path, head_path)?;
    // Meta is written LAST: a crash between the renames leaves a stale/absent
    // meta, which just forces a rebuild — never a valid meta over wrong data.
    let meta_tmp = model_dir.join(format!("{HEAD_CACHE_META}.tmp.{pid}"));
    std::fs::write(&meta_tmp, meta_want)?;
    std::fs::rename(&meta_tmp, meta_path)?;
    eprintln!(
        "dgq layered blob: built head cache {} ({:.2} GiB)",
        head_path.display(),
        head_len as f64 / 1073741824.0,
    );
    let f = File::open(head_path)?;
    let mmap = unsafe { Mmap::map(&f)? };
    Ok((Some(f), mmap))
}

/// Quantized linear weight view into a shared blob buffer (PyTorch `[out, in]`).
#[derive(Clone)]
pub struct Q4LinearGpu {
    pub kind: QuantKind,
    pub blob: Arc<DgqGpuBlob>,
    pub byte_offset: u64,
    pub out_dim: usize,
    pub in_dim: usize,
}

impl Q4LinearGpu {
    pub fn from_entry(
        blob: Arc<DgqGpuBlob>,
        offset: u64,
        out_dim: usize,
        in_dim: usize,
        kind: QuantKind,
    ) -> Self {
        Self {
            kind,
            blob,
            byte_offset: offset,
            out_dim,
            in_dim,
        }
    }

    pub fn is_nvfp4(&self) -> bool {
        self.kind == QuantKind::Nvfp4Block
    }

    pub fn quant_kind(&self) -> QuantKind {
        self.kind
    }

    pub fn matrix_byte_len(&self) -> usize {
        match self.kind {
            QuantKind::Q4Block => q4_matrix_bytes(self.out_dim, self.in_dim),
            QuantKind::Q6Block => crate::dgq::layout::q6_matrix_bytes(self.out_dim, self.in_dim),
            QuantKind::Nvfp4Block => nvfp4_matrix_bytes(self.out_dim, self.in_dim),
            _ => panic!("not a block linear"),
        }
    }

    pub fn q4_byte_len(&self) -> usize {
        self.matrix_byte_len()
    }

    pub fn groups_per_row(&self) -> u32 {
        match self.kind {
            QuantKind::Q4Block | QuantKind::Q6Block => self.in_dim.div_ceil(32) as u32,
            QuantKind::Nvfp4Block => self.in_dim.div_ceil(16) as u32,
            _ => panic!("not a block linear"),
        }
    }

    pub fn global_scale_f32(&self) -> f32 {
        if !self.is_nvfp4() {
            return 1.0;
        }
        let ptr = self.blob.host_ptr(self.byte_offset);
        f32::from_le_bytes([
            unsafe { *ptr },
            unsafe { *ptr.add(1) },
            unsafe { *ptr.add(2) },
            unsafe { *ptr.add(3) },
        ])
    }

    pub fn weight_buffer(&self) -> (&ProtocolObject<dyn MTLBuffer>, u64) {
        self.blob.buffer_for(self.byte_offset)
    }

    /// CPU-readable view of Q4 bytes (shared mmap; matches GPU blob layout).
    pub fn src_slice(&self) -> &[u8] {
        let len = self.q4_byte_len();
        unsafe { std::slice::from_raw_parts(self.blob.host_ptr(self.byte_offset), len) }
    }

    /// (MTLBuffer, rebased offset) for GPU binds — region-aware.
    #[allow(dead_code)]
    pub fn mtl_buffer_and_offset(&self) -> (&ProtocolObject<dyn MTLBuffer>, u64) {
        self.blob.buffer_for(self.byte_offset)
    }
}

/// Raw bf16/f32 payload in the blob (norms, router).
#[derive(Clone)]
pub struct RawBlobView {
    pub blob: Arc<DgqGpuBlob>,
    pub byte_offset: u64,
    pub byte_len: u64,
    pub numel: usize,
}

/// Expert stack `[E, out, in]` q4 tensor; slice per expert without copy.
#[derive(Clone)]
pub struct Q4ExpertStackGpu {
    pub kind: QuantKind,
    pub blob: Arc<DgqGpuBlob>,
    pub byte_offset: u64,
    pub out_dim: usize,
    pub in_dim: usize,
}

impl Q4ExpertStackGpu {
    pub fn matrix_stride(&self) -> usize {
        match self.kind {
            QuantKind::Q4Block => q4_matrix_bytes(self.out_dim, self.in_dim),
            QuantKind::Q6Block => crate::dgq::layout::q6_matrix_bytes(self.out_dim, self.in_dim),
            QuantKind::Nvfp4Block => nvfp4_matrix_bytes(self.out_dim, self.in_dim),
            _ => panic!("not a block expert stack"),
        }
    }

    pub fn expert_linear(&self, expert: usize) -> Q4LinearGpu {
        let per = self.matrix_stride();
        Q4LinearGpu {
            kind: self.kind,
            blob: Arc::clone(&self.blob),
            byte_offset: self.byte_offset + expert as u64 * per as u64,
            out_dim: self.out_dim,
            in_dim: self.in_dim,
        }
    }
}

/// Q8 row-major matrix view into the shared blob (embed / lm_head).
#[derive(Clone)]
pub struct Q8LinearGpu {
    pub blob: Arc<DgqGpuBlob>,
    pub byte_offset: u64,
    pub row_offset: usize,
    pub out_dim: usize,
    pub in_dim: usize,
}

impl Q8LinearGpu {
    pub fn from_entry(blob: Arc<DgqGpuBlob>, offset: u64, out_dim: usize, in_dim: usize) -> Self {
        Self {
            blob,
            byte_offset: offset,
            row_offset: 0,
            out_dim,
            in_dim,
        }
    }

    pub fn row_slice(&self, row_offset: usize, out_dim: usize) -> Self {
        assert!(row_offset + out_dim <= self.out_dim);
        Self {
            blob: Arc::clone(&self.blob),
            byte_offset: self.byte_offset,
            row_offset: self.row_offset + row_offset,
            out_dim,
            in_dim: self.in_dim,
        }
    }

    pub fn row_stride(&self) -> usize {
        q8_row_bytes(self.in_dim)
    }

    pub fn weight_buffer(&self) -> (&ProtocolObject<dyn MTLBuffer>, u64) {
        let off = self.byte_offset + self.row_offset as u64 * self.row_stride() as u64;
        self.blob.buffer_for(off)
    }

    /// Decode per-row bf16 scale at row `r` (CPU reference for q8 row layout).
    pub fn row_scale_f32(&self, row: usize) -> f32 {
        let stride = self.row_stride();
        let (_, base) = self.weight_buffer();
        let byte = blob_offset_for_mtl(base) + row * stride;
        let ptr = unsafe { self.blob.buffer.contents().as_ptr().add(byte) as *const u8 };
        let bits = u16::from_le_bytes([unsafe { *ptr }, unsafe { *ptr.add(1) }]);
        bf16_bits_to_f32(bits)
    }
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    if exp == 0 {
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let val = (mant as f32) * 2f32.powi(-24);
        return if sign == 1 { -val } else { val };
    }
    if exp == 0x1f {
        return if mant == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        };
    }
    f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
}

/// Histogram q8 row scales for chunk; also show scales the old f32_q8_linear kernel would read.
pub fn log_q8_chunk_scale_histogram(w: &Q8LinearGpu, k_dim: usize, hidden: usize) {
    let rows = w.out_dim;
    let mut ok = 0usize;
    let mut huge = 0usize;
    let mut max_scale = 0.0f32;
    for r in 0..rows {
        let s = w.row_scale_f32(r);
        max_scale = max_scale.max(s.abs());
        if s.is_finite() && s.abs() < 1e3 {
            ok += 1;
        } else {
            huge += 1;
        }
    }
    eprintln!(
        "q8 chunk scales (correct row_stride={}): rows={rows} ok={ok} huge={huge} max_abs={max_scale:.6}",
        w.row_stride()
    );

    // Old kernel: row index = hidden col, row_stride = 2 + k_dim (vocab chunk size).
    let wrong_stride = 2 + k_dim;
    let wrong_rows = hidden.min(rows);
    let mut wrong_huge = 0usize;
    let mut wrong_max = 0.0f32;
    let (_, base) = w.weight_buffer();
    for col in 0..wrong_rows {
        let byte = base as usize + col * wrong_stride;
        let ptr = unsafe { w.blob.buffer.contents().as_ptr().add(byte) as *const u8 };
        let bits = u16::from_le_bytes([unsafe { *ptr }, unsafe { *ptr.add(1) }]);
        let s = bf16_bits_to_f32(bits);
        wrong_max = wrong_max.max(s.abs());
        if !s.is_finite() || s.abs() >= 1e3 {
            wrong_huge += 1;
        }
    }
    eprintln!(
        "q8 chunk scales (old f32_q8_linear indexing, stride={wrong_stride}): sampled={wrong_rows} huge={wrong_huge} max_abs={wrong_max:.6e}"
    );
}

pub fn parse_kind(s: &str) -> Result<QuantKind, Error> {
    match s {
        "q4_block" => Ok(QuantKind::Q4Block),
        "q6_block" => Ok(QuantKind::Q6Block),
        "nvfp4_block" => Ok(QuantKind::Nvfp4Block),
        "q8_row" => Ok(QuantKind::Q8Row),
        "raw" => Ok(QuantKind::Raw),
        _ => Err(Error::Format("unknown dgq tensor kind")),
    }
}

pub fn load_q8_linear(
    store: &DgqStore,
    blob: Arc<DgqGpuBlob>,
    name: &str,
) -> Result<Q8LinearGpu, Error> {
    let entry = store
        .get_entry(name)
        .ok_or_else(|| Error::NotFound(name.to_string()))?;
    if parse_kind(&entry.meta.kind)? != QuantKind::Q8Row {
        return Err(Error::Format("expected q8_row linear"));
    }
    if entry.meta.shape.len() != 2 {
        return Err(Error::Format("q8 linear expects rank 2"));
    }
    let out = entry.meta.shape[0] as usize;
    let inp = entry.meta.shape[1] as usize;
    Ok(Q8LinearGpu::from_entry(blob, entry.meta.offset, out, inp))
}

pub fn load_block_linear(
    store: &DgqStore,
    blob: Arc<DgqGpuBlob>,
    name: &str,
) -> Result<Q4LinearGpu, Error> {
    let entry = store
        .get_entry(name)
        .ok_or_else(|| Error::NotFound(name.to_string()))?;
    let kind = parse_kind(&entry.meta.kind)?;
    if kind != QuantKind::Q4Block && kind != QuantKind::Nvfp4Block {
        return Err(Error::Format("expected q4_block or nvfp4_block linear"));
    }
    if entry.meta.shape.len() != 2 {
        return Err(Error::Runtime("block linear expects rank 2"));
    }
    let out = entry.meta.shape[0] as usize;
    let inp = entry.meta.shape[1] as usize;
    Ok(Q4LinearGpu::from_entry(
        blob,
        entry.meta.offset,
        out,
        inp,
        kind,
    ))
}

pub fn load_block_expert_stack(
    store: &DgqStore,
    blob: Arc<DgqGpuBlob>,
    name: &str,
) -> Result<Q4ExpertStackGpu, Error> {
    let entry = store
        .get_entry(name)
        .ok_or_else(|| Error::NotFound(name.to_string()))?;
    let kind = parse_kind(&entry.meta.kind)?;
    if kind != QuantKind::Q4Block && kind != QuantKind::Q6Block && kind != QuantKind::Nvfp4Block {
        return Err(Error::Format(
            "expected q4_block/q6_block/nvfp4_block expert stack",
        ));
    }
    if entry.meta.shape.len() != 3 {
        return Err(Error::Runtime("block expert expects rank 3"));
    }
    Ok(Q4ExpertStackGpu {
        kind,
        blob,
        byte_offset: entry.meta.offset,
        out_dim: entry.meta.shape[1] as usize,
        in_dim: entry.meta.shape[2] as usize,
    })
}

pub fn load_raw_view(
    store: &DgqStore,
    blob: Arc<DgqGpuBlob>,
    name: &str,
) -> Result<RawBlobView, Error> {
    let entry = store
        .get_entry(name)
        .ok_or_else(|| Error::NotFound(name.to_string()))?;
    if parse_kind(&entry.meta.kind)? != QuantKind::Raw {
        return Err(Error::Format("expected raw tensor"));
    }
    let numel: usize = entry.meta.shape.iter().product::<i64>() as usize;
    Ok(RawBlobView {
        blob,
        byte_offset: entry.meta.offset,
        byte_len: entry.meta.byte_len,
        numel,
    })
}

#[cfg(all(test, target_os = "macos"))]
mod q4_gpu_tests {
    use super::*;
    use crate::metal::batch::GpuBatch;
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use crate::metal::linear::f32_q4_linear_gpu_bufs;

    #[test]
    fn q8_gpu_linear_matches_cpu_dequant() {
        let dgq_dir_buf = match crate::shaders::test_util::dgq_model_dir() {
            Some(d) => d,
            None => std::path::PathBuf::from("/tmp/quantized-weights"),
        };
        let dgq_dir = dgq_dir_buf.as_path();
        if !dgq_dir.join("model.dgq.json").exists() {
            eprintln!("skip: /tmp/quantized-weights missing");
            return;
        }
        let store = DgqStore::open(dgq_dir).expect("open dgq");
        let ctx = MetalContext::new().expect("metal");
        let blob = DgqGpuBlob::from_store(&store, &ctx.device).expect("blob");
        let q8 = load_q8_linear(
            &store,
            Arc::clone(&blob),
            "model.decoder.self_conditioning.down_proj.weight",
        )
        .expect("q8 embed");
        let q8 = q8.row_slice(100, 8);

        let m = 4usize;
        let k = q8.in_dim;
        let n = q8.out_dim;
        let a: Vec<f32> = (0..m * k)
            .map(|i| (i as f32 * 0.001 - 0.5).sin() * 0.01)
            .collect();

        let f32_w = store
            .tensor_f32("model.decoder.self_conditioning.down_proj.weight")
            .expect("dequant");
        let row_off = 100usize;
        let mut cpu_out = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[row * k + p] * f32_w[(row_off + col) * k + p];
                }
                cpu_out[row * n + col] = sum;
            }
        }

        let mut pool = BufferPool::new();
        let prod = crate::shaders::variant::KernelVariant::PRODUCTION;
        let pipeline =
            crate::shaders::gemm_q8_linear_f32::pipeline_for(&ctx, prod).expect("pipeline");
        let mut batch = GpuBatch::begin_with_telemetry(&ctx.queue, &mut pool, &ctx.device, None)
            .expect("batch");
        let buf_a = batch.alloc_f32(&a).expect("a");
        let buf_c = crate::metal::linear::f32_q8_linear_gpu_bufs(
            &mut batch, &pipeline, &buf_a, &q8, m, k, n,
        )
        .expect("gemm");
        let mut gpu_out = vec![0.0f32; m * n];
        batch.register_read(buf_c, &mut gpu_out);
        batch.end().expect("end");

        let mut max_err = 0.0f32;
        for (a, b) in cpu_out.iter().zip(gpu_out.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        eprintln!("q8 gpu vs cpu dequant max_err={max_err:.6}");
        assert!(max_err < 0.05, "max_err={max_err}");
    }

    #[test]
    fn nvfp4_gpu_linear_matches_cpu_dequant() {
        let dgq_dir = std::path::Path::new("/tmp/nvfp4-weights");
        if !dgq_dir.join("model.dgq.json").exists() {
            eprintln!("skip: /tmp/nvfp4-weights missing");
            return;
        }
        let store = DgqStore::open(dgq_dir).expect("open dgq");
        let ctx = MetalContext::new().expect("metal");
        let blob = DgqGpuBlob::from_store(&store, &ctx.device).expect("blob");
        let q4 = load_block_linear(
            &store,
            Arc::clone(&blob),
            "model.decoder.layers.0.self_attn.q_proj.weight",
        )
        .expect("nvfp4 view");
        assert!(q4.is_nvfp4());

        let m = 4usize;
        let k = q4.in_dim;
        let n = q4.out_dim;
        let a: Vec<f32> = (0..m * k)
            .map(|i| (i as f32 * 0.001 - 0.5).sin() * 0.01)
            .collect();

        let f32_w = store
            .tensor_f32("model.decoder.layers.0.self_attn.q_proj.weight")
            .expect("dequant");
        let mut cpu_out = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[row * k + p] * f32_w[col * k + p];
                }
                cpu_out[row * n + col] = sum;
            }
        }

        let mut pool = BufferPool::new();
        let prod = crate::shaders::variant::KernelVariant::PRODUCTION;
        let q4_pipeline = crate::shaders::gemm_linear_f32::pipeline_for(
            &ctx,
            crate::shaders::QuantFormat::Q4Affine,
            prod,
        )
        .expect("pipeline");
        let nvfp4_pipeline = crate::shaders::gemm_linear_f32::pipeline_for(
            &ctx,
            crate::shaders::QuantFormat::NvFp4,
            prod,
        )
        .expect("nvfp4 pipeline");
        let mut batch = GpuBatch::begin_with_telemetry(&ctx.queue, &mut pool, &ctx.device, None)
            .expect("batch");
        let buf_a = batch.alloc_f32(&a).expect("a");
        let buf_c = f32_q4_linear_gpu_bufs(
            &mut batch,
            &q4_pipeline,
            &nvfp4_pipeline,
            &buf_a,
            &q4,
            m,
            k,
            n,
        )
        .expect("gemm");
        let mut gpu_out = vec![0.0f32; m * n];
        batch.register_read(buf_c, &mut gpu_out);
        batch.end().expect("end");

        let mut max_err = 0.0f32;
        for (a, b) in cpu_out.iter().zip(gpu_out.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        eprintln!("nvfp4 gpu vs cpu dequant max_err={max_err:.6}");
        assert!(max_err < 0.08, "max_err={max_err}");
    }

    /// Encoder prefill uses `seq_len` batched linears (e.g. 22–25 prompt tokens), not M=4.
    #[test]
    fn nvfp4_gpu_linear_prefill_seq_len() {
        let dgq_dir = std::path::Path::new("/tmp/nvfp4-weights");
        if !dgq_dir.join("model.dgq.json").exists() {
            eprintln!("skip: /tmp/nvfp4-weights missing");
            return;
        }
        let store = DgqStore::open(dgq_dir).expect("open dgq");
        let ctx = MetalContext::new().expect("metal");
        let blob = DgqGpuBlob::from_store(&store, &ctx.device).expect("blob");
        let tensor = "model.decoder.layers.1.self_attn.k_proj.weight";
        let q4 = load_block_linear(&store, Arc::clone(&blob), tensor).expect("nvfp4 view");
        assert!(q4.is_nvfp4());

        let m = 25usize;
        let k = q4.in_dim;
        let n = q4.out_dim;
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32 * 0.0009).sin() * 0.03 + (i % 17) as f32 * 0.0001))
            .collect();
        let f32_w = store.tensor_f32(tensor).expect("dequant");
        let mut cpu_out = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[row * k + p] * f32_w[col * k + p];
                }
                cpu_out[row * n + col] = sum;
            }
        }

        let prod = crate::shaders::variant::KernelVariant::PRODUCTION;
        let q4_pipeline = crate::shaders::gemm_linear_f32::pipeline_for(
            &ctx,
            crate::shaders::QuantFormat::Q4Affine,
            prod,
        )
        .expect("pipeline");
        let nvfp4_pipeline = crate::shaders::gemm_linear_f32::pipeline_for(
            &ctx,
            crate::shaders::QuantFormat::NvFp4,
            prod,
        )
        .expect("nvfp4 pipeline");
        let mut pool = BufferPool::new();
        let mut batch = GpuBatch::begin_with_telemetry(&ctx.queue, &mut pool, &ctx.device, None)
            .expect("batch");
        let buf_a = batch.alloc_f32(&a).expect("a");
        let buf_c = f32_q4_linear_gpu_bufs(
            &mut batch,
            &q4_pipeline,
            &nvfp4_pipeline,
            &buf_a,
            &q4,
            m,
            k,
            n,
        )
        .expect("gemm");
        let mut gpu_out = vec![0.0f32; m * n];
        batch.register_read(buf_c, &mut gpu_out);
        batch.end().expect("end");

        let mut max_err = 0.0f32;
        let mut nan = 0usize;
        for (a, b) in cpu_out.iter().zip(gpu_out.iter()) {
            if !b.is_finite() {
                nan += 1;
            }
            max_err = max_err.max((a - b).abs());
        }
        eprintln!("nvfp4 m={m} linear max_err={max_err:.6} nan={nan}");
        assert_eq!(nan, 0, "gpu linear produced {nan} non-finite outputs");
        assert!(max_err < 0.08, "max_err={max_err}");
    }

    fn grouped_stats(cpu: &[f32], gpu: &[f32]) -> (f32, f64) {
        assert_eq!(cpu.len(), gpu.len());
        let mut max_err = 0.0f32;
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (a, b) in cpu.iter().zip(gpu.iter()) {
            let err = (*a - *b).abs();
            if err.is_finite() {
                max_err = max_err.max(err);
            }
            if a.is_finite() && b.is_finite() {
                dot += *a as f64 * *b as f64;
                na += *a as f64 * *a as f64;
                nb += *b as f64 * *b as f64;
            }
        }
        let cos = if na > 0.0 && nb > 0.0 {
            dot / (na.sqrt() * nb.sqrt())
        } else {
            0.0
        };
        (max_err, cos)
    }

    /// Converter writes one 4-byte FP32 global scale per expert matrix; kernel stride must match.
    #[test]
    fn nvfp4_expert_stack_per_expert_header_on_disk() {
        use crate::dgq::layout::nvfp4_matrix_bytes;

        let dgq_dir = std::path::Path::new("/tmp/nvfp4-weights");
        if !dgq_dir.join("model.dgq.json").exists() {
            eprintln!("skip: /tmp/nvfp4-weights missing");
            return;
        }
        let store = DgqStore::open(dgq_dir).expect("open dgq");
        let name = "model.decoder.layers.0.experts.gate_up_proj";
        let entry = store.get_entry(name).expect("gate_up");
        let out_dim = entry.meta.shape[1] as usize;
        let in_dim = entry.meta.shape[2] as usize;
        let experts = entry.meta.shape[0] as usize;
        let per_expert = nvfp4_matrix_bytes(out_dim, in_dim);
        assert_eq!(entry.meta.byte_len as usize, experts * per_expert);

        let blob_path = dgq_dir.join(crate::dgq::layout::BLOB_FILE);
        let blob = std::fs::read(&blob_path).expect("read blob");
        let base = crate::dgq::layout::blob_offset_usize(entry.meta.offset).expect("tensor offset");
        for e in [0usize, 1, 2, experts - 1] {
            let off = base + e * per_expert;
            let gscale = f32::from_le_bytes(blob[off..off + 4].try_into().expect("header"));
            eprintln!("expert {e} gscale={gscale:.6} off={off}");
            assert!(
                gscale.is_finite() && gscale > 0.0,
                "expert {e} missing per-expert header at off={off}"
            );
        }
        // If stride omitted the 4-byte header, E1 would start 4 bytes early inside E0 body.
        let wrong_e1 = base + per_expert - 4;
        let bogus = f32::from_le_bytes(blob[wrong_e1..wrong_e1 + 4].try_into().expect("tail"));
        let actual_e1 = f32::from_le_bytes(
            blob[base + per_expert..base + per_expert + 4]
                .try_into()
                .expect("e1"),
        );
        eprintln!("misaligned E1 gscale (no-header stride)={bogus:.6} actual E1={actual_e1:.6}");
        assert!((actual_e1 - 1.0).abs() < 1e-5);
        assert!(
            (bogus - 1.0).abs() > 1e-3,
            "tail bytes must not look like gscale=1"
        );
    }

    /// `.dgq` blob offsets exceed 4 GiB; grouped MoE jobs on late layers must use u64/ulong paths.
    #[test]
    fn dgq_blob_offset_width_audit() {
        let dgq_dir = std::path::Path::new("/tmp/nvfp4-weights");
        if !dgq_dir.join("model.dgq.json").exists() {
            eprintln!("skip: /tmp/nvfp4-weights missing");
            return;
        }
        use crate::dgq::layout::nvfp4_matrix_bytes;
        use crate::metal::step_kernel::{HID, MOE_FF, build_layout, build_offsets_from_store};

        let store = DgqStore::open(dgq_dir).expect("open dgq");
        let offsets = build_offsets_from_store(&store);
        let layout = build_layout(&offsets, 512);
        let gu_stride = nvfp4_matrix_bytes(MOE_FF as usize * 2, HID) as u64;
        let dn_stride = nvfp4_matrix_bytes(HID, MOE_FF as usize) as u64;

        let mut max_end = 0u64;
        for entry in store.tensor_entries() {
            max_end = max_end.max(entry.meta.offset + entry.meta.byte_len);
        }
        eprintln!(
            "dgq blob end={max_end} bytes ({:.3} GiB) exceeds u32={}",
            max_end as f64 / (1024.0_f64.powi(3)),
            max_end > u32::MAX as u64
        );
        assert!(max_end > u32::MAX as u64, "fixture should exceed 4 GiB");
        assert_eq!(
            std::mem::size_of::<usize>(),
            8,
            "dgq metal path requires 64-bit host"
        );

        for entry in store.tensor_entries() {
            let off = crate::dgq::layout::blob_offset_usize(entry.meta.offset)
                .expect("manifest tensor offset");
            let end = off
                + crate::dgq::layout::blob_offset_usize(entry.meta.byte_len)
                    .expect("manifest tensor byte_len");
            assert!(
                end <= store.blob_bytes() as usize,
                "tensor {} OOB",
                entry.name
            );
        }
        assert!(
            store
                .tensor_entries()
                .iter()
                .any(|e| e.meta.offset > u32::MAX as u64),
            "fixture should contain tensors past 4 GiB"
        );
        let late = store
            .tensor_entries()
            .iter()
            .filter(|e| e.name.contains("layers.29.experts"))
            .count();
        assert!(late >= 2, "expected L29 expert tensors in fixture");

        let l29 = &layout.layers[29];
        let gate127 = l29.experts_gate_up + 127 * gu_stride;
        let down127 = l29.experts_down + 127 * dn_stride;
        for (label, off) in [("gate127", gate127), ("down127", down127)] {
            let idx = crate::dgq::layout::blob_offset_usize(off).expect(label);
            assert!(
                idx > u32::MAX as usize,
                "{label} must exceed u32 after conversion"
            );
        }
        eprintln!(
            "L29 E127 gate_off={gate127} down_off={down127} (>u32: gate={} down={})",
            gate127 > u32::MAX as u64,
            down127 > u32::MAX as u64
        );
        assert!(gate127 > u32::MAX as u64);
        assert!(down127 > u32::MAX as u64);

        assert_eq!(std::mem::size_of::<crate::metal::BlockGroupedJob>(), 16);
    }

    /// Real nvfp4 MoE gate/up and down grouped GEMM vs CPU oracle (includes >4 GiB expert offsets).
    #[test]
    fn nvfp4_block_grouped_real_moe_weights_match_cpu() {
        use crate::metal::BlockGroupedJob;
        use crate::metal::step_kernel::{HID, MOE_FF, build_layout, build_offsets_from_store};
        use crate::shaders::QuantFormat;
        use crate::shaders::cpu::gemm_linear_grouped::gemm_linear_grouped_cpu;
        use crate::shaders::gemm_block_grouped::{BlobGroupedParams, gpu_on_blob};

        let dgq_dir = std::path::Path::new("/tmp/nvfp4-weights");
        if !dgq_dir.join("model.dgq.json").exists() {
            eprintln!("skip: /tmp/nvfp4-weights missing");
            return;
        }
        let store = DgqStore::open(dgq_dir).expect("open dgq");
        let ctx = MetalContext::new().expect("metal");
        let blob = DgqGpuBlob::from_store(&store, &ctx.device).expect("blob");
        let offsets = build_offsets_from_store(&store);
        let layout = build_layout(&offsets, 512);
        let layer = 29usize;
        let l = &layout.layers[layer];
        let hidden = HID;
        let moe_ff = MOE_FF as usize;
        let gu_stride = crate::dgq::layout::nvfp4_matrix_bytes(moe_ff * 2, hidden) as u64;
        let dn_stride = crate::dgq::layout::nvfp4_matrix_bytes(hidden, moe_ff) as u64;
        let gu_gpr = (hidden as u32).div_ceil(16);
        let dn_gpr = (moe_ff as u32).div_ceil(16);

        let blob_bytes = unsafe {
            std::slice::from_raw_parts(blob.buffer.contents().as_ptr().cast::<u8>(), blob.len)
        };
        let variant = crate::shaders::variant::KernelVariant::PRODUCTION;

        struct Case {
            label: &'static str,
            k: usize,
            n: usize,
            experts: [usize; 2],
            w_base: u64,
            stride: u64,
            gpr: u32,
        }

        let cases = [
            Case {
                label: "gate_up L0",
                k: hidden,
                n: moe_ff * 2,
                experts: [34, 77],
                w_base: layout.layers[0].experts_gate_up,
                stride: gu_stride,
                gpr: gu_gpr,
            },
            Case {
                label: "gate_up L29 (>4GiB)",
                k: hidden,
                n: moe_ff * 2,
                experts: [34, 127],
                w_base: l.experts_gate_up,
                stride: gu_stride,
                gpr: gu_gpr,
            },
            Case {
                label: "down L29 (>4GiB)",
                k: moe_ff,
                n: hidden,
                experts: [77, 127],
                w_base: l.experts_down,
                stride: dn_stride,
                gpr: dn_gpr,
            },
        ];

        for case in cases {
            let jobs = [
                BlockGroupedJob {
                    w_byte_off: case.w_base + case.experts[0] as u64 * case.stride,
                    groups_per_row: case.gpr,
                    _pad: 0,
                },
                BlockGroupedJob {
                    w_byte_off: case.w_base + case.experts[1] as u64 * case.stride,
                    groups_per_row: case.gpr,
                    _pad: 0,
                },
            ];
            eprintln!(
                "{} job offs [{}, {}] (>u32: [{}, {}])",
                case.label,
                jobs[0].w_byte_off,
                jobs[1].w_byte_off,
                jobs[0].w_byte_off > u32::MAX as u64,
                jobs[1].w_byte_off > u32::MAX as u64
            );

            let row_starts = [0u32, 50, 100];
            let total_m = 100usize;
            let mut a = vec![0.0f32; total_m * case.k];
            for (row, slot) in a.chunks_mut(case.k).enumerate() {
                for (i, v) in slot.iter_mut().enumerate() {
                    *v = ((row * 17 + i) as f32 * 0.0007).sin() * 0.02;
                }
            }

            let cpu = gemm_linear_grouped_cpu(
                &a,
                total_m,
                case.k,
                case.n,
                blob_bytes,
                &jobs,
                &row_starts,
                QuantFormat::NvFp4,
            );
            let params = BlobGroupedParams {
                blob: &blob.buffer,
                a: &a,
                jobs: &jobs,
                row_starts: &row_starts,
                k: case.k,
                n: case.n,
                format: QuantFormat::NvFp4,
            };
            let gpu = gpu_on_blob(&params, variant).expect("gpu block grouped");
            let cpu_max = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let gpu_max = gpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let (max_err, cos) = grouped_stats(&cpu, &gpu);
            eprintln!(
                "{}: cpu_max={cpu_max:.6} gpu_max={gpu_max:.6} max_err={max_err:.6} cos={cos:.6}",
                case.label
            );
            assert!(cpu_max > 1e-4, "{} cpu oracle produced zeros", case.label);
            assert!(
                gpu_max > 1e-4,
                "{} gpu block grouped produced zeros",
                case.label
            );
            assert!(max_err < 0.08, "{} max_err={max_err}", case.label);
            assert!(cos > 0.999, "{} cos={cos}", case.label);
        }
    }
}
