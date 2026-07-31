//! Zero-copy `.dgq` blob → `MTLBuffer` and per-tensor GPU views.

use crate::Error;
use crate::dgq::DgqStore;
use crate::dgq::layout::{
    DgqManifest, DgqTensorEntry, MANIFEST_FILE, QuantKind, TensorSource, blob_offset_for_mtl,
    blob_offset_usize, dgq_version_supported, nvfp4_matrix_bytes, q4_matrix_bytes, q8_row_bytes,
};
use memmap2::{Mmap, MmapMut};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::collections::HashMap;
use std::ffi::c_void;
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

/// Apple Silicon page size — both the VM granule `MAP_FIXED`/`mmap` offsets
/// must respect and the "page-congruent" unit the head-splice plan and
/// writer agree on.
const HEAD_SPLICE_PAGE: u64 = 16384;

/// Keeps mmap + file alive while the GPU buffer references the mapping.
/// `_file`/`_mmap` back region 1 (head); `_tail_file`/`_tail_mmap` are only
/// `Some` when region 2 (experts) is backed by a DIFFERENT mapping than
/// region 1 — the layered split-source path below, where the (large) expert
/// tail is wrapped straight off this pack's own blob file instead of being
/// gather-copied. `_file` is `None` for an anonymous (materialized) mapping
/// or a VA-spliced head (see `HeadRegion`).
pub struct DgqGpuBlob {
    _file: Option<File>,
    _mmap: HeadRegion,
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

/// Minimum canonical-offset alignment every `.dgq` tensor entry has always
/// had, unconditionally, since before layered packs existed: several
/// production kernels reinterpret `blob + w_off` as a typed pointer (e.g.
/// `gemm_rowk.metal`: `device const ushort *w = (device const ushort
/// *)(blob + w_off)`, then indexed) — correctness of that read depends on
/// `w_off` being sufficiently aligned, and 64 bytes is the value every
/// working pack (self-contained or layered) has always used via the writer's
/// unconditional `align_offset`.
const TENSOR_OFFSET_ALIGN: u64 = 64;

/// Load-time tripwire for the failure class a byte-content check CANNOT see:
/// a `w_off` that is byte-CORRECT (right tensor, right value once read) but
/// insufficiently aligned for the typed-pointer reads several kernels do.
/// This is cheap (manifest-only, no I/O) and unconditional — it runs for
/// every pack, not just layered ones, and would have caught, at load time
/// with a clear diagnostic instead of silently wrong generation, the writer
/// regression `writer_plan_integration_tests` predates (a VA-splice writer
/// draft dropped per-tensor alignment inside a shard-run to mirror the
/// source file's zero-gap layout — safe for byte CONTENT, unsafe for GPU
/// reads; `gpu_buffer_matches_store_for_every_tensor`'s host_ptr-vs-
/// `DgqStore` comparison does NOT catch this, because both sides read
/// through untyped byte pointers with no alignment requirement of their
/// own — only the actual GPU kernel's typed reinterpret-cast cares).
fn assert_tensor_offset_alignment(manifest: &DgqManifest) -> Result<(), Error> {
    let offenders: Vec<&str> = manifest
        .tensors
        .iter()
        .filter(|t| !t.meta.offset.is_multiple_of(TENSOR_OFFSET_ALIGN))
        .map(|t| t.name.as_str())
        .collect();
    if offenders.is_empty() {
        return Ok(());
    }
    eprintln!(
        "dgq: {} tensor(s) have a canonical offset not aligned to {TENSOR_OFFSET_ALIGN} bytes \
         — this pack is unsafe to load (GPU kernels read weight bytes through a typed pointer \
         cast at that offset); first few: {:?}",
        offenders.len(),
        &offenders[..offenders.len().min(10)]
    );
    Err(Error::Format(
        "dgq: manifest has misaligned tensor offset(s) — refusing to load",
    ))
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
        assert_tensor_offset_alignment(&manifest)?;
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

    /// Layered split-source path: head served via VA-splice when the plan
    /// allows it (`build_head_splice`), expert tail wrapped directly off
    /// `manifest.blob_file` at `local_expert_split` (page-aligned, so the
    /// pointer arithmetic mirrors the self-contained split-buffer wrap in
    /// [`Self::wrap_mmap`]). On ANY failure in the splice attempt
    /// (unsatisfiable plan, OS/Metal step failure) falls back to
    /// materializing ONLY the head into a private anonymous mapping — the
    /// (large) expert tail STAYS the cheap file-backed no-copy mmap either
    /// way. This is deliberately NOT the whole-blob `materialize_layered_blob`
    /// fallback: that path re-gathers the entire canonical blob (head AND
    /// the multi-GiB expert tail) into anonymous memory, which is fine for
    /// its OTHER caller (a manifest missing `local_expert_split` entirely,
    /// where there's no cheap tail wrap to preserve) but would silently
    /// double the resident/dirty memory of every load whenever splicing
    /// fails — measured live: with zero splice coverage on the shipped
    /// model (see ARCHITECTURE.md §8.1), routing here through the whole-blob
    /// path made every model-gated test pay a private ~18.84 GiB anonymous
    /// buffer instead of a shared ~13.3 GiB mmap, and two such loads
    /// overlapping (membudget permits it; each looks like one "load") OOM-
    /// killed the process the full suite ran in. Correctness never depends
    /// on the splice succeeding either way — only which fallback shape a
    /// failure takes.
    fn from_store_layered_split(
        model_dir: &Path,
        manifest: &DgqManifest,
        expert_split: u64,
        local_expert_split: u64,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Result<Arc<Self>, Error> {
        let head = match build_head_splice(model_dir, manifest, expert_split) {
            Ok(spliced) => HeadRegion::Spliced(spliced),
            Err(err) => {
                eprintln!(
                    "dgq layered blob: head splice unavailable ({err}); materializing just the \
                     head anonymously (expert tail stays the cheap file-backed no-copy mmap) — \
                     correctness never depends on splice"
                );
                HeadRegion::Mmap(materialize_layered_head_only(model_dir, manifest, expert_split)?)
            }
        };
        Self::finish_layered_split(model_dir, manifest, expert_split, local_expert_split, head, device)
    }

    fn finish_layered_split(
        model_dir: &Path,
        manifest: &DgqManifest,
        expert_split: u64,
        local_expert_split: u64,
        head: HeadRegion,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Result<Arc<Self>, Error> {
        let head_len = head.len();

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
        let head_ptr =
            NonNull::new(head.as_ptr() as *mut c_void).ok_or(Error::Runtime("dgq mmap null"))?;
        let buffer = unsafe {
            device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    head_ptr,
                    head_len,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
                .ok_or(Error::Gpu("dgq gpu blob region1 (head) alloc failed"))?
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
            _file: None,
            _mmap: head,
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
                _mmap: HeadRegion::Mmap(mmap),
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
            _mmap: HeadRegion::Mmap(mmap),
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

/// Gather-copy ONLY the canonical HEAD region `[0, expert_split)` into one
/// anonymous mapping — the fallback `from_store_layered_split` uses when
/// `build_head_splice` can't prove a splice safe. Deliberately narrower than
/// `materialize_layered_blob`: the (large) expert tail is NEVER touched
/// here, because the caller always keeps it as the existing cheap
/// file-backed `local_expert_split` mmap regardless of whether the head
/// splices or falls back — gathering the tail too would needlessly double
/// the anonymous (dirty, non-shared, non-evictable-and-re-readable)
/// resident memory of every load whenever the head can't be spliced.
fn materialize_layered_head_only(
    model_dir: &Path,
    manifest: &DgqManifest,
    expert_split: u64,
) -> Result<Mmap, Error> {
    let head_len = blob_offset_usize(expert_split)?;
    let local_bin = File::open(model_dir.join(&manifest.blob_file))?;
    let local_mmap = unsafe { Mmap::map(&local_bin)? };
    let external = resolve_external_mmaps(model_dir, manifest)?;

    let mut anon = MmapMut::map_anon(head_len)?;
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
        anon[dst_start..dst_end].copy_from_slice(src);
    }
    drop(external);
    drop(local_mmap);
    Ok(anon.make_read_only()?)
}

// ---------------------------------------------------------------------------
// VA-splice: the layered-head mechanism (replaces the old model.dgq.head
// gather-cache). See `plan_head_splice` for the pure planning logic and
// `build_head_splice` for the unsafe execution; `src/metal/va_splice_probe.rs`
// is the canary that verified Metal accepts a no-copy `MTLBuffer` spanning a
// VA range spliced from multiple distinct mappings.
// ---------------------------------------------------------------------------

/// Backing for `DgqGpuBlob`'s region-1 (head) mapping: either a single
/// `memmap2::Mmap` (self-contained pack, or the full-materialize fallback),
/// or a VA-spliced reservation stitched from multiple `MAP_FIXED` sub-
/// mappings (`HeadSplice`, the layered-pack fast path). Both expose ONE base
/// pointer + length for the WHOLE head range — every consumer (`host_ptr`,
/// the no-copy `MTLBuffer` wrap) already treats "the head mapping" as one
/// opaque contiguous region, so this is a drop-in swap.
enum HeadRegion {
    Mmap(Mmap),
    Spliced(HeadSplice),
}

impl HeadRegion {
    fn as_ptr(&self) -> *const u8 {
        match self {
            Self::Mmap(m) => m.as_ptr(),
            Self::Spliced(s) => s.base.as_ptr().cast(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Mmap(m) => m.len(),
            Self::Spliced(s) => s.len,
        }
    }
}

/// RAII VA reservation stitched from multiple `MAP_FIXED` sub-mappings
/// (files + anon fill) — see `build_head_splice`. `Drop` tears the WHOLE
/// range down in one `munmap`, which correctly unmaps every sub-mapping
/// inside it regardless of how many distinct `MAP_FIXED` calls built it (the
/// kernel tracks VMA splits internally; one `munmap(addr, len)` over a range
/// spanning several VMAs unmaps all of them).
struct HeadSplice {
    base: NonNull<c_void>,
    len: usize,
}

// SAFETY: `base` points at read-only memory after `build_head_splice`
// finishes (every sub-mapping is either a read-only file mapping, or an
// anon mapping we `mprotect`ed to PROT_READ once done writing) — no
// interior mutability, no aliasing writer, same reasoning `memmap2::Mmap`
// itself relies on for its own Send+Sync impl.
unsafe impl Send for HeadSplice {}
unsafe impl Sync for HeadSplice {}

impl Drop for HeadSplice {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.base.as_ptr(), self.len);
        }
    }
}

/// One maximal run of head tensors safe to `MAP_FIXED`-splice straight off
/// an external HF shard: contiguous in BOTH the canonical (manifest
/// `offset`) and source-file address spaces, with a page-congruent start
/// (`canonical_start ≡ file_start mod PAGE`) — see `plan_head_splice`.
#[derive(Debug, Clone)]
struct SpliceRun {
    file_key: String,
    canonical_start: u64,
    canonical_end: u64,
    file_start: u64,
    tensor_names: Vec<String>,
}

/// The loader's plan for the head region `[0, expert_split)`: which maximal
/// runs are safe to splice, and which entries fall back to a plain memcpy
/// into the reservation's anonymous backdrop (every `Local`-sourced entry —
/// by design, never spliced, see the writer's "local head bytes... grouped
/// into their own run" — plus any `External` entry this loader can't PROVE
/// is safe: non-congruent, or whose rounded page range would clobber a
/// neighboring tensor's real bytes).
#[derive(Debug, Default)]
struct SplicePlan {
    splice_runs: Vec<SpliceRun>,
    anon_entries: Vec<String>,
}

/// Pure planning logic (no I/O, no unsafe): derive the splice plan from the
/// manifest's own entries. Never trusts the writer's intent — every run is
/// independently re-validated for (a) page congruence and (b) NOT
/// overlapping, once rounded to page boundaries, any OTHER head tensor's
/// true declared byte range. (b) is what makes this safe regardless of what
/// the writer did: even a run that IS page-congruent gets rejected (falls
/// back to anon-fill for its members) if splicing it would read one of its
/// rounding fringes over a neighbor's real bytes. Two runs' fringes may
/// still overlap EACH OTHER — harmless, since no declared tensor lives in
/// either fringe once both pass their own neighbor check.
fn plan_head_splice(manifest: &DgqManifest, expert_split: u64) -> Result<SplicePlan, Error> {
    const PAGE: u64 = HEAD_SPLICE_PAGE;

    let mut head: Vec<&DgqTensorEntry> = manifest
        .tensors
        .iter()
        .filter(|t| t.meta.offset < expert_split)
        .collect();
    head.sort_by_key(|t| t.meta.offset);

    // Defensive: head entries must not overlap each other in canonical
    // space — every assumption below (and every downstream `w_off`
    // consumer) depends on this. A corrupt/hand-edited manifest must fail
    // loud here, not feed garbage into a splice plan.
    for w in head.windows(2) {
        let (a, b) = (w[0], w[1]);
        if a.meta.offset + a.meta.byte_len > b.meta.offset {
            return Err(Error::Layered(format!(
                "dgq head splice plan: tensors {} and {} overlap in canonical space \
                 ({}..{} vs {}..)",
                a.name,
                b.name,
                a.meta.offset,
                a.meta.offset + a.meta.byte_len,
                b.meta.offset
            )));
        }
    }

    struct Building<'a> {
        file_key: &'a str,
        canonical_start: u64,
        canonical_end: u64,
        file_start: u64,
        file_end: u64,
        names: Vec<&'a str>,
    }

    let mut plan = SplicePlan::default();
    let mut candidates: Vec<Building> = Vec::new();
    let mut building: Option<Building> = None;

    for entry in &head {
        match &entry.meta.source {
            Some(TensorSource::External { file, offset }) => {
                let extends = building.as_ref().is_some_and(|b| {
                    b.file_key == file.as_str()
                        && b.canonical_end == entry.meta.offset
                        && b.file_end == *offset
                });
                if extends {
                    let b = building.as_mut().expect("checked above");
                    b.canonical_end = entry.meta.offset + entry.meta.byte_len;
                    b.file_end = offset + entry.meta.byte_len;
                    b.names.push(&entry.name);
                } else {
                    candidates.extend(building.take());
                    building = Some(Building {
                        file_key: file.as_str(),
                        canonical_start: entry.meta.offset,
                        canonical_end: entry.meta.offset + entry.meta.byte_len,
                        file_start: *offset,
                        file_end: offset + entry.meta.byte_len,
                        names: vec![&entry.name],
                    });
                }
            }
            // Local-sourced (or, defensively, a bare `None` source — should
            // never appear in a layered pack's head, but handled the same
            // safe way rather than assumed away) tensors are never spliced.
            _ => {
                candidates.extend(building.take());
                plan.anon_entries.push(entry.name.clone());
            }
        }
    }
    candidates.extend(building.take());

    for cand in candidates {
        let congruent = (cand.canonical_start % PAGE) == (cand.file_start % PAGE);
        if !congruent {
            plan.anon_entries.extend(cand.names.iter().map(|s| s.to_string()));
            continue;
        }
        let map_start = (cand.canonical_start / PAGE) * PAGE;
        let map_end = cand.canonical_end.div_ceil(PAGE) * PAGE;
        let overlaps_other = head.iter().any(|e| {
            if cand.names.contains(&e.name.as_str()) {
                return false;
            }
            let e_start = e.meta.offset;
            let e_end = e.meta.offset + e.meta.byte_len;
            e_start < map_end && map_start < e_end
        });
        if overlaps_other {
            plan.anon_entries.extend(cand.names.iter().map(|s| s.to_string()));
            continue;
        }
        plan.splice_runs.push(SpliceRun {
            file_key: cand.file_key.to_string(),
            canonical_start: cand.canonical_start,
            canonical_end: cand.canonical_end,
            file_start: cand.file_start,
            tensor_names: cand.names.iter().map(|s| s.to_string()).collect(),
        });
    }

    Ok(plan)
}

/// Build the head region `[0, expert_split)` by VA-splice: reserve a
/// contiguous range, `MAP_FIXED` every validated `SpliceRun` straight off
/// its resolved external HF shard, and memcpy everything else (`Local`
/// tensors, any run this loader couldn't prove safe) into the reservation's
/// anonymous backdrop. Returns `Err` for anything that keeps this from
/// being provably correct — including the degenerate "old non-congruent
/// pack" case (external head tensors present, zero runs qualified) — so the
/// caller can fall back to the whole-blob `materialize_layered_blob` path.
fn build_head_splice(
    model_dir: &Path,
    manifest: &DgqManifest,
    expert_split: u64,
) -> Result<HeadSplice, Error> {
    const PAGE: u64 = HEAD_SPLICE_PAGE;
    let head_len = blob_offset_usize(expert_split)?.max(1);
    let plan = plan_head_splice(manifest, expert_split)?;

    let has_external_head_tensor = manifest
        .tensors
        .iter()
        .any(|t| t.meta.offset < expert_split && matches!(t.meta.source, Some(TensorSource::External { .. })));
    if plan.splice_runs.is_empty() && has_external_head_tensor {
        return Err(Error::Layered(
            "zero splice runs qualified despite external head tensors — this pack's canonical \
             head layout doesn't mirror shard byte order (an old pre-splice overlay pack?); \
             re-run quantize/repack --overlay to get a spliceable layout"
                .to_string(),
        ));
    }

    let local_bin_path = model_dir.join(&manifest.blob_file);
    let local_file = File::open(&local_bin_path)?;
    let local_mmap = unsafe { Mmap::map(&local_file)? };
    let external = resolve_external_mmaps(model_dir, manifest)?;

    unsafe {
        let base = libc::mmap(
            std::ptr::null_mut(),
            head_len,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        );
        if base == libc::MAP_FAILED {
            return Err(Error::Runtime("dgq head splice: VA reservation failed"));
        }
        // Any early return below must release the reservation — disarmed
        // (via `is_committed`) only once `HeadSplice` is about to own it.
        struct ReserveGuard {
            base: *mut c_void,
            len: usize,
            committed: bool,
        }
        impl Drop for ReserveGuard {
            fn drop(&mut self) {
                if !self.committed {
                    unsafe {
                        libc::munmap(self.base, self.len);
                    }
                }
            }
        }
        let mut guard = ReserveGuard {
            base,
            len: head_len,
            committed: false,
        };

        let anon = libc::mmap(
            base,
            head_len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED,
            -1,
            0,
        );
        if anon == libc::MAP_FAILED || anon != base {
            return Err(Error::Runtime("dgq head splice: anon backdrop mmap failed"));
        }

        for name in &plan.anon_entries {
            let entry = manifest
                .tensors
                .iter()
                .find(|t| &t.name == name)
                .ok_or_else(|| Error::NotFound(name.clone()))?;
            let len = blob_offset_usize(entry.meta.byte_len)?;
            let dst_start = blob_offset_usize(entry.meta.offset)?;
            if dst_start + len > head_len {
                return Err(Error::Runtime(
                    "dgq head splice: anon-fill tensor extends past expert_split",
                ));
            }
            let src = source_slice(entry, &local_mmap, &external, len)?;
            std::ptr::copy_nonoverlapping(src.as_ptr(), base.cast::<u8>().add(dst_start), len);
        }

        for run in &plan.splice_runs {
            let ext_file = manifest.external_files.get(&run.file_key).ok_or_else(|| {
                Error::Layered(format!(
                    "dgq head splice: run references unresolved external file '{}'",
                    run.file_key
                ))
            })?;
            let path = crate::dgq::hf_resolve::resolve_external_file(
                model_dir,
                manifest.base_model.as_ref(),
                &run.file_key,
                ext_file,
            )?;
            let file = File::open(&path)?;
            let file_len = file.metadata()?.len();

            let map_start = (run.canonical_start / PAGE) * PAGE;
            let map_end = run.canonical_end.div_ceil(PAGE) * PAGE;
            let file_map_start = (run.file_start / PAGE) * PAGE;
            if map_start > map_end || blob_offset_usize(map_end)? > head_len {
                return Err(Error::Runtime(
                    "dgq head splice: run's rounded range falls outside the head reservation",
                ));
            }
            let map_len = blob_offset_usize(map_end - map_start)?;
            if file_map_start + map_len as u64 > file_len {
                return Err(Error::Layered(format!(
                    "dgq head splice: run over '{}' ({} tensors) would read {} bytes from file \
                     offset {file_map_start}, but the shard is only {file_len} bytes",
                    run.file_key,
                    run.tensor_names.len(),
                    map_len
                )));
            }

            let dst = base.cast::<u8>().add(blob_offset_usize(map_start)?);
            let mapped = libc::mmap(
                dst.cast(),
                map_len,
                libc::PROT_READ,
                libc::MAP_SHARED | libc::MAP_FIXED,
                file.as_raw_fd(),
                file_map_start as libc::off_t,
            );
            if mapped == libc::MAP_FAILED || mapped != dst.cast() {
                return Err(Error::Runtime("dgq head splice: MAP_FIXED file splice failed"));
            }
            // `file` (the fd) can drop now — the MAP_SHARED mapping outlives it.
        }

        if libc::mprotect(base, head_len, libc::PROT_READ) != 0 {
            return Err(Error::Runtime("dgq head splice: mprotect(PROT_READ) failed"));
        }

        eprintln!(
            "dgq layered blob: head {:.2} GiB VA-spliced ({} run(s) over {} tensor(s), \
             {} tensor(s) anon-copied)",
            head_len as f64 / 1073741824.0,
            plan.splice_runs.len(),
            plan.splice_runs.iter().map(|r| r.tensor_names.len()).sum::<usize>(),
            plan.anon_entries.len(),
        );

        guard.committed = true;
        Ok(HeadSplice {
            base: NonNull::new(base).ok_or(Error::Runtime("dgq head splice: null base"))?,
            len: head_len,
        })
    }
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

/// Pure `plan_head_splice` tests: no I/O, no unsafe, no GPU — the plan is
/// just arithmetic over a hand-built manifest.
#[cfg(test)]
mod splice_plan_tests {
    use super::*;
    use crate::dgq::layout::{DgqTensorMeta, QuantProfile};

    fn manifest(tensors: Vec<DgqTensorEntry>) -> DgqManifest {
        DgqManifest {
            version: crate::dgq::layout::DGQ_VERSION_LAYERED,
            profile: QuantProfile::Q4,
            source_model: "src".to_string(),
            blob_file: "model.dgq.bin".to_string(),
            expert_split: None,
            local_expert_split: None,
            base_model: None,
            external_files: Default::default(),
            custom_classes: Default::default(),
            tensors,
        }
    }

    fn external(name: &str, file: &str, canonical: u64, file_off: u64, len: u64) -> DgqTensorEntry {
        DgqTensorEntry {
            name: name.to_string(),
            meta: DgqTensorMeta {
                kind: "raw".to_string(),
                dtype: "bf16".to_string(),
                shape: vec![1],
                offset: canonical,
                byte_len: len,
                source: Some(TensorSource::External {
                    file: file.to_string(),
                    offset: file_off,
                }),
            },
        }
    }

    fn local(name: &str, canonical: u64, len: u64) -> DgqTensorEntry {
        DgqTensorEntry {
            name: name.to_string(),
            meta: DgqTensorMeta {
                kind: "q8_row".to_string(),
                dtype: "bf16".to_string(),
                shape: vec![1],
                offset: canonical,
                byte_len: len,
                source: Some(TensorSource::Local { local_offset: 0 }),
            },
        }
    }

    #[test]
    fn contiguous_congruent_run_splices_as_one() {
        // Both tensors start at canonical/file offset 0 (trivially
        // congruent) and mirror the shard's own zero-gap layout.
        let a1 = external("a1", "shardA", 0, 0, 100);
        let a2 = external("a2", "shardA", 100, 100, 150); // extends the run
        let loc = local("local1", 20000, 64); // well clear of the run's rounded [0,16384)
        let m = manifest(vec![a1, a2, loc]);

        let plan = plan_head_splice(&m, 32768).expect("plan");
        assert_eq!(plan.splice_runs.len(), 1, "{plan:?}");
        assert_eq!(plan.splice_runs[0].tensor_names, vec!["a1", "a2"]);
        assert_eq!(plan.splice_runs[0].canonical_start, 0);
        assert_eq!(plan.splice_runs[0].canonical_end, 250);
        assert_eq!(plan.anon_entries, vec!["local1".to_string()]);
    }

    #[test]
    fn non_congruent_run_falls_back_to_anon() {
        // canonical starts at 0 (remainder 0) but the file offset's
        // remainder is nonzero — congruence fails, must anon-fill.
        let a = external("a", "shardA", 0, 5, 100);
        let m = manifest(vec![a]);
        let plan = plan_head_splice(&m, 16384).expect("plan");
        assert!(plan.splice_runs.is_empty(), "{plan:?}");
        assert_eq!(plan.anon_entries, vec!["a".to_string()]);
    }

    #[test]
    fn run_whose_rounded_fringe_would_clobber_a_neighbor_is_rejected() {
        // `a`: canonical [0, 20000), file [0, 20000) — congruent (both
        // remainder 0), but 20000 isn't page-aligned so `a`'s OWN rounded
        // range extends to 32768.
        let a = external("a", "shardA", 0, 0, 20000);
        // `b`: canonical [20000, 21000) from a DIFFERENT shard, chosen so
        // its own start is ALSO congruent (20000 % 16384 == 3616 == 3616 %
        // 16384) — a legitimate-looking run on its own, but its rounded
        // start (16384) falls INSIDE `a`'s true byte range [0, 20000).
        let b = external("b", "shardB", 20000, 3616, 1000);
        let m = manifest(vec![a, b]);

        let plan = plan_head_splice(&m, 32768).expect("plan");
        assert!(
            plan.splice_runs.is_empty(),
            "both runs must be rejected (each other's fringe clobbers real bytes): {plan:?}"
        );
        let mut anon = plan.anon_entries.clone();
        anon.sort();
        assert_eq!(anon, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn different_shards_break_the_run_even_with_zero_canonical_gap() {
        let a = external("a", "shardA", 0, 0, 16384); // page-aligned end
        let b = external("b", "shardB", 16384, 0, 100); // starts exactly where `a` ends
        let m = manifest(vec![a, b]);
        let plan = plan_head_splice(&m, 32768).expect("plan");
        // Two independent single-tensor runs — neither's rounded range
        // overlaps the other's true bytes (a's end IS the page boundary).
        assert_eq!(plan.splice_runs.len(), 2, "{plan:?}");
        assert!(plan.anon_entries.is_empty());
    }

    #[test]
    fn overlapping_manifest_entries_are_rejected_outright() {
        let a = external("a", "shardA", 0, 0, 200);
        let b = external("b", "shardA", 100, 100, 200); // overlaps a's [0,200)
        let m = manifest(vec![a, b]);
        let err = plan_head_splice(&m, 16384).expect_err("must reject overlapping entries");
        assert!(err.to_string().contains("overlap"), "{err}");
    }

    #[test]
    fn offset_alignment_tripwire_accepts_64_aligned_manifest() {
        let m = manifest(vec![external("a", "shardA", 0, 0, 128), local("b", 128, 64)]);
        assert_tensor_offset_alignment(&m).expect("64-byte-aligned manifest must pass");
    }

    #[test]
    fn offset_alignment_tripwire_rejects_misaligned_tensor() {
        // canonical offset 130 is not a multiple of 64 — exactly the class
        // of writer bug this tripwire exists to catch at load time (byte
        // content can be perfectly correct and this would still be unsafe
        // for a GPU kernel's typed pointer read).
        let m = manifest(vec![external("a", "shardA", 0, 0, 128), local("b", 130, 64)]);
        let err = assert_tensor_offset_alignment(&m).expect_err("must reject misaligned offset");
        assert!(err.to_string().contains("misaligned"), "{err}");
    }
}

/// Writer <-> loader integration: a freshly written overlay manifest (real
/// `quantize_model` writer, tiny synthetic HF fixture) must produce a
/// splice plan with ZERO External fallbacks — the writer's shard-order +
/// page-congruent layout is exactly what the loader's plan needs.
#[cfg(test)]
mod writer_plan_integration_tests {
    use super::*;
    use crate::dgq::layout::{QuantProfile, TensorSource as Src};
    use crate::dgq::test_fixtures::{bf16_payload, write_index, write_shard};
    use crate::dgq::{QuantizeOptions, quantize_model};

    #[test]
    fn quantize_overlay_writer_produces_zero_external_fallbacks() {
        let root = std::env::temp_dir().join(format!(
            "dgq-splice-plan-writer-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let snapshot_dir = root.join("hf_home/hub/models--acme--widgets/snapshots/cafef00d");
        std::fs::create_dir_all(&snapshot_dir).expect("mkdir snapshot");

        // Several attn/dense (Raw-classified) tensors from the SAME shard,
        // deliberately written out of alphabetical order in the file to
        // prove the writer reorders by actual shard byte position, not by
        // name — plus one local (SC q8) tensor and one expert tensor, per
        // real overlay pack shape.
        let t1 = ("model.decoder.layers.1.self_attn.q_proj.weight", vec![8, 32], bf16_payload(8 * 32, 1));
        let t0 = ("model.decoder.layers.0.self_attn.q_proj.weight", vec![8, 32], bf16_payload(8 * 32, 2));
        let t2 = ("model.decoder.layers.2.self_attn.q_proj.weight", vec![8, 32], bf16_payload(8 * 32, 3));
        let sc = (
            "model.decoder.self_conditioning.down_proj.weight",
            vec![4, 8],
            bf16_payload(32, 4),
        );
        let experts = (
            "model.decoder.layers.0.experts.gate_up_proj",
            vec![2, 4, 32],
            bf16_payload(2 * 4 * 32, 5),
        );
        let names: Vec<&str> = vec![t1.0, t0.0, t2.0, sc.0, experts.0];
        write_shard(
            &snapshot_dir.join("model-00001-of-00001.safetensors"),
            &[t1, t0, t2, sc, experts],
        );
        write_index(&snapshot_dir, "model-00001-of-00001.safetensors", &names);

        let cfg = crate::flags::RuntimeConfig::from_pairs(&[(
            "DGQ_HF_HOME".to_string(),
            root.join("hf_home").display().to_string(),
        )])
        .0;
        let _guard = crate::flags::install_for_test(cfg);

        let out_dir = root.join("overlay");
        let summary = quantize_model(QuantizeOptions {
            source_dir: snapshot_dir,
            output_prefix: out_dir.clone(),
            profile: QuantProfile::Q4,
            overlay_base: Some(crate::dgq::layout::BaseModelRef {
                repo: "acme/widgets".to_string(),
                revision: "cafef00d".to_string(),
            }),
            custom_overrides: Default::default(),
        })
        .expect("quantize --overlay");
        assert_eq!(summary.raw_tensors, 3, "3 attn tensors should be Raw");

        let manifest_json = std::fs::read_to_string(out_dir.join(MANIFEST_FILE)).expect("read manifest");
        let manifest: DgqManifest = serde_json::from_str(&manifest_json).expect("parse manifest");
        let expert_split = manifest.expert_split.expect("expert_split set");

        let plan = plan_head_splice(&manifest, expert_split).expect("plan");
        let external_anon: Vec<&String> = plan
            .anon_entries
            .iter()
            .filter(|name| {
                manifest
                    .tensors
                    .iter()
                    .any(|t| &t.name == *name && matches!(t.meta.source, Some(Src::External { .. })))
            })
            .collect();
        assert!(
            external_anon.is_empty(),
            "writer's layout should make every External head tensor spliceable; \
             fallbacks: {external_anon:?} full plan: {plan:?}"
        );
        // The three attn tensors should collapse into exactly one run (same
        // shard, writer visits them in FILE order regardless of name order).
        assert_eq!(plan.splice_runs.len(), 1, "{plan:?}");
        assert_eq!(plan.splice_runs[0].tensor_names.len(), 3);

        let _ = std::fs::remove_dir_all(&root);
    }

    /// Same property, for the OTHER writer: `repack --overlay` splitting a
    /// self-contained pack must ALSO produce a spliceable head layout — it
    /// reassigns canonical offsets fresh (not carried over from the
    /// self-contained source's alphabetical layout), using the same
    /// shard-order + page-congruent scheme as `quantize --overlay`.
    #[test]
    fn repack_overlay_writer_produces_zero_external_fallbacks() {
        use crate::dgq::overlay::{RepackOverlayOptions, repack_overlay};

        let root = std::env::temp_dir().join(format!("dgq-splice-plan-repack-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let snapshot_dir = root.join("hf_home/hub/models--acme--widgets/snapshots/cafef00d");
        std::fs::create_dir_all(&snapshot_dir).expect("mkdir snapshot");

        let t1 = ("model.decoder.layers.1.self_attn.q_proj.weight", vec![8, 32], bf16_payload(8 * 32, 1));
        let t0 = ("model.decoder.layers.0.self_attn.q_proj.weight", vec![8, 32], bf16_payload(8 * 32, 2));
        let t2 = ("model.decoder.layers.2.self_attn.q_proj.weight", vec![8, 32], bf16_payload(8 * 32, 3));
        let sc = (
            "model.decoder.self_conditioning.down_proj.weight",
            vec![4, 8],
            bf16_payload(32, 4),
        );
        let experts = (
            "model.decoder.layers.0.experts.gate_up_proj",
            vec![2, 4, 32],
            bf16_payload(2 * 4 * 32, 5),
        );
        let names: Vec<&str> = vec![t1.0, t0.0, t2.0, sc.0, experts.0];
        write_shard(
            &snapshot_dir.join("model-00001-of-00001.safetensors"),
            &[t1, t0, t2, sc, experts],
        );
        write_index(&snapshot_dir, "model-00001-of-00001.safetensors", &names);

        let cfg = crate::flags::RuntimeConfig::from_pairs(&[(
            "DGQ_HF_HOME".to_string(),
            root.join("hf_home").display().to_string(),
        )])
        .0;
        let _guard = crate::flags::install_for_test(cfg);

        // A plain self-contained pack (alphabetical layout, no splicing
        // relevance) from the same fixture — the input `repack --overlay`
        // must reshape.
        let self_contained = root.join("self_contained");
        quantize_model(QuantizeOptions {
            source_dir: snapshot_dir.clone(),
            output_prefix: self_contained.clone(),
            profile: QuantProfile::Q4,
            overlay_base: None,
            custom_overrides: Default::default(),
        })
        .expect("plain quantize");

        let overlay_out = root.join("repacked_overlay");
        repack_overlay(RepackOverlayOptions {
            pack_dir: self_contained,
            output_dir: overlay_out.clone(),
            hf_source_dir: Some(snapshot_dir),
            hf_repo_override: Some("acme/widgets".to_string()),
            hf_revision_override: Some("cafef00d".to_string()),
        })
        .expect("repack --overlay");

        let manifest_json = std::fs::read_to_string(overlay_out.join(MANIFEST_FILE)).expect("read manifest");
        let manifest: DgqManifest = serde_json::from_str(&manifest_json).expect("parse manifest");
        let expert_split = manifest.expert_split.expect("expert_split set");

        let plan = plan_head_splice(&manifest, expert_split).expect("plan");
        let external_anon: Vec<&String> = plan
            .anon_entries
            .iter()
            .filter(|name| {
                manifest
                    .tensors
                    .iter()
                    .any(|t| &t.name == *name && matches!(t.meta.source, Some(Src::External { .. })))
            })
            .collect();
        assert!(
            external_anon.is_empty(),
            "repack --overlay's layout should make every External head tensor spliceable; \
             fallbacks: {external_anon:?} full plan: {plan:?}"
        );
        assert_eq!(plan.splice_runs.len(), 1, "{plan:?}");
        assert_eq!(plan.splice_runs[0].tensor_names.len(), 3);

        let _ = std::fs::remove_dir_all(&root);
    }
}

#[cfg(all(test, target_os = "macos"))]
mod diagnostic_gpu_vs_store_tests {
    use super::*;
    use crate::dgq::DgqStore;
    use crate::metal::device::MetalContext;

    /// Ad hoc diagnostic (not a CI gate): DGQ_DIAG_PACK_DIR points at a pack;
    /// compares `DgqGpuBlob::host_ptr` (what the GPU buffer actually
    /// contains) against `DgqStore::tensor_bytes` (the independent CPU-side
    /// source-dispatch resolution) for every tensor. Skips silently if the
    /// env var is unset.
    #[test]
    fn gpu_buffer_matches_store_for_every_tensor() {
        let Ok(dir) = std::env::var("DGQ_DIAG_PACK_DIR") else {
            eprintln!("skip: set DGQ_DIAG_PACK_DIR to run this diagnostic");
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        let store = DgqStore::open(&dir).expect("open store");
        let ctx = MetalContext::new().expect("metal");
        let blob = DgqGpuBlob::from_store(&store, &ctx.device).expect("blob");

        let mut mismatches = Vec::new();
        let mut checked = 0usize;
        for entry in store.tensor_entries() {
            let want = store.tensor_bytes(&entry.name).expect("store bytes");
            let len = want.len();
            let got =
                unsafe { std::slice::from_raw_parts(blob.host_ptr(entry.meta.offset), len) };
            checked += 1;
            if got != want {
                let first_diff = want
                    .iter()
                    .zip(got.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(0);
                mismatches.push((entry.name.clone(), entry.meta.offset, len, first_diff));
            }
        }
        eprintln!(
            "diagnostic: checked {checked} tensors, {} mismatches",
            mismatches.len()
        );
        for (name, off, len, first_diff) in mismatches.iter().take(20) {
            eprintln!("  MISMATCH {name} offset={off} len={len} first_diff_at={first_diff}");
        }
        assert!(mismatches.is_empty(), "{} tensor(s) mismatched", mismatches.len());
    }
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
