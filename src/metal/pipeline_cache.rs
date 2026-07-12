//! Runtime `MTLBinaryArchive` cache for compiled compute pipelines.
//!
//! Persists device-specific pipeline ISA to disk so cold starts can skip
//! recompilation after the first run (pure runtime API; no Metal SDK required).

use crate::safetensors::Error;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSString, NSURL};
use objc2_metal::{
    MTLBinaryArchive, MTLBinaryArchiveDescriptor, MTLComputePipelineDescriptor,
    MTLComputePipelineState, MTLDevice, MTLFunction, MTLPipelineOption,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

const CACHE_BUNDLE_TAG: &str = "diffgemma-mps-v9-runtime-include";

fn shader_bundle_token() -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    // Whole-shader-tree hash from build.rs (walks shaders/**.metal). Replaces
    // a hand-maintained include_str! list that had drifted to cover only 60 of
    // 93 files — edits to unlisted kernels (qk_rope_kv, attention_device,
    // gemm_tunable, sample_commit, ...) were served STALE from the archive.
    env!("DGQ_SHADER_TREE_HASH").hash(&mut h);
    CACHE_BUNDLE_TAG.hash(&mut h);
    h.finish()
}

fn cache_enabled() -> bool {
    crate::flags::metal_pipeline_cache_enabled()
}

fn cache_root_dir() -> PathBuf {
    if let Some(dir) = crate::flags::metal_pipeline_cache_dir_override() {
        return dir;
    }
    std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
        .join("diffgemma-mps")
        .join("metal-pipelines")
}

fn sanitize_device_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

pub struct PipelineArchiveCache {
    /// All access to the non-thread-safe `MTLBinaryArchive` — the pipeline-lookup
    /// read during compilation, `addComputePipelineFunctions`, and `serialize` —
    /// goes through this Mutex, so the archive is never touched from two threads
    /// at once. The compiled-pipeline map lives here too so a repeat request
    /// returns a clone without re-entering the archive at all.
    inner: Mutex<ArchiveInner>,
    cache_file: PathBuf,
    persist: bool,
}

struct ArchiveInner {
    archive: Retained<ProtocolObject<dyn MTLBinaryArchive>>,
    /// Compiled pipelines keyed by the caller's label. The label encodes the
    /// entry point plus the full function-constant config (see
    /// `device.rs::compile_*`), so it uniquely identifies a pipeline — making
    /// this an exact-match dedup, not a lossy hash.
    compiled: HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    dirty: bool,
}

// SAFETY: the only non-`Send`/`Sync` members are Metal objects reached through
// `inner`'s Mutex. The archive is never used unsynchronized (see the field doc),
// and `MTLComputePipelineState` is immutable and safe to share once created.
unsafe impl Send for PipelineArchiveCache {}
unsafe impl Sync for PipelineArchiveCache {}

static GLOBAL_CACHE: OnceLock<Result<Arc<PipelineArchiveCache>, String>> = OnceLock::new();

impl PipelineArchiveCache {
    pub fn shared(device: &ProtocolObject<dyn MTLDevice>) -> Result<Arc<Self>, Error> {
        match GLOBAL_CACHE
            .get_or_init(|| Self::open(device).map(Arc::new).map_err(|e| e.to_string()))
        {
            Ok(c) => Ok(Arc::clone(c)),
            Err(msg) => Err(Error::NotFound(msg.clone())),
        }
    }

    pub fn flush_global() {
        if let Some(Ok(c)) = GLOBAL_CACHE.get() {
            c.flush_if_dirty();
        }
    }

    fn open(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self, Error> {
        let open_started = std::time::Instant::now();
        if !cache_enabled() {
            return Self::open_ephemeral(device);
        }

        let cache_dir = cache_root_dir();
        std::fs::create_dir_all(&cache_dir)
            .map_err(|e| Error::NotFound(format!("metal pipeline cache mkdir failed: {e}")))?;

        let device_name = sanitize_device_name(&device.name().to_string());
        let token = shader_bundle_token();
        let cache_file = cache_dir.join(format!("{device_name}-{token:016x}.metallibarchive"));
        let existed = cache_file.exists();

        let desc = MTLBinaryArchiveDescriptor::new();
        if existed {
            if let Some(url) = NSURL::from_file_path(&cache_file) {
                desc.setUrl(Some(&url));
            }
        }

        let archive = device
            .newBinaryArchiveWithDescriptor_error(&desc)
            .map_err(|e| {
                Error::NotFound(format!(
                    "MTLBinaryArchive open failed: {}",
                    e.localizedDescription()
                ))
            })?;

        if crate::flags::progress_enabled() {
            eprintln!(
                "metal pipeline cache: {} ({}, {:.2?})",
                cache_file.display(),
                if existed { "loaded" } else { "new" },
                open_started.elapsed()
            );
        }

        Ok(Self {
            inner: Mutex::new(ArchiveInner {
                archive,
                compiled: HashMap::new(),
                dirty: false,
            }),
            cache_file,
            persist: true,
        })
    }

    fn open_ephemeral(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self, Error> {
        let desc = MTLBinaryArchiveDescriptor::new();
        let archive = device
            .newBinaryArchiveWithDescriptor_error(&desc)
            .map_err(|_| Error::Format("MTLBinaryArchive create failed"))?;
        Ok(Self {
            inner: Mutex::new(ArchiveInner {
                archive,
                compiled: HashMap::new(),
                dirty: false,
            }),
            cache_file: PathBuf::new(),
            persist: false,
        })
    }

    pub fn compile_compute(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        function: &ProtocolObject<dyn MTLFunction>,
        label: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, Error> {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());

        // Fast path: already compiled this exact pipeline — hand back a clone
        // (a cheap retain) without re-entering the archive.
        if let Some(pipeline) = inner.compiled.get(label) {
            return Ok(pipeline.clone());
        }

        let desc = MTLComputePipelineDescriptor::new();
        desc.setComputeFunction(Some(function));
        desc.setLabel(Some(&NSString::from_str(label)));
        desc.setSupportIndirectCommandBuffers(true);
        let archives = NSArray::from_slice(&[&*inner.archive]);
        desc.setBinaryArchives(Some(&archives));

        let pipeline = device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &desc,
                MTLPipelineOption::empty(),
                None,
            )
            .map_err(|_| Error::Format("Metal pipeline compile failed"))?;

        if self.persist
            && inner
                .archive
                .addComputePipelineFunctionsWithDescriptor_error(&desc)
                .is_ok()
        {
            inner.dirty = true;
        }

        inner.compiled.insert(label.to_string(), pipeline.clone());
        Ok(pipeline)
    }

    pub fn flush_if_dirty(&self) {
        if !self.persist {
            return;
        }
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if !std::mem::replace(&mut inner.dirty, false) {
            return;
        }
        // Serialize to a pid-tagged temp then atomically rename, so a crash or a
        // second process writing the same cache file can never leave a
        // half-written (corrupt) archive that would fail to reopen.
        let tmp = self
            .cache_file
            .with_extension(format!("tmp-{}", std::process::id()));
        let Some(url) = NSURL::from_file_path(&tmp) else {
            return;
        };
        let save_started = std::time::Instant::now();
        match inner.archive.serializeToURL_error(&url) {
            Ok(()) => {
                if let Err(e) = std::fs::rename(&tmp, &self.cache_file) {
                    let _ = std::fs::remove_file(&tmp);
                    eprintln!("warning: metal pipeline cache rename failed: {e}");
                    return;
                }
                if crate::flags::progress_enabled() {
                    eprintln!(
                        "metal pipeline cache: saved {} ({:.2?})",
                        self.cache_file.display(),
                        save_started.elapsed()
                    );
                }
            }
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                eprintln!(
                    "warning: metal pipeline cache serialize failed: {}",
                    e.localizedDescription()
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::device::MetalContext;

    const SRC: &str = "\
#include <metal_stdlib>
using namespace metal;
kernel void pc_k0(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 0.0f; }
kernel void pc_k1(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 1.0f; }
kernel void pc_k2(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 2.0f; }
kernel void pc_k3(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 3.0f; }
";

    /// Hammer the process-global pipeline archive from many threads at once.
    /// Same entry across threads exercises the dedup fast path; different entries
    /// race concurrent `addComputePipelineFunctions`. Pre-Mutex this raced the
    /// non-thread-safe `MTLBinaryArchive` and SIGSEGV'd — the reason the test
    /// suite ran `--test-threads=1`. Manages its own threads, so it still
    /// exercises concurrency under the serial test harness.
    #[test]
    fn concurrent_compile_is_race_free() {
        if MetalContext::new().is_err() {
            eprintln!("skip: no Metal device");
            return;
        }
        let entries = ["pc_k0", "pc_k1", "pc_k2", "pc_k3"];
        let handles: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(move || {
                    let ctx = MetalContext::new().expect("device");
                    for _ in 0..4 {
                        for e in entries {
                            ctx.compile_kernel(SRC, e).expect("pipeline compile");
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("worker panicked — pipeline cache race?");
        }
    }
}
