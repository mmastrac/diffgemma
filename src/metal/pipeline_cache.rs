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
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

const CACHE_BUNDLE_TAG: &str = "diffgemma-mps-v8-stacked-seg-fc";

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
    archive: Retained<ProtocolObject<dyn MTLBinaryArchive>>,
    cache_file: PathBuf,
    dirty: AtomicBool,
    persist: bool,
}

// Metal pipeline objects are used from the main inference thread only.
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
            archive,
            cache_file,
            dirty: AtomicBool::new(false),
            persist: true,
        })
    }

    fn open_ephemeral(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self, Error> {
        let desc = MTLBinaryArchiveDescriptor::new();
        let archive = device
            .newBinaryArchiveWithDescriptor_error(&desc)
            .map_err(|_| Error::Format("MTLBinaryArchive create failed"))?;
        Ok(Self {
            archive,
            cache_file: PathBuf::new(),
            dirty: AtomicBool::new(false),
            persist: false,
        })
    }

    pub fn compile_compute(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        function: &ProtocolObject<dyn MTLFunction>,
        label: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, Error> {
        let desc = MTLComputePipelineDescriptor::new();
        desc.setComputeFunction(Some(function));
        desc.setLabel(Some(&NSString::from_str(label)));
        desc.setSupportIndirectCommandBuffers(true);
        let archives = NSArray::from_slice(&[&*self.archive]);
        desc.setBinaryArchives(Some(&archives));

        let pipeline = device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &desc,
                MTLPipelineOption::empty(),
                None,
            )
            .map_err(|_| Error::Format("Metal pipeline compile failed"))?;

        if self.persist
            && self
                .archive
                .addComputePipelineFunctionsWithDescriptor_error(&desc)
                .is_ok()
        {
            self.dirty.store(true, Ordering::Relaxed);
        }

        Ok(pipeline)
    }

    pub fn flush_if_dirty(&self) {
        if !self.persist || !self.dirty.swap(false, Ordering::Relaxed) {
            return;
        }
        let Some(url) = NSURL::from_file_path(&self.cache_file) else {
            return;
        };
        let save_started = std::time::Instant::now();
        match self.archive.serializeToURL_error(&url) {
            Ok(()) => {
                if crate::flags::progress_enabled() {
                    eprintln!(
                        "metal pipeline cache: saved {} ({:.2?})",
                        self.cache_file.display(),
                        save_started.elapsed()
                    );
                }
            }
            Err(e) => eprintln!(
                "warning: metal pipeline cache serialize failed: {}",
                e.localizedDescription()
            ),
        }
    }
}
