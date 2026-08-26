//! Runtime `MTLBinaryArchive` cache for compiled compute pipelines.
//!
//! Persists device-specific pipeline ISA to disk so cold starts can skip
//! recompilation after the first run (pure runtime API; no Metal SDK required).

use crate::Error;
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

/// On-disk pipeline cache settings, supplied by the application.
#[derive(Clone)]
pub struct CacheConfig {
    /// `false` keeps an in-memory archive only (nothing touches disk).
    pub enabled: bool,
    /// Cache directory; `None` resolves `$XDG_CACHE_HOME` (else `~/.cache`,
    /// else `/tmp`) + `<namespace>/metal-pipelines`.
    pub dir: Option<PathBuf>,
    pub namespace: &'static str,
    /// Version key baked into the cache file name. Must change whenever any
    /// shader source the archive may hold changes (hash the whole shader
    /// tree), or an edited kernel is served stale from the old archive.
    pub key: u64,
    /// Print cache load/save lines to stderr.
    pub verbose: bool,
}

impl CacheConfig {
    fn root_dir(&self) -> PathBuf {
        if let Some(dir) = &self.dir {
            return dir.clone();
        }
        std::env::var("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|_| std::env::var("HOME").map(|h| PathBuf::from(h).join(".cache")))
            .unwrap_or_else(|_| PathBuf::from("/tmp"))
            .join(self.namespace)
            .join("metal-pipelines")
    }
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
    verbose: bool,
}

struct ArchiveInner {
    archive: Retained<ProtocolObject<dyn MTLBinaryArchive>>,
    /// Compiled pipelines keyed by the caller's label. The label must encode
    /// the full specialization (see `Context::compile_specialized`), making
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
    /// The process-global cache. The first call fixes the configuration;
    /// later calls return the same cache and ignore `config`.
    pub fn shared(
        device: &ProtocolObject<dyn MTLDevice>,
        config: &CacheConfig,
    ) -> Result<Arc<Self>, Error> {
        match GLOBAL_CACHE.get_or_init(|| {
            Self::open(device, config)
                .map(Arc::new)
                .map_err(|e| e.to_string())
        }) {
            Ok(c) => Ok(Arc::clone(c)),
            Err(msg) => Err(Error::Cache(msg.clone())),
        }
    }

    pub fn flush_global() {
        if let Some(Ok(c)) = GLOBAL_CACHE.get() {
            c.flush_if_dirty();
        }
    }

    fn open(device: &ProtocolObject<dyn MTLDevice>, config: &CacheConfig) -> Result<Self, Error> {
        let open_started = std::time::Instant::now();
        if !config.enabled {
            return Self::open_ephemeral(device);
        }

        let cache_dir = config.root_dir();
        std::fs::create_dir_all(&cache_dir)
            .map_err(|e| Error::Cache(format!("mkdir failed: {e}")))?;

        let device_name = sanitize_device_name(&device.name().to_string());
        let cache_file =
            cache_dir.join(format!("{device_name}-{:016x}.metallibarchive", config.key));
        let existed = cache_file.exists();

        let desc = MTLBinaryArchiveDescriptor::new();
        if existed && let Some(url) = NSURL::from_file_path(&cache_file) {
            desc.setUrl(Some(&url));
        }

        let archive = device
            .newBinaryArchiveWithDescriptor_error(&desc)
            .map_err(|e| {
                Error::Cache(format!(
                    "MTLBinaryArchive open failed: {}",
                    e.localizedDescription()
                ))
            })?;

        if config.verbose {
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
            verbose: config.verbose,
        })
    }

    fn open_ephemeral(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self, Error> {
        let desc = MTLBinaryArchiveDescriptor::new();
        let archive = device
            .newBinaryArchiveWithDescriptor_error(&desc)
            .map_err(|_| Error::Gpu("MTLBinaryArchive create failed"))?;
        Ok(Self {
            inner: Mutex::new(ArchiveInner {
                archive,
                compiled: HashMap::new(),
                dirty: false,
            }),
            cache_file: PathBuf::new(),
            persist: false,
            verbose: false,
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
            .map_err(|_| Error::Gpu("Metal pipeline compile failed"))?;

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
                if self.verbose {
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
