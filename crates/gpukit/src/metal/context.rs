use crate::Error;
use crate::metal::expand::expand;
use crate::metal::pipeline_cache::{CacheConfig, PipelineArchiveCache};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLCommandQueue, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice,
    MTLFunctionConstantValues, MTLLibrary,
};
use std::sync::Arc;

/// Stable (no-random-seed) hash of a shader source for cache-label derivation.
pub fn source_hash(source: &str) -> u64 {
    // FNV-1a: deterministic across runs (std DefaultHasher is randomized).
    let mut h: u64 = 0xcbf29ce484222325;
    for b in source.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Function-constant assignments for one pipeline specialization.
///
/// [`Context::compile_specialized`] folds every value here, plus the source
/// hash, into the cache label — so two specializations can only share a
/// pipeline when their full input set is identical. An axis that bypasses
/// this struct (a source `#define`, a differing header table) must differ in
/// source text, which the hash covers.
#[derive(Clone, Default)]
pub struct FcValues {
    bools: Vec<(u32, bool)>,
    uints: Vec<(u32, u32)>,
    ulongs: Vec<(u32, u64)>,
}

impl FcValues {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_bool(&mut self, index: u32, value: bool) -> &mut Self {
        self.bools.push((index, value));
        self
    }

    pub fn set_uint(&mut self, index: u32, value: u32) -> &mut Self {
        self.uints.push((index, value));
        self
    }

    pub fn set_ulong(&mut self, index: u32, value: u64) -> &mut Self {
        self.ulongs.push((index, value));
        self
    }

    fn apply(&self, fc: &MTLFunctionConstantValues) {
        unsafe {
            for (index, value) in &self.bools {
                fc.setConstantValue_type_atIndex(
                    std::ptr::NonNull::from_ref(value).cast(),
                    MTLDataType::Bool,
                    *index as usize,
                );
            }
            for (index, value) in &self.uints {
                fc.setConstantValue_type_atIndex(
                    std::ptr::NonNull::from_ref(value).cast(),
                    MTLDataType::UInt,
                    *index as usize,
                );
            }
            for (index, value) in &self.ulongs {
                fc.setConstantValue_type_atIndex(
                    std::ptr::NonNull::from_ref(value).cast(),
                    MTLDataType::ULong,
                    *index as usize,
                );
            }
        }
    }

    fn label_suffix(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        for (index, value) in &self.bools {
            let _ = write!(s, "_b{index}={}", u8::from(*value));
        }
        for (index, value) in &self.uints {
            let _ = write!(s, "_u{index}={value}");
        }
        for (index, value) in &self.ulongs {
            let _ = write!(s, "_l{index}={value}");
        }
        s
    }
}

/// A Metal device + command queue plus the compile/cache configuration.
pub struct Context {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    includes: Vec<(&'static str, &'static str)>,
    cache: Arc<PipelineArchiveCache>,
}

impl Context {
    /// Open the system default device. Shared headers come from the folders
    /// registered with [`crate::register_includes!`]. The first context in
    /// the process fixes the pipeline cache's settings; later contexts share
    /// that cache and their `cache` config is ignored.
    pub fn new(cache: CacheConfig) -> Result<Self, Error> {
        let includes = crate::includes::include_table()?;
        let device = MTLCreateSystemDefaultDevice().ok_or(Error::Gpu("no Metal device"))?;
        let queue = device
            .newCommandQueue()
            .ok_or(Error::Gpu("failed to create Metal command queue"))?;
        let cache = PipelineArchiveCache::shared(&device, &cache)?;
        Ok(Self {
            device,
            queue,
            includes,
            cache,
        })
    }

    /// Compile a source string (quoted includes expanded) to a library.
    pub fn compile_library(
        &self,
        source: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, Error> {
        let ns_source = NSString::from_str(&expand(source, &self.includes));
        self.device
            .newLibraryWithSource_options_error(&ns_source, None)
            .map_err(compile_error)
    }

    /// Compile one unspecialized kernel. Cached by entry name alone, so an
    /// entry name must map to one source body process-wide.
    pub fn compile_kernel(&self, source: &str, entry: &str) -> Result<ComputePipeline, Error> {
        let library = self.compile_library(source)?;
        self.compile_kernel_from_library(&library, entry)
    }

    /// As [`Self::compile_kernel`], for an already-compiled library.
    pub fn compile_kernel_from_library(
        &self,
        library: &ProtocolObject<dyn MTLLibrary>,
        entry: &str,
    ) -> Result<ComputePipeline, Error> {
        let name = NSString::from_str(entry);
        let function = library
            .newFunctionWithName(&name)
            .ok_or(Error::Gpu("Metal kernel entry not found"))?;
        let pipeline = self.cache.compile_compute(&self.device, &function, entry)?;
        Ok(ComputePipeline { pipeline })
    }

    /// Compile several entry points from one source.
    pub fn compile_kernels(
        &self,
        source: &str,
        entries: &[&str],
    ) -> Result<Vec<ComputePipeline>, Error> {
        let library = self.compile_library(source)?;
        entries
            .iter()
            .map(|entry| self.compile_kernel_from_library(&library, entry))
            .collect()
    }

    /// Compile a kernel specialized by function constants. The cache label is
    /// `label_prefix` + every constant value + the source hash (see
    /// [`FcValues`]); `label_prefix` is for human readability, not uniqueness.
    pub fn compile_specialized(
        &self,
        source: &str,
        entry: &str,
        values: &FcValues,
        label_prefix: &str,
    ) -> Result<ComputePipeline, Error> {
        let library = self.compile_library(source)?;
        let fc = MTLFunctionConstantValues::new();
        values.apply(&fc);
        let name = NSString::from_str(entry);
        let function = library
            .newFunctionWithName_constantValues_error(&name, &fc)
            .map_err(compile_error)?;
        let label = format!(
            "{label_prefix}{}_s{:x}",
            values.label_suffix(),
            source_hash(source)
        );
        let pipeline = self
            .cache
            .compile_compute(&self.device, &function, &label)?;
        Ok(ComputePipeline { pipeline })
    }
}

pub struct ComputePipeline {
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl Clone for ComputePipeline {
    fn clone(&self) -> Self {
        Self {
            pipeline: self.pipeline.clone(),
        }
    }
}

fn compile_error(err: Retained<NSError>) -> Error {
    Error::Compile(err.localizedDescription().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = "\
#include <metal_stdlib>
using namespace metal;
kernel void pc_k0(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 0.0f; }
kernel void pc_k1(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 1.0f; }
kernel void pc_k2(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 2.0f; }
kernel void pc_k3(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 3.0f; }
";

    fn test_config() -> CacheConfig {
        CacheConfig {
            enabled: true,
            dir: Some(std::env::temp_dir().join("gpukit-test-pipeline-cache")),
            namespace: "gpukit-test",
            key: 0x67706b74,
            verbose: false,
        }
    }

    /// Hammer the process-global pipeline archive from many threads at once.
    /// Same entry across threads exercises the dedup fast path; different entries
    /// race concurrent `addComputePipelineFunctions`. Unsynchronized, this races
    /// the non-thread-safe `MTLBinaryArchive` and SIGSEGVs.
    #[test]
    fn concurrent_compile_is_race_free() {
        if Context::new(test_config()).is_err() {
            eprintln!("skip: no Metal device");
            return;
        }
        let entries = ["pc_k0", "pc_k1", "pc_k2", "pc_k3"];
        let handles: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(move || {
                    let ctx = Context::new(test_config()).expect("device");
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
