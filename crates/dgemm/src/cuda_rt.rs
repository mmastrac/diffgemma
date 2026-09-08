//! Process-global CUDA context and kernel cache for the GEMM bodies.
//!
//! Context creation is expensive and module loading is not free, so both are
//! cached for the process. The context makes itself current per operation, so
//! parallel tests may dispatch concurrently.

use gpukit::cuda::{Context, ContextConfig, Kernel};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// The process-wide CUDA context, opened on first use.
pub fn context() -> Result<&'static Context, gpukit::Error> {
    static CTX: OnceLock<Result<Context, String>> = OnceLock::new();
    match CTX.get_or_init(|| Context::new(ContextConfig::default()).map_err(|e| e.to_string())) {
        Ok(ctx) => Ok(ctx),
        Err(msg) => Err(gpukit::Error::Cuda(msg.clone())),
    }
}

/// Resolve a kernel entry point from the cubin image, caching by
/// (image address, entry).
pub fn kernel(cubin: &'static [u8], entry: &'static str) -> Result<Kernel, gpukit::Error> {
    static CACHE: OnceLock<Mutex<HashMap<(usize, &'static str), Kernel>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (cubin.as_ptr() as usize, entry);
    if let Some(existing) = cache.lock().unwrap().get(&key) {
        return Ok(existing.clone());
    }
    let ctx = context()?;
    let module = ctx.load_module(cubin)?;
    let kernel = module.function(entry)?;
    cache.lock().unwrap().insert(key, kernel.clone());
    Ok(kernel)
}

/// Bytes of a repr(C) POD value, for binding as one kernel argument.
pub fn pod_bytes<T: Copy>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(std::ptr::from_ref(value).cast::<u8>(), size_of::<T>()) }
}
