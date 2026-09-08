//! Shared Metal context and per-thread pipeline cache for the GEMM bodies.
//!
//! Context creation and MSL compilation are both expensive, and a model step
//! dispatches the same kernel thousands of times, so both are cached per
//! thread (a Metal context and its pipeline objects are not Send). No include
//! table is used; no on-disk archive is touched.

use gpukit::metal::{CacheConfig, ComputePipeline, Context, ContextConfig};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

thread_local! {
    static CONTEXT: RefCell<Option<Rc<Context>>> = const { RefCell::new(None) };
    static PIPELINES: RefCell<HashMap<(&'static str, &'static str), Rc<ComputePipeline>>> =
        RefCell::new(HashMap::new());
}

/// This thread's Metal context, created on first use.
pub fn context() -> Result<Rc<Context>, gpukit::Error> {
    CONTEXT.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            let ctx = Context::new(ContextConfig {
                includes: &[],
                cache: CacheConfig {
                    enabled: false,
                    dir: None,
                    namespace: "dgemm",
                    key: 0x6467_6d6d,
                    verbose: false,
                },
            })?;
            *slot = Some(Rc::new(ctx));
        }
        Ok(Rc::clone(slot.as_ref().expect("just set")))
    })
}

/// A compiled pipeline for (source, entry), compiled at most once per thread.
pub fn pipeline(
    ctx: &Context,
    source: &'static str,
    entry: &'static str,
) -> Result<Rc<ComputePipeline>, gpukit::Error> {
    PIPELINES.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(existing) = cache.get(&(source, entry)) {
            return Ok(Rc::clone(existing));
        }
        let compiled = Rc::new(ctx.compile_kernel(source, entry)?);
        cache.insert((source, entry), Rc::clone(&compiled));
        Ok(compiled)
    })
}
