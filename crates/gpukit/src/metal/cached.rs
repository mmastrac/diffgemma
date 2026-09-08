//! Per-thread memoized Metal context and pipeline cache.
//!
//! Context creation and MSL compilation are both expensive, and a step
//! dispatches the same handful of kernels thousands of times, so both are
//! cached per thread (a Metal context and its pipeline objects are not Send).
//! Callers that pass the same [`CacheConfig`] share one context and one
//! pipeline cache per thread.

use super::{CacheConfig, ComputePipeline, Context};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

thread_local! {
    static CONTEXT: RefCell<Option<Rc<Context>>> = const { RefCell::new(None) };
    static PIPELINES: RefCell<HashMap<(&'static str, &'static str), Rc<ComputePipeline>>> =
        RefCell::new(HashMap::new());
}

/// This thread's Metal context, created on first use.
///
/// The first caller's `cache` wins for the thread; a later caller asking for a
/// different configuration reuses the cached context.
pub fn cached_context(cache: CacheConfig) -> Result<Rc<Context>, crate::Error> {
    CONTEXT.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            let ctx = Context::new(cache)?;
            *slot = Some(Rc::new(ctx));
        }
        Ok(Rc::clone(slot.as_ref().expect("just set")))
    })
}

/// A compiled pipeline for (source, entry), compiled at most once per thread.
pub fn cached_pipeline(
    ctx: &Context,
    source: &'static str,
    entry: &'static str,
) -> Result<Rc<ComputePipeline>, crate::Error> {
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
