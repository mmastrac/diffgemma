//! Metal backend: context, specialization, pipeline cache, buffers, dispatch.

mod buffer;
mod context;
mod dispatch;
mod expand;
mod pipeline_cache;

pub use buffer::BufferPool;
pub use context::{ComputePipeline, Context, FcValues, source_hash};
pub use dispatch::{
    dispatch_1d, dispatch_1d_ranged, dispatch_grid, dispatch_rows, div_up, set_bytes,
};
pub use expand::expand;
pub use pipeline_cache::{CacheConfig, PipelineArchiveCache};
