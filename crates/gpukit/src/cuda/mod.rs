//! CUDA backend: driver-API context, module/kernel cache, buffers, dispatch.
//!
//! The driver is resolved at runtime (see the driver module), so this backend
//! compiles -- and type-checks -- on hosts without CUDA. Only a dispatch needs
//! the real libcuda.

mod buffer;
mod cached;
mod context;
mod dispatch;
pub mod driver;

pub use buffer::{BufferPool, DeviceBuffer};
pub use cached::{cached_context, cached_kernel, pod_bytes};
pub use context::{Context, ContextConfig, Kernel, Module};
pub use dispatch::{
    KernelArgs, THREADS_PER_BLOCK, div_up, launch_1d, launch_1d_ranged, launch_grid, launch_rows,
};
