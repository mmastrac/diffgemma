//! Portable GPU ops with CPU oracles, dispatched to Metal on macOS and CUDA
//! elsewhere when the cuda feature is on.
//!
//! Every op is one logical kernel with one CPU reference implementation. The
//! per-backend source bodies live beside it (op.metal / op.cu) and are expected
//! to compute the same function; the tier-1 tests in each module pin the GPU
//! result to the CPU reference, which is the oracle.

// Fixtures always have a length and are never empty; an is_empty twin would
// be meaningless.
#![allow(clippy::len_without_is_empty)]

pub mod backend;
#[cfg(feature = "cuda")]
pub(crate) mod cuda_rt;
pub mod kernel;
#[cfg(target_os = "macos")]
pub(crate) mod metal_rt;
pub mod ops;
pub mod testing;

pub use gpukit::Error;
