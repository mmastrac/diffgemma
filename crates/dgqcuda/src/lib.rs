//! Library surface for tests: the CUDA kernel source and a cuBLAS probe.

#[cfg(feature = "cuda")]
pub const KERNELS: &str = include_str!("kernels.cu");

pub mod chat_template;
pub mod config;
pub mod denoise;
pub mod forward;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod weights;

#[cfg(feature = "cuda")]
pub use gpu::cublas_probe;
pub mod moe_grouped;
pub mod tokenizer;
