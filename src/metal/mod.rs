//! Metal GPU runtime (macOS only, `metal` feature).

mod attention;
mod batch;
mod batched_kernels;
mod buffer;
mod decoder;
mod decoder_layer;
mod device;
mod engine;
mod gemm;
mod kernels;
mod linear;
mod memory;
mod moe;
mod weights;

pub use memory::{estimate_decoder_forward, estimate_weight_cache, MemoryEstimate};

pub use attention::GpuAttention;
pub use decoder::{
    bench_forward, forward as decoder_forward, load_weight_cache, BenchConfig, GpuDecoderScratch,
};
pub use weights::GpuDecoderWeightCache;
pub use engine::GpuDecoderEngine;
pub use gemm::{bf16_matmul_cpu, f32_to_bf16, Bf16Gemm};
