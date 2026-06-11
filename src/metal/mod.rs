//! Metal GPU runtime (macOS only, `metal` feature).

mod attention;
mod buffer;
mod decoder;
mod decoder_layer;
mod device;
mod engine;
mod gemm;
mod kernels;
mod linear;
mod moe;

pub use attention::GpuAttention;
pub use decoder::{forward as decoder_forward, GpuDecoderScratch};
pub use engine::GpuDecoderEngine;
pub use gemm::{bf16_matmul_cpu, f32_to_bf16, Bf16Gemm};
