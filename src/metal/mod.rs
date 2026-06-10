//! Metal GPU runtime (macOS only, `metal` feature).

mod buffer;
mod device;
mod gemm;

pub use gemm::{bf16_matmul_cpu, f32_to_bf16, Bf16Gemm};
