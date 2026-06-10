//! Metal GPU runtime (macOS only, `metal` feature).

mod attention;
mod buffer;
mod device;
mod gemm;

pub use attention::GpuAttention;
pub use gemm::{bf16_matmul_cpu, f32_to_bf16, Bf16Gemm};
