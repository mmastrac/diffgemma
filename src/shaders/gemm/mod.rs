//! Dense / grouped / tunable GEMM kernels. `gemm.metal` holds the engine's
//! simple f32×bf16 linear entry points (compiled via include_str! in
//! src/metal/{engine,gemm}.rs).
pub mod common;
pub mod gemm_linear_f32;
pub mod gemm_linear_grouped;
pub mod gemm_q8_linear_f32;
pub mod gemm_rowk;
pub mod gemm_tunable;
